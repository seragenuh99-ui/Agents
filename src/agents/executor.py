"""Executor Agent: processes data, runs code in sandbox, and performs computations."""

from __future__ import annotations

import json
import subprocess
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

from .base import BaseAgent, LLMBackend, LLMError
from ..task_intent import is_open_qa
from ..protocol import Message, MessageType, ActionType
from ..protocol.scheduler import Scheduler
from ..state.embeddings import EmbeddingEngine
from ..state.exchange import StateExchangeBus
from ..memory.store import MemoryStore


class ExecutorAgent(BaseAgent):
    """Agent responsible for data processing, computation, and code execution.

    Capabilities: execute, process, compute, transform, store_memory
    Supports CodeAct mode: LLM generates Python code that runs in a sandbox.
    """

    def __init__(
        self,
        scheduler: Scheduler,
        llm: LLMBackend,
        embedding_engine: EmbeddingEngine,
        state_bus: StateExchangeBus,
        memory_store: MemoryStore,
        use_structured_protocol: bool = True,
        sandbox_enabled: bool = True,
    ):
        super().__init__(
            agent_id="executor",
            role="executor",
            capabilities=["execute", "process", "compute", "transform", "store_memory"],
            scheduler=scheduler,
            llm=llm,
            embedding_engine=embedding_engine,
            state_bus=state_bus,
            memory_store=memory_store,
            use_structured_protocol=use_structured_protocol,
        )
        self.sandbox_enabled = sandbox_enabled

    def handle_message(self, message: Message) -> Optional[Message]:
        if message.msg_type == MessageType.REQUEST:
            if message.action == ActionType.EXECUTE:
                return self._handle_execute_request(message)
        return None

    def _handle_execute_request(self, message: Message) -> Optional[Message]:
        result = self.execute_task(message.params)
        return self.scheduler.send_response(
            message,
            result=result,
            embedding=self.embedding_engine.encode_state(result),
            memory_refs=result.get("memory_refs", []),
        )

    def execute_task(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a processing or computation task.

        Args:
            task_input: Contains 'input' (what to process), 'data' (input data),
                       and optionally 'code' for CodeAct mode.
        """
        t0 = time.time()
        task_desc = task_input.get("input", task_input.get("description", ""))
        input_data = task_input.get("data", {})
        task_id = task_input.get("task_id", "unknown")
        provided_code = task_input.get("code")
        retrieval_results = task_input.get("retrieval_results", {})
        self._context["task_id"] = task_id

        result: Dict[str, Any] = {
            "task": task_desc,
            "memory_refs": [],
            "output": "",
            "error": None,
            "code_executed": False,
        }

        # If retrieval results are provided, use them
        if retrieval_results:
            result["input_from_retrieval"] = retrieval_results

        task_lower = task_desc.lower()
        task_tags = task_input.get("tags", [])
        open_qa = is_open_qa(task_desc, task_tags)
        needs_code = any(
            k in task_lower
            for k in ("code", "compute", "script", "calculate", "implement", "run ")
        )
        kb_hits = (
            retrieval_results.get("knowledge_base_hits", [])
            if isinstance(retrieval_results, dict)
            else []
        )

        if open_qa and not needs_code and not provided_code:
            self._emit_status("③ 执行：开放题，调用模型整理检索要点…")
            result["output"] = self._llm_open_qa_organize(
                task_desc, retrieval_results
            )
            result["code_executed"] = False
        # AgentPrune-style: skip CodeAct LLM when KB already answers (research tasks)
        elif kb_hits and not needs_code and not provided_code:
            result["output"] = self._process_manually(
                task_desc, input_data, retrieval_results
            )
            result["code_executed"] = False
        elif self.sandbox_enabled:
            code = provided_code or self._generate_code(task_desc, input_data)
            if code:
                exec_result = self._run_sandbox(code, {
                    "input_data": input_data,
                    "retrieval_results": retrieval_results,
                })
                result["output"] = exec_result.get("output", "")
                result["error"] = exec_result.get("error")
                result["code"] = code
                result["code_executed"] = True
            else:
                result["output"] = self._process_manually(
                    task_desc, input_data, retrieval_results
                )
        else:
            result["output"] = self._process_manually(
                task_desc, input_data, retrieval_results
            )

        result["elapsed_ms"] = (time.time() - t0) * 1000

        # Store execution results in shared memory
        memory_id = self.store_memory(
            topic=f"Execution: {task_desc[:80]}",
            summary=f"Executed: {task_desc[:150]}",
            content=json.dumps(result, ensure_ascii=False, indent=2),
            tags=["execution", "processing"],
            memory_type="result",
        )
        result["memory_refs"].append(memory_id)

        # Transfer execution state embedding
        self.transfer_state(
            target_agent="summarizer",
            state_data={
                "execution_result": result.get("output", ""),
                "task": task_desc,
                "success": result.get("error") is None,
            },
            context=f"Execution result for: {task_desc[:100]}",
        )

        self._task_history.append({
            "task_id": task_id,
            "task": task_desc,
            "success": result.get("error") is None,
        })
        return result

    def _generate_code(self, task_desc: str, input_data: Dict[str, Any]) -> Optional[str]:
        """Use LLM to generate Python code for the task (CodeAct mode)."""
        system_prompt = """You are a code generation agent. Write Python code to accomplish the given task.
Output ONLY the Python code, no explanations. The code should:
1. Be safe and not access network or file system beyond what's needed
2. Return results by printing JSON to stdout
3. Handle errors gracefully"""
        user_prompt = f"Task: {task_desc}\nAvailable input data: {json.dumps(input_data)[:500]}"

        try:
            self._emit_status("③ 执行：调用 DeepSeek 生成代码…")
            code = self._call_llm(
                system_prompt,
                user_prompt,
                temperature=0.1,
                max_tokens=1024,
                status_hint="③ 执行：DeepSeek 代码生成",
            )
        except LLMError:
            return None

        # Clean up code block markers
        if code.startswith("```python"):
            code = code[9:]
        if code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]

        return code.strip() or None

    def _run_sandbox(self, code: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute code in a lightweight sandbox.

        Uses a subprocess with restricted namespace for isolation.
        """
        # Build execution context with limited builtins
        safe_builtins = {
            "print": print,
            "len": len,
            "range": range,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "type": type,
            "isinstance": isinstance,
            "enumerate": enumerate,
            "zip": zip,
            "map": map,
            "filter": filter,
            "sorted": sorted,
            "reversed": reversed,
            "min": min,
            "max": max,
            "sum": sum,
            "abs": abs,
            "round": round,
            "json": json,
            "any": any,
            "all": all,
        }

        try:
            import io
            stdout = io.StringIO()
            stderr = io.StringIO()

            # Execute in subprocess for proper isolation
            proc = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=10,
                env={
                    "PYTHONPATH": "",
                    "HOME": "/tmp",
                    "PATH": "/usr/bin:/usr/local/bin",
                },
                cwd="/tmp",
            )

            return {
                "output": (proc.stdout or "") + (proc.stderr or ""),
                "returncode": proc.returncode,
                "error": proc.stderr if proc.returncode != 0 else None,
            }
        except subprocess.TimeoutExpired:
            return {"output": "", "error": "Code execution timed out (10s limit)"}
        except Exception as e:
            return {"output": "", "error": f"Sandbox error: {str(e)}"}

    def _llm_open_qa_organize(
        self, task_desc: str, retrieval_results: Dict[str, Any]
    ) -> str:
        """Turn retrieval + LLM research into an outline for the summarizer."""
        evidence = ""
        if isinstance(retrieval_results, dict):
            evidence = retrieval_results.get("combined_results", "") or ""
        cjk = any("\u4e00" <= c <= "\u9fff" for c in task_desc)
        if cjk:
            system = (
                "将下列检索材料整理成条理清晰的中文提纲（分条），供后续写最终回答。"
                "保留有用信息，删去无关英文技术内容。"
            )
        else:
            system = (
                "Organize the evidence into a clear bullet outline for a final answer. "
                "Drop irrelevant material."
            )
        user = f"问题：{task_desc[:400]}\n\n材料：\n{evidence[:2000]}"
        try:
            return self._call_llm(
                system,
                user,
                max_tokens=600,
                temperature=0.3,
                status_hint="③ 执行：开放题要点整理",
            )
        except LLMError:
            return self._process_manually(task_desc, {}, retrieval_results)

    def _process_manually(
        self,
        task_desc: str,
        input_data: Dict[str, Any],
        retrieval_results: Dict[str, Any],
    ) -> str:
        """Process data without code execution (textual processing)."""
        parts = [f"Processing task: {task_desc}"]

        if retrieval_results:
            kb_hits = retrieval_results.get("knowledge_base_hits", [])
            memory_hits = retrieval_results.get("memory_hits", [])
            if kb_hits:
                parts.append("\nFrom Knowledge Base:")
                for hit in kb_hits[:3]:
                    parts.append(f"  - {hit.get('summary', str(hit)[:200])}")
            if memory_hits:
                parts.append("\nFrom Shared Memory:")
                for hit in memory_hits[:3]:
                    parts.append(f"  - {hit.get('summary', '')[:200]}")

        if input_data:
            parts.append(f"\nInput data: {json.dumps(input_data)[:500]}")

        return "\n".join(parts)
