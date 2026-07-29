"""Executor Agent — 数据处理、沙盒代码执行与计算。

ExecutorAgent 负责执行任务中的数据处理和计算步骤，支持两种模式：

1. CodeAct 模式：LLM 生成 Python 代码 → AST 安全验证 → 子进程沙盒执行
   安全机制：
   - AST 级别：CodeValidator 拦截危险导入和调用（eval/exec/open/__import__ 等）
   - OS 级别：子进程隔离（受限环境变量、/tmp 工作目录、10s 超时）
   - 双重防御确保代码安全执行

2. 手动处理模式：文本级数据整理（无需代码执行时）

状态消费：
  从 StateExchangeBus 拉取 Retriever 的嵌入向量，用于主动搜索
  相关历史执行结果，跳过冗余的"文本→编码"步骤。

开放题处理：
  整理检索结果和 LLM 研究为结构化提纲，供 Summarizer 生成最终回答。
"""

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
from ..sandbox.executor import validate_code


class ExecutorAgent(BaseAgent):
    """数据处理与代码执行 Agent。

    核心能力：execute（执行）, process（处理）, compute（计算）, transform（转换）, store_memory（记忆存储）

    执行模式：
    - CodeAct: LLM 生成代码 → validate_code() AST 验证 → subprocess 子进程隔离执行
    - 手动处理: 文本级数据整理（无代码需求或沙盒禁用时）
    - 开放题整理: 将检索材料整理为提纲
    - AgentPrune 跳过: KB 已充分回答时（研究类任务），跳过 CodeAct LLM
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
        self.sandbox_enabled = sandbox_enabled  # 沙盒执行开关

    # ---- 消息处理 ----

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

    # ---- 核心任务执行 ----

    def execute_task(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """执行数据处理或计算任务。

        Args:
            task_input: 包含 input（任务描述）, data（输入数据）,
                       可选 code（CodeAct 模式）, retrieval_results（检索结果）

        Returns:
            包含 output, error, code_executed 等字段的执行结果
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

        # ---- 状态消费：使用 Retriever 的嵌入向量进行主动记忆搜索 ----
        # 跳过冗余的"文本→编码"步骤，直接使用上游已计算的嵌入向量
        _state_packets = self.receive_state_packets(limit=2)
        if _state_packets:
            _search_embs = (
                [p.embedding for p in _state_packets]
                if len(_state_packets) > 1
                else _state_packets[0].embedding
            )
            self.state_bus.mark_consumed([p.packet_id for p in _state_packets])
            try:
                _related = self.memory_store.search_by_similarity(
                    _search_embs, limit=3, memory_type="result"
                )
                if _related and not retrieval_results:
                    retrieval_results = {}
                if _related and isinstance(retrieval_results, dict):
                    _mb = retrieval_results.setdefault("memory_hits", [])
                    for _mem, _score in _related:
                        if _score > 0.4:
                            _mb.append({
                                "memory_id": _mem.memory_id,
                                "summary": _mem.summary[:200],
                                "type": _mem.memory_type,
                                "relevance": round(_score, 3),
                            })
            except Exception:
                pass

        if retrieval_results:
            result["input_from_retrieval"] = retrieval_results

        # ---- 路由：根据任务特征选择执行模式 ----
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
            # 开放题：LLM 整理检索要点
            self._emit_status("③ 执行：开放题，调用模型整理检索要点…")
            result["output"] = self._llm_open_qa_organize(task_desc, retrieval_results)
            result["code_executed"] = False
        elif kb_hits and not needs_code and not provided_code:
            # AgentPrune：KB 已充分回答，跳过 CodeAct LLM
            result["output"] = self._process_manually(task_desc, input_data, retrieval_results)
            result["code_executed"] = False
        elif self.sandbox_enabled:
            # CodeAct：LLM 生成代码 → AST 验证 → 子进程执行
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
                result["output"] = self._process_manually(task_desc, input_data, retrieval_results)
        else:
            # 沙盒禁用时手动处理
            result["output"] = self._process_manually(task_desc, input_data, retrieval_results)

        result["elapsed_ms"] = (time.time() - t0) * 1000

        # ---- 存储执行结果到共享记忆 ----
        memory_id = self.store_memory(
            topic=f"Execution: {task_desc[:80]}",
            summary=f"Executed: {task_desc[:150]}",
            content=json.dumps(result, ensure_ascii=False, indent=2),
            tags=["execution", "processing"],
            memory_type="result",
        )
        result["memory_refs"].append(memory_id)

        # ---- 状态传递：将执行嵌入发送给 Summarizer ----
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

    # ---- 代码生成 ----

    def _generate_code(self, task_desc: str, input_data: Dict[str, Any]) -> Optional[str]:
        """LLM 生成 Python 代码（CodeAct 模式）。

        要求 LLM 输出纯 Python 代码（无解释），以 JSON 格式打印结果到 stdout。
        """
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

        # 清理代码块标记
        if code.startswith("```python"):
            code = code[9:]
        if code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]

        return code.strip() or None

    # ---- 沙盒执行 ----

    def _run_sandbox(self, code: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """在轻量沙盒中执行代码。

        双重安全防御：
        1. AST 级别：CodeValidator.validate() 拦截危险调用
        2. OS 级别：子进程隔离（受限环境变量、/tmp 目录、10s 超时）
        """
        # AST 安全验证
        is_safe, violations = validate_code(code)
        if not is_safe:
            return {
                "output": "",
                "error": f"Code validation failed: {'; '.join(violations)}",
            }

        try:
            # 子进程隔离执行
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

    # ---- 开放题整理 ----

    def _llm_open_qa_organize(
        self, task_desc: str, retrieval_results: Dict[str, Any]
    ) -> str:
        """将检索结果 + LLM 研究整理为结构化提纲，供 Summarizer 生成最终回答。"""
        evidence = ""
        if isinstance(retrieval_results, dict):
            evidence = retrieval_results.get("combined_results", "") or ""
        cjk = any("一" <= c <= "鿿" for c in task_desc)
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

    # ---- 手动处理 ----

    def _process_manually(
        self,
        task_desc: str,
        input_data: Dict[str, Any],
        retrieval_results: Dict[str, Any],
    ) -> str:
        """无需代码执行时的文本级数据处理。"""
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
