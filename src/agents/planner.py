"""Planner Agent: decomposes complex tasks into subtask plans."""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from .base import BaseAgent, LLMBackend
from ..protocol import Message, MessageType, ActionType
from ..protocol.scheduler import Scheduler
from ..state.embeddings import EmbeddingEngine
from ..state.exchange import StateExchangeBus
from ..memory.store import MemoryStore
from ..memory.models import MemoryUnit
from ..task_intent import cache_intents_compatible, is_open_qa, is_open_qa


class PlannerAgent(BaseAgent):
    """Agent responsible for task decomposition and workflow planning.

    Capabilities: plan, decompose, prioritize, route
    """

    def __init__(
        self,
        scheduler: Scheduler,
        llm: LLMBackend,
        embedding_engine: EmbeddingEngine,
        state_bus: StateExchangeBus,
        memory_store: MemoryStore,
        use_structured_protocol: bool = True,
    ):
        super().__init__(
            agent_id="planner",
            role="planner",
            capabilities=["plan", "decompose", "route", "query_memory"],
            scheduler=scheduler,
            llm=llm,
            embedding_engine=embedding_engine,
            state_bus=state_bus,
            memory_store=memory_store,
            use_structured_protocol=use_structured_protocol,
        )
        self.run_options = None

    def handle_message(self, message: Message) -> Optional[Message]:
        if message.msg_type == MessageType.REQUEST:
            if message.action == ActionType.PLAN:
                return self._handle_plan_request(message)
            elif message.action == ActionType.QUERY_MEMORY:
                return self._handle_memory_query(message)
        return None

    def _handle_plan_request(self, message: Message) -> Optional[Message]:
        result = self.execute_task(message.params)
        return self.scheduler.send_response(
            message,
            result=result,
            embedding=self.embedding_engine.encode_state(result),
            memory_refs=result.get("memory_refs", []),
        )

    def _handle_memory_query(self, message: Message) -> Optional[Message]:
        query = message.params.get("query", "")
        tags = message.params.get("tags")
        memories = self.query_memory(query, tags)
        result = {
            "memories": [m.to_dict() for m in memories],
            "count": len(memories),
        }
        return self.scheduler.send_response(message, result=result)

    def execute_task(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """Decompose a complex task into subtasks and create execution plan.

        Args:
            task_input: Contains 'task_description', 'task_id', and optional 'context'

        Returns:
            A plan with ordered subtasks, each assigned to a specific agent role.
        """
        t0 = time.time()
        task_desc = task_input.get("task_description", "")
        task_id = task_input.get("task_id", "unknown")
        task_tags = task_input.get("tags", [])
        self._context["task_id"] = task_id

        # Query shared memory for relevant past plans
        memories = self.query_memory(
            query=f"plan {task_desc}",
            tags=["plan", "strategy"],
            limit=2,
        )
        # Merge proactive suggestions from orchestrator
        suggested = task_input.get("suggested_memories", [])
        suggested_memories = self.retrieve_memories([s["memory_id"] for s in suggested])
        seen_ids = {m.memory_id for m in memories}
        for sm in suggested_memories:
            if sm.memory_id not in seen_ids:
                memories.append(sm)
                seen_ids.add(sm.memory_id)

        # --- P1: Search for domain templates (abstraction_level >= 1) ---
        template_memories: List[MemoryUnit] = []
        if task_tags:
            template_memories = self.memory_store.get_templates(
                task_tags, memory_type="strategy", limit=3
            )
            # Filter to same-domain only: require primary tag (first tag) match
            primary_tag = task_tags[0] if task_tags else ""
            if primary_tag and len(task_tags) > 1:
                filtered = [t for t in template_memories if primary_tag in (t.tags or [])]
                template_memories = filtered
        memory_context = self._context_from_memories(memories)

        # --- Cache check: 4-tier plan retrieval (P0: LLM-as-judge for borderline) ---
        # Tier 1 (cos > 0.85): direct reuse, no LLM call
        # Tier 2 (cos 0.70-0.85): template fill, LLM adapts differences only
        # Tier 3 (cos 0.50-0.70): LLM-judged → if YES, template fill; if NO, full gen
        # Tier 4 (cos < 0.50): full LLM generation
        plan = None
        template_plan = None
        reused_from = None
        source_strategy_id: Optional[str] = None

        self._emit_status("① 规划：在记忆库中查找可复用计划…")

        # P1: Check domain templates first (tag-based, no embedding needed)
        for tmem in template_memories:
            if template_plan is not None:
                break
            try:
                candidate = json.loads(tmem.content)
                if isinstance(candidate, dict) and "subtask_pattern" in candidate:
                    # Adapt template format to plan format
                    adapted = {
                        "plan_id": f"template_{tmem.memory_id[:8]}",
                        "subtasks": candidate.get("subtask_pattern", []),
                        "expected_outcome": candidate.get("notes", ""),
                        "_template_score": 1.0,
                        "_from_template": True,
                        "_template_id": tmem.memory_id,
                    }
                    template_plan = adapted
                    self._emit_status("① 规划：命中领域模板，将做模板填充…")
                    break
            except (json.JSONDecodeError, TypeError):
                pass

        opts = self.run_options
        forced = task_input.get("forced_template_plan")
        forced_score = float(task_input.get("forced_template_score", 0.75))
        forced_mem_id = task_input.get("forced_template_memory_id")
        if isinstance(forced, dict) and template_plan is None:
            allow_forced = True
            if opts and opts.enable_intent_cache_gate:
                if forced_mem_id:
                    fmem = self.memory_store.get(forced_mem_id)
                    if fmem is not None:
                        allow_forced = cache_intents_compatible(
                            self._mem_task_desc(fmem),
                            task_desc,
                            fmem.tags if hasattr(fmem, "tags") else [],
                            task_tags,
                        )
                elif forced.get("task_description"):
                    allow_forced = cache_intents_compatible(
                        forced.get("task_description", ""),
                        task_desc,
                        forced.get("tags", []),
                        task_tags,
                    )
            if allow_forced:
                template_plan = forced.copy()
                template_plan["_template_score"] = forced_score
                template_plan["_source_strategy_id"] = (
                    forced.get("memory_id") or forced.get("plan_id")
                )

        use_cache = (
            opts is None or opts.enable_planner_cache
        ) and self.memory_store._index is not None and self.memory_store._index.ntotal > 0
        if use_cache:
            self._emit_status("① 规划：向量检索相似历史计划…")
            try:
                task_emb = self.embedding_engine.encode(task_desc)
                similar = self.memory_store.search_by_similarity(
                    task_emb, limit=10, memory_type="strategy"
                )
                for mem, score in similar:
                    cached_tags = mem.tags if hasattr(mem, "tags") else []
                    if not self._tags_overlap(task_tags, cached_tags):
                        continue
                    direct_th = opts.planner_direct_threshold if opts else 0.82
                    template_th = opts.planner_template_threshold if opts else 0.78
                    judge_th = opts.planner_judge_threshold if opts else 0.55
                    use_sieve = opts is None or opts.enable_safe_sieve
                    eff = (
                        self.memory_store.effective_similarity(mem, score)
                        if use_sieve
                        else score
                    )
                    if eff > direct_th and plan is None:
                        if opts and opts.enable_intent_cache_gate:
                            cached_task = self._mem_task_desc(mem)
                            if not cache_intents_compatible(
                                cached_task, task_desc, cached_tags, task_tags
                            ):
                                continue
                        try:
                            candidate = json.loads(mem.content)
                            if isinstance(candidate, dict) and "subtasks" in candidate:
                                plan = candidate.copy()
                                reused_from = mem.memory_id
                                source_strategy_id = mem.memory_id
                                self._emit_status(
                                    "① 规划：直接复用高相似度计划（跳过 LLM）"
                                )
                                break
                        except (json.JSONDecodeError, TypeError):
                            pass
                    elif eff > template_th and template_plan is None:
                        if opts and opts.enable_intent_cache_gate:
                            cached_task = self._mem_task_desc(mem)
                            if not cache_intents_compatible(
                                cached_task, task_desc, cached_tags, task_tags
                            ):
                                continue
                        try:
                            candidate = json.loads(mem.content)
                            if isinstance(candidate, dict) and "subtasks" in candidate:
                                template_plan = candidate.copy()
                                template_plan["_template_score"] = round(eff, 3)
                                template_plan["_source_strategy_id"] = mem.memory_id
                                source_strategy_id = mem.memory_id
                        except (json.JSONDecodeError, TypeError):
                            pass
                    elif eff > judge_th and template_plan is None:
                        cached_task = self._mem_task_desc(mem)
                        if opts and opts.enable_intent_cache_gate:
                            if not cache_intents_compatible(
                                cached_task, task_desc, cached_tags, task_tags
                            ):
                                continue
                        if self._llm_judge_similar(task_desc, cached_task):
                            try:
                                candidate = json.loads(mem.content)
                                if isinstance(candidate, dict) and "subtasks" in candidate:
                                    template_plan = candidate.copy()
                                    template_plan["_template_score"] = round(eff, 3)
                                    template_plan["_llm_judged"] = True
                                    template_plan["_source_strategy_id"] = mem.memory_id
                                    source_strategy_id = mem.memory_id
                                    break
                            except (json.JSONDecodeError, TypeError):
                                pass
            except Exception:
                pass

        # Open Q&A: fixed 3-step pipeline (no planner LLM; avoids wrong template reuse)
        if plan is None and is_open_qa(task_desc, task_tags):
            plan = self._generate_open_qa_plan(task_desc, task_id)
            self._emit_status("① 规划：开放题使用标准检索→整理→作答流程（无规划 API）")

        # If no direct cache hit, generate using LLM (template fill or full)
        if plan is None:
            # Unified system_prompt for DeepSeek KV-cache reuse — same for ALL calls.
            # Template structure and task specifics go in user_prompt.
            # Compact SNS-style system_prompt — short, identical for all calls.
            # Long prompts for DeepSeek caching were counterproductive:
            # they primed verbose completions and cost more than caching saved.
            system_prompt = "Plan task→subtasks. Output JSON:{subtasks:[{step,desc,role:retriever|executor|summarizer,action,params:{query},depends_on}],expected_outcome}"

            if memory_context and "No relevant memories" not in memory_context:
                ref_context = f"\nRef:{memory_context}"
            else:
                ref_context = ""

            if template_plan is not None:
                # Template fill: skeleton + prior task delta (Plan-Caching style, fewer tokens)
                steps = []
                for s in template_plan.get("subtasks", []):
                    role = s.get("role", s.get("agent_role", "?"))
                    action = s.get("action", "?")
                    deps = ",".join(str(d) for d in s.get("depends_on", [])) or "-"
                    steps.append(f"{role}/{action}[deps:{deps}]")
                compact = "|".join(steps)
                prior = (template_plan.get("task_description") or "")[:160]
                user_prompt = (
                    f"Task: {task_desc[:350]}\n"
                    f"Prior: {prior}\n"
                    f"Skeleton: {compact}\n"
                    f"Keep steps/roles; update params.query only.{ref_context}"
                )
            else:
                # Full generation
                user_prompt = f"Task: {task_desc[:400]}{ref_context}"

            try:
                if opts is not None and template_plan is not None:
                    max_tok = opts.planner_template_max_tokens
                elif opts is not None:
                    max_tok = opts.planner_full_max_tokens
                else:
                    max_tok = 256 if template_plan is not None else 448
                if template_plan is not None:
                    self._emit_status("① 规划：调用 DeepSeek 做模板填充…")
                    llm_hint = "① 规划：DeepSeek 模板填充"
                else:
                    self._emit_status("① 规划：调用 DeepSeek 生成完整子任务计划…")
                    llm_hint = "① 规划：DeepSeek 完整分解"
                plan = self._call_llm_structured(
                    system_prompt,
                    user_prompt,
                    {},
                    max_tokens=max_tok,
                    status_hint=llm_hint,
                )
                if isinstance(plan, list):
                    plan = {
                        "subtasks": plan,
                        "expected_outcome": task_desc[:120],
                    }
                if not isinstance(plan, dict) or "parse_error" in plan:
                    plan = self._generate_fallback_plan(task_desc, task_id)
            except Exception:
                plan = self._generate_fallback_plan(task_desc, task_id)

        if not isinstance(plan, dict):
            plan = self._generate_fallback_plan(task_desc, task_id)

        # Ensure every plan has a summarizer step (templates may omit it)
        subtasks = plan.get("subtasks", [])
        has_summarizer = any(
            (s.get("role") or s.get("agent_role", "")) == "summarizer"
            for s in subtasks
        )
        if not has_summarizer:
            max_step = max((s.get("step", 0) for s in subtasks), default=0)
            subtasks.append({
                "step": max_step + 1,
                "description": "Synthesize all findings into a final report",
                "role": "summarizer",
                "agent_role": "summarizer",
                "action": "summarize",
                "params": {"input": "Summarize all findings"},
                "depends_on": [s["step"] for s in subtasks],
            })
            plan["subtasks"] = subtasks

        # Override plan metadata for current task
        plan["plan_id"] = f"plan_{task_id}"
        plan["task_id"] = task_id
        plan["task_description"] = task_desc
        plan["generated_by"] = self.agent_id
        plan["elapsed_ms"] = (time.time() - t0) * 1000
        plan["llm_skipped"] = reused_from is not None or plan.get("_open_qa_plan", False)
        plan["_template_filled"] = template_plan is not None and reused_from is None
        if template_plan is not None:
            plan["_from_template"] = template_plan.get("_from_template", False)
        if template_plan is not None and reused_from is None:
            plan["_template_from"] = template_plan.get("plan_id", "unknown")
            plan["_template_score"] = template_plan.get("_template_score", 0)
        if source_strategy_id:
            plan["_source_strategy_id"] = source_strategy_id
        elif template_plan and template_plan.get("_source_strategy_id"):
            plan["_source_strategy_id"] = template_plan["_source_strategy_id"]
        plan["memory_refs"] = [m.memory_id for m in memories]
        if reused_from:
            plan["_reused_from"] = reused_from
            plan["memory_refs"].append(reused_from)

        # Store plan in shared memory (use task description for embedding match)
        memory_id = self.store_memory(
            topic=f"Plan: {task_desc[:80]}",
            summary=plan.get("expected_outcome", task_desc)[:200],
            content=json.dumps(plan, ensure_ascii=False, indent=2),
            tags=["plan", "strategy"] + task_input.get("tags", []),
            memory_type="strategy",
            evidence_chain=[m.memory_id for m in memories],
            embedding_text=task_desc,
        )
        plan["memory_id"] = memory_id

        # P1: Trigger template promotion after storing a strategy (async-capable)
        template_id = self._maybe_promote_to_template(
            tags=["plan", "strategy"] + task_input.get("tags", []),
            memory_type="strategy",
            threshold=2,
        )
        if template_id:
            plan["_promoted_template"] = template_id

        # Transfer plan state to coordinator via embedding
        self.transfer_state(
            target_agent="summarizer",
            state_data={"plan": plan, "task_id": task_id},
            context=f"Plan for task: {task_desc[:100]}",
        )

        self._task_history.append({"task_id": task_id, "plan": plan, "elapsed_ms": plan["elapsed_ms"]})
        return plan

    @staticmethod
    def _tags_overlap(task_tags: List[str], cached_tags: List[str]) -> bool:
        if not task_tags or not cached_tags:
            return True
        return any(t in cached_tags for t in task_tags)

    def _generate_open_qa_plan(self, task_desc: str, task_id: str) -> Dict[str, Any]:
        """Standard pipeline for custom / open-ended questions."""
        return {
            "plan_id": f"plan_{task_id}",
            "task_description": task_desc,
            "subtasks": [
                {
                    "step": 1,
                    "description": f"检索与问题相关的知识：{task_desc[:120]}",
                    "role": "retriever",
                    "agent_role": "retriever",
                    "action": "retrieve",
                    "params": {"query": task_desc},
                    "depends_on": [],
                },
                {
                    "step": 2,
                    "description": f"整理要点：{task_desc[:120]}",
                    "role": "executor",
                    "agent_role": "executor",
                    "action": "execute",
                    "params": {"input": task_desc},
                    "depends_on": [1],
                },
                {
                    "step": 3,
                    "description": f"生成完整回答：{task_desc[:120]}",
                    "role": "summarizer",
                    "agent_role": "summarizer",
                    "action": "summarize",
                    "params": {"input": task_desc},
                    "depends_on": [2],
                },
            ],
            "expected_outcome": task_desc[:200],
            "_open_qa_plan": True,
        }

    def _generate_fallback_plan(self, task_desc: str, task_id: str) -> Dict[str, Any]:
        """Generate a deterministic fallback plan when LLM is unavailable."""
        # Determine task type from description keywords
        task_lower = task_desc.lower()

        if any(w in task_lower for w in ["research", "report", "summarize", "analyze", "information"]):
            return {
                "plan_id": f"plan_{task_id}",
                "task_description": task_desc,
                "subtasks": [
                    {
                        "step": 1,
                        "description": f"Search for information about: {task_desc}",
                        "agent_role": "retriever",
                        "action": "retrieve",
                        "params": {"query": task_desc},
                        "depends_on": [],
                    },
                    {
                        "step": 2,
                        "description": "Process and organize retrieved information",
                        "agent_role": "executor",
                        "action": "execute",
                        "params": {"input": "Organize and filter results from step 1"},
                        "depends_on": [1],
                    },
                    {
                        "step": 3,
                        "description": "Generate summary report from processed data",
                        "agent_role": "summarizer",
                        "action": "summarize",
                        "params": {"input": "Create comprehensive report from step 2 results"},
                        "depends_on": [2],
                    },
                ],
                "expected_outcome": f"Comprehensive report on: {task_desc}",
            }
        elif any(w in task_lower for w in ["code", "program", "function", "debug", "fix"]):
            return {
                "plan_id": f"plan_{task_id}",
                "task_description": task_desc,
                "subtasks": [
                    {
                        "step": 1,
                        "description": "Analyze requirements and search for relevant code patterns",
                        "agent_role": "retriever",
                        "action": "retrieve",
                        "params": {"query": f"code pattern {task_desc}"},
                        "depends_on": [],
                    },
                    {
                        "step": 2,
                        "description": f"Write and execute code for: {task_desc}",
                        "agent_role": "executor",
                        "action": "execute",
                        "params": {"input": task_desc},
                        "depends_on": [1],
                    },
                    {
                        "step": 3,
                        "description": "Review and summarize the solution",
                        "agent_role": "summarizer",
                        "action": "summarize",
                        "params": {"input": "Review execution results and produce final report"},
                        "depends_on": [2],
                    },
                ],
                "expected_outcome": f"Working solution for: {task_desc}",
            }
        else:
            return {
                "plan_id": f"plan_{task_id}",
                "task_description": task_desc,
                "subtasks": [
                    {
                        "step": 1,
                        "description": f"Retrieve relevant information for: {task_desc}",
                        "agent_role": "retriever",
                        "action": "retrieve",
                        "params": {"query": task_desc},
                        "depends_on": [],
                    },
                    {
                        "step": 2,
                        "description": f"Execute processing for: {task_desc}",
                        "agent_role": "executor",
                        "action": "execute",
                        "params": {"input": task_desc},
                        "depends_on": [1],
                    },
                    {
                        "step": 3,
                        "description": f"Summarize results for: {task_desc}",
                        "agent_role": "summarizer",
                        "action": "summarize",
                        "params": {"input": "Summarize all findings"},
                        "depends_on": [2],
                    },
                ],
                "expected_outcome": f"Completed: {task_desc}",
            }
