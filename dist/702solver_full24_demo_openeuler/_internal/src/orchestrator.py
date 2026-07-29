"""多 Agent 任务编排器 — 协调 Planner、Retriever、Executor、Summarizer。

Orchestrator 是整个系统的中央协调器，管理 Agent 生命周期、任务管线执行、
指标收集和缓存策略。

核心执行管线（execute_task）：
  1. E2E 缓存检查 → 精确匹配时跳过整个管线
  2. 领域计划种子 → 从相似策略加载计划模板，避免完整 LLM 生成
  3. 规划（Planner）→ 任务分解为子任务
  4. 按依赖层级执行子任务 → 同角色同级并行（ThreadPoolExecutor）
  5. 综合（Summarizer）→ 生成最终报告

三级缓存体系：
  - E2E 精确缓存（cos >= 0.85）：跳过整个管线
  - 领域计划播种（cos >= 0.68）：模板填入，跳过完整规划
  - Executor 剔除（模板填入分 >= 0.65）：跳过执行 Agent

优化机制：
  - 主动记忆建议：执行前向 Agent 注入相关历史记忆
  - 批量嵌入搜索：encode_batch() 单次模型前向传递
  - SafeSieve-lite：记录模板使用结果，低成功率模板降权
  - 意图门控：防止不同意图类型间的缓存误用
  - AgentPrune：限制下游 Agent 的记忆建议数量
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import time
from typing import Any, Dict, List, Optional

from .agents.base import LLMBackend
from .agents.planner import PlannerAgent
from .agents.retriever import RetrieverAgent
from .agents.executor import ExecutorAgent
from .agents.summarizer import SummarizerAgent
from .protocol import ActionType
from .protocol import ProtocolParser
from .protocol.scheduler import Scheduler, AgentRegistry, MessageBus
from .state.embeddings import EmbeddingEngine
from .state.exchange import StateExchangeBus
from .memory.store import MemoryStore
from .evaluation.metrics import MetricsCollector, ComparisonReport
from .run_options import RunOptions, DEFAULT_OPTIONS, options_for_task
from .evaluation.quality_validator import validate_task
from .paths import DEFAULT_SHARED_MEMORY_DB
from .task_intent import (
    blocks_executor_dropout,
    cache_intents_compatible,
    is_open_qa,
    memory_task_description,
    requires_fresh_synthesis,
)


class Orchestrator:
    """多 Agent 任务编排器，支持结构化协议和文本模式。

    两种模式：
    - "structured"：紧凑结构化协议 + 非文本状态传递（本系统的核心创新）
    - "text"：传统自然语言 Agent 通信（基线对照）
    """

    def __init__(
        self,
        llm: Optional[LLMBackend] = None,
        mode: str = "structured",
        use_real_embeddings: bool = True,
        sandbox_enabled: bool = True,
        memory_db_path: Optional[str] = None,
        run_options: Optional[RunOptions] = None,
        embedding_model: Optional[str] = None,
    ):
        self.mode = mode
        self.run_options = run_options or DEFAULT_OPTIONS
        if memory_db_path is None:
            memory_db_path = DEFAULT_SHARED_MEMORY_DB
        model_name = embedding_model or os.environ.get(
            "EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"
        )

        # ---- 核心基础设施 ----
        self.registry = AgentRegistry()
        self.message_bus = MessageBus()
        self.scheduler = Scheduler(self.registry, self.message_bus)
        self.embedding_engine = EmbeddingEngine(
            model_name=model_name,
            use_real_model=use_real_embeddings,
        )
        self.state_bus = StateExchangeBus(self.embedding_engine)
        self.memory_store = MemoryStore(db_path=memory_db_path)
        self.metrics = MetricsCollector()
        self._progress_fn: Optional[Any] = None

        # LLM 后端
        self.llm = llm or LLMBackend()

        # ---- 创建 Agent ----
        use_structured = (mode == "structured")
        self.planner = PlannerAgent(
            self.scheduler, self.llm, self.embedding_engine,
            self.state_bus, self.memory_store, use_structured,
        )
        self.retriever = RetrieverAgent(
            self.scheduler, self.llm, self.embedding_engine,
            self.state_bus, self.memory_store, use_structured,
        )
        self.executor = ExecutorAgent(
            self.scheduler, self.llm, self.embedding_engine,
            self.state_bus, self.memory_store, use_structured,
            sandbox_enabled=sandbox_enabled,
        )
        self.summarizer = SummarizerAgent(
            self.scheduler, self.llm, self.embedding_engine,
            self.state_bus, self.memory_store, use_structured,
        )

        # 记忆索引开关
        if not self.run_options.enable_memory_index:
            self.memory_store._index = None
            self.memory_store._id_to_idx.clear()
            self.memory_store._idx_to_id.clear()

        # 注入 run_options 到各 Agent
        for agent in (self.planner, self.retriever, self.executor, self.summarizer):
            agent.run_options = self.run_options

        self.agents = {
            "planner": self.planner,
            "retriever": self.retriever,
            "executor": self.executor,
            "summarizer": self.summarizer,
        }

    # ============================================================
    # 依赖层级构建
    # ============================================================

    @staticmethod
    def _build_dependency_levels(subtasks: List[Dict]) -> List[List[Dict]]:
        """将子任务按拓扑排序分组为依赖层级，同级可并行执行。

        无依赖（depends_on 为空）的子任务在第一层；
        依赖已完成步骤的子任务在后续层级。
        """
        if not subtasks:
            return []

        completed: set = set()
        remaining = list(subtasks)
        levels: List[List[Dict]] = []

        while remaining:
            current_level = []
            next_remaining = []
            for st in remaining:
                deps = set(st.get("depends_on", []))
                if deps.issubset(completed):
                    current_level.append(st)
                else:
                    next_remaining.append(st)

            if not current_level:
                # 循环依赖或错误 → 剩余任务顺序执行
                levels.append(list(remaining))
                break

            levels.append(current_level)
            for st in current_level:
                completed.add(st["step"])
            remaining = next_remaining

        return levels

    # ============================================================
    # 主动记忆建议（单个 + 批量）
    # ============================================================

    def _suggest_memories(
        self, text: str, limit: int = 2, min_score: float = 0.35
    ) -> List[Dict[str, Any]]:
        """在执行前为 Agent 主动检索相关历史记忆。

        编码文本为嵌入向量 → FAISS 语义搜索 → 返回简洁建议字典列表。
        """
        if self.memory_store._index is None or self.memory_store._index.ntotal == 0:
            return []
        try:
            emb = self.embedding_engine.encode(text)
            results = self.memory_store.search_by_similarity(emb, limit=limit)
        except Exception:
            return []
        suggestions = []
        for mem, score in results:
            if score > min_score:
                suggestions.append({
                    "memory_id": mem.memory_id,
                    "summary": mem.summary[:200],
                    "type": mem.memory_type,
                    "relevance": round(score, 3),
                })
        return suggestions

    def _suggest_memories_batch(
        self, texts: List[str], limit: int = 2, min_score: float = 0.35
    ) -> List[List[Dict[str, Any]]]:
        """为多段文本批量检索记忆建议。

        使用 EmbeddingEngine.encode_batch() 单次模型前向传递，
        然后分别对每个嵌入做 FAISS 搜索。
        """
        if (not texts or self.memory_store._index is None
                or self.memory_store._index.ntotal == 0):
            return [[] for _ in texts]
        try:
            embs = self.embedding_engine.encode_batch(texts)
        except Exception:
            return [[] for _ in texts]
        all_suggestions: List[List[Dict[str, Any]]] = []
        for emb in embs:
            try:
                results = self.memory_store.search_by_similarity(emb, limit=limit)
            except Exception:
                all_suggestions.append([])
                continue
            suggestions = []
            for mem, score in results:
                if score > min_score:
                    suggestions.append({
                        "memory_id": mem.memory_id,
                        "summary": mem.summary[:200],
                        "type": mem.memory_type,
                        "relevance": round(score, 3),
                    })
            all_suggestions.append(suggestions)
        return all_suggestions

    # ============================================================
    # LLM 判断相似性
    # ============================================================

    def _llm_judge_similar(self, task_a: str, task_b: str) -> bool:
        """用 LLM 判断两个任务是否属于同一类工作。

        P0 "LLM-as-judge" 机制：FAISS 嵌入提供快速粗粒度召回（cos > 0.50），
        LLM 判断提供高精度验证。成本：约 50 prompt tokens + 3 completion tokens/次。
        """
        if not task_a or not task_b:
            return False
        try:
            system = "Task similarity judge. Answer YES if two tasks are the same kind of work (share domain, goal, or methodology). Answer NO if they are fundamentally different."
            user = f"Task A: {task_a[:200]}\nTask B: {task_b[:200]}\nSame kind of task? YES/NO:"
            response = self.llm.chat(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_tokens=5,
            )
            return "YES" in response.upper()
        except Exception:
            return False

    # ============================================================
    # 辅助工具
    # ============================================================

    def _extract_task_desc(self, mem) -> str:
        """从记忆的 task_topic 中提取干净的任务描述。"""
        topic = getattr(mem, 'task_topic', '') or ''
        for prefix in ["Summary: ", "Plan: ", "Execution: ", "Retrieval: "]:
            if topic.startswith(prefix):
                return topic[len(prefix):]
        return topic

    @staticmethod
    def _tags_overlap(task_tags: List[str], cached_tags: List[str]) -> bool:
        """检查两组标签是否有交集。"""
        if not task_tags or not cached_tags:
            return True
        return any(t in cached_tags for t in task_tags)

    def _parse_result_payload(self, mem) -> Optional[Dict[str, Any]]:
        """解析记忆内容中的结果负载。"""
        try:
            cached = json.loads(mem.content)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(cached, dict):
            return None
        summ = cached.get("summary")
        if isinstance(summ, dict) and not summ.get("_error"):
            return cached
        return None

    def _effective_score(self, mem, raw_score: float) -> float:
        """计算记忆的有效相似度分数（策略类型应用 SafeSieve 降权）。"""
        if mem.memory_type == "strategy":
            return self.memory_store.effective_similarity(mem, raw_score)
        return raw_score

    def _progress(self, msg: str) -> None:
        """向进度回调发送消息。"""
        if self._progress_fn:
            self._progress_fn(msg)

    def _llm_calls(self) -> int:
        """获取累计 LLM API 调用次数。"""
        return self.llm.get_usage_stats().get("call_count", 0)

    # ============================================================
    # E2E 缓存
    # ============================================================

    def _finish_e2e(
        self,
        *,
        task_id: str,
        task_description: str,
        cached: Dict[str, Any],
        mem,
        match_type: str,
        score: float,
        t0: float,
        msg_count_before: int,
    ) -> Dict[str, Any]:
        """完成 E2E 缓存命中：构建结果并结束指标记录。"""
        result = {
            "task_id": task_id,
            "task_description": task_description,
            "mode": self.mode,
            "steps": {"summary": cached},
            "retrieval_refs_count": len(cached.get("evidence_refs", [])),
            "execution_refs_count": 0,
            "elapsed_ms": (time.time() - t0) * 1000,
            "_e2e_cached": True,
            "_e2e_match": match_type,
            "_e2e_score": round(score, 3),
            "_e2e_from": mem.memory_id,
            "_e2e_judge": "cos",
        }
        self.metrics.end_task()
        return result

    def _try_e2e_reuse(
        self,
        task_id: str,
        task_description: str,
        tags: Optional[List[str]],
        t0: float,
        msg_count_before: int,
    ) -> Optional[Dict[str, Any]]:
        """尝试 E2E 缓存复用：任务嵌入足够相似时直接返回历史结果。

        两级匹配：
        - exact（cos >= e2e_threshold，默认 0.85）：完全复用
        - summarizer（cos >= summarizer_e2e_threshold）：Summarizer E2E 复用
        """
        if (
            not self.run_options.enable_e2e_cache
            or self.mode != "structured"
            or self.memory_store._index is None
            or self.memory_store._index.ntotal == 0
        ):
            return None

        task_emb = self.embedding_engine.encode(task_description)
        similar = self.memory_store.search_by_similarity(
            task_emb, limit=12, memory_type="result"
        )
        for mem, raw in similar:
            cached_tags = mem.tags if hasattr(mem, "tags") else []
            if tags and cached_tags and not self._tags_overlap(tags, cached_tags):
                continue
            if self.run_options.enable_intent_cache_gate:
                if requires_fresh_synthesis(task_description, tags):
                    continue
                if not cache_intents_compatible(
                    memory_task_description(mem),
                    task_description,
                    cached_tags,
                    tags,
                ):
                    continue
            score = self._effective_score(mem, raw)
            cached = self._parse_result_payload(mem)
            if not cached:
                continue

            if score >= self.run_options.e2e_threshold:
                return self._finish_e2e(
                    task_id=task_id, task_description=task_description,
                    cached=cached, mem=mem, match_type="exact",
                    score=score, t0=t0, msg_count_before=msg_count_before,
                )

            if (
                self.run_options.enable_summarizer_e2e
                and score >= self.run_options.summarizer_e2e_threshold
            ):
                return self._finish_e2e(
                    task_id=task_id, task_description=task_description,
                    cached=cached, mem=mem, match_type="summarizer",
                    score=score, t0=t0, msg_count_before=msg_count_before,
                )
        return None

    # ============================================================
    # 领域计划播种
    # ============================================================

    def _load_domain_plan_hint(
        self, task_description: str, tags: Optional[List[str]]
    ) -> tuple:
        """查找相似策略计划作为 Planner 的种子模板。

        跳过 LLM-as-judge（~50 tok/次），直接按 cos >= domain_plan_threshold
        匹配，大幅减少 API 调用。
        """
        if self.memory_store._index is None or self.memory_store._index.ntotal == 0:
            return None, 0.0, ""
        try:
            task_emb = self.embedding_engine.encode(task_description)
            similar = self.memory_store.search_by_similarity(
                task_emb, limit=8, memory_type="strategy"
            )
            for mem, raw in similar:
                cached_tags = mem.tags if hasattr(mem, "tags") else []
                if tags and cached_tags and not self._tags_overlap(tags, cached_tags):
                    continue
                if self.run_options.enable_intent_cache_gate:
                    if not cache_intents_compatible(
                        memory_task_description(mem),
                        task_description,
                        cached_tags,
                        tags,
                    ):
                        continue
                score = self._effective_score(mem, raw)
                if score < self.run_options.domain_plan_threshold:
                    continue
                try:
                    candidate = json.loads(mem.content)
                    if isinstance(candidate, dict) and "subtasks" in candidate:
                        return candidate, score, mem.memory_id
                except (json.JSONDecodeError, TypeError):
                    pass
        except Exception:
            pass
        return None, 0.0, ""

    # ============================================================
    # 子任务执行
    # ============================================================

    def _execute_subtask(
        self,
        subtask: Dict[str, Any],
        task_id: str,
        task_description: str,
        tags: List[str],
        plan: Dict[str, Any],
        retrieval_results: Dict[int, Any],
        execution_results: Dict[int, Any],
        retrieval_refs: Optional[List[str]] = None,
        execution_refs: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """执行单个子任务并返回结果。

        路由逻辑：
        - retriever → 检索子任务，含主动记忆建议
        - executor → 执行子任务（AgentPrune：executor 无记忆建议）
        - summarizer → 综合子任务（限制 1 条记忆建议，min_score=0.40）
        """
        role = subtask.get("agent_role", "")
        step = subtask.get("step", 0)
        role_cn = {
            "retriever": "② 检索",
            "executor": "③ 执行",
            "summarizer": "④ 总结",
        }.get(role, role)
        desc_snip = (subtask.get("description") or "")[:36]
        self._progress(f"{role_cn}：子任务 {step}（{desc_snip}…）")

        # 主动记忆建议 — 按角色限制数量（AgentPrune 风格）
        suggestion_text = subtask.get("description", "") + " " + str(subtask.get("params", {}))
        if role == "executor":
            suggested = []
        elif role == "summarizer":
            suggested = self._suggest_memories(suggestion_text, limit=1, min_score=0.40)
        else:
            suggested = self._suggest_memories(suggestion_text, limit=2, min_score=0.35)

        if role == "retriever":
            capable = self.scheduler.discover_capable_agents("retrieve")
            if capable and "retriever" not in capable:
                self._progress(f"警告：retriever 不支持 retrieve 能力，可用: {capable}")
            msg = self.scheduler.route_task(
                action=ActionType.RETRIEVE,
                params={
                    "query": subtask.get("params", {}).get("query", ""),
                    "task_id": task_id,
                    "tags": tags or [],
                    "suggested_memories": suggested,
                },
                from_agent="orchestrator",
                to_agent="retriever",
            )
            if self.run_options.enable_message_dispatch:
                responses = self.retriever.process_pending_messages()
                result = responses[0].result if responses else {}
            else:
                result = self.retriever.execute_task(msg.params)
                if result.get("error"):
                    self.scheduler.bus.send(ProtocolParser.create_error(msg, str(result["error"])))
                self.scheduler.send_response(msg, result=result)
            n = len(result.get("memory_hits", [])) + len(result.get("knowledge_base_hits", []))
            self._progress(f"② 检索：子任务 {step} 完成（命中 {n} 条）")
            return {"step": step, "role": role, "result": result}

        elif role == "executor":
            msg = self.scheduler.route_task(
                action=ActionType.EXECUTE,
                params={
                    "input": subtask.get("params", {}).get("input", subtask.get("description", "")),
                    "data": {"plan": plan, "retrieval_results": retrieval_results},
                    "task_id": task_id,
                    "suggested_memories": suggested,
                },
                from_agent="orchestrator",
                to_agent="executor",
            )
            if self.run_options.enable_message_dispatch:
                responses = self.executor.process_pending_messages()
                result = responses[0].result if responses else {}
            else:
                result = self.executor.execute_task(msg.params)
                if result.get("error"):
                    self.scheduler.bus.send(ProtocolParser.create_error(msg, str(result["error"])))
                self.scheduler.send_response(msg, result=result)
            self._progress(f"③ 执行：子任务 {step} 完成")
            return {"step": step, "role": role, "result": result}

        elif role == "summarizer":
            msg = self.scheduler.route_task(
                action=ActionType.SUMMARIZE,
                params={
                    "task_id": task_id,
                    "task_description": task_description or plan.get("task_description", ""),
                    "plan": plan,
                    "plan_ref": plan.get("memory_id", ""),
                    "retrieval_refs": retrieval_refs or [],
                    "execution_refs": execution_refs or [],
                    "tags": tags or [],
                    "retrieval_results": retrieval_results,
                    "execution_results": execution_results,
                    "suggested_memories": suggested,
                },
                from_agent="orchestrator",
                to_agent="summarizer",
            )
            if self.run_options.enable_message_dispatch:
                responses = self.summarizer.process_pending_messages()
                result = responses[0].result if responses else {}
            else:
                result = self.summarizer.execute_task(msg.params)
                if result.get("error"):
                    self.scheduler.bus.send(ProtocolParser.create_error(msg, str(result["error"])))
                self.scheduler.send_response(msg, result=result)
            self._progress(f"④ 总结：子任务 {step} 完成")
            return {"step": step, "role": role, "result": result}

        return {"step": step, "role": role, "result": {}}

    # ============================================================
    # 任务执行主入口
    # ============================================================

    def execute_task(self, task_id: str, task_description: str, tags: List[str] = None) -> Dict[str, Any]:
        """执行完整的多 Agent 任务管线。

        管线：Plan → [Retrieve || Execute]* → Summarize
        同级独立子任务并行执行（ThreadPoolExecutor）。
        """
        self.metrics.start_task(task_id, task_description, self.mode)
        t0 = time.time()
        msg_count_before = len(self.message_bus._history)
        calls_at_start = self._llm_calls()
        memories_before = self.memory_store.get_stats()["total_memories"]
        state_transfers_before = self.state_bus.get_stats()["total_transfers"]

        result = {
            "task_id": task_id,
            "task_description": task_description,
            "mode": self.mode,
            "steps": {},
        }

        # 禁止 E2E 缓存的场景
        forbid_e2e = (tags and "no-e2e-cache" in tags) or is_open_qa(task_description, tags)

        # 按任务类型调整 run_options
        task_opts = options_for_task(self.run_options, task_description, tags)
        for agent in self.agents.values():
            agent.run_options = task_opts
        if is_open_qa(task_description, tags):
            self._progress("开放题模式：跳过演示知识库，使用模型检索与长回答…")

        try:
            return self._execute_task_body(
                task_id, task_description, tags, t0,
                msg_count_before, calls_at_start,
                memories_before, state_transfers_before,
                result, forbid_e2e,
            )
        except Exception:
            self.metrics.end_task()  # 异常时也要结束指标记录
            raise
        finally:
            # 恢复默认 run_options
            for agent in self.agents.values():
                agent.run_options = self.run_options

    def _execute_task_body(
        self,
        task_id: str,
        task_description: str,
        tags: Optional[List[str]],
        t0: float,
        msg_count_before: int,
        calls_at_start: int,
        memories_before: int,
        state_transfers_before: int,
        result: Dict[str, Any],
        forbid_e2e: bool,
    ) -> Dict[str, Any]:
        """任务管线的主体逻辑。"""

        domain_template_plan: Optional[Dict[str, Any]] = None
        domain_match_score: float = 0.0
        domain_match_from: str = ""

        # ---- E2E 缓存检查 ----
        if not forbid_e2e:
            self._progress("检查是否可复用历史整题答案…")
            e2e_hit = self._try_e2e_reuse(task_id, task_description, tags, t0, msg_count_before)
            if e2e_hit is not None:
                self._progress("命中整题缓存，跳过 API 调用。")
                return e2e_hit

        # ---- 领域计划播种：从相似策略中加载计划模板 ----
        if not is_open_qa(task_description, tags):
            hint, hint_score, hint_id = self._load_domain_plan_hint(task_description, tags)
            if hint is not None:
                domain_template_plan = hint
                domain_match_score = hint_score
                domain_match_from = hint_id
        if (
            not is_open_qa(task_description, tags)
            and domain_template_plan is None
            and self.run_options.enable_domain_llm_judge
            and self.memory_store._index is not None
            and self.memory_store._index.ntotal > 0
        ):
            # 备选路径：通过 LLM 判断从 result 反查 strategy（默认关闭）
            try:
                task_emb = self.embedding_engine.encode(task_description)
                similar = self.memory_store.search_by_similarity(task_emb, limit=6, memory_type="result")
                for mem, raw in similar:
                    if raw <= self.run_options.planner_judge_threshold:
                        continue
                    cached_tags = mem.tags if hasattr(mem, "tags") else []
                    if tags and cached_tags and not self._tags_overlap(tags, cached_tags):
                        continue
                    if self.run_options.enable_intent_cache_gate:
                        if not cache_intents_compatible(
                            memory_task_description(mem), task_description, cached_tags, tags,
                        ):
                            continue
                    if self._llm_judge_similar(task_description, self._extract_task_desc(mem)):
                        try:
                            payload = json.loads(mem.content)
                        except (json.JSONDecodeError, TypeError):
                            payload = {}
                        for ref_id in payload.get("evidence_refs", []):
                            plan_mem = self.memory_store.get(ref_id)
                            if plan_mem and plan_mem.memory_type == "strategy":
                                domain_template_plan = json.loads(plan_mem.content)
                                domain_match_score = raw
                                domain_match_from = mem.memory_id
                                break
                    if domain_template_plan is not None:
                        break
            except Exception:
                pass

        # ---- Step 1：规划 ----
        plan_suggested = self._suggest_memories(task_description, limit=2, min_score=0.40)

        # E2E 领域匹配：将种子计划注入为最高优先级建议
        if domain_template_plan is not None:
            domain_plan_id = domain_template_plan.get("plan_id", "")
            domain_plan_mem_id = domain_template_plan.get("memory_id", "")
            plan_suggested.insert(0, {
                "memory_id": domain_plan_mem_id,
                "summary": f"Domain-matched plan (cos={domain_match_score:.3f}): "
                          f"{domain_template_plan.get('expected_outcome', '')[:150]}",
                "type": "strategy",
                "relevance": domain_match_score,
            })

        plan_params: Dict[str, Any] = {
            "task_description": task_description,
            "task_id": task_id,
            "tags": tags or [],
            "suggested_memories": plan_suggested,
        }
        if domain_template_plan is not None:
            plan_params["forced_template_plan"] = domain_template_plan
            plan_params["forced_template_score"] = domain_match_score
            plan_params["forced_template_memory_id"] = domain_match_from

        self.llm.on_status = self._progress
        t_plan = time.time()
        self._progress("① 规划：分解任务为子步骤…")
        plan_msg = self.scheduler.route_task(
            action=ActionType.PLAN,
            params=plan_params,
            from_agent="orchestrator",
            to_agent="planner",
        )
        if self.run_options.enable_message_dispatch:
            responses = self.planner.process_pending_messages()
            plan = responses[0].result if responses else self.planner._generate_fallback_plan(
                task_description, task_id
            )
        else:
            plan = self.planner.execute_task(plan_msg.params)
            self.scheduler.send_response(plan_msg, result=plan)
        result["steps"]["plan"] = plan
        if plan.get("llm_skipped"):
            result["llm_calls_skipped"] = result.get("llm_calls_skipped", 0) + 1
            self._progress(
                f"① 规划完成（复用缓存，{time.time() - t_plan:.1f}s，"
                f"累计 API {self._llm_calls() - calls_at_start} 次）"
            )
        else:
            self._progress(
                f"① 规划完成（{time.time() - t_plan:.1f}s，"
                f"累计 API {self._llm_calls() - calls_at_start} 次）"
            )

        # 跟踪领域匹配
        if domain_template_plan is not None:
            result["_e2e_domain_match"] = True
            result["_e2e_domain_score"] = round(domain_match_score, 3)
            result["_e2e_domain_from"] = domain_match_from

        # ---- 构建依赖层级并规范化子任务 ----
        subtasks = plan.get("subtasks", [])
        for st in subtasks:
            if "agent_role" not in st and "role" in st:
                st["agent_role"] = st["role"]

        # ---- AgentDropout：模板填入分足够高时跳过 Executor ----
        template_score = plan.get("_template_score", 0)
        intent_blocks_drop = (
            self.run_options.enable_intent_cache_gate
            and blocks_executor_dropout(task_description, tags)
        )
        if (
            self.run_options.enable_executor_dropout
            and not intent_blocks_drop
            and plan.get("_template_filled")
            and template_score >= self.run_options.executor_dropout_threshold
        ):
            executor_steps = {s["step"] for s in subtasks if s.get("agent_role") == "executor"}
            if executor_steps:
                subtasks = [s for s in subtasks if s.get("agent_role") != "executor"]
                # 重连 Summarizer 依赖：将 Executor 步骤替换为 Retriever 步骤
                retriever_steps = {s["step"] for s in subtasks if s.get("agent_role") == "retriever"}
                for s in subtasks:
                    if s.get("agent_role") == "summarizer":
                        old_deps = set(s.get("depends_on", []))
                        new_deps = (old_deps - executor_steps) | retriever_steps
                        s["depends_on"] = sorted(new_deps)
                plan["_executor_dropped"] = True

        levels = self._build_dependency_levels(subtasks)

        retrieval_results: Dict[int, Any] = {}
        execution_results: Dict[int, Any] = {}
        retrieval_refs: List[str] = []
        execution_refs: List[str] = []

        # ---- Step 2-4：按依赖层级执行子任务 ----
        for level in levels:
            level_retrievers = [s for s in level if s.get("agent_role") == "retriever"]
            level_executors = [s for s in level if s.get("agent_role") == "executor"]
            level_summarizers = [s for s in level if s.get("agent_role") == "summarizer"]
            level_others = [s for s in level if s.get("agent_role") not in ("retriever", "executor", "summarizer")]

            all_level_tasks = level_retrievers + level_executors + level_others + level_summarizers

            # 同层内按 retriever → executor → summarizer 顺序执行，
            # 确保 Summarizer 始终看到已完成的上游结果（避免并行竞态）
            for phase_roles in ("retriever", "executor", "summarizer"):
                phase_tasks = [s for s in all_level_tasks if s.get("agent_role") == phase_roles]
                if not phase_tasks:
                    continue
                phase_cn = {"retriever": "② 检索", "executor": "③ 执行", "summarizer": "④ 总结"}
                self._progress(f"{phase_cn.get(phase_roles, phase_roles)}：{len(phase_tasks)} 个子任务…")
                if len(phase_tasks) == 1:
                    for subtask in phase_tasks:
                        r = self._execute_subtask(
                            subtask, task_id, task_description, tags or [], plan,
                            retrieval_results, execution_results,
                            retrieval_refs, execution_refs,
                        )
                        if r["role"] == "retriever":
                            retrieval_results[r["step"]] = r["result"]
                            retrieval_refs.extend(r["result"].get("memory_refs", []))
                        elif r["role"] == "executor":
                            execution_results[r["step"]] = r["result"]
                            execution_refs.extend(r["result"].get("memory_refs", []))
                        elif r["role"] == "summarizer":
                            result["steps"]["summary"] = r["result"]
                else:
                    # 多子任务并行执行
                    with concurrent.futures.ThreadPoolExecutor(max_workers=len(phase_tasks)) as pool:
                        futures = {
                            pool.submit(
                                self._execute_subtask, st, task_id, task_description,
                                tags or [], plan, retrieval_results, execution_results,
                                retrieval_refs, execution_refs,
                            ): st
                            for st in phase_tasks
                        }
                        for future in concurrent.futures.as_completed(futures):
                            r = future.result()
                            if r["role"] == "retriever":
                                retrieval_results[r["step"]] = r["result"]
                                retrieval_refs.extend(r["result"].get("memory_refs", []))
                            elif r["role"] == "executor":
                                execution_results[r["step"]] = r["result"]
                                execution_refs.extend(r["result"].get("memory_refs", []))
                            elif r["role"] == "summarizer":
                                result["steps"]["summary"] = r["result"]

        result["steps"]["retrieval"] = retrieval_results
        result["steps"]["execution"] = execution_results
        result["retrieval_refs_count"] = len(retrieval_refs)
        result["execution_refs_count"] = len(execution_refs)
        result["_executor_dropped"] = plan.get("_executor_dropped", False)

        result["elapsed_ms"] = (time.time() - t0) * 1000

        # ---- 记录指标 ----

        # LLM 用量（来自 API 响应）
        llm_stats = self.llm.get_usage_stats()
        self.metrics.record_llm_usage(
            prompt_tokens=llm_stats["total_prompt_tokens"],
            completion_tokens=llm_stats["total_completion_tokens"],
            cached_tokens=llm_stats["total_cached_tokens"],
        )
        # 记录 LLM 调用次数（last_elapsed_ms 反映最后一次调用）
        calls_this_task = max(0, self._llm_calls() - calls_at_start)
        if calls_this_task > 0:
            self.metrics.record_llm_call(elapsed_ms=self.llm.last_elapsed_ms)

        # 通信记录（仅本次任务的消息）
        self._record_communication(since_index=msg_count_before)

        # 状态传输记录（本次任务的增量）
        state_stats = self.state_bus.get_stats()
        transfers_this_task = max(0, state_stats["total_transfers"] - state_transfers_before)
        for _ in range(transfers_this_task):
            self.metrics.record_state_transfer(
                int(state_stats.get("avg_packet_size_bytes", 384 * 4)),
                state_stats.get("avg_generation_ms", 0),
            )

        # 记忆操作
        for subtask_result in retrieval_results.values():
            mem_hits = subtask_result.get("memory_hits", [])
            proactive_count = sum(1 for h in mem_hits if h.get("source") == "proactive_suggestion")
            cross_task_count = max(0, len(mem_hits) - proactive_count)
            self.metrics.record_memory_query(hits=len(mem_hits), cross_task=cross_task_count)
        memories_after = self.memory_store.get_stats()["total_memories"]
        for _ in range(max(0, memories_after - memories_before)):
            self.metrics.record_memory_store()

        # ---- SafeSieve-lite：策略模板复用质量反馈 ----
        plan_step = result.get("steps", {}).get("plan", {})
        strategy_id = plan_step.get("_source_strategy_id") or plan_step.get("_reused_from")
        if (
            self.run_options.enable_safe_sieve
            and strategy_id
            and (plan_step.get("_template_filled") or plan_step.get("_reused_from"))
        ):
            summary_step = result.get("steps", {}).get("summary", {})
            summ = summary_step.get("summary") if isinstance(summary_step, dict) else None
            ok = (
                isinstance(summ, dict)
                and not summ.get("_error")
                and not summ.get("_fallback")
                and bool(summ.get("key_findings"))
            )
            try:
                if summary_step:
                    v, _, _ = validate_task(
                        {"task_id": task_id, "description": task_description, "tags": tags or [], "expected_topics": []},
                        {"steps": {"summary": summary_step}},
                        encode_fn=self.embedding_engine.encode,
                        pass_threshold=0.65,
                        use_llm_judge=False,
                    )
                    ok = ok and v.heuristic_pass
            except Exception:
                pass
            try:
                self.memory_store.record_template_outcome(strategy_id, ok)
            except Exception:
                pass

        self.metrics.end_task()
        return result

    # ============================================================
    # 文本模式
    # ============================================================

    def execute_task_text_mode(self, task_id: str, task_description: str, tags: List[str] = None) -> Dict[str, Any]:
        """以文本模式执行任务（基线对照）。

        文本模式下 Agent 通过自然语言通信，不使用结构化协议和非文本状态传递。
        """
        original_mode = self.mode
        self.mode = "text"

        for agent in self.agents.values():
            agent.use_structured_protocol = False

        result = self.execute_task(task_id, task_description, tags)

        self.mode = original_mode
        for agent in self.agents.values():
            agent.use_structured_protocol = (original_mode == "structured")

        return result

    # ============================================================
    # 指标记录
    # ============================================================

    def _record_communication(self, since_index: int = 0) -> None:
        """记录自 since_index 以来的消息通信指标。"""
        history = self.message_bus._history
        for msg in history[since_index:]:
            self.metrics.record_message(
                token_count=msg.estimated_token_count(),
                text_token_count=msg.text_token_count(),
                char_count=len(str(msg.to_dict())),
            )

    # ============================================================
    # 任务组与对比实验
    # ============================================================

    def execute_task_group(
        self, tasks: List[Dict[str, Any]], group_name: str = "default"
    ) -> List[Dict[str, Any]]:
        """顺序执行一组关联任务。

        组内任务可复用之前任务的记忆（跨任务记忆复用）。
        """
        results = []
        for task in tasks:
            result = self.execute_task(
                task_id=task["task_id"],
                task_description=task["description"],
                tags=task.get("tags", []),
            )
            results.append(result)
        return results

    def run_comparison_experiment(
        self, tasks: List[Dict[str, Any]], group_name: str
    ) -> ComparisonReport:
        """在两种模式下运行相同任务并对比结果。

        验证系统改进的核心实验方法：
        1. 结构化模式运行 → 收集指标
        2. 文本模式运行 → 收集指标
        3. 生成 ComparisonReport
        """
        # 结构化模式
        self.mode = "structured"
        for agent in self.agents.values():
            agent.use_structured_protocol = True
        self.message_bus._history.clear()
        self.state_bus.clear()
        self.metrics.clear()

        structured_results = self.execute_task_group(tasks, f"{group_name}_structured")

        structured_msg_count = self.message_bus.message_count
        structured_tokens = self.message_bus.structured_token_count
        structured_latency = sum(r.get("elapsed_ms", 0) for r in structured_results)
        structured_state_bytes = self.state_bus.get_stats().get("total_data_bytes", 0)

        # 文本模式
        self.mode = "text"
        for agent in self.agents.values():
            agent.use_structured_protocol = False
        self.message_bus._history.clear()
        self.state_bus.clear()
        self.metrics.clear()

        text_results = self.execute_task_group(tasks, f"{group_name}_text")

        text_msg_count = self.message_bus.message_count
        text_tokens = self.message_bus.text_token_count
        text_latency = sum(r.get("elapsed_ms", 0) for r in text_results)
        text_state_bytes = 0  # 文本模式无状态传递

        report = ComparisonReport(
            task_group=group_name,
            structured_messages=structured_msg_count,
            structured_tokens=structured_tokens,
            structured_latency_ms=structured_latency,
            structured_state_bytes=structured_state_bytes,
            text_messages=text_msg_count,
            text_tokens=text_tokens,
            text_latency_ms=text_latency,
            text_state_bytes=text_state_bytes,
        )
        self.metrics.add_comparison(report)

        # 恢复结构化模式
        self.mode = "structured"
        for agent in self.agents.values():
            agent.use_structured_protocol = True

        return report

    # ============================================================
    # 系统状态与清理
    # ============================================================

    def get_system_stats(self) -> Dict[str, Any]:
        """获取综合系统统计信息。"""
        return {
            "scheduler": self.scheduler.get_statistics(),
            "memory": self.memory_store.get_stats(),
            "state_exchange": self.state_bus.get_stats(),
            "metrics": self.metrics.get_aggregate_metrics(),
            "agents": {
                agent_id: agent.get_stats()
                for agent_id, agent in self.agents.items()
            },
        }

    def print_system_status(self) -> str:
        """生成人类可读的系统状态报告。"""
        stats = self.get_system_stats()
        mem = stats["memory"]
        se = stats["state_exchange"]
        sch = stats["scheduler"]

        agent_lines = []
        for agent_id, info in self.registry.list_agents().items():
            agent_lines.append(
                f"    {agent_id} [{info.role}] status={info.status} "
                f"sent={info.messages_sent} recv={info.messages_received}"
            )
        planner_ids = self.registry.find_by_role("planner")
        retriever_ids = self.registry.find_by_role("retriever")

        lines = [
            "=" * 60,
            "  System Status",
            "=" * 60,
            f"  Mode: {self.mode}",
            f"  Agents: {sch['agents']}",
            f"    Planners: {planner_ids}",
            f"    Retrievers: {retriever_ids}",
            f"  Active Agent Details:",
        ] + agent_lines + [
            f"  Total Messages: {sch['total_messages']}",
            f"  Structured Tokens: {sch['structured_tokens']}",
            f"  Text Equivalent Tokens: {sch['text_equivalent_tokens']}",
            f"  Token Savings: {sch['token_savings_pct']:.1f}%",
            f"  Tasks Queued: {sch.get('tasks_queued', 0)}",
            f"  Tasks Completed: {sch.get('tasks_completed', 0)}",
            f"  Scheduler Metrics: {self.scheduler.get_metrics()}",
            "",
            "  --- Shared Memory ---",
            f"  Total Memories: {mem['total_memories']}",
            f"  Vector Index Size: {mem['vector_index_size']}",
            f"  Total Accesses: {mem['total_accesses']}",
            f"  Hit Rate: {mem.get('hit_rate', 0):.2%}",
            f"  By Type: {mem.get('by_type', {})}",
            f"  By Agent: {mem.get('by_agent', {})}",
            "",
            "  --- State Exchange ---",
            f"  Total Transfers: {se['transfer_count']}",
            f"  Total Data: {se['total_data_bytes']} bytes",
            f"  Avg Packet Size: {se['avg_packet_size_bytes']:.1f} bytes",
            f"  Total Compression Savings: {se['total_compression_saved_bytes']} bytes",
            "=" * 60,
        ]
        return "\n".join(lines)

    def cleanup(self) -> None:
        """清理资源：重置注册中心、消息总线、调度器和指标。"""
        self.registry = AgentRegistry()
        self.message_bus = MessageBus()
        self.scheduler = Scheduler(self.registry, self.message_bus)
        self.metrics.clear()
        self.state_bus.clear()
