"""Summarizer Agent — 多步结果综合与最终报告生成。

SummarizerAgent 负责综合 Planner、Retriever、Executor 的全部产出，
生成结构化的最终报告（key_findings + facts + conclusion）。

核心机制：
1. 非文本状态消费：从 StateExchangeBus 拉取 Planner 和 Executor 的嵌入向量，
   组合后直接用于 FAISS 语义记忆搜索，跳过"文本→编码"冗余步骤
2. 三级摘要缓存：
   - E2E 精确复用（cos >= 0.85）：直接复用已有摘要
   - 自适应改写（cos 0.72-0.85）：轻量 JSON 刷新，约 70% token 节省
   - 全新生成（cos < 0.72）：完整 LLM 生成
3. 意图门控：通过 task_intent 检测不应复用缓存的场景（对比类、合规类等）
4. 二级语义验证：对 E2E 候选进行 task_desc vs cached_content 逐句向量校验

输出格式：固定的 3 字段 JSON schema
  {"key_findings": [...], "facts": [...], "conclusion": "..."}
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

import numpy as np

from .base import BaseAgent, LLMBackend
from ..protocol import Message, MessageType, ActionType
from ..protocol.scheduler import Scheduler
from ..state.embeddings import EmbeddingEngine
from ..state.exchange import StateExchangeBus
from ..memory.store import MemoryStore
from ..memory.models import MemoryUnit
from ..task_intent import (
    cache_intents_compatible,
    memory_task_description,
    is_open_qa,
    requires_fresh_synthesis,
)


class SummarizerAgent(BaseAgent):
    """结果综合与报告生成 Agent。

    核心能力：summarize（综合）, synthesize（合成）, report（报告）,
            store_memory（记忆存储）, query_memory（记忆查询）

    输入来源（优先级：记忆引用 > 内联内容）：
    - plan_ref / plan: Planner 的执行计划
    - retrieval_refs / retrieval_results: Retriever 的检索结果
    - execution_refs / execution_results: Executor 的执行结果
    - suggested_memories: Orchestrator 的主动建议记忆
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
            agent_id="summarizer",
            role="summarizer",
            capabilities=["summarize", "synthesize", "report", "store_memory", "query_memory"],
            scheduler=scheduler,
            llm=llm,
            embedding_engine=embedding_engine,
            state_bus=state_bus,
            memory_store=memory_store,
            use_structured_protocol=use_structured_protocol,
        )
        self.run_options = None

    # ---- 消息处理 ----

    def handle_message(self, message: Message) -> Optional[Message]:
        if message.msg_type == MessageType.REQUEST:
            if message.action == ActionType.SUMMARIZE:
                return self._handle_summarize_request(message)
        return None

    def _handle_summarize_request(self, message: Message) -> Optional[Message]:
        result = self.execute_task(message.params)
        return self.scheduler.send_response(
            message,
            result=result,
            embedding=self.embedding_engine.encode_state(result),
            memory_refs=result.get("memory_refs", []),
        )

    # ---- Schema 验证与规范化 ----

    @staticmethod
    def _validate_summary_schema(summary: Any) -> Dict[str, Any]:
        """强制固定 schema：剔除未知键和散文包装。"""
        if not isinstance(summary, dict):
            return SummarizerAgent._fallback_summary(str(summary or ""), "")
        findings = summary.get("key_findings") or []
        if not isinstance(findings, list):
            findings = [str(findings)]
        facts = summary.get("facts") or []
        if not isinstance(facts, list):
            facts = [str(facts)] if facts else []
        conclusion = str(summary.get("conclusion") or "").strip()
        # 剔除散文前缀
        for prefix in ("Here is", "Below is", "Final Report", "---"):
            if conclusion.startswith(prefix):
                conclusion = conclusion.split("\n", 1)[-1].strip()[:400]
        return {
            "key_findings": [str(f)[:220] for f in findings[:5]],
            "facts": [str(f)[:160] for f in facts[:3]],
            "conclusion": conclusion[:400],
        }

    @staticmethod
    def _normalize_summary(summary: Any) -> Dict[str, Any]:
        """规范化摘要：截断超长字段（CodeAgents 风格紧凑 JSON）。"""
        if not isinstance(summary, dict):
            return summary
        return SummarizerAgent._validate_summary_schema(summary)

    @staticmethod
    def _fallback_summary(evidence: str, task_desc: str) -> Dict[str, Any]:
        """无 LLM 时从证据文本构建最小结构化摘要。"""
        lines = [ln.strip() for ln in evidence.splitlines() if ln.strip()]
        findings = []
        for ln in lines:
            if ln.startswith("- "):
                findings.append(ln[2:][:200])
            elif ln.startswith("Retrieved:") or ln.startswith("Analysis:"):
                continue
            elif len(findings) < 3 and len(ln) > 20:
                findings.append(ln[:200])
        if not findings and evidence.strip():
            findings = [evidence.strip()[:200]]
        conclusion = (
            f"Summary synthesized from available evidence for: {task_desc[:120]}."
            if findings
            else f"Insufficient evidence to summarize: {task_desc[:120]}."
        )
        return {
            "key_findings": findings[:5] or ["No structured evidence available"],
            "facts": [],
            "conclusion": conclusion,
            "_fallback": True,
        }

    # ---- 自适应缓存改写 ----

    def _adapt_cached_summary(
        self,
        cached: Dict[str, Any],
        task_desc: str,
        evidence_snippet: str,
        *,
        max_tokens: int = 280,
    ) -> Dict[str, Any]:
        """轻量 JSON 刷新：cos 在 [adapt_threshold, adapt_ceiling) 区间时，
        仅用约 70% token 改写已有摘要。"""
        system_prompt = (
            "Adapt JSON summary for NEW task. Same schema. "
            "key_findings max 5, facts max 3, conclusion max 35 words. "
            'Output ONLY JSON:{"key_findings":[],"facts":[],"conclusion":""}'
        )
        user_prompt = (
            f"New task: {task_desc[:280]}\n"
            f"Prior JSON: {json.dumps(cached, ensure_ascii=False)[:700]}\n"
            f"Evidence: {evidence_snippet[:500]}"
        )
        try:
            adapted = self._call_llm_structured(
                system_prompt,
                user_prompt,
                {},
                max_tokens=max_tokens,
                status_hint="④ 总结：DeepSeek 轻量改写缓存答案",
            )
            if isinstance(adapted, dict) and not adapted.get("parse_error"):
                return self._normalize_summary(adapted)
        except Exception:
            pass
        return self._normalize_summary(cached)

    # ---- 查询嵌入构建（非文本状态消费核心） ----

    def _build_query_embedding(self, task_desc: str) -> List[float]:
        """构建记忆搜索的查询嵌入，优先使用上游状态。

        检查 StateExchangeBus 中 Planner 和 Executor 发送的嵌入向量。
        若找到则直接组合使用（无需重新编码）。否则回退到 encode(task_desc)。

        这是非文本状态传递的"接收与使用"半部分：
        上游 Agent 已计算嵌入，跳过冗余的"文本→编码"步骤。
        """
        packets = self.receive_state_packets(limit=3)
        if packets:
            embeddings = [p.embedding for p in packets]
            self.state_bus.mark_consumed([p.packet_id for p in packets])
            if len(embeddings) == 1:
                return embeddings[0]
            return self.embedding_engine.combine(embeddings)

        return self.embedding_engine.encode(task_desc)

    # ---- 核心任务执行 ----

    def execute_task(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """综合所有上游步骤的结果，生成结构化最终报告。

        支持两种输入方式（优先级：记忆引用 > 内联内容）：
        - 记忆引用：plan_ref, retrieval_refs, execution_refs
        - 内联内容：plan, retrieval_results, execution_results

        三级摘要缓存：
        1. E2E 精确复用（cos >= e2e_threshold，默认 0.85）
        2. 自适应改写（cos 在 [adapt_threshold, adapt_ceiling)）
        3. 全新 LLM 生成
        """
        t0 = time.time()
        task_id = task_input.get("task_id", "unknown")
        self._context["task_id"] = task_id

        # ---- 解析上游结果（记忆引用优先，内联内容回退） ----
        plan = task_input.get("plan", {})
        plan_ref = task_input.get("plan_ref", "")
        if plan_ref:
            mem = self.memory_store.get(plan_ref)
            if mem:
                try:
                    plan = json.loads(mem.content)
                except (json.JSONDecodeError, TypeError):
                    pass

        retrieval_refs = task_input.get("retrieval_refs", [])
        retrieval_memories = self.retrieve_memories(retrieval_refs) if retrieval_refs else []
        retrieval_results = task_input.get("retrieval_results", {})

        execution_refs = task_input.get("execution_refs", [])
        execution_memories = self.retrieve_memories(execution_refs) if execution_refs else []
        execution_results = task_input.get("execution_results", {})

        # 收集所有证据引用
        evidence_refs: List[str] = list(retrieval_refs) + list(execution_refs)
        evidence_refs.extend(plan.get("memory_refs", []))

        # 合并 Orchestrator 的主动建议记忆
        suggested = task_input.get("suggested_memories", [])
        suggested_memories = self.retrieve_memories([s["memory_id"] for s in suggested])

        # 查询共享记忆中的相关历史摘要和策略
        memories = self.query_memory(
            query=f"summary {plan.get('task_description', '')}",
            tags=["summary", "report", "conclusion"],
            limit=2,
        )
        seen_ids = {m.memory_id for m in memories}
        for sm in suggested_memories:
            if sm.memory_id not in seen_ids:
                memories.append(sm)
                seen_ids.add(sm.memory_id)
        memory_context = self._context_from_memories(memories)

        # ---- 构建综合摘要文本 ----
        summary_parts = []
        all_retrieved_memories: List[MemoryUnit] = []

        if plan:
            summary_parts.append(f"Plan: {plan.get('task_description', 'N/A')[:200]}")
            expected = plan.get("expected_outcome", "")
            if expected:
                summary_parts.append(f"Goal: {expected[:200]}")

        if retrieval_memories:
            all_retrieved_memories.extend(retrieval_memories)
            parts = [f"- {mem.summary[:150]}" for mem in retrieval_memories[:3]]
            if parts:
                summary_parts.append("Retrieved:\n" + "\n".join(parts))
        elif retrieval_results:
            for step, result in retrieval_results.items():
                if isinstance(result, dict):
                    evidence_refs.extend(result.get("memory_refs", []))
                    if result.get("combined_results"):
                        summary_parts.append(f"Retrieved: {result['combined_results'][:1000]}")
                        break

        if execution_memories:
            all_retrieved_memories.extend(execution_memories)
            parts = [f"- {mem.summary[:150]}" for mem in execution_memories[:3]]
            if parts:
                summary_parts.append("Analysis:\n" + "\n".join(parts))
        elif execution_results:
            for step, result in execution_results.items():
                if isinstance(result, dict):
                    evidence_refs.extend(result.get("memory_refs", []))
                    if result.get("output"):
                        summary_parts.append(f"Analysis: {result['output'][:1000]}")
                        break

        if memory_context and "No relevant memories" not in memory_context:
            summary_parts.append(f"Past: {memory_context}")

        combined_summary = "\n\n".join(summary_parts)

        # ---- 摘要缓存：语义匹配（嵌入 + 可选 LLM 验证） ----
        final_summary = None
        cached_from = None

        task_tags = task_input.get("tags", [])
        opts = self.run_options
        summ_cache = opts is None or opts.enable_summarizer_cache
        max_tok = opts.summarizer_max_tokens if opts else 640

        task_desc = (
            task_input.get("task_description", "")
            or (plan.get("task_description", "") if plan else "")
            or task_id
        )
        evidence_cap = 1400
        if opts is not None:
            evidence_cap = min(1400, opts.evidence_max_chars * 2)
        if is_open_qa(task_desc, task_tags):
            evidence_cap = max(evidence_cap, 2400)

        intent_gate = opts is None or opts.enable_intent_cache_gate
        skip_summary_cache = intent_gate and requires_fresh_synthesis(task_desc, task_tags)

        if combined_summary.strip() and summ_cache and not skip_summary_cache:
            try:
                if (
                    task_desc
                    and self.memory_store._index is not None
                    and self.memory_store._index.ntotal > 0
                ):
                    # 非文本状态消费：使用上游嵌入进行记忆搜索
                    query_emb = self._build_query_embedding(task_desc)

                    similar = self.memory_store.search_by_similarity(
                        query_emb, limit=6, memory_type="result"
                    )
                    adapt_candidate = None
                    adapt_score = 0.0
                    adapt_mem = None

                    min_rel = opts.summarizer_cache_min_relevance if opts else 0.72

                    for mem, raw in similar:
                        cached_tags = mem.tags if hasattr(mem, "tags") else []
                        if task_tags and cached_tags and not any(t in cached_tags for t in task_tags):
                            continue
                        if intent_gate and not cache_intents_compatible(
                            memory_task_description(mem), task_desc, cached_tags, task_tags,
                        ):
                            continue
                        score = (
                            self.memory_store.effective_similarity(mem, raw)
                            if mem.memory_type == "strategy"
                            else raw
                        )
                        try:
                            cached = json.loads(mem.content)
                            cached_sum = cached.get("summary") if isinstance(cached, dict) else None
                        except (json.JSONDecodeError, TypeError):
                            continue
                        if not isinstance(cached_sum, dict) or cached_sum.get("_error"):
                            continue

                        e2e_th = opts.summarizer_e2e_threshold if opts else 0.85
                        if score >= e2e_th:
                            # 二级语义验证：防止高维相似但语义不同的误匹配
                            if intent_gate and score < 1.0:
                                try:
                                    c_text = " ".join(
                                        cached_sum.get("key_findings", [])
                                        + [cached_sum.get("conclusion", "")]
                                    )
                                    te = self.embedding_engine.encode(task_desc)
                                    ce = self.embedding_engine.encode(c_text[:500])
                                    na, nb = np.linalg.norm(te), np.linalg.norm(ce)
                                    rel = float(np.dot(te, ce) / (na * nb)) if na and nb else 0.0
                                    if rel < min_rel:
                                        continue
                                except Exception:
                                    pass
                            final_summary = cached_sum
                            cached_from = mem.memory_id
                            break

                        if (
                            opts
                            and opts.enable_summarizer_adapt
                            and opts.summarizer_adapt_threshold <= score < opts.summarizer_adapt_ceiling
                        ):
                            if score > adapt_score:
                                adapt_score = score
                                adapt_candidate = cached_sum
                                adapt_mem = mem

                    # 自适应改写
                    if final_summary is None and adapt_candidate is not None:
                        final_summary = self._adapt_cached_summary(
                            adapt_candidate,
                            task_desc,
                            combined_summary[: min(600, evidence_cap)],
                            max_tokens=opts.summarizer_adapt_max_tokens if opts else 280,
                        )
                        cached_from = adapt_mem.memory_id if adapt_mem else None
                        if isinstance(final_summary, dict):
                            final_summary["_summarizer_adapt"] = True
            except Exception:
                pass

        # ---- LLM 全新生成 ----
        if final_summary is None and combined_summary.strip():
            open_qa = is_open_qa(task_desc, task_tags)
            cjk = any("一" <= c <= "鿿" for c in task_desc)
            if open_qa and cjk:
                system_prompt = (
                    "只输出合法 JSON，不要 markdown。"
                    "字段：key_findings（最多8条完整中文要点）、facts（最多5条补充事实）、"
                    "conclusion（完整中文回答段落，至少120字，覆盖推荐与理由）。"
                    '格式:{"key_findings":[],"facts":[],"conclusion":""}'
                )
            elif open_qa:
                system_prompt = (
                    "Output ONLY valid JSON. "
                    "key_findings: up to 8 complete bullet strings; "
                    "facts: up to 5 strings; "
                    "conclusion: a full answer paragraph (at least 80 words). "
                    'Schema:{"key_findings":[],"facts":[],"conclusion":""}'
                )
            else:
                system_prompt = (
                    "Output ONLY valid JSON, no markdown. "
                    "Exactly 3 keys: key_findings (max 5 strings), facts (max 3 strings), "
                    "conclusion (one sentence, max 35 words). "
                    'Schema:{"key_findings":[],"facts":[],"conclusion":""}'
                )
            user_prompt = f"Task: {task_desc[:400]}\n\nEvidence:\n{combined_summary[:evidence_cap]}"

            try:
                self._emit_status("④ 总结：调用 DeepSeek 生成最终报告…")
                final_summary = self._call_llm_structured(
                    system_prompt,
                    user_prompt,
                    {},
                    max_tokens=max_tok,
                    status_hint="④ 总结：DeepSeek 生成最终报告",
                )
            except Exception:
                try:
                    retry_prompt = f"Task: {task_desc[:300]}\n\nEvidence:\n{combined_summary[:1000]}"
                    final_summary = self._call_llm_structured(
                        system_prompt, retry_prompt, {}, max_tokens=max_tok
                    )
                except Exception:
                    try:
                        raw = self._call_llm(
                            system_prompt,
                            f"Task: {task_desc[:300]}\n\n{combined_summary[:800]}",
                            max_tokens=max_tok,
                            temperature=0.1,
                        )
                        from .base import _extract_json
                        final_summary = _extract_json(raw)
                    except Exception:
                        final_summary = self._fallback_summary(combined_summary, task_desc)
        elif final_summary is None and not combined_summary.strip():
            final_summary = self._fallback_summary("", task_desc)
        if isinstance(final_summary, dict):
            final_summary = self._normalize_summary(final_summary)

        # 构建存储用摘要文本
        if isinstance(final_summary, dict):
            summary_for_storage = "; ".join(final_summary.get("key_findings", []))
            if final_summary.get("conclusion"):
                summary_for_storage += " " + final_summary["conclusion"]
        else:
            summary_for_storage = final_summary

        result = {
            "task_id": task_id,
            "summary": final_summary,
            "source_sections": summary_parts,
            "evidence_count": len(evidence_refs),
            "memory_refs_from_prior": [m.memory_id for m in memories],
            "_summ_cached_from": cached_from,
            "evidence_refs": list(set(evidence_refs)),
            "elapsed_ms": (time.time() - t0) * 1000,
        }

        # ---- 存储最终摘要到共享记忆（跳过错误兜底） ----
        topic = task_desc[:80]
        if isinstance(final_summary, dict) and not final_summary.get("_error"):
            memory_id = self.store_memory(
                topic=f"Summary: {topic}",
                summary=summary_for_storage[:300],
                content=json.dumps(result, ensure_ascii=False, indent=2),
                tags=["summary", "report", "conclusion", "final"] + task_tags,
                memory_type="result",
                evidence_chain=result["evidence_refs"],
                embedding_text=task_desc,
            )
            result["memory_refs"] = [memory_id] + result["evidence_refs"]
        else:
            result["memory_refs"] = list(result["evidence_refs"])

        # ---- P1：触发结果模板提升 ----
        result_tags = ["summary", "report", "conclusion", "final"]
        plan_tags = plan.get("tags", []) if plan else []
        self._maybe_promote_to_template(
            tags=result_tags + plan_tags,
            memory_type="result",
            threshold=2,
        )

        summary_len = (
            len(json.dumps(final_summary, ensure_ascii=False))
            if isinstance(final_summary, dict)
            else len(final_summary)
        )
        self._task_history.append({
            "task_id": task_id,
            "summary_length": summary_len,
            "evidence_count": len(evidence_refs),
        })
        return result
