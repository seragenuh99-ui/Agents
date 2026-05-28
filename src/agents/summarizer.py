"""Summarizer Agent: synthesizes results into coherent final outputs."""

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
    """Agent responsible for synthesizing and summarizing execution results.

    Capabilities: summarize, synthesize, report, store_memory, query_memory
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

    def handle_message(self, message: Message) -> Optional[Message]:
        if message.msg_type == MessageType.REQUEST:
            if message.action == ActionType.SUMMARIZE:
                return self._handle_summarize_request(message)
        return None

    @staticmethod
    def _validate_summary_schema(summary: Any) -> Dict[str, Any]:
        """Enforce fixed schema; drop unknown keys and prose wrappers."""
        if not isinstance(summary, dict):
            return SummarizerAgent._fallback_summary(str(summary or ""), "")
        findings = summary.get("key_findings") or []
        if not isinstance(findings, list):
            findings = [str(findings)]
        facts = summary.get("facts") or []
        if not isinstance(facts, list):
            facts = [str(facts)] if facts else []
        conclusion = str(summary.get("conclusion") or "").strip()
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
        """Cap fields for compact JSON (CodeAgents-style density)."""
        if not isinstance(summary, dict):
            return summary
        validated = SummarizerAgent._validate_summary_schema(summary)
        return validated

    @staticmethod
    def _fallback_summary(evidence: str, task_desc: str) -> Dict[str, Any]:
        """Build a minimal structured summary from evidence without LLM."""
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

    def _adapt_cached_summary(
        self,
        cached: Dict[str, Any],
        task_desc: str,
        evidence_snippet: str,
        *,
        max_tokens: int = 280,
    ) -> Dict[str, Any]:
        """Lightweight JSON refresh for cos in [adapt_threshold, adapt_ceiling) — ~70% fewer tokens than full gen."""
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

    def _handle_summarize_request(self, message: Message) -> Optional[Message]:
        result = self.execute_task(message.params)
        return self.scheduler.send_response(
            message,
            result=result,
            embedding=self.embedding_engine.encode_state(result),
            memory_refs=result.get("memory_refs", []),
        )

    def execute_task(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize results from all prior steps into a coherent final report.

        Accepts either inline content (backward compat) or memory references.
        Memory references are preferred — the summarizer pulls content from
        shared memory, avoiding redundant transmission of full text in messages.
        """
        t0 = time.time()
        task_id = task_input.get("task_id", "unknown")
        self._context["task_id"] = task_id

        # --- Resolve upstream results from memory refs (new) or inline (fallback) ---
        plan = task_input.get("plan", {})
        plan_ref = task_input.get("plan_ref", "")
        if plan_ref:
            mem = self.memory_store.get(plan_ref)
            if mem:
                try:
                    plan = json.loads(mem.content)
                except (json.JSONDecodeError, TypeError):
                    pass

        # Resolve retrieval results from refs
        retrieval_refs = task_input.get("retrieval_refs", [])
        retrieval_memories = self.retrieve_memories(retrieval_refs) if retrieval_refs else []
        retrieval_results = task_input.get("retrieval_results", {})

        # Resolve execution results from refs
        execution_refs = task_input.get("execution_refs", [])
        execution_memories = self.retrieve_memories(execution_refs) if execution_refs else []
        execution_results = task_input.get("execution_results", {})

        # Collect all evidence refs from both refs (new) and inline (fallback)
        evidence_refs: List[str] = list(retrieval_refs) + list(execution_refs)
        evidence_refs.extend(plan.get("memory_refs", []))

        # Collect suggested memories from orchestrator (proactive injection)
        suggested = task_input.get("suggested_memories", [])
        suggested_memories = self.retrieve_memories([s["memory_id"] for s in suggested])

        # Check shared memory for related past summaries and strategies
        memories = self.query_memory(
            query=f"summary {plan.get('task_description', '')}",
            tags=["summary", "report", "conclusion"],
            limit=2,
        )
        # Merge proactive suggestions with query results, dedup by memory_id
        seen_ids = {m.memory_id for m in memories}
        for sm in suggested_memories:
            if sm.memory_id not in seen_ids:
                memories.append(sm)
                seen_ids.add(sm.memory_id)
        memory_context = self._context_from_memories(memories)

        # --- Build synthesis from memories (new path) or inline content (fallback) ---
        summary_parts = []
        all_retrieved_memories: List[MemoryUnit] = []

        # Plan — compact format
        if plan:
            summary_parts.append(f"Plan: {plan.get('task_description', 'N/A')[:200]}")
            expected = plan.get("expected_outcome", "")
            if expected:
                summary_parts.append(f"Goal: {expected[:200]}")

        # Retrieval results — prefer from memory
        if retrieval_memories:
            all_retrieved_memories.extend(retrieval_memories)
            parts = [f"- {mem.summary[:150]}" for mem in retrieval_memories[:3]]
            if parts:
                summary_parts.append("Retrieved:\n" + "\n".join(parts))
        elif retrieval_results:
            # Fallback: inline content
            for step, result in retrieval_results.items():
                if isinstance(result, dict):
                    evidence_refs.extend(result.get("memory_refs", []))
                    if result.get("combined_results"):
                        summary_parts.append(f"Retrieved: {result['combined_results'][:1000]}")
                        break

        # Execution results — prefer from memory
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

        # Include past memory insights if available
        if memory_context and "No relevant memories" not in memory_context:
            summary_parts.append(f"Past: {memory_context}")

        combined_summary = "\n\n".join(summary_parts)

        # Generate final structured summary using LLM
        final_summary = None
        cached_from = None

        # --- Summarizer cache: semantic matching (embedding + optional LLM) ---
        # Replaces Jaccard-on-evidence_refs (which never hit because each task
        # generates new memory IDs). Now uses task-description embedding to find
        # semantically similar past summaries, with LLM verification for borderline cases.
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
        skip_summary_cache = intent_gate and requires_fresh_synthesis(
            task_desc, task_tags
        )

        if combined_summary.strip() and summ_cache and not skip_summary_cache:
            try:
                if (
                    task_desc
                    and self.memory_store._index is not None
                    and self.memory_store._index.ntotal > 0
                ):
                    query_emb = self.embedding_engine.encode(task_desc)
                    similar = self.memory_store.search_by_similarity(
                        query_emb, limit=6, memory_type="result"
                    )
                    adapt_candidate = None
                    adapt_score = 0.0
                    adapt_mem = None

                    min_rel = (
                        opts.summarizer_cache_min_relevance if opts else 0.72
                    )

                    for mem, raw in similar:
                        cached_tags = mem.tags if hasattr(mem, "tags") else []
                        if task_tags and cached_tags and not any(
                            t in cached_tags for t in task_tags
                        ):
                            continue
                        if intent_gate and not cache_intents_compatible(
                            memory_task_description(mem),
                            task_desc,
                            cached_tags,
                            task_tags,
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

                        e2e_th = (
                            opts.summarizer_e2e_threshold
                            if opts
                            else 0.85
                        )
                        if score >= e2e_th:
                            if intent_gate and score < 1.0:
                                try:
                                    c_text = " ".join(
                                        cached_sum.get("key_findings", [])
                                        + [cached_sum.get("conclusion", "")]
                                    )
                                    te = self.embedding_engine.encode(task_desc)
                                    ce = self.embedding_engine.encode(c_text[:500])
                                    na, nb = np.linalg.norm(te), np.linalg.norm(ce)
                                    rel = (
                                        float(np.dot(te, ce) / (na * nb))
                                        if na and nb
                                        else 0.0
                                    )
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
                            and opts.summarizer_adapt_threshold
                            <= score
                            < opts.summarizer_adapt_ceiling
                        ):
                            if score > adapt_score:
                                adapt_score = score
                                adapt_candidate = cached_sum
                                adapt_mem = mem

                    if final_summary is None and adapt_candidate is not None:
                        final_summary = self._adapt_cached_summary(
                            adapt_candidate,
                            task_desc,
                            combined_summary[: min(600, evidence_cap)],
                            max_tokens=(
                                opts.summarizer_adapt_max_tokens if opts else 280
                            ),
                        )
                        cached_from = adapt_mem.memory_id if adapt_mem else None
                        if isinstance(final_summary, dict):
                            final_summary["_summarizer_adapt"] = True
            except Exception:
                pass

        if final_summary is None and combined_summary.strip():
            open_qa = is_open_qa(task_desc, task_tags)
            cjk = any("\u4e00" <= c <= "\u9fff" for c in task_desc)
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
                    retry_prompt = (
                        f"Task: {task_desc[:300]}\n\nEvidence:\n{combined_summary[:1000]}"
                    )
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
                        final_summary = self._fallback_summary(
                            combined_summary, task_desc
                        )
        elif final_summary is None and not combined_summary.strip():
            final_summary = self._fallback_summary("", task_desc)
        if isinstance(final_summary, dict):
            final_summary = self._normalize_summary(final_summary)

        # Normalize: structured output becomes dict, fallback stays as string
        if isinstance(final_summary, dict):
            summary_for_storage = "; ".join(
                final_summary.get("key_findings", [])
            )
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

        # Store final summary in shared memory for future reuse (skip error fallbacks)
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

        # P1: Trigger template promotion for result type
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
