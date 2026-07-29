"""Shared benchmark runner for experiment matrix variants."""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from experiments.benchmark_tasks import BENCHMARK_TASKS
from experiments.benchmark_suites import get_suite
from src.agents.base import LLMBackend
from src.evaluation.quality_validator import validate_task
from src.orchestrator import Orchestrator
from src.run_options import RunOptions, DEFAULT_OPTIONS


@dataclass
class BatchSession:
    """Reusable orchestrator + LLM across multiple tasks (e.g. REPL multi-turn)."""

    orch: Optional[Orchestrator] = None
    llm: Optional[LLMBackend] = None


def clear_db(path: str) -> None:
    for p in [path, path + "-shm", path + "-wal"]:
        try:
            os.unlink(p)
        except OSError:
            pass


def classify_strategy(result: Dict[str, Any], plan: Dict[str, Any]) -> str:
    dropped = result.get("_executor_dropped", False)
    if result.get("_e2e_cached"):
        match = result.get("_e2e_match", "exact")
        if match in ("exact", "summarizer"):
            return "E2E_EXACT"
        if match == "adapt":
            return "SUMMARIZER_ADAPT"
    summary_step = result.get("steps", {}).get("summary", {})
    if isinstance(summary_step, dict):
        summ = summary_step.get("summary")
        if isinstance(summ, dict) and summ.get("_summarizer_adapt"):
            return "SUMMARIZER_ADAPT"
    if plan.get("_from_template"):
        return "P1_TEMPLATE" + ("_DROP" if dropped else "")
    if plan.get("llm_skipped"):
        return "CACHE_REUSE"
    if plan.get("_template_filled"):
        return "TEMPLATE_FILL" + ("_DROP" if dropped else "")
    return "FULL_GEN"


def run_task_batch(
    api_key: str,
    tasks: List[Dict[str, Any]],
    *,
    db_path: str,
    options: RunOptions = DEFAULT_OPTIONS,
    cold_db_per_task: bool = False,
    reset_llm_each_task: bool = True,
    run_quality: bool = False,
    use_llm_judge: bool = False,
    embedding_model: Optional[str] = None,
    verbose: bool = False,
    session: Optional[BatchSession] = None,
    clear_db_before: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run tasks through Orchestrator with given options.

    When *session* is provided and *clear_db_before* is False, reuses the same
    orchestrator, memory DB, and embedding model across tasks (multi-turn chat).
    """
    if clear_db_before:
        clear_db(db_path)
        if session is not None:
            session.orch = None

    if session is not None and session.llm is not None:
        llm = session.llm
    else:
        llm = LLMBackend(provider="deepseek", api_key=api_key, model="deepseek-chat")
        if session is not None:
            session.llm = llm

    results: List[Dict[str, Any]] = []
    quality_pass = 0
    adversarial_false_e2e = 0
    adversarial_total = 0
    orch: Optional[Orchestrator] = session.orch if session else None

    n_tasks = len(tasks)
    for i, task in enumerate(tasks):
        if task.get("must_not_e2e"):
            adversarial_total += 1
        if cold_db_per_task:
            clear_db(db_path)
            orch = None

        if reset_llm_each_task:
            llm.reset_stats()
        stats_before = llm.get_usage_stats()

        if orch is None:
            if verbose:
                print(
                    "\n  [系统] 正在加载语义向量模型（出现 Loading weights 属正常，完成后继续）…",
                    flush=True,
                )
                print(
                    "  [系统] 正在初始化多 Agent 运行时与共享记忆库…",
                    flush=True,
                )
            orch = Orchestrator(
                llm=llm,
                mode="structured",
                use_real_embeddings=True,
                sandbox_enabled=True,
                memory_db_path=db_path,
                run_options=options,
                embedding_model=embedding_model,
            )
            if verbose:
                print("  [系统] 初始化完成。\n", flush=True)

        if verbose:
            orch._progress_fn = lambda m: print(f"        {m}", flush=True)

        desc_short = (task.get("description") or "")[:40]
        if verbose:
            print(
                f"  [{i + 1}/{n_tasks}] {task['task_id']}: {desc_short}…",
                flush=True,
            )
            print(
                "        完整流水线：规划→检索→执行→总结；下方会实时显示当前步骤…",
                flush=True,
            )

        t0 = time.time()
        try:
            result = orch.execute_task(
                task_id=task["task_id"],
                task_description=task["description"],
                tags=task.get("tags", []),
            )
        except Exception as exc:
            if verbose:
                print(f"        执行失败: {exc}", flush=True)
            raise
        elapsed = time.time() - t0

        stats_after = llm.get_usage_stats()
        prompt_t = stats_after["total_prompt_tokens"] - stats_before["total_prompt_tokens"]
        compl_t = stats_after["total_completion_tokens"] - stats_before["total_completion_tokens"]
        calls = stats_after["call_count"] - stats_before["call_count"]
        cached = stats_after["total_cached_tokens"] - stats_before["total_cached_tokens"]

        if verbose:
            print(
                f"        本题完成，耗时 {elapsed:.1f} 秒，"
                f"本题 API 调用 {calls} 次，token {prompt_t + compl_t}。",
                flush=True,
            )

        plan = result.get("steps", {}).get("plan", {})
        summary_step = result.get("steps", {}).get("summary", {})
        strategy = classify_strategy(result, plan)

        summary_text = ""
        summary_obj: Dict[str, Any] = {}
        summary_full = ""
        if isinstance(summary_step, dict):
            from src.evaluation.quality_validator import (
                extract_summary_payload,
                format_answer_markdown,
            )

            summary_obj, summary_text = extract_summary_payload(summary_step)
            summary_full = format_answer_markdown(summary_obj, summary_text)

        relevance = 0.0
        try:
            task_emb = orch.embedding_engine.encode(task["description"])
            summ_emb = orch.embedding_engine.encode(summary_text[:500] if summary_text else " ")
            a, b = np.array(task_emb), np.array(summ_emb)
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na and nb:
                relevance = float(np.dot(a, b) / (na * nb))
        except Exception:
            pass

        composite = None
        composite_pass = None
        if task.get("must_not_e2e") and result.get("_e2e_cached"):
            adversarial_false_e2e += 1

        if run_quality and summary_step:
            v, _, _ = validate_task(
                task,
                summary_step,
                encode_fn=orch.embedding_engine.encode,
                pass_threshold=0.65,
                use_llm_judge=use_llm_judge,
                llm=llm if use_llm_judge else None,
            )
            composite = v.composite_score
            composite_pass = v.composite_pass
            if composite_pass:
                quality_pass += 1

        collab: Dict[str, Any] = {}
        if orch.metrics._task_history:
            tm = orch.metrics._task_history[-1]
            collab = {
                "messages_sent": tm.messages_sent,
                "structured_protocol_tokens": tm.structured_token_count,
                "text_equivalent_tokens": tm.text_equivalent_token_count,
                "communication_chars": tm.communication_char_count,
                "state_transfers": tm.state_transfers,
                "state_data_bytes": tm.state_data_bytes,
                "memory_queries": tm.memory_queries,
                "memory_hits": tm.memory_hits,
                "memory_hit_rate": round(tm.memory_hit_rate, 4),
                "cross_task_memories": tm.cross_task_memories_used,
                "llm_calls": tm.llm_call_count,
            }

        row = {
            "task_id": task["task_id"],
            "description": task.get("description", ""),
            "domain": task.get("domain", task.get("tags", ["?"])[0]),
            "strategy": strategy,
            "api_calls": calls,
            "prompt_tokens": prompt_t,
            "completion_tokens": compl_t,
            "total_tokens": prompt_t + compl_t,
            "cached_tokens": cached,
            "relevance": round(relevance, 3),
            "elapsed_ms": elapsed * 1000,
            "composite_score": composite,
            "composite_pass": composite_pass,
            "e2e_cached": bool(result.get("_e2e_cached")),
            "executor_dropped": bool(result.get("_executor_dropped")),
            "summary_preview": (summary_text or "")[:200],
            "summary_full": summary_full or summary_text or "",
            "summary_obj": summary_obj if summary_obj else None,
            "must_not_e2e": bool(task.get("must_not_e2e")),
            "false_e2e_violation": bool(
                task.get("must_not_e2e") and result.get("_e2e_cached")
            ),
            **collab,
        }
        results.append(row)

    if session is not None:
        session.orch = orch
        session.llm = llm

    totals_collab: Dict[str, Any] = {}
    if results and "messages_sent" in results[0]:
        q = sum(r.get("memory_queries", 0) for r in results)
        h = sum(r.get("memory_hits", 0) for r in results)
        totals_collab = {
            "total_messages": sum(r.get("messages_sent", 0) for r in results),
            "total_structured_protocol_tokens": sum(
                r.get("structured_protocol_tokens", 0) for r in results
            ),
            "total_text_equivalent_tokens": sum(
                r.get("text_equivalent_tokens", 0) for r in results
            ),
            "total_communication_chars": sum(
                r.get("communication_chars", 0) for r in results
            ),
            "total_state_transfers": sum(r.get("state_transfers", 0) for r in results),
            "total_state_data_bytes": sum(r.get("state_data_bytes", 0) for r in results),
            "memory_hit_rate": round(h / max(q, 1), 4),
            "total_memory_queries": q,
            "total_memory_hits": h,
            "total_cross_task_memories": sum(
                r.get("cross_task_memories", 0) for r in results
            ),
        }

    totals = {
        "total_api_calls": sum(r["api_calls"] for r in results),
        "total_prompt_tokens": sum(r["prompt_tokens"] for r in results),
        "total_completion_tokens": sum(r["completion_tokens"] for r in results),
        "total_tokens": sum(r["total_tokens"] for r in results),
        "total_cached_tokens": sum(r["cached_tokens"] for r in results),
        "total_elapsed_ms": sum(r["elapsed_ms"] for r in results),
        "avg_relevance": sum(r["relevance"] for r in results) / max(len(results), 1),
        "strategy_distribution": dict(Counter(r["strategy"] for r in results)),
        "e2e_hits": sum(1 for r in results if r.get("e2e_cached")),
        "dropout_hits": sum(1 for r in results if r.get("executor_dropped")),
        "quality_pass_count": quality_pass if run_quality else None,
        "adversarial_total": adversarial_total or None,
        "adversarial_false_e2e": adversarial_false_e2e if adversarial_total else None,
        "adversarial_e2e_safe": (
            adversarial_false_e2e == 0 if adversarial_total else None
        ),
        **totals_collab,
    }
    return results, totals


def run_suite(
    api_key: str,
    suite_name: str,
    variant_id: str,
    *,
    options: RunOptions = DEFAULT_OPTIONS,
    db_path: str = "_suite.db",
    cold_db_per_task: bool = False,
    run_quality: bool = True,
    use_llm_judge: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run a named task suite (core12, full24, adversarial6, ...)."""
    return run_task_batch(
        api_key,
        get_suite(suite_name),
        db_path=db_path,
        options=options,
        cold_db_per_task=cold_db_per_task,
        run_quality=run_quality,
        use_llm_judge=use_llm_judge,
    )


def save_variant_result(
    variant_id: str,
    results: List[Dict[str, Any]],
    totals: Dict[str, Any],
    options: RunOptions,
    out_dir: str,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{variant_id}.json")
    payload = {
        "variant_id": variant_id,
        "options": options.__dict__,
        "results": results,
        "totals": totals,
        "meta": extra_meta or {},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path
