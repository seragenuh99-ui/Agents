#!/usr/bin/env python3
"""Pure text baseline: verbose prompts, prose output, no structure, no cache.

Same task suite as structured system (control variable: task set only).
Per task: plan + retrieve + summarize (3 LLM calls), no memory, no executor.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

from experiments.benchmark_suites import get_suite
from src.evaluation.quality_validator import validate_task
from src.paths import result_path
from src.state.embeddings import EmbeddingEngine

MODEL = "deepseek-chat"
MAX_TOKENS = 1024


class TokenCounter:
    def __init__(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0

    def add(self, usage: Dict[str, Any]) -> None:
        self.calls += 1
        self.prompt_tokens += usage.get("prompt_tokens", 0)
        self.completion_tokens += usage.get("completion_tokens", 0)

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def snapshot(self) -> Dict[str, int]:
        return {
            "api_calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total,
        }

    def reset(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0


def call_llm(system_prompt: str, user_prompt: str, api_key: str, max_tokens: int = MAX_TOKENS):
    last_err = None
    for attempt in range(4):
        try:
            resp = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.3,
                    "max_tokens": max_tokens,
                },
                timeout=180,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"], data.get("usage", {})
            last_err = RuntimeError(f"API error {resp.status_code}: {resp.text[:300]}")
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_err = e
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"LLM call failed after retries: {last_err}")


def run_text_baseline_task(
    task: Dict[str, Any], api_key: str, encode_fn
) -> Dict[str, Any]:
    """One task: 3 verbose LLM calls, isolated (no cross-task state)."""
    counter = TokenCounter()
    t0 = time.time()

    plan_system = (
        "You are a task planning agent for a multi-agent collaboration system. "
        "Decompose the task into subtasks with roles, actions, and dependencies. "
        "Output a clear numbered plan in prose."
    )
    plan_user = (
        f"Create a detailed execution plan for:\n\n{task['description']}\n\n"
        "List each subtask step by step."
    )
    plan_text, plan_usage = call_llm(plan_system, plan_user, api_key)
    counter.add(plan_usage)

    retrieve_system = (
        "You are a research assistant. Provide relevant information, facts, "
        "and context to complete the task. Be thorough."
    )
    retrieve_user = f"Research and provide information for:\n\n{task['description']}"
    retrieve_text, retrieve_usage = call_llm(retrieve_system, retrieve_user, api_key)
    counter.add(retrieve_usage)

    summary_system = (
        "You are a report writer. Synthesize into a comprehensive final report: "
        "executive summary, key findings, conclusions, recommendations. "
        "Professional prose."
    )
    summary_user = (
        f"Write a final report.\n\nTASK:\n{task['description']}\n\n"
        f"PLAN:\n{plan_text[:1500]}\n\nRESEARCH:\n{retrieve_text[:1500]}"
    )
    summary_text, summary_usage = call_llm(summary_system, summary_user, api_key)
    counter.add(summary_usage)

    elapsed_ms = (time.time() - t0) * 1000
    snap = counter.snapshot()

    relevance = 0.0
    try:
        task_emb = encode_fn(task["description"])
        summ_emb = encode_fn(summary_text[:500])
        a, b = np.array(task_emb), np.array(summ_emb)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na and nb:
            relevance = float(np.dot(a, b) / (na * nb))
    except Exception:
        pass

    summary_step = {"summary": summary_text, "task_id": task["task_id"]}
    validation, _, answer_text = validate_task(
        task,
        summary_step,
        encode_fn=encode_fn,
        pass_threshold=0.65,
        use_llm_judge=False,
    )

    return {
        "task_id": task["task_id"],
        "domain": task.get("domain", task.get("tags", ["?"])[0]),
        "strategy": "PURE_TEXT_3CALL",
        **snap,
        "relevance": round(relevance, 3),
        "elapsed_ms": elapsed_ms,
        "composite_score": validation.composite_score,
        "composite_pass": validation.composite_pass,
        "summary_preview": summary_text[:200],
        "summary_full": summary_text,
    }


def run_text_baseline(
    api_key: str,
    suite: str = "full24",
    *,
    run_quality: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], float]:
    tasks = get_suite(suite)
    n = len(tasks)
    embed = EmbeddingEngine(model_name="BAAI/bge-small-en-v1.5", use_real_model=True)
    encode_fn = embed.encode

    results: List[Dict[str, Any]] = []
    wall_t0 = time.time()
    quality_pass = 0

    for i, task in enumerate(tasks):
        print(
            f"  [{i+1:2d}/{n}] {task['task_id']}: {task['description'][:50]}...",
            end=" ",
            flush=True,
        )
        row = run_text_baseline_task(task, api_key, encode_fn)
        if row.get("composite_pass"):
            quality_pass += 1
        print(
            f"calls={row['api_calls']} tok={row['total_tokens']} "
            f"rel={row['relevance']:.3f} "
            f"{'PASS' if row.get('composite_pass') else 'FAIL'} "
            f"{row['elapsed_ms']:.0f}ms"
        )
        results.append(row)

    wall_sec = time.time() - wall_t0
    totals = {
        "total_api_calls": sum(r["api_calls"] for r in results),
        "total_prompt_tokens": sum(r["prompt_tokens"] for r in results),
        "total_completion_tokens": sum(r["completion_tokens"] for r in results),
        "total_tokens": sum(r["total_tokens"] for r in results),
        "total_elapsed_ms": sum(r["elapsed_ms"] for r in results),
        "avg_relevance": sum(r["relevance"] for r in results) / max(n, 1),
        "quality_pass_count": quality_pass if run_quality else None,
        "wall_clock_sec": round(wall_sec, 1),
        "suite": suite,
        "model": MODEL,
        "variant": "A_pure_text",
    }
    return results, totals, wall_sec


def main() -> int:
    parser = argparse.ArgumentParser(description="Pure text baseline (no cache)")
    parser.add_argument(
        "--suite", default="full24",
        choices=["core12", "full24", "extended6", "adversarial6"],
    )
    parser.add_argument(
        "--output", default=None,
        help="JSON path (default: pure_text_baseline_{suite}.json)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set")
        return 1

    n = len(get_suite(args.suite))
    print("=" * 70)
    print(f"  Pure Text Baseline — {args.suite} ({n} tasks)")
    print("  3 LLM calls/task (plan+retrieve+summarize), no cache, verbose prose")
    print("=" * 70)

    results, totals, _ = run_text_baseline(api_key, args.suite)

    print(f"\n  === TOTALS ===")
    print(f"  Tokens: {totals['total_tokens']:,}")
    print(f"  Avg rel: {totals['avg_relevance']:.3f}")
    print(f"  Quality: {totals['quality_pass_count']}/{n}")
    print(f"  Wall: {totals['wall_clock_sec']}s")

    out = args.output or result_path(f"pure_text_baseline_{args.suite}.json")
    payload = {
        "config": {
            "variant": "A_pure_text",
            "suite": args.suite,
            "tasks": n,
            "model": MODEL,
            "calls_per_task": 3,
        },
        "results": results,
        "totals": totals,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"  Exported: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
