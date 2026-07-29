#!/usr/bin/env python3
"""Single-agent strong baseline: one LLM chain per task (no multi-agent cache).

Answers: "Is multi-agent necessary?" — compare tokens/quality vs C_full.
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

import numpy as np

from experiments.benchmark_suites import get_suite
from src.agents.base import LLMBackend, _extract_json
from src.evaluation.quality_validator import validate_task
from src.state.embeddings import EmbeddingEngine


SYSTEM = (
    "You are a research assistant. Answer the task with JSON only: "
    '{"key_findings":["..."],"facts":["..."],"conclusion":"one sentence"} '
    "Max 5 findings, 3 facts. Be concise and on-topic."
)


def run_single_agent(api_key: str, suite: str = "core12") -> dict:
    llm = LLMBackend(provider="deepseek", api_key=api_key, model="deepseek-chat")
    eng = EmbeddingEngine(use_real_model=True)
    tasks = get_suite(suite)
    results = []

    for i, task in enumerate(tasks):
        llm.reset_stats()
        t0 = time.time()
        user = f"Task:\n{task['description']}\n\nRespond in JSON."
        raw = llm.chat(
            [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user},
            ],
            max_tokens=1024,
            temperature=0.2,
        )
        elapsed = (time.time() - t0) * 1000
        stats = llm.get_usage_stats()

        try:
            summary_obj = _extract_json(raw)
        except Exception:
            summary_obj = {"key_findings": [raw[:300]], "facts": [], "conclusion": ""}

        summary_step = {"summary": summary_obj}
        v, _, text = validate_task(
            task,
            summary_step,
            encode_fn=eng.encode,
            use_llm_judge=False,
        )

        rel = 0.0
        if text:
            a, b = np.array(eng.encode(task["description"])), np.array(eng.encode(text[:500]))
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na and nb:
                rel = float(np.dot(a, b) / (na * nb))

        results.append({
            "task_id": task["task_id"],
            "domain": task.get("domain"),
            "total_tokens": stats["total_prompt_tokens"] + stats["total_completion_tokens"],
            "prompt_tokens": stats["total_prompt_tokens"],
            "completion_tokens": stats["total_completion_tokens"],
            "api_calls": stats["call_count"],
            "elapsed_ms": elapsed,
            "composite_score": v.composite_score,
            "composite_pass": v.composite_pass,
            "relevance": round(rel, 3),
            "summary_preview": text[:200],
        })
        print(
            f"  [{i+1}/{len(tasks)}] {task['task_id']}: "
            f"{results[-1]['total_tokens']} tok pass={v.composite_pass}"
        )

    totals = {
        "total_tokens": sum(r["total_tokens"] for r in results),
        "total_api_calls": sum(r["api_calls"] for r in results),
        "avg_composite": sum(r["composite_score"] for r in results) / len(results),
        "quality_pass_count": sum(1 for r in results if r["composite_pass"]),
    }
    return {"suite": suite, "results": results, "totals": totals}


def main() -> int:
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set")
        return 1

    suite = "core12"
    if len(sys.argv) > 1:
        suite = sys.argv[1]

    print(f"=== Single-agent baseline (suite={suite}) ===")
    out = run_single_agent(api_key, suite)
    from src.paths import result_path

    path = result_path("single_agent_baseline_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nTotal tokens: {out['totals']['total_tokens']:,}")
    print(f"Quality pass: {out['totals']['quality_pass_count']}/{len(out['results'])}")
    print(f"Saved: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
