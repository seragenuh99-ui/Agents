#!/usr/bin/env python3
"""Run optimized benchmark N times and report mean ± std (addresses run variance)."""

from __future__ import annotations

import argparse
import json
import os
import statistics
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

from experiments.benchmark_tasks import BENCHMARK_TASKS
from experiments.optimized_benchmark import run_optimized, clear_db
from src.paths import resolve_path, result_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--runs", type=int, default=3, help="Number of runs (default 3)")
    parser.add_argument(
        "--out", default="multi_run_benchmark_results.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set")
        return 1

    print("=" * 70)
    print(f"  Multi-run Benchmark — {args.runs} independent runs × 12 tasks")
    print("=" * 70)

    all_runs = []
    for run_idx in range(args.runs):
        db = f"_multirun_{run_idx}.db"
        print(f"\n--- Run {run_idx + 1}/{args.runs} ---")
        t0 = time.time()
        results, totals = run_optimized(api_key)
        totals["run_index"] = run_idx
        totals["wall_ms"] = (time.time() - t0) * 1000
        all_runs.append({"results": results, "totals": totals})
        for p in [db, db + "-shm", db + "-wal"]:
            try:
                os.unlink(p)
            except OSError:
                pass

    tokens = [r["totals"]["total_tokens"] for r in all_runs]
    rels = [r["totals"]["avg_relevance"] for r in all_runs]
    calls = [r["totals"]["total_api_calls"] for r in all_runs]

    def mean_std(xs):
        if len(xs) < 2:
            return xs[0], 0.0
        return statistics.mean(xs), statistics.stdev(xs)

    tok_m, tok_s = mean_std(tokens)
    rel_m, rel_s = mean_std(rels)
    call_m, call_s = mean_std(calls)

    baseline_tok = 49076
    pt = resolve_path("pure_text_baseline_results.json")
    if os.path.exists(pt):
        with open(pt) as f:
            baseline_tok = json.load(f).get("totals", {}).get("total_tokens", 49076)

    save_m = (1 - tok_m / baseline_tok) * 100

    summary = {
        "runs": args.runs,
        "tasks_per_run": len(BENCHMARK_TASKS),
        "aggregates": {
            "total_tokens": {"mean": tok_m, "std": tok_s, "values": tokens},
            "avg_relevance": {"mean": rel_m, "std": rel_s, "values": rels},
            "total_api_calls": {"mean": call_m, "std": call_s, "values": calls},
            "token_savings_vs_pure_text_pct": {"mean": save_m, "baseline_tokens": baseline_tok},
        },
        "per_run": all_runs,
    }

    out_path = result_path(os.path.basename(args.out)) if not os.path.isabs(args.out) else args.out
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n  === AGGREGATE ({args.runs} runs) ===")
    print(f"  Tokens:     {tok_m:,.0f} ± {tok_s:,.0f}")
    print(f"  Relevance:  {rel_m:.3f} ± {rel_s:.3f}")
    print(f"  API calls:  {call_m:.1f} ± {call_s:.1f}")
    print(f"  Savings:    {save_m:.1f}% vs pure text ({baseline_tok:,})")
    print(f"\n  Saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
