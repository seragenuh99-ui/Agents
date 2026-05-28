"""Optimized system benchmark — default suite is full24 (see benchmark_full24.py).

Legacy 12-task output: use --suite core12
"""

import argparse
import json
import os
import sys
import time
import tiktoken

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
from src.paths import resolve_path, result_path
from experiments.experiment_common import clear_db, run_task_batch
from src.run_options import DEFAULT_OPTIONS

ENCODER = tiktoken.get_encoding("cl100k_base")
CODE_VERSION = "v0.11.1"
DEFAULT_SUITE = "full24"


def run_optimized(api_key, suite: str = DEFAULT_SUITE, embedding_model: str | None = None):
    """Run tasks through the optimized structured system."""
    tasks = get_suite(suite)
    n = len(tasks)
    clear_db("_opt_bench.db")
    label = embedding_model or os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    print("=" * 70)
    print(f"  Optimized System Benchmark — {CODE_VERSION}")
    print(f"  Suite: {suite} ({n} tasks)")
    print(f"  Embedding: {label}")
    print("=" * 70)

    results, totals = run_task_batch(
        api_key,
        tasks,
        db_path="_opt_bench.db",
        options=DEFAULT_OPTIONS,
        cold_db_per_task=False,
        run_quality=False,
        embedding_model=embedding_model,
    )

    for i, r in enumerate(results):
        print(
            f"  [{i+1:2d}/{n}] {r['task_id']}: {r['strategy']} "
            f"total={r['total_tokens']} rel={r['relevance']:.3f}"
        )

    return results, totals, n


def main():
    parser = argparse.ArgumentParser(description="Optimized benchmark (default: full24)")
    parser.add_argument(
        "--suite",
        default=DEFAULT_SUITE,
        choices=["core12", "full24", "extended6", "adversarial6"],
        help="Task suite (default: full24)",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Sentence-transformers model (default: EMBEDDING_MODEL env or bge-small)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="JSON output path (default: optimized_benchmark_results.json)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set")
        return 1

    emb = args.embedding_model or os.environ.get("EMBEDDING_MODEL")
    n_tasks = len(get_suite(args.suite))
    print("=" * 70)
    print("  Optimized System Benchmark — compact templates + compressed memory")
    print(f"  {n_tasks} tasks, DeepSeek API — prefer: python3 experiments/benchmark_full24.py")
    print("=" * 70)

    results, totals, n = run_optimized(api_key, suite=args.suite, embedding_model=emb)

    print(f"\n  === OPTIMIZED SYSTEM TOTALS ===")
    print(f"  Total API calls:         {totals['total_api_calls']}")
    print(f"  Total prompt tokens:     {totals['total_prompt_tokens']:,}")
    print(f"  Total completion tokens: {totals['total_completion_tokens']:,}")
    print(f"  Total LLM tokens:        {totals['total_tokens']:,}")
    print(f"  Total cached tokens:     {totals['total_cached_tokens']:,}")
    print(f"  Avg relevance:           {totals['avg_relevance']:.3f}")
    print(f"  Total elapsed:           {totals['total_elapsed_ms']:,.0f}ms")

    # Per-task table
    print(f"\n  Per-task breakdown:")
    print(f"  {'Task':<22} {'Domain':<10} {'Strategy':<14} {'Calls':>6} {'Prompt':>8} {'Compl':>8} {'Total':>8} {'Rel':>6}")
    print(f"  {'─'*22} {'─'*10} {'─'*14} {'─'*6} {'─'*8} {'─'*8} {'─'*8} {'─'*6}")
    for r in results:
        print(f"  {r['task_id']:<22} {r['domain']:<10} {r['strategy']:<14} {r['api_calls']:>6} {r['prompt_tokens']:>8,} {r['completion_tokens']:>8,} {r['total_tokens']:>8,} {r['relevance']:>6.3f}")

    # Strategy distribution
    from collections import Counter
    strat_counts = Counter(r["strategy"] for r in results)
    print(f"\n  Strategy distribution:")
    for s, c in strat_counts.most_common():
        print(f"    {s}: {c}/{n}")

    # Export
    emb_label = emb or os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    output = {
        "config": {
            "model": "deepseek-chat",
            "tasks": n,
            "suite": args.suite,
            "version": CODE_VERSION,
            "embedding_model": emb_label,
        },
        "results": results,
        "totals": totals,
    }
    out_path = args.output or result_path("optimized_benchmark_results.json")
    if not os.path.isabs(out_path):
        out_path = result_path(os.path.basename(out_path))
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Results exported to: {out_path}")

    # Compare with previous baselines if available
    for baseline_file in ["pure_text_baseline_results.json", "comparison_benchmark_results.json"]:
        bp = resolve_path(baseline_file)
        if os.path.exists(bp):
            with open(bp) as f:
                prev = json.load(f)
            prev_totals = prev.get("totals", prev.get("with_cache", {}).get("totals", {}))
            if prev_totals:
                prev_tok = prev_totals.get("total_tokens", 0)
                if prev_tok > 0:
                    save_pct = (1 - totals["total_tokens"] / prev_tok) * 100
                    print(f"  vs {baseline_file}: {totals['total_tokens']:,} vs {prev_tok:,} tokens (save {save_pct:.1f}%)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
