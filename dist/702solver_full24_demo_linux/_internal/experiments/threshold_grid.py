#!/usr/bin/env python3
"""Grid search RunOptions thresholds on core12 (records JSON for analysis)."""

from __future__ import annotations

import argparse
import itertools
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

from dataclasses import replace

from experiments.benchmark_suites import get_suite
from experiments.experiment_common import clear_db, run_task_batch
from src.paths import result_path
from src.run_options import DEFAULT_OPTIONS


def main() -> None:
    parser = argparse.ArgumentParser(description="Threshold grid on core12")
    parser.add_argument("--dry-run", action="store_true", help="Print combos only")
    parser.add_argument("--db", default="_threshold_grid.db")
    args = parser.parse_args()

    e2e_vals = [0.82, 0.85, 0.88]
    drop_vals = [0.60, 0.65, 0.70]
    combos = list(itertools.product(e2e_vals, drop_vals))

    if args.dry_run:
        print(f"Would run {len(combos)} combinations")
        for e2e, drop in combos:
            print(f"  e2e={e2e} dropout={drop}")
        return

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("DEEPSEEK_API_KEY required")
        sys.exit(1)

    tasks = get_suite("core12")
    rows = []
    t_all = time.time()

    for e2e_th, drop_th in combos:
        clear_db(args.db)
        opts = replace(
            DEFAULT_OPTIONS,
            e2e_threshold=e2e_th,
            summarizer_e2e_threshold=e2e_th,
            executor_dropout_threshold=drop_th,
        )
        label = f"e2e{e2e_th}_drop{drop_th}"
        print(f"\n=== {label} ===")
        results, totals = run_task_batch(
            api_key,
            tasks,
            db_path=args.db,
            options=opts,
            cold_db_per_task=False,
            reset_llm_each_task=True,
            run_quality=False,
        )
        row = {
            "label": label,
            "e2e_threshold": e2e_th,
            "executor_dropout_threshold": drop_th,
            "total_tokens": totals.get("total_tokens"),
            "avg_relevance": totals.get("avg_relevance"),
            "e2e_hits": totals.get("e2e_hits"),
            "dropout_hits": totals.get("dropout_hits"),
            "strategy_distribution": totals.get("strategy_distribution"),
        }
        rows.append(row)
        print(
            f"  tokens={row['total_tokens']} rel={row['avg_relevance']:.3f} "
            f"e2e={row['e2e_hits']} drop={row['dropout_hits']}"
        )

    out = {
        "elapsed_sec": round(time.time() - t_all, 1),
        "combinations": rows,
    }
    path = result_path("threshold_grid_results.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
