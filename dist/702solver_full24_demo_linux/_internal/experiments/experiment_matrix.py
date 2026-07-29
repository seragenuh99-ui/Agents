#!/usr/bin/env python3
"""Full experiment matrix: ablations, warm/cold, domain sequence.

Outputs: experiment_matrix_report.json + docs/experiment_matrix_results.md

Usage:
  python3 experiments/experiment_matrix.py --tier1     # ablations + warm/cold (~5 runs)
  python3 experiments/experiment_matrix.py --all       # + pure text + multi-run
  python3 experiments/experiment_matrix.py --variant C_full
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

env_path = os.path.join(_ROOT, ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

from experiments.benchmark_tasks import BENCHMARK_TASKS
from experiments.experiment_common import (
    clear_db,
    run_task_batch,
    save_variant_result,
)
from src.run_options import RunOptions, DEFAULT_OPTIONS
from src.paths import (
    EXPERIMENT_MATRIX_DIR,
    RESULTS_DIR,
    database_path,
    resolve_path,
    result_path,
)

OUT_DIR = EXPERIMENT_MATRIX_DIR
ENERGY_TASKS = [t for t in BENCHMARK_TASKS if t.get("domain") == "energy"]


def _load_pure_text_tokens() -> Optional[int]:
    path = resolve_path("pure_text_baseline_results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f).get("totals", {}).get("total_tokens")


def _run_variant(
    variant_id: str,
    options: RunOptions,
    *,
    tasks: Optional[List[Dict]] = None,
    cold_db_per_task: bool = False,
    db_name: Optional[str] = None,
    run_quality: bool = True,
) -> Dict[str, Any]:
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not set")

    tasks = tasks or BENCHMARK_TASKS
    db_path = database_path(db_name or f"_matrix_{variant_id}.db")
    if not cold_db_per_task:
        clear_db(db_path)

    print(f"\n{'='*60}\n  Variant: {variant_id}\n{'='*60}")
    results, totals = run_task_batch(
        api_key,
        tasks,
        db_path=db_path,
        options=options,
        cold_db_per_task=cold_db_per_task,
        run_quality=run_quality,
    )
    path = save_variant_result(
        variant_id,
        results,
        totals,
        options,
        OUT_DIR,
        extra_meta={
            "task_count": len(tasks),
            "cold_db_per_task": cold_db_per_task,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
    )
    print(
        f"  -> {totals['total_tokens']:,} tokens, "
        f"rel={totals['avg_relevance']:.3f}, "
        f"E2E={totals.get('e2e_hits', 0)}, DROP={totals.get('dropout_hits', 0)}"
    )
    return {"variant_id": variant_id, "path": path, "totals": totals, "results": results}


def build_matrix_report(variants: List[Dict[str, Any]]) -> Dict[str, Any]:
    base_tok = _load_pure_text_tokens()
    report: Dict[str, Any] = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "version": "v0.9.0",
            "pure_text_baseline_tokens": base_tok,
        },
        "variants": {},
        "hypotheses": {},
    }
    by_id = {v["variant_id"]: v for v in variants}
    for vid, v in by_id.items():
        t = v["totals"]
        save = None
        if base_tok and t.get("total_tokens"):
            save = round((1 - t["total_tokens"] / base_tok) * 100, 2)
        report["variants"][vid] = {
            **t,
            "save_vs_pure_text_pct": save,
        }

    # H1/H2 style deltas
    if "B_nocache" in by_id and "C_full" in by_id:
        b, c = by_id["B_nocache"]["totals"]["total_tokens"], by_id["C_full"]["totals"]["total_tokens"]
        report["hypotheses"]["H2_cache_layer"] = {
            "B_tokens": b,
            "C_tokens": c,
            "delta_pct": round((1 - c / b) * 100, 2) if b else None,
        }
    if "C_no_dropout" in by_id and "C_full" in by_id:
        nd, full = by_id["C_no_dropout"]["totals"]["total_tokens"], by_id["C_full"]["totals"]["total_tokens"]
        report["hypotheses"]["H_dropout"] = {
            "no_dropout_tokens": nd,
            "full_tokens": full,
            "dropout_saves_tokens": nd - full,
        }
    if "C_warm_12" in by_id and "C_cold_12" in by_id:
        w, c = by_id["C_warm_12"]["totals"], by_id["C_cold_12"]["totals"]
        report["hypotheses"]["H4_warm_memory"] = {
            "warm_tokens": w["total_tokens"],
            "cold_tokens": c["total_tokens"],
            "warm_e2e_hits": w.get("e2e_hits"),
            "cold_e2e_hits": c.get("e2e_hits"),
        }
    if "C_energy_seq" in by_id:
        report["hypotheses"]["H4_domain_sequence"] = {
            "energy_only_tokens": by_id["C_energy_seq"]["totals"]["total_tokens"],
            "per_task": [
                {"task_id": r["task_id"], "tokens": r["total_tokens"], "strategy": r["strategy"]}
                for r in by_id["C_energy_seq"]["results"]
            ],
        }
    return report


def write_results_md(report: Dict[str, Any]) -> str:
    lines = [
        "# 实验矩阵结果",
        "",
        f"- 生成时间: {report['meta']['generated_at']}",
        f"- 版本: {report['meta']['version']}",
        "",
        "## 变体汇总",
        "",
        "| 变体 | Token | vs 纯文本 | Relevance | E2E | Dropout | 质量通过 |",
        "|------|------:|----------:|----------:|----:|--------:|---------:|",
    ]
    for vid, t in sorted(report.get("variants", {}).items()):
        save = t.get("save_vs_pure_text_pct")
        save_s = f"{save:.1f}%" if save is not None else "—"
        qp = t.get("quality_pass_count")
        qp_s = f"{qp}/{t.get('strategy_distribution', {}) and 12}" if qp is not None else "—"
        lines.append(
            f"| {vid} | {t.get('total_tokens', 0):,} | {save_s} | "
            f"{t.get('avg_relevance', 0):.3f} | {t.get('e2e_hits', 0)} | "
            f"{t.get('dropout_hits', 0)} | {qp_s} |"
        )
    lines.extend(["", "## 假设检验", ""])
    for hid, data in report.get("hypotheses", {}).items():
        lines.append(f"### {hid}")
        lines.append(f"```json\n{json.dumps(data, indent=2, ensure_ascii=False)}\n```\n")
    lines.append("设计文档: `docs/experiments/design_v1/matrix_v1.md`")
    return "\n".join(lines)


VARIANTS: Dict[str, RunOptions] = {
    "C_full": DEFAULT_OPTIONS,
    "C_no_dropout": RunOptions(enable_executor_dropout=False),
    "C_no_e2e": RunOptions(enable_e2e_cache=False),
    "B_nocache": RunOptions(
        enable_memory_index=False,
        enable_e2e_cache=False,
        enable_planner_cache=False,
        enable_summarizer_cache=False,
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier1", action="store_true", help="Ablation + warm/cold (no pure text)")
    parser.add_argument("--all", action="store_true", help="Full matrix including expensive baselines")
    parser.add_argument("--variant", type=str, help="Run single variant id")
    parser.add_argument("--skip-micro", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("ERROR: DEEPSEEK_API_KEY not set")
        return 1

    os.makedirs(OUT_DIR, exist_ok=True)
    ran: List[Dict[str, Any]] = []
    t0 = time.time()

    if not args.skip_micro:
        print("\n>>> Micro: system_metrics + embedding_matrix")
        subprocess.call([sys.executable, "experiments/collect_system_metrics.py"], cwd=_ROOT)
        subprocess.call([sys.executable, "experiments/embedding_matrix.py"], cwd=_ROOT)

    if args.variant:
        opts = VARIANTS.get(args.variant, DEFAULT_OPTIONS)
        ran.append(_run_variant(args.variant, opts))
    elif args.tier1 or args.all:
        for vid, opts in VARIANTS.items():
            ran.append(_run_variant(vid, opts, db_name=f"_matrix_{vid}.db"))

        ran.append(
            _run_variant(
                "C_warm_12", DEFAULT_OPTIONS,
                db_name="_matrix_warm.db", cold_db_per_task=False,
            )
        )
        ran.append(
            _run_variant(
                "C_cold_12", DEFAULT_OPTIONS,
                db_name="_matrix_cold.db", cold_db_per_task=True,
            )
        )
        ran.append(
            _run_variant(
                "C_energy_seq", DEFAULT_OPTIONS,
                tasks=ENERGY_TASKS,
                db_name="_matrix_energy.db",
                cold_db_per_task=False,
            )
        )

        if args.all:
            if not os.path.exists(resolve_path("pure_text_baseline_results.json")):
                print("\n>>> E1 pure_text_baseline")
                subprocess.call([sys.executable, "experiments/pure_text_baseline.py"], cwd=_ROOT)
            print("\n>>> E6 multi_run x3")
            subprocess.call(
                [sys.executable, "experiments/multi_run_benchmark.py", "-n", "3"],
                cwd=_ROOT,
            )
    else:
        parser.print_help()
        return 0

    report = build_matrix_report(ran)
    report["meta"]["elapsed_sec"] = round(time.time() - t0, 1)

    out_json = result_path("experiment_matrix_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    md_path = os.path.join(_ROOT, "docs", "experiment_matrix_results.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(write_results_md(report))

    print(f"\n=== MATRIX DONE ({report['meta']['elapsed_sec']}s) ===")
    print(f"JSON: {out_json}")
    print(f"MD:   {md_path}")

    subprocess.call([sys.executable, "experiments/analyze_experiment_matrix.py"], cwd=_ROOT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
