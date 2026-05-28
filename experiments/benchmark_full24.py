#!/usr/bin/env python3
"""Standard 24-task benchmark (v0.11.1) — default for all future experiments.

Runs:
  1. C_full on full24 (warm memory, quality validation)
  2. B_nocache on full24 (ablation: no cache)
  3. adversarial6 after core12 warm-up (false-E2E check)

Outputs:
  - output/results/full24_benchmark_report.json
  - docs/full24_benchmark_results.md
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

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

from experiments.benchmark_suites import get_suite
from experiments.experiment_common import clear_db, run_task_batch, save_variant_result
from src.paths import EXPERIMENT_MATRIX_V2_DIR, database_path, resolve_path, result_path
from src.run_options import DEFAULT_OPTIONS, RunOptions

OUT_DIR = EXPERIMENT_MATRIX_V2_DIR
CODE_VERSION = "v0.11.2"


def _api_key() -> str:
    k = os.environ.get("DEEPSEEK_API_KEY", "")
    if not k:
        raise RuntimeError("DEEPSEEK_API_KEY not set")
    return k


def _run(
    variant_id: str,
    suite: str,
    options: RunOptions,
    *,
    cold: bool = False,
    db: str | None = None,
    use_llm_judge: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], float]:
    db_path = database_path(db or f"_full24_{variant_id}.db")
    if not cold:
        clear_db(db_path)
    t0 = time.time()
    print(f"\n=== {variant_id} [{suite}] ===")
    results, totals = run_task_batch(
        _api_key(),
        get_suite(suite),
        db_path=db_path,
        options=options,
        cold_db_per_task=cold,
        run_quality=True,
        use_llm_judge=use_llm_judge,
    )
    elapsed_sec = time.time() - t0
    save_variant_result(
        variant_id, results, totals, options, OUT_DIR,
        extra_meta={"suite": suite, "code_version": CODE_VERSION},
    )
    n = len(results)
    qp = totals.get("quality_pass_count", 0)
    print(
        f"  tokens={totals['total_tokens']:,}  rel={totals['avg_relevance']:.3f}  "
        f"quality={qp}/{n}  wall={elapsed_sec:.0f}s  "
        f"avg/task={totals['total_elapsed_ms']/max(n,1)/1000:.1f}s"
    )
    return results, totals, elapsed_sec


def _nocache_options() -> RunOptions:
    return RunOptions(
        enable_memory_index=False,
        enable_e2e_cache=False,
        enable_planner_cache=False,
        enable_summarizer_cache=False,
    )


def build_comparison(report: Dict[str, Any]) -> None:
    variants = report["variants"]
    full = variants.get("C_full_full24", {})
    nocache = variants.get("B_nocache_full24", {})

    ctrl_path = resolve_path("full24_controlled_comparison.json")
    pt24 = None
    if os.path.exists(ctrl_path):
        with open(ctrl_path, encoding="utf-8") as f:
            ctrl = json.load(f)
        pt24 = ctrl.get("variants", {}).get("A_pure_text_full24", {}).get("total_tokens")
    full_tok = full.get("total_tokens", 0)
    if full_tok and pt24:
        full["save_vs_pure_text_full24_pct"] = round((1 - full_tok / pt24) * 100, 2)
    if full_tok and nocache.get("total_tokens"):
        full["save_vs_nocache_pct"] = round(
            (1 - full_tok / nocache["total_tokens"]) * 100, 2
        )
    report["reference"] = {
        "pure_text_full24": pt24,
        "note": "Use full24_controlled_comparison for same-condition A vs C.",
    }


def write_md(report: Dict[str, Any]) -> str:
    ref = report.get("reference", {})
    lines = [
        "# full24 标准基准结果（当前系统）",
        "",
        f"- **生成时间**: {report['meta']['generated_at']}",
        f"- **代码版本**: {report['meta']['code_version']}",
        f"- **任务集**: full24（core12 + extended6 + adversarial6）",
        f"- **命令**: `python3 experiments/benchmark_full24.py`",
        "",
        "## 汇总对比",
        "",
        "| 变体 | Token | vs 纯文本(按24题折算) | vs 无缓存 | 平均相关度 | 质量通过 | 总耗时 | 平均每题 |",
        "|------|------:|----------------------:|----------:|----------:|---------:|-------:|---------:|",
    ]
    for vid in ("C_full_full24", "B_nocache_full24", "C_adversarial6_warm"):
        t = report["variants"].get(vid, {})
        if not t:
            continue
        n = sum((t.get("strategy_distribution") or {}).values()) or (
            t.get("adversarial_total") or 24
        )
        save_pt = t.get("save_vs_pure_text_scaled_pct")
        save_nc = t.get("save_vs_nocache_pct") if vid == "C_full_full24" else None
        qp = t.get("quality_pass_count")
        wall = t.get("wall_clock_sec", 0)
        avg_task = t.get("total_elapsed_ms", 0) / max(n, 1) / 1000
        lines.append(
            f"| {vid} | {t.get('total_tokens', 0):,} | "
            f"{f'{save_pt:.1f}%' if save_pt is not None else '—'} | "
            f"{f'{save_nc:.1f}%' if save_nc is not None else '—'} | "
            f"{t.get('avg_relevance', 0):.3f} | "
            f"{qp}/{n} | {wall:.0f}s | {avg_task:.1f}s |"
        )
    adv = report["variants"].get("C_adversarial6_warm", {})
    if adv.get("adversarial_total"):
        lines.extend([
            "",
            "## 对抗集（陷阱题）",
            "",
            f"- 误用 E2E: **{adv.get('adversarial_false_e2e', '?')}/{adv.get('adversarial_total')}** "
            f"（目标 0）",
            f"- 安全: **{'是' if adv.get('adversarial_e2e_safe') else '否'}**",
        ])
    lines.extend([
        "",
        "## 说明",
        "",
        f"- 纯文本基线（12 题实测）: **{ref.get('pure_text_12_tasks', PURE_TEXT_12):,}** Token",
        f"- 折算 24 题对比: **{ref.get('pure_text_scaled_24', _scaled_pure_text(24)):,}**（{ref.get('note', '')}）",
        "- **以后实验默认用 full24**；12 题 core12 仅作历史对照。",
    ])
    path = os.path.join(_ROOT, "docs", "full24_benchmark_results.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Standard full24 benchmark")
    parser.add_argument(
        "--skip-adversarial", action="store_true",
        help="Skip warm-up + adversarial6 (faster)",
    )
    parser.add_argument(
        "--only", choices=["full", "nocache", "adversarial", "all"], default="all",
    )
    args = parser.parse_args()

    t_all = time.time()
    report: Dict[str, Any] = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "code_version": CODE_VERSION,
            "suite": "full24",
            "design": "docs/experiment_design_v2.md",
        },
        "variants": {},
    }

    print("=" * 70)
    print(f"  full24 Standard Benchmark — {CODE_VERSION}")
    print("=" * 70)

    if args.only in ("all", "full"):
        _, totals, wall = _run(
            "C_full_full24", "full24", DEFAULT_OPTIONS, cold=False
        )
        totals["wall_clock_sec"] = round(wall, 1)
        report["variants"]["C_full_full24"] = totals

    if args.only in ("all", "nocache"):
        _, totals, wall = _run(
            "B_nocache_full24", "full24", _nocache_options(), cold=False
        )
        totals["wall_clock_sec"] = round(wall, 1)
        report["variants"]["B_nocache_full24"] = totals

    if args.only in ("all", "adversarial") and not args.skip_adversarial:
        db = database_path("_full24_adversarial_warm.db")
        clear_db(db)
        print("\n=== warm_core12_for_adv (warm-up) ===")
        run_task_batch(
            _api_key(), get_suite("core12"), db_path=db,
            options=DEFAULT_OPTIONS, run_quality=False,
        )
        _, totals, wall = _run(
            "C_adversarial6_warm", "adversarial6", DEFAULT_OPTIONS,
            cold=False, db=db,
        )
        totals["wall_clock_sec"] = round(wall, 1)
        report["variants"]["C_adversarial6_warm"] = totals

    build_comparison(report)
    report["meta"]["elapsed_sec"] = round(time.time() - t_all, 1)

    out_json = result_path("full24_benchmark_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    md_path = write_md(report)
    print(f"\n  Total wall time: {report['meta']['elapsed_sec']}s")
    print(f"  JSON: {out_json}")
    print(f"  MD:   {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
