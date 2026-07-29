#!/usr/bin/env python3
"""Experiment matrix v2 — addresses prior limitations (see docs/experiment_design_v2.md).

Phases (select with --phase):
  micro     — system_metrics + embedding (full24)
  core      — A/B/C ablations on core12 + single-agent baseline
  extended  — C_full on full24
  adversarial — warm core12 then adversarial6 (false-E2E check)
  quality   — core12 with LLM judge
  factcheck — export human review pack

  all       — run micro + core + extended + adversarial (no pure_text by default)

Usage:
  python3 experiments/experiment_matrix_v2.py --phase all
  python3 experiments/experiment_matrix_v2.py --phase core --skip-existing
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

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

from experiments.benchmark_suites import get_suite, HUMAN_FACTCHECK_TASK_IDS
from experiments.experiment_common import (
    clear_db,
    run_task_batch,
    save_variant_result,
)
from src.run_options import RunOptions, DEFAULT_OPTIONS
from src.paths import (
    EXPERIMENT_MATRIX_V2_DIR,
    database_path,
    resolve_path,
    result_path,
)

OUT_DIR = EXPERIMENT_MATRIX_V2_DIR


def _api_key() -> str:
    k = os.environ.get("DEEPSEEK_API_KEY", "")
    if not k:
        raise RuntimeError("DEEPSEEK_API_KEY not set")
    return k


def _exists(name: str) -> bool:
    return os.path.exists(os.path.join(OUT_DIR, f"{name}.json"))


def _run_variant(
    variant_id: str,
    suite: str,
    options: RunOptions,
    *,
    cold: bool = False,
    db: str | None = None,
    quality: bool = True,
    llm_judge: bool = False,
) -> Dict[str, Any]:
    db_path = database_path(db or f"_v2_{variant_id}.db")
    if not cold:
        clear_db(db_path)
    results, totals = run_task_batch(
        _api_key(),
        get_suite(suite),
        db_path=db_path,
        options=options,
        cold_db_per_task=cold,
        run_quality=quality,
        use_llm_judge=llm_judge,
    )
    save_variant_result(
        variant_id, results, totals, options, OUT_DIR,
        extra_meta={"suite": suite, "cold_db_per_task": cold},
    )
    print(
        f"  {variant_id} [{suite}]: {totals['total_tokens']:,} tok, "
        f"rel={totals['avg_relevance']:.3f}, "
        f"E2E={totals.get('e2e_hits', 0)}, "
        f"Q={totals.get('quality_pass_count', '?')}"
    )
    if totals.get("adversarial_total"):
        safe = totals.get("adversarial_e2e_safe")
        print(f"    adversarial false-E2E: {totals.get('adversarial_false_e2e', 0)}/{totals['adversarial_total']} safe={safe}")
    return {"variant_id": variant_id, "totals": totals, "results": results}


def phase_micro(skip: bool) -> None:
    if skip:
        return
    subprocess.call([sys.executable, "experiments/collect_system_metrics.py"], cwd=_ROOT)
    subprocess.call(
        [sys.executable, "experiments/embedding_matrix.py", "--suite", "full24"],
        cwd=_ROOT,
    )


def phase_core(skip: bool) -> List[Dict[str, Any]]:
    ran = []
    variants = [
        ("C_full_core12", "core12", DEFAULT_OPTIONS, False),
        ("B_nocache_core12", "core12", RunOptions(
            enable_memory_index=False,
            enable_e2e_cache=False,
            enable_planner_cache=False,
            enable_summarizer_cache=False,
        ), False),
        ("C_no_dropout_core12", "core12", RunOptions(enable_executor_dropout=False), False),
        ("C_no_e2e_core12", "core12", RunOptions(enable_e2e_cache=False), False),
        ("C_cold_core12", "core12", DEFAULT_OPTIONS, True),
        ("C_warm_core12", "core12", DEFAULT_OPTIONS, False),
    ]
    for vid, suite, opts, cold in variants:
        if skip and _exists(vid):
            continue
        ran.append(_run_variant(vid, suite, opts, cold=cold))

    if not (skip and os.path.exists(resolve_path("single_agent_baseline_results.json"))):
        subprocess.call(
            [sys.executable, "experiments/single_agent_baseline.py", "core12"],
            cwd=_ROOT,
        )
    return ran


def phase_extended(skip: bool) -> List[Dict[str, Any]]:
    if skip and _exists("C_full_full24"):
        return []
    return [_run_variant("C_full_full24", "full24", DEFAULT_OPTIONS)]


def phase_adversarial(skip: bool) -> List[Dict[str, Any]]:
    """Warm-run core12 into shared DB, then adversarial6 on same DB."""
    ran = []
    db = database_path("_v2_adversarial_warm.db")
    if not (skip and _exists("warm_core12_for_adv")):
        clear_db(db)
        run_task_batch(
            _api_key(), get_suite("core12"), db_path=db,
            options=DEFAULT_OPTIONS, run_quality=False,
        )
        save_variant_result(
            "warm_core12_for_adv", [], {"note": "warmup only"}, DEFAULT_OPTIONS, OUT_DIR,
        )
    if skip and _exists("C_adversarial6_warm"):
        return ran
    ran.append(_run_variant(
        "C_adversarial6_warm", "adversarial6", DEFAULT_OPTIONS,
        cold=False, db=db, quality=True,
    ))
    return ran


def phase_quality(skip: bool) -> List[Dict[str, Any]]:
    if skip and _exists("C_full_core12_llm_judge"):
        return []
    return [_run_variant(
        "C_full_core12_llm_judge", "core12", DEFAULT_OPTIONS,
        quality=True, llm_judge=True,
    )]


def phase_factcheck(skip: bool) -> None:
    if skip and os.path.exists(os.path.join(_ROOT, "docs/human_factcheck_pack.md")):
        return
    subprocess.call([sys.executable, "experiments/factcheck_export.py"], cwd=_ROOT)


def build_report(ran: List[Dict[str, Any]]) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "design": "experiment_design_v2.md",
            "version": "v2.0",
        },
        "variants": {},
    }
    pt_path = resolve_path("pure_text_baseline_results.json")
    if os.path.exists(pt_path):
        with open(pt_path) as f:
            report["pure_text_tokens"] = json.load(f).get("totals", {}).get("total_tokens")
    sa_path = resolve_path("single_agent_baseline_results.json")
    if os.path.exists(sa_path):
        with open(sa_path) as f:
            report["single_agent"] = json.load(f).get("totals", {})

    for fn in os.listdir(OUT_DIR):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(OUT_DIR, fn), encoding="utf-8") as f:
            block = json.load(f)
        vid = block.get("variant_id", fn[:-5])
        t = block.get("totals", {})
        if report.get("pure_text_tokens") and t.get("total_tokens"):
            t["save_vs_pure_text_pct"] = round(
                (1 - t["total_tokens"] / report["pure_text_tokens"]) * 100, 2
            )
        report["variants"][vid] = t

    return report


def write_md(report: Dict[str, Any]) -> None:
    lines = [
        "# 实验矩阵 v2 结果",
        "",
        f"- 时间: {report['meta']['generated_at']}",
        f"- 设计: `docs/experiment_design_v2.md`",
        "",
        "## 变体汇总",
        "",
        "| 变体 | Token | vs 纯文本 | Relevance | E2E | 质量通过 | 对抗 E2E 安全 |",
        "|------|------:|----------:|----------:|----:|---------:|:-------------:|",
    ]
    for vid, t in sorted(report.get("variants", {}).items()):
        save = t.get("save_vs_pure_text_pct")
        save_s = f"{save:.1f}%" if save is not None else "—"
        qp = t.get("quality_pass_count")
        strat = t.get("strategy_distribution") or {}
        n = sum(strat.values()) if strat else (t.get("adversarial_total") or 12)
        qp_s = f"{qp}/{n}" if qp is not None else "—"
        adv = "✓" if t.get("adversarial_e2e_safe") else (
            "✗" if t.get("adversarial_total") else "—"
        )
        lines.append(
            f"| {vid} | {t.get('total_tokens', 0):,} | {save_s} | "
            f"{t.get('avg_relevance', 0):.3f} | {t.get('e2e_hits', 0)} | {qp_s} | {adv} |"
        )
    if report.get("single_agent"):
        sa = report["single_agent"]
        lines.extend([
            "",
            "## 单 Agent 强基线",
            "",
            f"- Token: {sa.get('total_tokens', 0):,}",
            f"- 质量通过: {sa.get('quality_pass_count', '?')}",
        ])
    lines.append("\n人工抽检: `docs/human_factcheck_pack.md`")
    path = os.path.join(_ROOT, "docs", "experiment_matrix_v2_results.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=["micro", "core", "extended", "adversarial", "quality", "factcheck", "all"],
        default="all",
    )
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    ran: List[Dict[str, Any]] = []

    phases = {
        "micro": lambda: phase_micro(args.skip_existing),
        "core": lambda: ran.extend(phase_core(args.skip_existing)),
        "extended": lambda: ran.extend(phase_extended(args.skip_existing)),
        "adversarial": lambda: ran.extend(phase_adversarial(args.skip_existing)),
        "quality": lambda: ran.extend(phase_quality(args.skip_existing)),
        "factcheck": lambda: phase_factcheck(args.skip_existing),
    }

    if args.phase == "all":
        for fn in phases.values():
            fn()
    else:
        phases[args.phase]()

    report = build_report(ran)
    report["meta"]["elapsed_sec"] = round(time.time() - t0, 1)
    out = result_path("experiment_matrix_v2_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    write_md(report)
    print(f"\n=== V2 DONE ({report['meta']['elapsed_sec']}s) ===\nJSON: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
