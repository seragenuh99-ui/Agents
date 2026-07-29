#!/usr/bin/env python3
"""Controlled comparison on full24 (same tasks, same model, same API).

Variants (control: task set = full24, model = deepseek-chat):
  A  pure_text   — verbose prose, 3 calls/task, no memory
  C  structured  — v0.11.2 full pipeline + warm memory
  B  nocache     — structured, all caches off

Output:
  output/results/full24_controlled_comparison.json
  output/results/pure_text_baseline_full24.json
  output/results/structured_full24_controlled.json
  output/results/structured_nocache_full24.json
  docs/full24_controlled_comparison.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

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

from experiments.experiment_common import clear_db, run_task_batch
from experiments.benchmark_suites import get_suite
from experiments.pure_text_baseline import run_text_baseline
from src.paths import result_path
from src.run_options import DEFAULT_OPTIONS, RunOptions

SUITE = "full24"
CODE_VERSION = "v0.11.2"


def _nocache() -> RunOptions:
    return RunOptions(
        enable_memory_index=False,
        enable_e2e_cache=False,
        enable_planner_cache=False,
        enable_summarizer_cache=False,
        enable_executor_dropout=False,
    )


def _pct(save: float, base: float) -> float:
    if not base:
        return 0.0
    return round((1 - save / base) * 100, 2)


def _row_from_totals(vid: str, totals: Dict[str, Any], n: int) -> Dict[str, Any]:
    return {
        "variant_id": vid,
        "total_tokens": totals.get("total_tokens", 0),
        "total_api_calls": totals.get("total_api_calls", 0),
        "total_prompt_tokens": totals.get("total_prompt_tokens", 0),
        "total_completion_tokens": totals.get("total_completion_tokens", 0),
        "avg_relevance": round(totals.get("avg_relevance", 0), 4),
        "quality_pass": totals.get("quality_pass_count"),
        "quality_rate": (
            f"{totals['quality_pass_count']}/{n}"
            if totals.get("quality_pass_count") is not None
            else "—"
        ),
        "wall_clock_sec": totals.get("wall_clock_sec", 0),
        "avg_sec_per_task": round(
            totals.get("wall_clock_sec", 0) / max(n, 1), 2
        ),
        "e2e_hits": totals.get("e2e_hits"),
        "dropout_hits": totals.get("dropout_hits"),
    }


def _pass_flag(rec: Dict[str, Any]) -> bool:
    if "composite_pass" in rec:
        return bool(rec["composite_pass"])
    return bool(rec.get("quality_pass", True))


def build_per_task(
    a_results: List[Dict[str, Any]],
    c_results: List[Dict[str, Any]],
    b_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    a_by = {r["task_id"]: r for r in a_results}
    c_by = {r["task_id"]: r for r in c_results}
    b_by = {r["task_id"]: r for r in b_results}
    rows: List[Dict[str, Any]] = []
    for tid in [r["task_id"] for r in a_results]:
        a, c, b = a_by[tid], c_by[tid], b_by[tid]
        at, ct, bt = a["total_tokens"], c["total_tokens"], b["total_tokens"]
        rows.append(
            {
                "task_id": tid,
                "A_tokens": at,
                "B_tokens": bt,
                "C_tokens": ct,
                "C_vs_A_save_pct": round((1 - ct / at) * 100, 1) if at else 0,
                "A_relevance": round(a.get("relevance", 0), 3),
                "B_relevance": round(b.get("relevance", 0), 3),
                "C_relevance": round(c.get("relevance", 0), 3),
                "A_quality_pass": _pass_flag(a),
                "B_quality_pass": _pass_flag(b),
                "C_quality_pass": _pass_flag(c),
                "C_strategy": c.get("strategy"),
            }
        )
    return rows


def _per_task_md_table(per_task: List[Dict[str, Any]]) -> str:
    lines = [
        "| 题号 | A Token | B Token | C Token | C省% | A相关 | C相关 | A | B | C | C策略 |",
        "|------|--------:|--------:|--------:|-----:|------:|------:|:---:|:---:|:---:|------|",
    ]
    for r in per_task:
        lines.append(
            f"| `{r['task_id']}` | {r['A_tokens']:,} | {r['B_tokens']:,} | "
            f"{r['C_tokens']:,} | {r['C_vs_A_save_pct']}% | "
            f"{r['A_relevance']:.3f} | {r['C_relevance']:.3f} | "
            f"{'✓' if r['A_quality_pass'] else '✗'} | "
            f"{'✓' if r['B_quality_pass'] else '✗'} | "
            f"{'✓' if r['C_quality_pass'] else '✗'} | {r['C_strategy']} |"
        )
    return "\n".join(lines)


def write_md(report: Dict[str, Any], path: str) -> None:
    n = report["meta"]["task_count"]
    rows = report["variants"]
    a_tok = rows["A_pure_text_full24"]["total_tokens"]
    c_tok = rows["C_structured_full24"]["total_tokens"]
    b_tok = rows["B_nocache_full24"]["total_tokens"]

    lines = [
        "# full24 对照实验（控制变量）",
        "",
        f"- **时间**: {report['meta']['generated_at']}",
        f"- **代码**: {report['meta']['code_version']}",
        f"- **控制**: 同一任务集 `{SUITE}`（{n} 题）、同一模型 `deepseek-chat`、真 API",
        f"- **命令**: `python3 experiments/full24_controlled_comparison.py`",
        "",
        "## 实验设计",
        "",
        "| 变体 | 条件 |",
        "|------|------|",
        "| **A 纯文本** | 每题 3 次 LLM（规划+检索+综合），长 prompt、散文输出、**无记忆、无结构化协议** |",
        f"| **C 结构化全功能** | Orchestrator + 记忆 + 模板/E2E/Dropout（{CODE_VERSION}） |",
        "| **B 结构化无缓存** | 同 C，但关闭全部缓存与 Dropout |",
        "",
        "## 主表（同条件对比）",
        "",
        "| 变体 | Token | API 调用 | 平均相关度 | 质量通过 | 墙钟总耗时 | 平均每题 |",
        "|------|------:|---------:|----------:|---------:|-----------:|---------:|",
    ]
    for vid in ("A_pure_text_full24", "C_structured_full24", "B_nocache_full24"):
        r = rows[vid]
        lines.append(
            f"| {vid} | {r['total_tokens']:,} | {r['total_api_calls']} | "
            f"{r['avg_relevance']:.3f} | {r['quality_rate']} | "
            f"{r['wall_clock_sec']}s | {r['avg_sec_per_task']}s |"
        )

    lines.extend([
        "",
        "## Token 节省（相对对照）",
        "",
        "| 对比 | 节省 Token | 说明 |",
        "|------|----------:|------|",
        f"| C vs A（主结论） | **{_pct(c_tok, a_tok)}%** | {a_tok:,} → {c_tok:,} |",
        f"| C vs B | **{_pct(c_tok, b_tok)}%** | 记忆+剪枝相对无缓存 |",
        f"| B vs A | {_pct(b_tok, a_tok)}% | 仅结构化协议、无记忆 |",
        "",
        "## 准确性",
        "",
    ])
    for vid in ("A_pure_text_full24", "C_structured_full24", "B_nocache_full24"):
        r = rows[vid]
        lines.append(
            f"- **{vid}**: 质量 {r['quality_rate']}，相关度 {r['avg_relevance']:.3f}"
        )

    adv = report.get("adversarial")
    if adv:
        lines.extend([
            "",
            "## 对抗子集（C 结构化，热记忆后 6 题）",
            "",
            f"- 误用 E2E: {adv.get('adversarial_false_e2e', '?')}/"
            f"{adv.get('adversarial_total', 6)}",
            f"- 对抗质量通过: {adv.get('quality_pass_count', '?')}/"
            f"{adv.get('adversarial_total', 6)}",
            f"- 安全: {adv.get('adversarial_e2e_safe')}",
        ])

    failed = [
        r["task_id"]
        for r in report.get("per_task", [])
        if not r.get("C_quality_pass")
    ]
    if failed:
        lines.extend([
            "",
            "## C 未通过题",
            "",
        ])
        for tid in failed:
            lines.append(f"- `{tid}`")

    per_task = report.get("per_task")
    if per_task:
        lines.extend([
            "",
            "## 逐题对比（A / B / C）",
            "",
            _per_task_md_table(per_task),
        ])

    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _load_pure_text_baseline() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    path = result_path("pure_text_baseline_full24.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["results"], data["totals"]


def main() -> int:
    parser = argparse.ArgumentParser(description="full24 controlled comparison")
    parser.add_argument(
        "--skip-pure-text",
        action="store_true",
        help="Reuse output/results/pure_text_baseline_full24.json (A unchanged)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set")
        return 1

    tasks = get_suite(SUITE)
    n = len(tasks)
    t_all = time.time()

    print("=" * 70)
    print(f"  full24 Controlled Comparison — {CODE_VERSION}")
    print(f"  {n} tasks, same suite, deepseek-chat")
    print("=" * 70)

    report: Dict[str, Any] = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "code_version": CODE_VERSION,
            "suite": SUITE,
            "task_count": n,
            "model": "deepseek-chat",
            "control_variables": {
                "task_set": SUITE,
                "model": "deepseek-chat",
                "api": "DeepSeek real",
            },
        },
        "variants": {},
    }

    pt_results: List[Dict[str, Any]]
    pt_totals: Dict[str, Any]

    if args.skip_pure_text:
        print("\n>>> A_pure_text_full24 (skipped — loading prior results)")
        pt_path = result_path("pure_text_baseline_full24.json")
        if not os.path.isfile(pt_path):
            print(f"ERROR: missing {pt_path}; run without --skip-pure-text first")
            return 1
        pt_results, pt_totals = _load_pure_text_baseline()
        report["meta"]["A_pure_text_source"] = "cached"
    else:
        print("\n>>> A_pure_text_full24")
        pt_results, pt_totals, pt_wall = run_text_baseline(api_key, SUITE)
        pt_totals["wall_clock_sec"] = round(pt_wall, 1)
        pt_path = result_path("pure_text_baseline_full24.json")
        with open(pt_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "config": {"variant": "A_pure_text", "suite": SUITE, "tasks": n},
                    "results": pt_results,
                    "totals": pt_totals,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    report["variants"]["A_pure_text_full24"] = _row_from_totals(
        "A_pure_text_full24", pt_totals, n
    )

    print("\n>>> C_structured_full24")
    clear_db("_controlled_C.db")
    t0 = time.time()
    c_results, c_totals = run_task_batch(
        api_key,
        tasks,
        db_path="_controlled_C.db",
        options=DEFAULT_OPTIONS,
        cold_db_per_task=False,
        run_quality=True,
    )
    c_totals["wall_clock_sec"] = round(time.time() - t0, 1)
    report["variants"]["C_structured_full24"] = _row_from_totals(
        "C_structured_full24", c_totals, n
    )
    with open(result_path("structured_full24_controlled.json"), "w") as f:
        json.dump({"results": c_results, "totals": c_totals}, f, indent=2)

    print("\n>>> B_nocache_full24")
    clear_db("_controlled_B.db")
    t0 = time.time()
    b_results, b_totals = run_task_batch(
        api_key,
        tasks,
        db_path="_controlled_B.db",
        options=_nocache(),
        cold_db_per_task=False,
        run_quality=True,
    )
    b_totals["wall_clock_sec"] = round(time.time() - t0, 1)
    report["variants"]["B_nocache_full24"] = _row_from_totals(
        "B_nocache_full24", b_totals, n
    )
    with open(result_path("structured_nocache_full24.json"), "w") as f:
        json.dump({"results": b_results, "totals": b_totals}, f, indent=2)

    report["per_task"] = build_per_task(pt_results, c_results, b_results)

    a_tok = report["variants"]["A_pure_text_full24"]["total_tokens"]
    c_tok = report["variants"]["C_structured_full24"]["total_tokens"]
    b_tok = report["variants"]["B_nocache_full24"]["total_tokens"]
    report["comparisons"] = {
        "C_vs_A_save_pct": _pct(c_tok, a_tok),
        "C_vs_B_save_pct": _pct(c_tok, b_tok),
        "B_vs_A_save_pct": _pct(b_tok, a_tok),
        "tokens": {"A": a_tok, "C": c_tok, "B": b_tok},
    }

    print("\n>>> adversarial6 (warm core12 then traps)")
    adv_db = "_controlled_adv.db"
    clear_db(adv_db)
    run_task_batch(
        api_key,
        get_suite("core12"),
        db_path=adv_db,
        options=DEFAULT_OPTIONS,
        run_quality=False,
    )
    t0 = time.time()
    adv_results, adv_totals = run_task_batch(
        api_key,
        get_suite("adversarial6"),
        db_path=adv_db,
        options=DEFAULT_OPTIONS,
        run_quality=True,
    )
    adv_totals["wall_clock_sec"] = round(time.time() - t0, 1)
    with open(result_path("adversarial6_controlled.json"), "w") as f:
        json.dump({"results": adv_results, "totals": adv_totals}, f, indent=2)
    report["adversarial"] = {
        "total_tokens": adv_totals.get("total_tokens"),
        "quality_pass_count": adv_totals.get("quality_pass_count"),
        "adversarial_false_e2e": adv_totals.get("adversarial_false_e2e"),
        "adversarial_total": adv_totals.get("adversarial_total"),
        "adversarial_e2e_safe": adv_totals.get("adversarial_e2e_safe"),
        "avg_relevance": adv_totals.get("avg_relevance"),
    }

    report["meta"]["elapsed_sec"] = round(time.time() - t_all, 1)

    out_json = result_path("full24_controlled_comparison.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    md_path = os.path.join(_ROOT, "docs", "full24_controlled_comparison.md")
    write_md(report, md_path)

    print("\n" + "=" * 70)
    print("  CONTROLLED COMPARISON SUMMARY")
    print("=" * 70)
    for vid in ("A_pure_text_full24", "C_structured_full24", "B_nocache_full24"):
        r = report["variants"][vid]
        print(
            f"  {vid}: {r['total_tokens']:,} tok, "
            f"Q={r['quality_rate']}, rel={r['avg_relevance']:.3f}, "
            f"{r['wall_clock_sec']}s"
        )
    cmp_ = report["comparisons"]
    print(f"\n  C vs A (pure text): save {cmp_['C_vs_A_save_pct']}%")
    print(f"  C vs B (no cache):  save {cmp_['C_vs_B_save_pct']}%")
    failed = [r["task_id"] for r in report["per_task"] if not r["C_quality_pass"]]
    if failed:
        print(f"  C failures: {failed}")
    print(f"  JSON: {out_json}")
    print(f"  MD:   {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
