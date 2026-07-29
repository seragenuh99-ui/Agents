#!/usr/bin/env python3
"""One-shot experiment suite: ablation + optimized + quality summary.

Usage:
  python3 experiments/full_evaluation_suite.py
  python3 experiments/full_evaluation_suite.py --skip-existing
  python3 experiments/full_evaluation_suite.py --quick   # optimized + embedding only
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

from src.paths import resolve_path, result_path


def _load_json(path: str) -> Optional[Dict[str, Any]]:
    full = resolve_path(path)
    if not os.path.exists(full):
        return None
    with open(full, encoding="utf-8") as f:
        return json.load(f)


def _run_script(rel_path: str, extra_args: List[str] = None) -> int:
    cmd = [sys.executable, os.path.join(_ROOT, "experiments", rel_path)]
    if extra_args:
        cmd.extend(extra_args)
    print(f"\n>>> {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=_ROOT)


def _pct_save(opt: int, base: int) -> float:
    if base <= 0:
        return 0.0
    return (1 - opt / base) * 100


def build_summary_md(report: Dict[str, Any]) -> str:
    lines = [
        "# 完整实验汇总",
        "",
        f"- 生成时间: {report['meta']['generated_at']}",
        f"- 版本: {report['meta'].get('version', 'v0.6.0')}",
        "",
        "## 1. Token 对照（赛题通信效率）",
        "",
        "| 模式 | 总 Token | vs 纯文本 | 平均 Relevance | API 调用 |",
        "|------|----------|----------|----------------|----------|",
    ]
    base_tok = report.get("pure_text", {}).get("total_tokens")
    for key, label in [
        ("pure_text", "A 纯文本基线"),
        ("structured_nocache", "B 结构化无缓存"),
        ("optimized", "C 全功能优化"),
    ]:
        block = report.get(key)
        if not block:
            lines.append(f"| {label} | — | — | — | — |")
            continue
        tok = block.get("total_tokens", 0)
        save = f"{_pct_save(tok, base_tok):.1f}%" if base_tok else "—"
        lines.append(
            f"| {label} | {tok:,} | {save} | {block.get('avg_relevance', '—')} | "
            f"{block.get('total_api_calls', '—')} |"
        )

    lines.extend([
        "",
        "## 2. 质量验证",
        "",
    ])
    q = report.get("quality")
    if q:
        lines.append(f"- 综合通过: {q.get('composite_pass_count')}/{q.get('task_count')}")
        lines.append(f"- 平均综合分: {q.get('avg_composite_score', 0):.3f}")
        lines.append(f"- 详见: `docs/quality_records.md`")
    else:
        lines.append("- 未运行（执行 `quality_benchmark.py`）")

    lines.extend([
        "",
        "## 3. 缓存相似度（E2E 候选对）",
        "",
    ])
    emb = report.get("embedding_pairs")
    if emb:
        for p in emb[:10]:
            lines.append(f"- {p['a']} ↔ {p['b']}: cos={p['cos']}")
    else:
        lines.append("- 未运行 `embedding_matrix.py`")

    lines.extend([
        "",
        "## 4. 实验文件索引",
        "",
        "| 实验 | 文件 |",
        "|------|------|",
        "| E1 纯文本 | `pure_text_baseline_results.json` |",
        "| E2 全功能 | `output/results/optimized_benchmark_results.json` |",
        "| E3 无缓存 | `comparison_benchmark_results.json` |",
        "| E4 质量 | `output/results/quality_report.json`, `docs/quality_records.md` |",
        "| E6 多轮 | `multi_run_benchmark_results.json` |",
        "| E7 相似度 | `embedding_similarity_matrix.json` |",
        "",
        "设计说明: `docs/experiments/design_v1/comprehensive_suite.md`",
        "文献调研: `docs/strategy/literature_review_2024_2026.md`",
        "产物目录: `output/README.md`",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Only optimized + embedding")
    parser.add_argument("--with-multi-run", action="store_true", help="Also run 3x benchmark")
    parser.add_argument("--with-quality", action="store_true", help="Run quality_benchmark")
    args = parser.parse_args()

    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("ERROR: DEEPSEEK_API_KEY not set")
        return 1

    report: Dict[str, Any] = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "version": "v0.6.0",
        }
    }

    def should_run(name: str) -> bool:
        if not args.skip_existing:
            return True
        paths = {
            "pure": "pure_text_baseline_results.json",
            "cmp": "comparison_benchmark_results.json",
            "opt": "optimized_benchmark_results.json",
            "qual": "quality_report.json",
            "multi": "multi_run_benchmark_results.json",
            "emb": "embedding_similarity_matrix.json",
        }
        p = paths.get(name, "")
        return not os.path.exists(resolve_path(p)) if p else True

    t0 = time.time()

    if not args.quick:
        if should_run("pure"):
            _run_script("pure_text_baseline.py")
        if should_run("cmp"):
            _run_script("comparison_benchmark.py")

    if should_run("opt"):
        _run_script("optimized_benchmark.py")

    if should_run("emb"):
        _run_script("embedding_matrix.py")

    if args.with_quality and should_run("qual"):
        _run_script("quality_benchmark.py", ["--no-judge"])

    if args.with_multi_run and should_run("multi"):
        _run_script("multi_run_benchmark.py", ["-n", "3"])

    # Aggregate
    pt = _load_json("pure_text_baseline_results.json")
    if pt:
        report["pure_text"] = {**pt.get("totals", {}), "source": "pure_text_baseline_results.json"}

    cache_cmp = _load_json("cache_comparison_results.json")
    if cache_cmp and cache_cmp.get("no_cache", {}).get("totals"):
        report["structured_nocache"] = {
            **cache_cmp["no_cache"]["totals"],
            "source": "cache_comparison_results.json",
        }

    cmp = _load_json("comparison_benchmark_results.json")
    if cmp and "structured_nocache" not in report:
        for key in ("no_cache", "text", "without_cache"):
            block = cmp.get(key, {})
            totals = block.get("totals", block) if isinstance(block, dict) else {}
            if totals.get("total_tokens"):
                report["structured_nocache"] = {
                    **totals,
                    "source": f"comparison_benchmark_results.json ({key})",
                }
                break

    opt = _load_json("optimized_benchmark_results.json")
    if opt:
        report["optimized"] = {**opt.get("totals", {}), "source": "optimized_benchmark_results.json"}
        report["optimized"]["strategy"] = {}
        from collections import Counter
        c = Counter(r.get("strategy") for r in opt.get("results", []))
        report["optimized"]["strategy"] = dict(c)

    qual = _load_json("quality_report.json")
    if qual:
        report["quality"] = qual.get("meta", {})

    emb = _load_json("embedding_similarity_matrix.json")
    if emb:
        report["embedding_pairs"] = emb.get("pairs_above_085", [])

    multi = _load_json("multi_run_benchmark_results.json")
    if multi:
        report["multi_run"] = multi.get("aggregates", {})

    report["meta"]["elapsed_sec"] = round(time.time() - t0, 1)

    out_json = result_path("full_evaluation_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    md = build_summary_md(report)
    md_path = os.path.join(_ROOT, "docs", "full_evaluation_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"\n=== SUITE DONE ({report['meta']['elapsed_sec']}s) ===")
    print(f"JSON: {out_json}")
    print(f"MD:   {md_path}")
    if report.get("pure_text") and report.get("optimized"):
        save = _pct_save(
            report["optimized"]["total_tokens"],
            report["pure_text"]["total_tokens"],
        )
        print(f"Token savings (optimized vs pure): {save:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
