#!/usr/bin/env python3
"""Analyze experiment_matrix_report.json and emit optimization recommendations."""

from __future__ import annotations

import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from src.paths import resolve_path


def main() -> int:
    path = resolve_path("experiment_matrix_report.json")
    if not os.path.exists(path):
        print("No experiment_matrix_report.json — run experiment_matrix.py first")
        return 1

    with open(path, encoding="utf-8") as f:
        report = json.load(f)

    variants = report.get("variants", {})
    recs: list[str] = []
    actions: list[str] = []

    full = variants.get("C_full", {})
    no_drop = variants.get("C_no_dropout", {})
    no_e2e = variants.get("C_no_e2e", {})
    nocache = variants.get("B_nocache", {})
    warm = variants.get("C_warm_12", {})
    cold = variants.get("C_cold_12", {})

    if nocache and full:
        save = (1 - full["total_tokens"] / nocache["total_tokens"]) * 100
        recs.append(
            f"缓存层（B→C）贡献约 **{save:.1f}%** token 节省 "
            f"({nocache['total_tokens']:,} → {full['total_tokens']:,})。"
        )

    if no_drop and full:
        delta = no_drop["total_tokens"] - full["total_tokens"]
        if delta > 200:
            recs.append(
                f"AgentDropout 节省 **{delta:,}** tokens；保持 enable_executor_dropout=True。"
            )
            actions.append("保持 Executor Dropout（threshold=0.70）")
        elif delta < -100:
            recs.append("Dropout 反而增加 token，考虑提高 executor_dropout_threshold 至 0.75。")
            actions.append("将 executor_dropout_threshold 从 0.70 提至 0.75")

    if no_e2e and full:
        delta = no_e2e["total_tokens"] - full["total_tokens"]
        recs.append(f"E2E 缓存节省约 **{delta:,}** tokens（关闭 E2E 时更高）。")
        if full.get("e2e_hits", 0) < 2:
            actions.append("考虑 bge-base 或 Summarizer-E2E 扩大 E2E 命中（当前 E2E 偏少）")

    if warm and cold:
        tok_delta = cold["total_tokens"] - warm["total_tokens"]
        e2e_delta = warm.get("e2e_hits", 0) - cold.get("e2e_hits", 0)
        recs.append(
            f"热记忆 vs 冷启动：warm 省 **{tok_delta:,}** tokens，E2E 多 **{e2e_delta}** 次。"
        )
        if tok_delta < 500:
            actions.append("12 题顺序已能积累记忆；可增加同域序列 benchmark 展示")

    if full.get("avg_relevance", 1) < 0.82:
        actions.append("relevance 偏低：检查误缓存或收紧 e2e_threshold")

    if full.get("quality_pass_count") is not None and full["quality_pass_count"] < 12:
        actions.append("质量未全通过：SafeSieve 已接 heuristic，可接 LLM judge 反馈")

    lines = [
        "# 实验矩阵驱动的优化建议",
        "",
        f"基于 `{path}` 自动生成。",
        "",
        "## 结论摘要",
        "",
    ]
    lines.extend(recs or ["- （数据不足，请先运行 experiment_matrix.py --tier1）"])
    lines.extend(["", "## 建议代码动作", ""])
    lines.extend([f"- {a}" for a in actions] or ["- 维持 v0.9 默认 RunOptions"])

    out = os.path.join(_ROOT, "docs", "optimization_from_matrix.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n=== OPTIMIZATION RECOMMENDATIONS ===")
    for r in recs:
        print(f"  • {r}")
    for a in actions:
        print(f"  → {a}")
    print(f"\nWritten: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
