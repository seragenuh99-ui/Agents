#!/usr/bin/env python3
"""Export human fact-check pack from quality_report or fresh core12 run."""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.benchmark_suites import HUMAN_FACTCHECK_TASK_IDS, task_by_id
from src.paths import resolve_path


def main() -> int:
    qr_path = resolve_path("quality_report.json")
    records = {}
    if os.path.exists(qr_path):
        with open(qr_path, encoding="utf-8") as f:
            for rec in json.load(f).get("records", []):
                records[rec["task_id"]] = rec

    lines = [
        "# 人工事实核查包（6 题样本）",
        "",
        "请评审员对每题勾选：**事实正确** / **部分正确** / **错误**，并备注。",
        "自动指标（composite、主题覆盖）不能替代本表。",
        "",
    ]

    for tid in HUMAN_FACTCHECK_TASK_IDS:
        task = task_by_id(tid)
        rec = records.get(tid, {})
        lines.extend([
            f"## {tid}",
            "",
            f"**领域**: {task.get('domain')}",
            "",
            "### 题目",
            "",
            task["description"],
            "",
            "### 系统回答（自动）",
            "",
            rec.get("answer_markdown") or rec.get("answer_text", "_未找到 quality_report，请先运行 quality_benchmark.py_"),
            "",
            "### 自动评分",
            "",
            f"- composite: {rec.get('validation', {}).get('composite_score', '—')}",
            f"- topic_coverage: {rec.get('validation', {}).get('topic_coverage', '—')}",
            f"- strategy: {rec.get('strategy', '—')}",
            "",
            "### 人工核查",
            "",
            "| 项目 | 勾选 |",
            "|------|------|",
            "| 事实正确 | [ ] |",
            "| 部分正确 | [ ] |",
            "| 错误 | [ ] |",
            "",
            "备注：",
            "",
            "---",
            "",
        ])

    out = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "docs",
        "human_factcheck_pack.md",
    )
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
