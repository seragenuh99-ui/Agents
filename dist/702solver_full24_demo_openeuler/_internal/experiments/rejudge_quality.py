#!/usr/bin/env python3
"""Re-run LLM judge on an existing quality_report.json (no task re-execution)."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(_ROOT, ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

from src.agents.base import LLMBackend
from experiments.quality_benchmark import build_markdown_report
from src.paths import ROOT, resolve_path, result_path
from src.evaluation.quality_validator import (
    llm_judge_answer,
    composite_score,
    format_validation_markdown,
    ValidationResult,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input_path", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set")
        return 1

    path = resolve_path("quality_report.json") if args.input_path is None else (
        result_path(os.path.basename(args.input_path))
        if not os.path.isabs(args.input_path) and "/" not in args.input_path
        else args.input_path
    )
    with open(path, encoding="utf-8") as f:
        report = json.load(f)

    llm = LLMBackend(provider="deepseek", api_key=api_key, model="deepseek-chat")
    threshold = report.get("meta", {}).get("pass_threshold", 0.65)

    for rec in report["records"]:
        judge = llm_judge_answer(
            llm,
            rec["problem"],
            rec["answer_text"],
            rec.get("summary_object", {}),
        )
        vdict = rec["validation"]
        v = ValidationResult(**{k: vdict[k] for k in vdict if k in ValidationResult.__dataclass_fields__})
        v.llm_on_topic = judge.get("on_topic")
        v.llm_score = judge.get("score")
        v.llm_reason = judge.get("reason", "")
        v.llm_issues = judge.get("issues", [])
        v.composite_score = round(composite_score(v, use_llm=True), 3)
        v.composite_pass = v.composite_score >= threshold and not v.has_error_flag
        rec["validation"] = v.to_dict()
        rec["validation_markdown"] = format_validation_markdown(v)
        llm.reset_stats()

    report["meta"]["llm_judge"] = True
    vals = [r["validation"] for r in report["records"]]
    n = len(vals)
    report["meta"]["composite_pass_count"] = sum(1 for v in vals if v["composite_pass"])
    report["meta"]["avg_composite_score"] = sum(v["composite_score"] for v in vals) / n

    out = result_path("quality_report.json") if args.out is None else (
        result_path(os.path.basename(args.out))
        if not os.path.isabs(args.out) and "/" not in args.out
        else args.out
    )
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    md_path = os.path.join(ROOT, "docs", "quality_records.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(build_markdown_report(report["records"], report["meta"]))
    print(f"Updated: {out}")
    print(f"Updated: {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
