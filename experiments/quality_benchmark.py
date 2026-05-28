#!/usr/bin/env python3
"""Run 12 tasks, validate answers, export Q&A records + quality scores.

Outputs:
  - quality_report.json   (machine-readable, full problem + answer + validation)
  - docs/quality_records.md (human-readable per-task Q&A)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

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

from experiments.benchmark_tasks import BENCHMARK_TASKS, CROSS_DOMAIN_MARKERS
from src.paths import ROOT, result_path
from src.agents.base import LLMBackend
from src.orchestrator import Orchestrator
from src.evaluation.quality_validator import (
    validate_task,
    format_answer_markdown,
    format_validation_markdown,
    extract_summary_payload,
)


def clear_db(path: str) -> None:
    for p in [path, path + "-shm", path + "-wal"]:
        try:
            os.unlink(p)
        except OSError:
            pass


def infer_strategy(result: Dict[str, Any]) -> str:
    plan = result.get("steps", {}).get("plan", {})
    e2e = result.get("_e2e_cached", False)
    dropped = result.get("_executor_dropped", False)
    if e2e and result.get("_e2e_match") == "exact":
        return "E2E_EXACT"
    if plan.get("_from_template"):
        return "P1_TEMPLATE" + ("_DROP" if dropped else "")
    if plan.get("llm_skipped"):
        return "CACHE_REUSE"
    if plan.get("_template_filled"):
        return "TEMPLATE_FILL" + ("_DROP" if dropped else "")
    return "FULL_GEN"


def build_markdown_report(records: List[Dict[str, Any]], meta: Dict[str, Any]) -> str:
    lines = [
        "# 任务问答与质量验证记录",
        "",
        f"- **生成时间**: {meta.get('generated_at', '')}",
        f"- **系统版本**: {meta.get('version', 'v0.5.0')}",
        f"- **模型**: {meta.get('model', 'deepseek-chat')}",
        f"- **LLM 评判**: {'开启' if meta.get('llm_judge') else '关闭（仅启发式）'}",
        "",
        "## 汇总",
        "",
        f"| 指标 | 值 |",
        f"|------|-----|",
        f"| 任务数 | {meta.get('task_count', 0)} |",
        f"| 综合通过 | {meta.get('composite_pass_count', 0)}/{meta.get('task_count', 0)} |",
        f"| 平均综合分 | {meta.get('avg_composite_score', 0):.3f} |",
        f"| 平均嵌入相关性 | {meta.get('avg_embedding_relevance', 0):.3f} |",
        f"| 平均主题覆盖 | {meta.get('avg_topic_coverage', 0):.1%} |",
        f"| 总 Token | {meta.get('total_tokens', 0):,} |",
        "",
        "---",
        "",
    ]
    for rec in records:
        v = rec["validation"]
        lines.extend([
            f"## {rec['task_id']}（{rec.get('domain', '')} · {rec.get('strategy', '')}）",
            "",
            "### 问题（任务描述）",
            "",
            rec["problem"],
            "",
            "### 最终回答",
            "",
            rec["answer_markdown"],
            "",
            "### 验证",
            "",
            rec["validation_markdown"],
            "",
            "<details><summary>原始 JSON 摘要</summary>",
            "",
            "```json",
            json.dumps(rec.get("summary_object", {}), ensure_ascii=False, indent=2)[:4000],
            "```",
            "",
            "</details>",
            "",
            "---",
            "",
        ])
    return "\n".join(lines)


def run_quality_benchmark(
    api_key: str,
    use_llm_judge: bool = True,
    db_path: str = "_quality_bench.db",
    pass_threshold: float = 0.65,
) -> Dict[str, Any]:
    clear_db(db_path)
    llm = LLMBackend(provider="deepseek", api_key=api_key, model="deepseek-chat")
    judge_llm = llm if use_llm_judge else None

    orch = Orchestrator(
        llm=llm,
        mode="structured",
        use_real_embeddings=True,
        sandbox_enabled=True,
        memory_db_path=db_path,
    )

    records: List[Dict[str, Any]] = []
    total_tokens = 0

    for i, task in enumerate(BENCHMARK_TASKS):
        print(f"  [{i+1:2d}/12] {task['task_id']}...", end=" ", flush=True)
        t0 = time.time()
        result = orch.execute_task(
            task_id=task["task_id"],
            task_description=task["description"],
            tags=task.get("tags", []),
        )
        elapsed_ms = (time.time() - t0) * 1000
        llm_stats = llm.get_usage_stats()
        total_tokens += llm_stats["total_prompt_tokens"] + llm_stats["total_completion_tokens"]

        summary_step = result.get("steps", {}).get("summary", {})
        validation, summary_obj, answer_text = validate_task(
            task,
            summary_step,
            orch.embedding_engine.encode,
            cross_domain_markers=CROSS_DOMAIN_MARKERS,
            llm=judge_llm,
            use_llm_judge=use_llm_judge,
            pass_threshold=pass_threshold,
        )

        strategy = infer_strategy(result)
        status = "PASS" if validation.composite_pass else "FAIL"
        print(
            f"{strategy} composite={validation.composite_score:.3f} "
            f"rel={validation.embedding_relevance:.3f} {status} {elapsed_ms:.0f}ms"
        )

        records.append({
            "task_id": task["task_id"],
            "domain": task.get("domain", task["tags"][0]),
            "tags": task["tags"],
            "strategy": strategy,
            "problem": task["description"],
            "answer_text": answer_text,
            "answer_markdown": format_answer_markdown(summary_obj, answer_text),
            "summary_object": summary_obj if isinstance(summary_obj, dict) else {"text": answer_text},
            "validation": validation.to_dict(),
            "validation_markdown": format_validation_markdown(validation),
            "metrics": {
                "api_calls": llm_stats["call_count"],
                "prompt_tokens": llm_stats["total_prompt_tokens"],
                "completion_tokens": llm_stats["total_completion_tokens"],
                "total_tokens": llm_stats["total_prompt_tokens"] + llm_stats["total_completion_tokens"],
                "elapsed_ms": elapsed_ms,
                "e2e_cached": result.get("_e2e_cached", False),
            },
            "pipeline": {
                "executor_dropped": result.get("_executor_dropped", False),
                "plan_template_filled": result.get("steps", {}).get("plan", {}).get("_template_filled"),
            },
        })
        llm.reset_stats()

    n = len(records)
    vals = [r["validation"] for r in records]
    meta = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "version": "v0.11.1",
        "model": "deepseek-chat",
        "llm_judge": use_llm_judge,
        "pass_threshold": pass_threshold,
        "task_count": n,
        "composite_pass_count": sum(1 for v in vals if v["composite_pass"]),
        "avg_composite_score": sum(v["composite_score"] for v in vals) / n,
        "avg_embedding_relevance": sum(v["embedding_relevance"] for v in vals) / n,
        "avg_topic_coverage": sum(v["topic_coverage"] for v in vals) / n,
        "total_tokens": total_tokens,
    }

    return {"meta": meta, "records": records}


def main() -> int:
    parser = argparse.ArgumentParser(description="Quality benchmark with Q&A records")
    parser.add_argument(
        "--no-judge", action="store_true",
        help="Skip LLM judge (heuristic validation only, faster/cheaper)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.65,
        help="Composite pass threshold (default 0.65)",
    )
    parser.add_argument(
        "--json-out", default=None,
        help="JSON output path (default output/results/quality_report.json)",
    )
    parser.add_argument(
        "--md-out", default="docs/quality_records.md",
        help="Markdown output path (default docs/quality_records.md)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set")
        return 1

    use_judge = not args.no_judge
    print("=" * 70)
    print("  Quality Benchmark — Q&A records + validation")
    print(f"  LLM judge: {'ON' if use_judge else 'OFF (heuristic only)'}")
    print("=" * 70)

    report = run_quality_benchmark(
        api_key, use_llm_judge=use_judge, pass_threshold=args.threshold
    )

    json_path = args.json_out or result_path("quality_report.json")
    if not os.path.isabs(json_path) and not json_path.startswith("docs"):
        json_path = result_path(os.path.basename(json_path))
    md_path = args.md_out if os.path.isabs(args.md_out) else os.path.join(_ROOT, args.md_out)
    os.makedirs(os.path.dirname(md_path) or ".", exist_ok=True)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    md_content = build_markdown_report(report["records"], report["meta"])
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    m = report["meta"]
    print(f"\n  === QUALITY SUMMARY ===")
    print(f"  Composite pass: {m['composite_pass_count']}/{m['task_count']}")
    print(f"  Avg composite:  {m['avg_composite_score']:.3f}")
    print(f"  Avg relevance:  {m['avg_embedding_relevance']:.3f}")
    print(f"  Avg topic cov:  {m['avg_topic_coverage']:.1%}")
    print(f"  Total tokens:   {m['total_tokens']:,}")
    print(f"\n  JSON: {json_path}")
    print(f"  MD:   {md_path}")

    failed = [r["task_id"] for r in report["records"] if not r["validation"]["composite_pass"]]
    if failed:
        print(f"\n  Failed tasks: {', '.join(failed)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
