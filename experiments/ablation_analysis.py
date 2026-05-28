#!/usr/bin/env python3
"""Offline ablation: decompose token savings from existing benchmark JSON files."""

import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from src.paths import resolve_path, result_path


def load(name):
    p = resolve_path(name)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def pct(a, b):
    return (1 - a / b) * 100 if b else 0.0


def main():
    pure = load("pure_text_baseline_results.json")
    nocache = load("cache_comparison_results.json")
    opt = load("optimized_benchmark_results.json")

    if not pure:
        print("Missing pure_text_baseline_results.json")
        return 1

    T_pure = pure["totals"]["total_tokens"]
    rows = []

    rows.append(("A 纯文本基线", T_pure, 0.0, "—"))

    if nocache:
        T_nc = nocache["no_cache"]["totals"]["total_tokens"]
        rows.append((
            "B 结构化(无FAISS缓存)",
            T_nc,
            pct(T_nc, T_pure),
            "SNS+JSON+refs，关闭跨任务缓存",
        ))
        if opt:
            T_opt = opt["totals"]["total_tokens"]
            rows.append((
                "C 全功能(v0.6)",
                T_opt,
                pct(T_opt, T_pure),
                f"+缓存/模板/E2E；再省 {pct(T_opt, T_nc):.1f}% vs B",
            ))
            rows.append((
                "Δ 缓存层贡献 (B→C)",
                T_nc - T_opt,
                pct(T_nc - T_opt, T_nc) if T_nc else 0,
                "E2E+模板填空+AgentDropout",
            ))
    elif opt:
        T_opt = opt["totals"]["total_tokens"]
        rows.append(("C 全功能", T_opt, pct(T_opt, T_pure), "—"))

    print("=" * 72)
    print("  Ablation: Token decomposition")
    print("=" * 72)
    print(f"  {'Mode':<28} {'Tokens':>10} {'vs A':>10}  Mechanism")
    print(f"  {'─'*28} {'─'*10} {'─'*10}  {'─'*20}")
    for label, tok, save, note in rows:
        print(f"  {label:<28} {tok:>10,} {save:>9.1f}%  {note}")

    # Per-component estimates (documented assumptions)
    if nocache and opt:
        T_nc = nocache["no_cache"]["totals"]["total_tokens"]
        T_opt = opt["totals"]["total_tokens"]
        cache_save = T_nc - T_opt
        struct_save = T_pure - T_nc
        print(f"\n  Estimated contribution to total savings ({pct(T_opt, T_pure):.1f}%):")
        print(f"    结构化协议+JSON (A→B):     ~{pct(T_nc, T_pure):.1f}%  ({struct_save:,} tok)")
        print(f"    记忆/缓存/剪枝 (B→C):       ~{pct(T_opt, T_nc):.1f}%  ({cache_save:,} tok)")

    report = {
        "pure_text_tokens": T_pure,
        "rows": [
            {"label": r[0], "tokens": r[1], "save_vs_pure_pct": r[2], "note": r[3]}
            for r in rows
        ],
    }
    if opt:
        report["optimized"] = opt["totals"]
        report["strategy"] = {}
        from collections import Counter
        report["strategy"] = dict(Counter(x.get("strategy") for x in opt.get("results", [])))

    out = result_path("ablation_analysis.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
