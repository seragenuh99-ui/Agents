"""Chinese terminal reporting for run.py sessions."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _fmt_pass(v: Optional[bool]) -> str:
    if v is None:
        return "未检测"
    return "通过" if v else "未通过"


def _fmt_na(v: Any, suffix: str = "") -> str:
    if v is None or v == "":
        return "—（本模式不适用）"
    return f"{v}{suffix}"


def print_banner(title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def get_row_answer_text(row: Dict[str, Any]) -> str:
    """Full answer text for display (not truncated)."""
    full = (row.get("summary_full") or "").strip()
    if full:
        return full
    preview = (row.get("summary_preview") or "").strip()
    if preview:
        return preview
    return "（无回答内容）"


def print_task_detail(
    row: Dict[str, Any],
    *,
    mode_label: str,
    index: int,
    total: int,
    mode_key: str,
    show_metrics: bool = True,
) -> None:
    tid = row.get("task_id", "?")
    desc = (row.get("description") or "")[:200]
    print()
    print("━" * 70)
    print(f"【第 {index}/{total} 题】{tid}")
    if desc:
        print(f"题目：{desc}{'…' if len(row.get('description') or '') > 200 else ''}")
    print(f"当前模式：{mode_label}")
    print("━" * 70)

    print("\n▶ 完整回答")
    print(get_row_answer_text(row))

    if not show_metrics:
        print("━" * 70)
        return

    print("\n▶ 执行摘要")
    print(f"  策略：{row.get('strategy', '—')}")
    e2e = row.get("e2e_cached")
    print(f"  整题 E2E 缓存：{'是' if e2e else '否'}")
    if row.get("executor_dropped"):
        print("  Executor：已跳过（Dropout）")
    comp = row.get("composite_score")
    if comp is not None:
        print(
            f"  质量：{_fmt_pass(row.get('composite_pass'))} | "
            f"综合分 {comp:.3f} | 相关度 {row.get('relevance', 0):.3f}"
        )
    else:
        print(f"  相关度：{row.get('relevance', 0):.3f}")

    print("\n▶ 协作与通信")
    if mode_key == "pure":
        print("  本模式为纯文本基线，无 Agent 结构化消息与向量传递。")
        print(f"  LLM 调用：{row.get('api_calls', 0)} 次")
    else:
        print(f"  Agent 消息次数：{row.get('messages_sent', 0)}")
        print(
            f"  协议 token ≈ {row.get('structured_protocol_tokens', 0)} | "
            f"等效长文本 token ≈ {row.get('text_equivalent_tokens', 0)}"
        )
        print(f"  通信字符量 ≈ {row.get('communication_chars', 0)}")
        st = row.get("state_transfers", 0)
        sb = row.get("state_data_bytes", 0)
        print(f"  非文本传递：{st} 次，约 {sb} 字节（384 维向量）")

    print("\n▶ 记忆与复用")
    if mode_key == "pure":
        print("  共享记忆：未使用")
    else:
        mq = row.get("memory_queries", 0)
        mh = row.get("memory_hits", 0)
        mhr = row.get("memory_hit_rate", 0)
        print(f"  记忆查询：{mq} | 命中：{mh} | 本题命中率：{mhr:.0%}")
        print(f"  跨任务复用：{row.get('cross_task_memories', 0)} 条")

    print("\n▶ 性能")
    ms = row.get("elapsed_ms", 0)
    print(f"  本题耗时：{ms / 1000:.1f} 秒")
    print(
        f"  LLM API token：{row.get('total_tokens', 0)} "
        f"（prompt {row.get('prompt_tokens', 0)} + "
        f"输出 {row.get('completion_tokens', 0)}）"
    )
    print("━" * 70)


def print_mode_summary(
    mode_num: int,
    mode_label: str,
    results: List[Dict[str, Any]],
    totals: Dict[str, Any],
    *,
    suite_label: str,
) -> None:
    n = len(results)
    qpass = totals.get("quality_pass_count")
    qline = (
        f"{qpass}/{n}"
        if qpass is not None
        else f"{sum(1 for r in results if r.get('composite_pass'))}/{n}"
    )
    wall = totals.get("wall_clock_sec") or totals.get("total_elapsed_ms", 0) / 1000

    print()
    print("╔" + "═" * 68 + "╗")
    print(f"║  模式 {mode_num} 完成 · {mode_label}")
    print(f"║  任务集：{suite_label}（{n} 题）")
    print("╠" + "═" * 68 + "╣")
    print(f"║  质量通过：{qline}")
    print(f"║  总耗时：{_fmt_time(wall)} | 平均每题：{wall / max(n, 1):.1f} 秒")
    print(f"║  LLM 总 token：{totals.get('total_tokens', 0):,}")
    avg_rel = totals.get("avg_relevance", 0)
    print(f"║  平均相关度：{avg_rel:.3f}")

    mode_key = MODE_KEY_BY_NUM.get(mode_num, "")
    if mode_key != "pure":
        print(f"║  Agent 消息总数：{totals.get('total_messages', 0)}")
        print(
            f"║  非文本传递：{totals.get('total_state_transfers', 0)} 次 / "
            f"{totals.get('total_state_data_bytes', 0):,} 字节"
        )
        print(
            f"║  记忆命中率：{totals.get('memory_hit_rate', 0):.1%} "
            f"（查询 {totals.get('total_memory_queries', 0)} / "
            f"命中 {totals.get('total_memory_hits', 0)}）"
        )
        e2e = totals.get("e2e_hits", 0)
        print(f"║  E2E 整题缓存命中：{e2e} 题")
    else:
        print("║  （纯文本模式无 Agent 消息 / 记忆统计）")

    print("╚" + "═" * 68 + "╝")


def print_comparison_table(sessions: List[Dict[str, Any]]) -> None:
    if len(sessions) < 2:
        return
    print()
    print_banner("多模式对比（同一批题目）")
    baseline = sessions[0]
    header = "指标".ljust(14) + "".join(
        f"{s['mode_label'][:10]:>14}" for s in sessions
    )
    print(header)
    print("-" * len(header))

    def row(name: str, key: str, fmt=lambda x: str(x)):
        vals = [fmt(s["totals"].get(key, s.get(key, "—"))) for s in sessions]
        print(name.ljust(14) + "".join(f"{v:>14}" for v in vals))

    n0 = len(sessions[0]["results"])

    def _qfmt(x, _s=sessions):
        return "—" if x is None else f"{int(x)}/{n0}"

    row("质量通过", "quality_pass_count", _qfmt)
    row("总 token", "total_tokens", lambda x: f"{int(x):,}")
    row("总耗时", "wall_clock_sec", _fmt_time)
    row("平均相关度", "avg_relevance", lambda x: f"{float(x):.3f}")
    row("记忆命中率", "memory_hit_rate", lambda x: "—" if x is None else f"{float(x):.1%}")

    ref_tok = baseline["totals"].get("total_tokens") or 1
    ref_time = baseline["totals"].get("wall_clock_sec") or (
        baseline["totals"].get("total_elapsed_ms", 1) / 1000
    )
    print("\n▶ 相对第一种模式：")
    for s in sessions[1:]:
        tok = s["totals"].get("total_tokens", 0)
        tsec = s["totals"].get("wall_clock_sec") or s["totals"].get("total_elapsed_ms", 0) / 1000
        save = (1 - tok / ref_tok) * 100 if ref_tok else 0
        speed = ref_time / tsec if tsec else 0
        print(
            f"  {s['mode_label']}：省 token {save:.1f}% | "
            f"耗时约为 {speed:.1f}x"
        )


def print_answer_brief(
    row: Dict[str, Any], *, mode_label: str, mode_key: str = "structured"
) -> None:
    """Full answer + key metrics (REPL single-mode)."""
    print_task_detail(
        row,
        mode_label=mode_label,
        index=1,
        total=1,
        mode_key=mode_key,
    )


def print_multi_mode_answers(
    sessions: List[Dict[str, Any]],
    *,
    question: str = "",
) -> None:
    """Side-by-side full answers for each mode (e.g. 纯文本 vs 结构化)."""
    if len(sessions) < 1:
        return
    print()
    print("=" * 70)
    print("  多模式完整回答对照")
    print("=" * 70)
    if question:
        q = question if len(question) <= 300 else question[:300] + "…"
        print(f"题目：{q}\n")

    for sess in sessions:
        results = sess.get("results") or []
        if not results:
            continue
        row = results[0]
        label = sess.get("mode_label", "?")
        mode_key = sess.get("mode_key", "")
        print()
        print("━" * 70)
        print(f"【{label}】")
        if mode_key == "pure":
            print("  （纯文本基线：Planner/Retriever/Summarizer 均为长文本 API 调用）")
        elif mode_key == "structured":
            print("  （结构化多 Agent：协议 + 共享记忆 + 缓存）")
        elif mode_key == "nocache":
            print("  （结构化无缓存对照）")
        print("━" * 70)
        print(get_row_answer_text(row))
        ms = row.get("elapsed_ms", 0) / 1000
        print(
            f"\n  本题：耗时 {ms:.1f}s | token {row.get('total_tokens', 0)} | "
            f"API {row.get('api_calls', 0)} 次 | 策略 {row.get('strategy', '—')}"
        )
    print()
    print("=" * 70)


def print_launch_intro() -> None:
    print()
    print("=" * 70)
    print("  702solver 实验控制台")
    print("=" * 70)
    print("  目标：在同一套 benchmark 题上，证明结构化多 Agent 比纯文本更省 token。")
    print("  答辩主数据：模式 2 跑 core12 / full24；模式 1 作对照基线。")
    print("  开放题体验仅作演示，不计入主实验数据。")
    print("=" * 70)


def print_experiment_footer(*, json_path: str | None = None, back_to_menu: bool = False) -> None:
    print()
    print("─" * 70)
    print("  实验汇报要点")
    print("─" * 70)
    if json_path:
        print(f"  · 详细 JSON 报告：{json_path}")
    print("  · 多模式对照时请看上方「多模式对比」表的省 token %")
    print("  · 文档：docs/STATUS_SUMMARY.md · docs/full24_benchmark_results.md")
    if back_to_menu:
        print("  · 已回到实验控制台主菜单")
    print("─" * 70)


def print_repl_help() -> None:
    print(
        """
【开放题体验】直接输入问题（模式 2/3 时多轮共用共享记忆，≠ benchmark 主数据）。
【新对话】    输入：新对话 / 清空记忆 / new — 清空共享记忆重新开始
【返回主菜单】输入：菜单 / 主菜单 / menu / back / 返回
【退出程序】  输入：退出 / quit / exit / q
【帮助】      输入：? / 帮助
"""
    )


def print_help_cn() -> None:
    print(
        """
702solver 实验汇报入口 — 中文帮助

【交互式实验控制台】
  python3 run.py
  ./启动实验.sh                    # 可选答辩快测 / 对照
  ./启动实验_答辩12题.sh           # core12 · 模式 1+2 对照
  ./启动实验_完整24题.sh           # full24 · 模式 2

【答辩一键预设 --preset】
  defense12           core12 · 仅模式 2（主结果）
  defense12-compare   core12 · 模式 1+2（token 对照）
  full24              full24 · 模式 2

【命令式示例】
  python3 run.py --preset defense12 -y
  python3 run.py --mode 2 --suite core12
  python3 run.py --mode 1,2 --suite continuous12
  python3 run.py --mode 2 --ask "研究太阳能并网政策"

【模式 --mode】
  1  纯文本对照（基线，最费 token）
  2  结构化全功能（答辩主结果）
  3  结构化无缓存
  可多选：1,2 或 1,2,3

【任务集 --suite】（benchmark 主数据）
  core12        标准 12 题（答辩快测）
  full24        完整 24 题（正式报告）
  continuous12  连续 12 轮（≥10 轮要求）
  energy_research / code_security  各 2 题记忆演示

【其它】
  --quality     开启质量检测（默认开）
  --json 路径   保存 JSON 报告
  --yes         跳过确认直接开跑

需要 .env 中配置 DEEPSEEK_API_KEY=
"""
    )


MODE_KEY_BY_NUM = {1: "pure", 2: "structured", 3: "nocache"}


def _fmt_time(sec: float) -> str:
    if sec is None:
        return "—"
    sec = float(sec)
    if sec >= 120:
        return f"{sec / 60:.1f} 分"
    return f"{sec:.1f} 秒"
