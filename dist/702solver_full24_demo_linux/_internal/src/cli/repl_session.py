"""Interactive REPL: experiment console for defense reporting and demos."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import List, Optional, Set

from experiments.benchmark_suites import get_suite

from .run_engine import (
    MODE_NUM,
    ReplChatSession,
    RunConfig,
    execute_session,
    mode_num_from_key,
    repl_supports_shared_memory,
    run_repl_chat_turn,
    start_repl_chat_session,
)
from .run_reporter import (
    print_answer_brief,
    print_banner,
    print_comparison_table,
    print_experiment_footer,
    print_help_cn,
    print_launch_intro,
    print_mode_summary,
    print_multi_mode_answers,
    print_repl_help,
    print_task_detail,
)
from .task_catalog import (
    MEMORY_DEMO_SUITES,
    MODE_CATALOG,
    SUITE_CATALOG,
    list_defense_suites,
    suite_desc,
    suite_eta,
    suite_name,
)

# 回主菜单（不退出程序）
CMD_MENU: Set[str] = {"菜单", "主菜单", "menu", "back", "返回", "/menu"}

# 退出整个程序
CMD_QUIT: Set[str] = {"退出", "quit", "exit", "q", "/quit", "再见"}

# 帮助
CMD_HELP: Set[str] = {"?", "帮助", "help", "/help"}

# 清空本对话共享记忆，开启新会话
CMD_NEW_CHAT: Set[str] = {
    "新对话",
    "清空记忆",
    "清空",
    "new",
    "/new",
    "重置",
}


@dataclass
class ReplState:
    modes: List[int] = field(default_factory=lambda: [2])
    run_quality: bool = True
    chat_session: Optional[ReplChatSession] = None


def _input_line(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print("\n（已中断）")
        raise SystemExit(0)


def _normalize_cmd(line: str) -> str:
    return line.strip().lower()


def _is_menu(line: str) -> bool:
    s = line.strip()
    return s in CMD_MENU or _normalize_cmd(s) in {_normalize_cmd(x) for x in CMD_MENU}


def _is_quit(line: str) -> bool:
    s = line.strip()
    return s in CMD_QUIT or _normalize_cmd(s) in {_normalize_cmd(x) for x in CMD_QUIT}


def _confirm(prompt: str, default_yes: bool = True) -> bool:
    hint = "y/n" if not default_yes else "y/n"
    default = "y" if default_yes else "n"
    ans = _input_line(f"{prompt} ({hint}) [{default}]: ").lower()
    if not ans:
        return default_yes
    return ans in ("y", "yes", "是")


def _pick_modes(current: List[int]) -> Optional[List[int]]:
    print("\n可选模式（可多选，逗号分隔）：")
    for num, (_, name, desc) in MODE_CATALOG.items():
        mark = " ← 当前" if num in current else ""
        print(f"  {num}  {name} — {desc}{mark}")
    raw = _input_line("请输入模式编号 [回车保持当前]: ")
    if not raw:
        return current
    if raw in CMD_HELP:
        print_repl_help()
        return _pick_modes(current)
    if _is_menu(raw):
        return None

    modes: List[int] = []
    for part in raw.replace("，", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            m = int(part)
            if m not in MODE_CATALOG:
                print(f"  无效模式：{m}")
                return None
            modes.append(m)
        except ValueError:
            print(f"  无法解析：{part}")
            return None
    return modes if modes else None


def _show_main_menu(state: ReplState) -> None:
    mode_names = "、".join(MODE_CATALOG[m][1] for m in state.modes)
    print()
    print("=" * 70)
    print("  702solver 实验控制台")
    print("=" * 70)
    print(f"  当前模式：{state.modes}（{mode_names}）")
    print()
    print("  【答辩 / 实验 — 推荐】")
    print("  1  跑 benchmark 题库（core12 / full24 / continuous12 …）")
    print("  2  多模式对照实验（选模式 + 同一题库，出 token 对比表）")
    print("  3  答辩快测：core12 · 仅模式 2（~12 分钟）")
    print("  4  答辩对照：core12 · 模式 1+2（~25 分钟）")
    print()
    print("  【体验 / 演示 — 非主评分数据】")
    print("  5  开放题体验（自定义中文问，看完整流水线）")
    print("  6  连续记忆演示（2～10 题关联，看缓存命中）")
    print()
    print("  【设置】")
    print("  7  切换默认运行模式")
    print("  8  帮助（赛题说明 / 命令行示例）")
    print("  0  退出程序")
    print()
    print("  提示：benchmark 数据用于答辩；开放题仅演示能力。")
    print("=" * 70)


def _invalidate_chat(state: ReplState) -> None:
    state.chat_session = None


def _ensure_chat_session(api_key: str, state: ReplState) -> ReplChatSession:
    mode_num = state.modes[0]
    key = tuple(state.modes)
    chat = state.chat_session
    if (
        isinstance(chat, ReplChatSession)
        and chat.modes_key == key
        and chat.mode_num == mode_num
    ):
        return chat
    state.chat_session = start_repl_chat_session(api_key, mode_num)
    return state.chat_session


def _print_suite_catalog(*, defense_only: bool = False, demo_only: bool = False) -> None:
    print("\n  可选题库：")
    for sid, row in SUITE_CATALOG.items():
        name, desc, eta, recommended = row
        if defense_only and not recommended:
            continue
        if demo_only and sid not in MEMORY_DEMO_SUITES:
            continue
        try:
            n = len(get_suite(sid))
        except KeyError:
            n = "?"
        tags: List[str] = []
        if recommended:
            tags.append("答辩推荐")
        if sid in MEMORY_DEMO_SUITES:
            tags.append("记忆演示")
        tag_str = f" [{' · '.join(tags)}]" if tags else ""
        print(f"    {sid:<16} {name}（{n} 题，{eta}）{tag_str}")
        print(f"    {'':16} {desc}")


def _run_batch_experiment(
    api_key: str,
    state: ReplState,
    *,
    suite_id: str,
    modes: Optional[List[int]] = None,
    task_ids: Optional[List[str]] = None,
    confirm: bool = True,
    title: str = "",
) -> None:
    run_modes = modes if modes is not None else state.modes
    n = len(task_ids) if task_ids else len(get_suite(suite_id))
    suite_label = suite_name(suite_id)
    eta = suite_eta(suite_id)

    if confirm:
        print(
            f"\n  将跑 {n} 题 · 套件 {suite_id}（{suite_label}，预计 {eta}）"
            f" · 模式 {run_modes}"
        )
        if not _confirm("确认开始"):
            print("  已取消。")
            return

    config = RunConfig(
        modes=run_modes,
        suite=None if task_ids else suite_id,
        task_ids=task_ids,
        run_quality=state.run_quality,
        save_json=True,
        db_prefix="_run_suite",
    )
    banner = title or f"benchmark 实验 | {suite_label} | {n} 题 | 模式 {run_modes}"
    print_banner(banner)

    def on_task(row, idx, total, mode_key):
        mode_num = mode_num_from_key(mode_key)
        print_task_detail(
            row,
            mode_label=MODE_CATALOG[mode_num][1],
            index=idx,
            total=total,
            mode_key=mode_key,
        )

    report = execute_session(api_key, config, on_task=on_task)
    for sess in report["sessions"]:
        print_mode_summary(
            sess["mode_num"],
            sess["mode_label"],
            sess["results"],
            sess["totals"],
            suite_label=f"{suite_label} ({suite_id})",
        )
    print_comparison_table(report["sessions"])
    jpath = report.get("json_path")
    print_experiment_footer(json_path=jpath, back_to_menu=True)


def _suite_loop(
    api_key: str,
    state: ReplState,
    *,
    defense_only: bool = False,
    default_modes: Optional[List[int]] = None,
) -> None:
    _print_suite_catalog(defense_only=defense_only)
    sid = _input_line("\n  请输入套件 ID [回车取消]: ")
    if not sid or _is_menu(sid):
        return
    if _is_quit(sid):
        raise SystemExit(0)
    if sid not in SUITE_CATALOG:
        print(f"  未知套件：{sid}")
        return

    pick = _input_line("  只跑部分题？task_id 逗号分隔，回车=全部: ")
    if _is_menu(pick):
        return
    task_ids = None
    if pick and not _is_quit(pick):
        task_ids = [x.strip() for x in pick.replace("，", ",").split(",") if x.strip()]

    _run_batch_experiment(
        api_key,
        state,
        suite_id=sid,
        modes=default_modes,
        task_ids=task_ids,
        confirm=True,
    )


def _compare_loop(api_key: str, state: ReplState) -> None:
    print("\n  多模式对照：将在同一套题上依次跑各模式，并输出 token 对比表。")
    picked = _pick_modes(state.modes)
    if picked is None:
        return
    if len(picked) < 2:
        print("  对照实验建议至少选 2 种模式（如 1,2）。仍用单模式继续。")
    _suite_loop(api_key, state, default_modes=picked)


def _run_preset(api_key: str, state: ReplState, *, modes: List[int], suite_id: str, title: str) -> None:
    old_modes = state.modes
    state.modes = modes
    _invalidate_chat(state)
    try:
        _run_batch_experiment(
            api_key,
            state,
            suite_id=suite_id,
            modes=modes,
            confirm=False,
            title=title,
        )
    finally:
        state.modes = old_modes
        _invalidate_chat(state)


def _memory_demo_loop(api_key: str, state: ReplState) -> None:
    print()
    print("─" * 70)
    print("  连续记忆演示")
    print("─" * 70)
    print("  用于答辩展示：第 2 题起应出现记忆命中 / 策略 CACHE。")
    print("  建议使用模式 2（结构化全功能）。")
    _print_suite_catalog(demo_only=True)
    print()
    print("  快捷选项：")
    print("    1  energy_research（能源 2 题，最快）")
    print("    2  code_security（安全 2 题）")
    print("    3  continuous_10（连续 10 轮）")
    choice = _input_line("  请选择 [1]: ")
    sid_map = {"1": "energy_research", "2": "code_security", "3": "continuous_10", "": "energy_research"}
    sid = sid_map.get(choice)
    if sid is None:
        sid = choice.strip()
    if sid not in MEMORY_DEMO_SUITES:
        print(f"  无效选项：{choice}")
        return

    demo_modes = [2] if state.modes != [2] else state.modes
    if state.modes != [2]:
        print("  演示将临时使用模式 2（结构化全功能）。")

    _run_batch_experiment(
        api_key,
        state,
        suite_id=sid,
        modes=demo_modes,
        confirm=True,
        title=f"记忆演示 | {suite_name(sid)} | 模式 2",
    )


def _run_one_question(api_key: str, state: ReplState, question: str) -> None:
    from .run_engine import build_tasks_from_asks

    n_modes = len(state.modes)
    print_banner(f"开放题体验 | 模式 {state.modes} | 1 题（非 benchmark 主数据）")

    if repl_supports_shared_memory(state.modes):
        chat = _ensure_chat_session(api_key, state)
        if chat.turn_count > 0:
            print("  [对话] 本题将复用本对话已有共享记忆（计划/检索等缓存）", flush=True)
        task = build_tasks_from_asks([question])[0]
        row = run_repl_chat_turn(
            api_key,
            chat,
            task,
            run_quality=state.run_quality,
        )
        mode_num = state.modes[0]
        print_answer_brief(
            row,
            mode_label=MODE_CATALOG[mode_num][1],
            mode_key=MODE_NUM[mode_num],
        )
        return

    config = RunConfig(
        modes=state.modes,
        suite=None,
        ask_texts=[question],
        run_quality=state.run_quality,
        save_json=False,
        db_prefix="_run_repl",
    )

    def on_task(row, idx, total, mode_key):
        mode_num = mode_num_from_key(mode_key)
        print(
            f"\n  >>> 模式 {mode_num}（{MODE_CATALOG[mode_num][1]}）流水线已完成，"
            "正在汇总完整回答…",
            flush=True,
        )

    report = execute_session(api_key, config, on_task=on_task)

    if n_modes > 1:
        print_multi_mode_answers(report["sessions"], question=question)
        print_comparison_table(report["sessions"])
        for sess in report["sessions"]:
            print_mode_summary(
                sess["mode_num"],
                sess["mode_label"],
                sess["results"],
                sess["totals"],
                suite_label="开放题对比（演示）",
            )
    else:
        mode_num = state.modes[0]
        row = report["sessions"][0]["results"][0]
        print_answer_brief(
            row,
            mode_label=MODE_CATALOG[mode_num][1],
            mode_key=MODE_NUM[mode_num],
        )


def _open_qa_loop(api_key: str, state: ReplState) -> None:
    print()
    print("─" * 70)
    print("  开放题体验（演示用，非 benchmark 主数据）")
    print("─" * 70)
    if len(state.modes) > 1 or state.modes[0] == 1:
        print("  提示：开放题演示建议单选模式 2；当前为多模式/纯文本，每题独立执行。")
    elif repl_supports_shared_memory(state.modes):
        print("  本对话内多轮问题共用同一共享记忆库（可复用计划/检索缓存）。")
        print("  输入「新对话」可清空记忆重新开始。")
    print_repl_help()
    print()

    while True:
        line = _input_line("\n你> ")
        if not line:
            continue
        if _is_menu(line):
            print("  已返回主菜单（对话记忆仍保留，再选「开放题体验」可接着聊）。")
            return
        if _is_quit(line):
            print("  再见。")
            raise SystemExit(0)
        if line in CMD_HELP:
            print_repl_help()
            continue
        if line in CMD_NEW_CHAT:
            if repl_supports_shared_memory(state.modes):
                _invalidate_chat(state)
                _ensure_chat_session(api_key, state)
                print("  已开始新对话（共享记忆已清空）。")
            else:
                print("  当前模式组合不支持共享记忆，无需清空。")
            continue

        try:
            _run_one_question(api_key, state, line)
        except KeyboardInterrupt:
            print("\n  （本题已中断，可继续输入下一题）")
            continue
        except Exception as exc:
            print(f"\n  本题执行失败：{exc}")
            continue

        print("\n  可继续输入问题；「菜单」回主菜单，「退出」结束程序。")


def run_repl(api_key: str) -> int:
    """Main experiment console; only exits on user choice or SystemExit."""
    state = ReplState()
    print_launch_intro()

    while True:
        _show_main_menu(state)
        choice = _input_line("请选择 [1]: ")

        if _is_quit(choice) or choice == "0":
            print("  再见。")
            return 0

        try:
            if not choice or choice == "1":
                _suite_loop(api_key, state, defense_only=False)
            elif choice == "2":
                _compare_loop(api_key, state)
            elif choice == "3":
                _run_preset(
                    api_key,
                    state,
                    modes=[2],
                    suite_id="core12",
                    title="答辩快测 | core12 | 模式 2",
                )
            elif choice == "4":
                _run_preset(
                    api_key,
                    state,
                    modes=[1, 2],
                    suite_id="core12",
                    title="答辩对照 | core12 | 模式 1+2",
                )
            elif choice == "5":
                _open_qa_loop(api_key, state)
            elif choice == "6":
                _memory_demo_loop(api_key, state)
            elif choice == "7":
                picked = _pick_modes(state.modes)
                if picked is not None and picked != state.modes:
                    state.modes = picked
                    _invalidate_chat(state)
                    print(f"  已切换为模式 {state.modes}（对话记忆已重置）")
                elif picked is not None:
                    print(f"  模式未变：{state.modes}")
            elif choice in ("8", "?", "帮助", "help"):
                print_help_cn()
                rec = list_defense_suites()
                print(f"\n  答辩推荐套件：{', '.join(rec)}")
                for sid in rec:
                    print(f"    · {sid}: {suite_name(sid)} — {suite_desc(sid)}（{suite_eta(sid)}）")
            else:
                print("  无效选项，请输入 0～8。")
        except SystemExit:
            return 0
