"""Chinese interactive wizard for run.py."""

from __future__ import annotations

from typing import List, Optional

from experiments.benchmark_suites import get_suite

from .run_engine import RunConfig
from .task_catalog import MODE_CATALOG, SUITE_CATALOG


def _input(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    line = input(f"{prompt}{suffix}: ").strip()
    return line or default


def _confirm(msg: str) -> bool:
    ans = _input(f"{msg} (y/n)", "y").lower()
    return ans in ("y", "yes", "是", "")


def run_wizard() -> Optional[RunConfig]:
    print()
    print("=" * 70)
    print("  702solver 实验向导（中文）")
    print("=" * 70)
    print("  提示：随时输入 ? 查看帮助；Ctrl+C 退出")
    print()

    print("【步骤 1/3】选择运行模式（可多选，逗号分隔）")
    for num, (_, name, desc) in MODE_CATALOG.items():
        print(f"  {num}  {name} — {desc}")
    print("  示例：2  或  1,2  或  1,2,3")
    mode_raw = _input("请输入模式编号", "2")
    if mode_raw == "?":
        from .run_reporter import print_help_cn

        print_help_cn()
        return run_wizard()

    modes: List[int] = []
    for part in mode_raw.replace("，", ",").split(","):
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
    if not modes:
        print("  至少选择一种模式。")
        return None

    print("\n【步骤 2/3】题目来源")
    print("  1  手动输入问题（一问一题，适合快速试）")
    print("  2  从题库模板选择")
    src = _input("请选择", "2")
    picked_suite: Optional[str] = "core12"
    ask_texts: Optional[List[str]] = None
    task_ids: Optional[List[str]] = None

    if src == "1":
        print("请输入问题（空行结束）：")
        lines = []
        while True:
            line = input("> ").strip()
            if not line:
                break
            lines.append(line)
        if not lines:
            print("  未输入问题，已取消。")
            return None
        ask_texts = lines
        picked_suite = None
    else:
        print("\n  可选题库：")
        for i, (sid, row) in enumerate(SUITE_CATALOG.items(), 1):
            name, desc = row[0], row[1]
            try:
                n = len(get_suite(sid))
            except KeyError:
                n = "?"
            print(f"  {sid:<16} {name}（{n} 题）— {desc}")
        picked_suite = _input("请输入套件 ID", "core12")
        if picked_suite not in SUITE_CATALOG:
            print(f"  未知套件：{picked_suite}")
            return None
        pick = _input("只跑部分题？输入 task_id 逗号分隔，回车=全部", "")
        if pick:
            task_ids = [x.strip() for x in pick.replace("，", ",").split(",") if x.strip()]
            picked_suite = None

    n_tasks = len(ask_texts) if ask_texts else (
        len(task_ids) if task_ids else len(get_suite(picked_suite or "core12"))
    )
    est = "约 2～15 分钟/模式（视题数与模式而定）"
    if ask_texts and n_tasks == 1:
        est = "单题约 30 秒～2 分钟；首题会先出现 Loading weights（加载向量模型），属正常"
    if picked_suite == "full24" or n_tasks >= 20:
        est = "较长，可能 30 分钟以上/模式"

    print("\n【步骤 3/3】确认")
    print(f"  模式：{', '.join(MODE_CATALOG[m][1] for m in modes)}")
    print(f"  题目：{n_tasks} 道")
    print(f"  预计：{est}")
    if not _confirm("确认开始"):
        print("  已取消。")
        return None

    final_suite = None if ask_texts or task_ids else picked_suite
    return RunConfig(
        modes=modes,
        suite=final_suite,
        task_ids=task_ids,
        ask_texts=ask_texts,
        run_quality=True,
    )
