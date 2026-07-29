#!/usr/bin/env python3
"""702solver 统一实验入口 — 中文提示、多模式、可选题库或自提问。

示例：
  python3 run.py
  python3 run.py --mode 2 --suite core12
  python3 run.py --mode 1,2 --suite continuous12
  python3 run.py --mode 2 --ask "对比太阳能与风电的LCOE"
"""

from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

env_path = os.path.join(_ROOT, ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

from src.cli.run_engine import MODE_NUM, RunConfig, execute_session, mode_num_from_key
from src.cli.repl_session import run_repl
from src.cli.run_reporter import (
    print_banner,
    print_comparison_table,
    print_help_cn,
    print_mode_summary,
    print_multi_mode_answers,
    print_task_detail,
)
from src.cli.task_catalog import EXPERIMENT_PRESETS, MODE_CATALOG, SUITE_CATALOG, suite_name


def _parse_modes(s: str) -> list[int]:
    modes = []
    for part in s.replace("，", ",").split(","):
        part = part.strip()
        if not part:
            continue
        m = int(part)
        if m not in MODE_CATALOG:
            raise argparse.ArgumentTypeError(
                f"无效模式 {m}，请用 1/2/3（{', '.join(MODE_CATALOG[k][1] for k in MODE_CATALOG)}）"
            )
        modes.append(m)
    return modes


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="702solver 统一实验入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
模式（--mode）：
  1  纯文本对照    2  结构化全功能    3  结构化无缓存
  可多选：--mode 1,2

任务：
  --suite core12 | continuous12 | full24 | energy_research | code_security ...
  --tasks e1_solar_basics,e4_renewable_comparison
  --ask "你的问题"

连续 ≥10 轮关联任务请用：--suite continuous12
        """,
    )
    p.add_argument(
        "--mode",
        type=_parse_modes,
        help="运行模式，逗号分隔，如 2 或 1,2,3",
    )
    p.add_argument("--suite", default=None, help="题库模板 ID")
    p.add_argument("--tasks", default=None, help="指定 task_id，逗号分隔")
    p.add_argument("--ask", default=None, help="自定义问题（单题）")
    p.add_argument("--quality", action="store_true", default=True, help="质量检测（默认开）")
    p.add_argument("--no-quality", action="store_false", dest="quality")
    p.add_argument("--json", default=None, help="保存 JSON 报告路径")
    p.add_argument("--yes", "-y", action="store_true", help="跳过确认")
    p.add_argument(
        "--preset",
        choices=sorted(EXPERIMENT_PRESETS.keys()),
        help="答辩一键预设：defense12 / defense12-compare / full24",
    )
    p.add_argument("--help-cn", action="store_true", help="中文帮助")
    return p


def _config_from_preset(preset: str) -> RunConfig:
    spec = EXPERIMENT_PRESETS[preset]
    return RunConfig(
        modes=list(spec["modes"]),
        suite=spec["suite"],
        run_quality=True,
        save_json=True,
    )


def config_from_args(args: argparse.Namespace) -> RunConfig | None:
    if args.help_cn:
        print_help_cn()
        return None

    if getattr(args, "preset", None):
        return _config_from_preset(args.preset)

    if args.mode is None and args.suite is None and args.ask is None:
        return None  # 交互 REPL，由 main() 启动

    modes = args.mode or [2]
    task_ids = None
    if args.tasks:
        task_ids = [x.strip() for x in args.tasks.replace("，", ",").split(",") if x.strip()]

    ask_texts = [args.ask] if args.ask else None
    suite = args.suite or ("core12" if not ask_texts and not task_ids else None)

    if not ask_texts and not task_ids and not suite:
        print("错误：请指定 --suite、--tasks 或 --ask")
        return None

    return RunConfig(
        modes=modes,
        suite=suite,
        task_ids=task_ids,
        ask_texts=ask_texts,
        run_quality=args.quality,
        json_path=args.json,
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("错误：未找到 DEEPSEEK_API_KEY。")
        print("  请在项目根目录 .env 中添加：DEEPSEEK_API_KEY=你的密钥")
        return 1

    if args.help_cn:
        print_help_cn()
        return 0

    if args.mode is None and args.suite is None and args.ask is None and not args.preset:
        return run_repl(api_key)

    config = config_from_args(args)
    if config is None:
        return 0

    from experiments.benchmark_suites import resolve_tasks

    tasks = (
        resolve_tasks(suite=config.suite, task_ids=config.task_ids)
        if not config.ask_texts
        else None
    )
    if config.ask_texts:
        from src.cli.run_engine import build_tasks_from_asks

        tasks = build_tasks_from_asks(config.ask_texts)
    n = len(tasks)

    suite_label = config.suite or "自定义"
    if config.suite and config.suite in SUITE_CATALOG:
        suite_label = f"{suite_name(config.suite)} ({config.suite})"

    preset_label = ""
    if args.preset:
        preset_label = EXPERIMENT_PRESETS[args.preset]["label"]
        print()
        print(f"预设实验：{preset_label}")

    if not args.yes and (args.mode is not None or args.preset):
        print()
        print(f"即将执行：模式 {config.modes} | {n} 题 | {suite_label}")
        ans = input("确认开始？(y/n) [y]: ").strip().lower()
        if ans not in ("", "y", "yes", "是"):
            print("已取消。")
            return 0

    print_banner(f"702solver 实验 | {n} 题 | 模式 {config.modes}")

    multi_single = len(config.modes) > 1 and n == 1

    def on_task(row, idx, total, mode_key):
        mode_num = mode_num_from_key(mode_key)
        if multi_single:
            print(
                f"\n  >>> 模式 {mode_num}（{MODE_CATALOG[mode_num][1]}）已完成…",
                flush=True,
            )
            return
        print_task_detail(
            row,
            mode_label=MODE_CATALOG[mode_num][1],
            index=idx,
            total=total,
            mode_key=mode_key,
        )

    report = execute_session(api_key, config, on_task=on_task)

    if multi_single and config.ask_texts:
        print_multi_mode_answers(report["sessions"], question=config.ask_texts[0])

    for sess in report["sessions"]:
        print_mode_summary(
            sess["mode_num"],
            sess["mode_label"],
            sess["results"],
            sess["totals"],
            suite_label=suite_label,
        )

    print_comparison_table(report["sessions"])

    jpath = report.get("json_path") or config.json_path
    from src.cli.run_reporter import print_experiment_footer

    print_experiment_footer(json_path=jpath)

    print("\n批处理完成。交互式实验控制台请运行：python3 run.py  或  ./启动实验.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
