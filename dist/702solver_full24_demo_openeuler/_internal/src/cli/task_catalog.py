"""Human-readable task suite catalog for run.py."""

from __future__ import annotations

from typing import Dict, List, Tuple

# suite_id -> (中文名, 说明, 预计耗时, 答辩推荐)
SuiteEntry = Tuple[str, str, str, bool]

SUITE_CATALOG: Dict[str, SuiteEntry] = {
    "core12": (
        "标准 12 题",
        "能源 / 安全 / 数据库各 4 题，适合答辩快测",
        "~12 分钟",
        True,
    ),
    "full24": (
        "完整 24 题",
        "含扩展题与 6 道陷阱题，正式报告主数据",
        "~35 分钟",
        True,
    ),
    "continuous12": (
        "连续 12 轮关联任务",
        "同一城市可再生能源规划链，满足「≥10 轮连续相关任务」要求",
        "~15 分钟",
        True,
    ),
    "continuous_10": (
        "连续 10 轮",
        "与 continuous12 前 10 题相同",
        "~12 分钟",
        False,
    ),
    "energy_research": (
        "能源关联 2 题",
        "太阳能 → 风光对比，演示记忆复用",
        "~3 分钟",
        False,
    ),
    "code_security": (
        "安全关联 2 题",
        "漏洞分析 → 审计，演示模式复用",
        "~3 分钟",
        False,
    ),
    "energy4": ("能源 4 题", "仅 core12 能源子集", "~5 分钟", False),
    "adversarial6": (
        "陷阱 6 题",
        "建议先跑 core12 热记忆",
        "~8 分钟",
        False,
    ),
    "extended6": ("扩展 6 题", "core12 之外的扩展覆盖", "~10 分钟", False),
}

# 记忆/连续任务演示用套件
MEMORY_DEMO_SUITES: Tuple[str, ...] = (
    "energy_research",
    "code_security",
    "continuous_10",
)

MODE_CATALOG = {
    1: ("pure", "纯文本对照", "每题 3 次 LLM，无记忆无协议，作基线"),
    2: ("structured", "结构化全功能", "记忆 + 模板 + 缓存（答辩主结果）"),
    3: ("nocache", "结构化无缓存", "有结构化流程，关闭全部缓存"),
}

# 一键实验预设（命令行 --preset 与启动脚本共用）
EXPERIMENT_PRESETS: Dict[str, Dict] = {
    "defense12": {
        "modes": [2],
        "suite": "core12",
        "label": "答辩快测 · 模式 2 · core12",
    },
    "defense12-compare": {
        "modes": [1, 2],
        "suite": "core12",
        "label": "答辩对照 · 模式 1+2 · core12",
    },
    "full24": {
        "modes": [2],
        "suite": "full24",
        "label": "完整 24 题 · 模式 2",
    },
}


def suite_name(suite_id: str) -> str:
    return SUITE_CATALOG[suite_id][0]


def suite_desc(suite_id: str) -> str:
    return SUITE_CATALOG[suite_id][1]


def suite_eta(suite_id: str) -> str:
    return SUITE_CATALOG[suite_id][2]


def suite_defense_recommended(suite_id: str) -> bool:
    return SUITE_CATALOG[suite_id][3]


def list_suites_menu() -> List[str]:
    return list(SUITE_CATALOG.keys())


def list_defense_suites() -> List[str]:
    return [sid for sid, row in SUITE_CATALOG.items() if row[3]]
