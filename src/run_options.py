"""运行时特性开关 — 可复现实验消融控制。

RunOptions 是一个 frozen dataclass，集中控制所有缓存、dropout、
阈值参数。每个实验变体可通过 replace() 创建变体，确保基准一致性。

设计决策：
- frozen=True：防止实验过程中意外修改参数
- 所有阈值为浮点数（0-1 范围），方便调参
- options_for_task() 提供开放问答的默认覆盖（更丰富的 token 预算）
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Sequence

from .task_intent import is_open_qa


# ═══════════════════════════════════════════════════════════════════════════════
# RunOptions — 运行时配置
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class RunOptions:
    """控制缓存、dropout 和阈值的一次实验运行配置。

    特性开关（按流水线阶段排列）：
    ┌─────────────────────────┬──────────────────────────────────────────────┐
    │ 开关                    │ 作用                                         │
    ├─────────────────────────┼──────────────────────────────────────────────┤
    │ enable_e2e_cache        │ 端到端缓存（cos≥threshold 跳过全部 pipeline） │
    │ enable_summarizer_e2e   │ 摘要 E2E 缓存（复用历史摘要）                 │
    │ enable_summarizer_adapt │ 摘要适配（微调历史摘要而非全新生成）          │
    │ enable_planner_cache    │ 计划缓存（模板填充 vs 完整 LLM 生成）         │
    │ enable_summarizer_cache │ 摘要缓存                                     │
    │ enable_executor_dropout │ 执行器 dropout（模板填充分数高时跳过执行器）  │
    │ enable_memory_index     │ FAISS 语义索引                                │
    │ enable_safe_sieve       │ SafeSieve-lite 模板成功率反馈               │
    │ enable_intent_cache_gate│ 意图门控（禁止不兼容意图间缓存复用）          │
    │ enable_domain_llm_judge │ 领域 LLM 评判（默认关闭，实验用）             │
    │ enable_message_dispatch │ 消息驱动调度（替代直接调用，默认关闭）        │
    └─────────────────────────┴──────────────────────────────────────────────┘
    """

    # ── 特性开关 ──
    enable_e2e_cache: bool = True
    enable_summarizer_e2e: bool = True
    enable_summarizer_adapt: bool = True
    enable_planner_cache: bool = True
    enable_summarizer_cache: bool = True
    enable_executor_dropout: bool = True
    enable_memory_index: bool = True
    enable_safe_sieve: bool = True
    enable_intent_cache_gate: bool = True
    enable_domain_llm_judge: bool = False    # 默认关闭
    enable_message_dispatch: bool = False    # 默认关闭

    # ── 缓存相关性阈值（余弦相似度）──
    e2e_threshold: float = 0.85              # E2E 缓存命中
    summarizer_e2e_threshold: float = 0.85   # 摘要直接复用
    summarizer_adapt_threshold: float = 0.72 # 摘要适配下限
    summarizer_adapt_ceiling: float = 0.84   # 摘要适配上限（超过则直接复用）
    planner_direct_threshold: float = 0.82   # 计划直接复用
    planner_template_threshold: float = 0.78 # 模板填充下限
    planner_judge_threshold: float = 0.55    # LLM 评判确认阈值
    domain_plan_threshold: float = 0.68      # 领域计划种子注入
    executor_dropout_threshold: float = 0.65 # 执行器跳过阈值（模板分数）
    retriever_cache_threshold: float = 0.88  # 检索缓存阈值

    # ── Token/字符预算 ──
    evidence_max_chars: int = 600            # 证据最大字符数
    summarizer_max_tokens: int = 512         # 摘要生成最大 token
    summarizer_adapt_max_tokens: int = 280   # 摘要适配最大 token
    planner_template_max_tokens: int = 256   # 计划模板填充最大 token
    planner_full_max_tokens: int = 448       # 完整计划生成最大 token

    def label(self) -> str:
        """生成人类可读的配置标签（用于实验报告命名）。

        Returns:
            如 "full"（全开）、"no_e2e_no_drop"（关闭部分特性）
        """
        parts = []
        if not self.enable_memory_index:
            parts.append("no_index")
        if not self.enable_e2e_cache:
            parts.append("no_e2e")
        if not self.enable_planner_cache:
            parts.append("no_plan_cache")
        if not self.enable_executor_dropout:
            parts.append("no_drop")
        if not self.enable_safe_sieve:
            parts.append("no_sieve")
        if not self.enable_summarizer_adapt:
            parts.append("no_adapt")
        return "_".join(parts) if parts else "full"


# ═══════════════════════════════════════════════════════════════════════════════
# 默认配置与任务级覆盖
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_OPTIONS = RunOptions()


def options_for_task(
    base: RunOptions,
    description: str,
    tags: Optional[Sequence[str]] = None,
) -> RunOptions:
    """为特定任务生成配置覆盖。

    开放问答任务的特殊处理：
    - 更丰富的证据（1200 字符）和摘要（1024 token）
    - 禁用计划缓存（防止无关任务污染模板）
    - 提高适配 token 预算

    非开放问答任务直接返回原始配置。
    """
    if not is_open_qa(description, tags):
        return base
    return replace(
        base,
        evidence_max_chars=max(base.evidence_max_chars, 1200),
        summarizer_max_tokens=max(base.summarizer_max_tokens, 1024),
        summarizer_adapt_max_tokens=max(base.summarizer_adapt_max_tokens, 480),
        planner_full_max_tokens=max(base.planner_full_max_tokens, 384),
        planner_template_max_tokens=max(base.planner_template_max_tokens, 320),
        enable_planner_cache=False,
    )
