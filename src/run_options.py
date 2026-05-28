"""Runtime flags for reproducible experiment ablations."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Sequence

from .task_intent import is_open_qa


@dataclass(frozen=True)
class RunOptions:
    """Controls caching, dropout, and threshold knobs for one benchmark run."""

    enable_e2e_cache: bool = True
    enable_summarizer_e2e: bool = True
    enable_summarizer_adapt: bool = True
    enable_planner_cache: bool = True
    enable_summarizer_cache: bool = True
    enable_executor_dropout: bool = True
    enable_memory_index: bool = True
    enable_safe_sieve: bool = True
    enable_intent_cache_gate: bool = True
    enable_domain_llm_judge: bool = False
    summarizer_cache_min_relevance: float = 0.75

    e2e_threshold: float = 0.85
    summarizer_e2e_threshold: float = 0.85
    summarizer_adapt_threshold: float = 0.72
    summarizer_adapt_ceiling: float = 0.84
    planner_direct_threshold: float = 0.82
    planner_template_threshold: float = 0.78
    planner_judge_threshold: float = 0.55
    domain_plan_threshold: float = 0.68
    executor_dropout_threshold: float = 0.65
    retriever_cache_threshold: float = 0.88

    evidence_max_chars: int = 600
    summarizer_max_tokens: int = 512
    summarizer_adapt_max_tokens: int = 280
    planner_template_max_tokens: int = 256
    planner_full_max_tokens: int = 448

    def label(self) -> str:
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


DEFAULT_OPTIONS = RunOptions()


def options_for_task(
    base: RunOptions,
    description: str,
    tags: Optional[Sequence[str]] = None,
) -> RunOptions:
    """Per-task overrides: open Q&A gets richer evidence/summary, no demo-KB reliance."""
    if not is_open_qa(description, tags):
        return base
    return replace(
        base,
        evidence_max_chars=max(base.evidence_max_chars, 1200),
        summarizer_max_tokens=max(base.summarizer_max_tokens, 1024),
        summarizer_adapt_max_tokens=max(base.summarizer_adapt_max_tokens, 480),
        planner_full_max_tokens=max(base.planner_full_max_tokens, 384),
        planner_template_max_tokens=max(base.planner_template_max_tokens, 320),
        # Avoid cross-task plan/summary reuse polluting unrelated questions
        enable_planner_cache=False,
    )
