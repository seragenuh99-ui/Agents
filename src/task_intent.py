"""Task intent detection for safe cross-task cache reuse."""

from __future__ import annotations

from enum import Enum
from typing import List, Optional, Sequence


class TaskIntent(str, Enum):
    COMPARE = "compare"
    POLICY = "policy"
    COMPLIANCE = "compliance"
    VULNERABILITY = "vulnerability"
    REVIEW = "review"
    ANALYZE = "analyze"
    GENERAL = "general"


_COMPARE_MARKERS = (
    "compare ",
    "comparison",
    " versus",
    " vs ",
    "across multiple",
    "trade-off",
    "tradeoff",
    "contrast ",
    "relative to",
    "optimal deployment",
)

_POLICY_MARKERS = (
    "policy only",
    "policy-only",
    "regulatory only",
    "compliance only",
)

_COMPLIANCE_MARKERS = (
    "soc2",
    "soc 2",
    "iso27001",
    "iso 27001",
    "control mapping",
    "compliance framework",
)

_VULN_MARKERS = (
    "vulnerability",
    "vulnerabilities",
    "sql injection",
    "command injection",
    "path traversal",
    "hardcoded credential",
    "insecure deserialization",
    "pickle.loads",
    "os.system",
    "cross-site scripting",
)

_REVIEW_MARKERS = (
    "code review",
    "review the",
    " audit",
    "security audit",
    "vulnerability assessment",
)

_STRICT_INTENTS = frozenset(
    {
        TaskIntent.COMPARE,
        TaskIntent.POLICY,
        TaskIntent.COMPLIANCE,
        TaskIntent.VULNERABILITY,
    }
)

# Benchmark suite domain tags — not open-ended Q&A
_BENCHMARK_DOMAIN_TAGS = frozenset(
    {
        "energy",
        "security",
        "database",
        "solar",
        "wind",
        "renewable",
        "code",
        "sql",
        "vulnerability",
        "compliance",
        "research",
        "continuous",
        "adversarial",
    }
)


def is_open_qa(
    description: str, tags: Optional[Sequence[str]] = None
) -> bool:
    """Open-ended questions (e.g. custom REPL) — skip demo KB, richer summarizer."""
    tag_set = {t.lower() for t in (tags or [])}
    if "custom" in tag_set or "open_qa" in tag_set:
        return True
    text = description or ""
    if not text.strip():
        return False
    # Chinese/general ask without benchmark domain tags
    has_cjk = any("\u4e00" <= c <= "\u9fff" for c in text)
    if has_cjk and not (tag_set & _BENCHMARK_DOMAIN_TAGS):
        return True
    # Obvious lifestyle / travel / food asks in English
    open_markers = (
        "recommend",
        "what to eat",
        "travel guide",
        "best places",
        "介绍一下",
        "有什么",
        "推荐",
        "攻略",
        "美食",
    )
    if any(m in text.lower() for m in open_markers) and not (
        tag_set & _BENCHMARK_DOMAIN_TAGS
    ):
        return True
    return False


def detect_intent(
    description: str, tags: Optional[Sequence[str]] = None
) -> TaskIntent:
    """Classify task intent from description and optional tags."""
    text = (description or "").lower()
    tag_set = {t.lower() for t in (tags or [])}

    if "comparison" in tag_set or "compare" in tag_set:
        return TaskIntent.COMPARE
    if any(m in text for m in _COMPARE_MARKERS):
        return TaskIntent.COMPARE

    if "policy" in tag_set and ("only" in text or "policy" in text):
        return TaskIntent.POLICY
    if any(m in text for m in _POLICY_MARKERS):
        return TaskIntent.POLICY

    if "compliance" in tag_set or any(m in text for m in _COMPLIANCE_MARKERS):
        return TaskIntent.COMPLIANCE

    if "vulnerability" in tag_set or "vuln" in tag_set:
        return TaskIntent.VULNERABILITY
    if any(m in text for m in _VULN_MARKERS):
        return TaskIntent.VULNERABILITY

    if "review" in tag_set or any(m in text for m in _REVIEW_MARKERS):
        return TaskIntent.REVIEW

    if any(
        w in text
        for w in ("analyze", "analyse", "research", "summarize", "explain", "describe")
    ):
        return TaskIntent.ANALYZE

    return TaskIntent.GENERAL


def cache_intents_compatible(
    source_description: str,
    target_description: str,
    source_tags: Optional[Sequence[str]] = None,
    target_tags: Optional[Sequence[str]] = None,
) -> bool:
    """Return False when reusing cache would mix incompatible task types."""
    src = detect_intent(source_description, source_tags)
    tgt = detect_intent(target_description, target_tags)
    if src in _STRICT_INTENTS or tgt in _STRICT_INTENTS:
        return src == tgt
    return True


def requires_fresh_synthesis(
    description: str, tags: Optional[Sequence[str]] = None
) -> bool:
    """Tasks that must not reuse a prior summary verbatim (E2E / summarizer cache)."""
    if is_open_qa(description, tags):
        return True
    return detect_intent(description, tags) in (
        TaskIntent.COMPARE,
        TaskIntent.COMPLIANCE,
        TaskIntent.REVIEW,
    )


def blocks_executor_dropout(
    description: str, tags: Optional[Sequence[str]] = None
) -> bool:
    """Keep executor for tasks that need a dedicated synthesis pass."""
    if is_open_qa(description, tags):
        return True
    return detect_intent(description, tags) in (
        TaskIntent.REVIEW,
        TaskIntent.COMPLIANCE,
    )


def memory_task_description(mem) -> str:
    """Best-effort task text from a memory unit."""
    topic = getattr(mem, "task_topic", "") or ""
    for prefix in ("Summary: ", "Plan: ", "Execution: ", "Retrieval: "):
        if topic.startswith(prefix):
            return topic[len(prefix) :]
    return topic
