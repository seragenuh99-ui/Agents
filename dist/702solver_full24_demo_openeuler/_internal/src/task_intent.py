"""任务意图检测 — 安全跨任务缓存复用门控。

意图分类体系：
- TaskIntent 枚举：COMPARE / POLICY / COMPLIANCE / VULNERABILITY / REVIEW / ANALYZE / GENERAL
- 基于描述文本关键词 + 标签匹配的确定性分类
- 严格意图（COMPARE/POLICY/COMPLIANCE/VULNERABILITY）不允许跨意图缓存复用

缓存门控函数：
- cache_intents_compatible(): 检查两个任务是否可以安全共享缓存
- requires_fresh_synthesis(): 判断任务是否必须全新生成（不能复用 E2E/摘要缓存）
- blocks_executor_dropout(): 判断任务是否必须保留执行器（不能跳过计算步骤）
- is_open_qa(): 判断是否为开放问答（跳演示 KB、更丰富的摘要）
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional, Sequence


# ═══════════════════════════════════════════════════════════════════════════════
# TaskIntent — 任务意图枚举
# ═══════════════════════════════════════════════════════════════════════════════

class TaskIntent(str, Enum):
    """任务意图分类，用于缓存策略决策。

    严格意图（STRICT）：缓存复用必须完全匹配，不能跨意图共享。
    宽松意图（GENERAL/ANALYZE/REVIEW）：可跨意图缓存复用。
    """
    COMPARE = "compare"           # 对比分析类
    POLICY = "policy"             # 政策/法规类
    COMPLIANCE = "compliance"     # 合规框架类（SOC2/ISO27001）
    VULNERABILITY = "vulnerability"  # 安全漏洞类
    REVIEW = "review"             # 评审/审计类
    ANALYZE = "analyze"           # 分析/研究/解释类
    GENERAL = "general"           # 通用类


# ═══════════════════════════════════════════════════════════════════════════════
# 意图检测标记词
# ═══════════════════════════════════════════════════════════════════════════════

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

# 严格意图集合 — 缓存复用必须完全匹配意图
_STRICT_INTENTS = frozenset(
    {
        TaskIntent.COMPARE,
        TaskIntent.POLICY,
        TaskIntent.COMPLIANCE,
        TaskIntent.VULNERABILITY,
    }
)

# 基准测试领域标签 — 不是开放问答
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


# ═══════════════════════════════════════════════════════════════════════════════
# 开放问答检测
# ═══════════════════════════════════════════════════════════════════════════════

def is_open_qa(
    description: str, tags: Optional[Sequence[str]] = None
) -> bool:
    """检测是否为开放问答（非基准测试任务）。

    判定规则：
    1. 标签含 "custom" 或 "open_qa" → 开放问答
    2. 中文文本且无基准测试领域标签 → 开放问答
    3. 含日常生活标记词（推荐、美食、旅游等）且无基准标签 → 开放问答

    开放问答的特殊处理：
    - 不使用演示知识库
    - 更丰富的证据和摘要 token 预算
    - 禁用计划缓存（防止不相关任务污染）
    """
    tag_set = {t.lower() for t in (tags or [])}
    if "custom" in tag_set or "open_qa" in tag_set:
        return True
    text = description or ""
    if not text.strip():
        return False
    # 中文文本且不在基准领域
    has_cjk = any("一" <= c <= "鿿" for c in text)
    if has_cjk and not (tag_set & _BENCHMARK_DOMAIN_TAGS):
        return True
    # 英文日常生活标记词
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


# ═══════════════════════════════════════════════════════════════════════════════
# 意图检测
# ═══════════════════════════════════════════════════════════════════════════════

def detect_intent(
    description: str, tags: Optional[Sequence[str]] = None
) -> TaskIntent:
    """从描述文本和标签中分类任务意图。

    检测顺序（优先级从高到低）：
    标签 > COMPARE > POLICY > COMPLIANCE > VULNERABILITY > REVIEW > ANALYZE > GENERAL
    """
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


# ═══════════════════════════════════════════════════════════════════════════════
# 缓存复用门控
# ═══════════════════════════════════════════════════════════════════════════════

def cache_intents_compatible(
    source_description: str,
    target_description: str,
    source_tags: Optional[Sequence[str]] = None,
    target_tags: Optional[Sequence[str]] = None,
) -> bool:
    """检查两个任务的意图是否兼容缓存复用。

    规则：
    - 如果源或目标是严格意图（COMPARE/POLICY/COMPLIANCE/VULNERABILITY），
      则必须意图完全匹配
    - 宽松意图之间可自由复用
    """
    src = detect_intent(source_description, source_tags)
    tgt = detect_intent(target_description, target_tags)
    if src in _STRICT_INTENTS or tgt in _STRICT_INTENTS:
        return src == tgt
    return True


def requires_fresh_synthesis(
    description: str, tags: Optional[Sequence[str]] = None
) -> bool:
    """判断任务是否必须全新生成（不能复用 E2E 或摘要缓存）。

    以下任务需要全新合成：
    - 开放问答（避免不相关缓存污染）
    - 对比分析、合规、评审类（结论不能跨任务复用）
    """
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
    """判断任务是否必须保留执行器（不能跳过计算步骤）。

    以下任务需要执行器：
    - 开放问答（需要整理检索要点）
    - 评审类、合规类（需要专门合成步骤）
    """
    if is_open_qa(description, tags):
        return True
    return detect_intent(description, tags) in (
        TaskIntent.REVIEW,
        TaskIntent.COMPLIANCE,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════════════════════

def memory_task_description(mem) -> str:
    """从记忆单元中尽力提取任务描述文本。

    处理前缀格式：Summary: / Plan: / Execution: / Retrieval:
    """
    topic = getattr(mem, "task_topic", "") or ""
    for prefix in ("Summary: ", "Plan: ", "Execution: ", "Retrieval: "):
        if topic.startswith(prefix):
            return topic[len(prefix):]
    return topic
