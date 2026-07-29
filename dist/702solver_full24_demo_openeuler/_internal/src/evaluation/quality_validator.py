"""任务答案质量验证 — 启发式规则 + 可选 LLM 评判。

验证流水线：
1. 提取摘要有效载荷 → (summary_obj, answer_text)
2. 嵌入相关性评分（task 与 answer 的余弦相似度）
3. 主题覆盖率（预期主题词命中/遗漏）
4. 跨域一致性检查（安全问题中不应出现光伏术语等）
5. 错误/回退标记检测
6. 启发式评分（加权 0-1）→ 可通过阈值
7. 可选 LLM 评判（1-5 分 + 切题判断）→ 综合评分

输出：ValidationResult（含各维度分数和问题列表）
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# 跨域标记 — 检测不应该出现在特定领域答案中的术语
# ═══════════════════════════════════════════════════════════════════════════════

_DEFAULT_CROSS_DOMAIN: Dict[str, List[str]] = {
    "security": ["photovoltaic", "wind turbine", "lcoe", "perovskite"],
    "database": ["csrf token", "xss attack", "os.system("],
    "energy": ["sql injection via string", "jwt validation failure"],
}


# ═══════════════════════════════════════════════════════════════════════════════
# ValidationResult — 单任务验证结果
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ValidationResult:
    """单任务的质量验证分数与标记。

    分层评分：
    - 基础维度：embedding_relevance, topic_coverage, domain_consistent
    - 状态标记：has_error_flag, has_fallback_flag, answer_nonempty
    - 评分结果：heuristic_score, composite_score
    - LLM 评判（可选）：llm_on_topic, llm_score, llm_reason
    """

    task_id: str

    # ── 基础维度 ──
    embedding_relevance: float = 0.0     # 嵌入余弦相似度
    topic_coverage: float = 0.0           # 预期主题覆盖率
    topics_hit: List[str] = field(default_factory=list)
    topics_missed: List[str] = field(default_factory=list)
    domain_consistent: bool = True        # 域一致性
    cross_domain_hits: List[str] = field(default_factory=list)

    # ── 状态标记 ──
    has_error_flag: bool = False
    has_fallback_flag: bool = False
    answer_nonempty: bool = False         # 回答非空（>40 字符）

    # ── 评分结果 ──
    heuristic_score: float = 0.0          # 纯启发式 0-1
    heuristic_pass: bool = False
    llm_on_topic: Optional[bool] = None   # LLM 评判：是否切题
    llm_score: Optional[int] = None       # LLM 评判：1-5 分
    llm_reason: str = ""
    llm_issues: List[str] = field(default_factory=list)
    composite_score: float = 0.0          # 综合评分（启发式 + LLM）
    composite_pass: bool = False
    issues: List[str] = field(default_factory=list)  # 汇总问题

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════════════════
# 有效载荷提取
# ═══════════════════════════════════════════════════════════════════════════════

def extract_summary_payload(summary_step: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """从摘要步骤中提取结构化对象和扁平文本。

    结构化对象用于状态标记检测（_error/_fallback），
    扁平文本用于嵌入计算和主题匹配。
    """
    raw = summary_step.get("summary", summary_step)
    if isinstance(raw, dict):
        parts = list(raw.get("key_findings", []) or [])
        if raw.get("facts"):
            parts.extend(raw["facts"] if isinstance(raw["facts"], list) else [str(raw["facts"])])
        if raw.get("conclusion"):
            parts.append(str(raw["conclusion"]))
        text = " ".join(str(p) for p in parts if p)
        return raw, text.strip()
    return {}, str(raw).strip()


# ═══════════════════════════════════════════════════════════════════════════════
# 嵌入相关性
# ═══════════════════════════════════════════════════════════════════════════════

def embedding_relevance(
    task_description: str, answer_text: str, encode_fn
) -> float:
    """计算任务描述与回答文本的嵌入余弦相似度。

    值越接近 1.0 表示回答与任务越相关。
    """
    if not task_description or not answer_text:
        return 0.0
    try:
        a = np.array(encode_fn(task_description))
        b = np.array(encode_fn(answer_text[:800]))
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))
    except Exception:
        return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 主题覆盖率检测
# ═══════════════════════════════════════════════════════════════════════════════

# 主题同义词表 — 避免因术语变体导致的假阴性
TOPIC_SYNONYMS: Dict[str, List[str]] = {
    "solar": ["photovoltaic", "pv", "monocrystalline", "polycrystalline", "thin-film", "thin film"],
    "photovoltaic": ["solar", "pv", "panel"],
    "wind": ["turbine", "offshore", "onshore", "dfig", "pmsg"],
    "compare": ["comparison", "versus", "vs", "contrast"],
    "security": ["vulnerab", "attack", "exploit", "audit"],
    "vulnerab": ["security", "flaw", "weakness"],
    "web": ["http", "csrf", "jwt", "cors", "oauth", "xss"],
    "database": ["sql", "nosql", "db", "postgres", "mysql", "redis", "mongodb"],
    "optim": ["optimization", "tuning", "performance"],
    "model": ["schema", "normalization", "entity", "er "],
}


def _topic_matches(lower_text: str, topic: str) -> bool:
    """检查主题词是否在文本中匹配（直接匹配或同义词匹配）。"""
    t = topic.lower()
    if t in lower_text:
        return True
    for syn in TOPIC_SYNONYMS.get(t, []):
        if syn.lower() in lower_text:
            return True
    return False


def topic_coverage(
    answer_text: str, expected_topics: List[str]
) -> Tuple[float, List[str], List[str]]:
    """检查回答覆盖了哪些预期主题词。

    Returns:
        (覆盖率, 命中列表, 遗漏列表)
    """
    if not expected_topics:
        return 1.0, [], []
    lower = answer_text.lower()
    hit, missed = [], []
    for topic in expected_topics:
        if _topic_matches(lower, topic):
            hit.append(topic)
        else:
            missed.append(topic)
    return len(hit) / len(expected_topics), hit, missed


# ═══════════════════════════════════════════════════════════════════════════════
# 跨域一致性检查
# ═══════════════════════════════════════════════════════════════════════════════

def check_cross_domain(
    answer_text: str, domain: str, markers: Optional[Dict[str, List[str]]] = None
) -> Tuple[bool, List[str]]:
    """检查回答中是否混入了其他领域的不相关术语。

    例如：安全问题的答案中不应出现光伏板效率数据。

    Returns:
        (是否一致, 跨域命中列表)
    """
    markers = markers or _DEFAULT_CROSS_DOMAIN
    forbidden = markers.get(domain, [])
    lower = answer_text.lower()
    hits = [m for m in forbidden if m.lower() in lower]
    return len(hits) == 0, hits


# ═══════════════════════════════════════════════════════════════════════════════
# 摘要标记检测
# ═══════════════════════════════════════════════════════════════════════════════

def detect_summary_flags(summary_obj: Dict[str, Any]) -> Tuple[bool, bool]:
    """检测摘要中的错误和回退标记。

    _error: 执行过程中发生错误
    _fallback: 使用了证据拼装而非 LLM 生成
    """
    if not isinstance(summary_obj, dict):
        return False, False
    return bool(summary_obj.get("_error")), bool(summary_obj.get("_fallback"))


# ═══════════════════════════════════════════════════════════════════════════════
# 评分函数
# ═══════════════════════════════════════════════════════════════════════════════

def heuristic_score(v: ValidationResult) -> float:
    """纯启发式加权评分（0-1），无需 LLM。

    权重分配：
    - 嵌入相关性 35%：答案与任务的语义相似度
    - 主题覆盖率 30%：预期主题词的覆盖程度
    - 域一致性 20%：未出现跨域污染
    - 干净度 15%：回答非空且无错误标记
    """
    w_rel, w_topic, w_domain, w_clean = 0.35, 0.30, 0.20, 0.15
    clean = 1.0 if (v.answer_nonempty and not v.has_error_flag) else 0.0
    domain = 1.0 if v.domain_consistent else 0.0
    return (
        w_rel * min(1.0, max(0.0, v.embedding_relevance))
        + w_topic * min(1.0, max(0.0, v.topic_coverage))
        + w_domain * domain
        + w_clean * clean
    )


def composite_score(v: ValidationResult, use_llm: bool = True) -> float:
    """综合评分：启发式 + LLM 评判（如启用）。

    权重：启发式 55% + LLM 标准化分 30% + LLM 切题 15%
    """
    h = v.heuristic_score
    if use_llm and v.llm_score is not None:
        llm_norm = min(5, max(1, v.llm_score)) / 5.0
        on_topic = 1.0 if v.llm_on_topic else 0.0
        return 0.55 * h + 0.30 * llm_norm + 0.15 * on_topic
    return h


# ═══════════════════════════════════════════════════════════════════════════════
# 主验证函数
# ═══════════════════════════════════════════════════════════════════════════════

def validate_task(
    task: Dict[str, Any],
    summary_step: Dict[str, Any],
    encode_fn,
    cross_domain_markers: Optional[Dict[str, List[str]]] = None,
    llm=None,
    use_llm_judge: bool = False,
    pass_threshold: float = 0.65,
) -> Tuple[ValidationResult, Dict[str, Any], str]:
    """验证单任务答案质量。

    验证流程：
    1. 提取摘要 → 2. 嵌入相关性 → 3. 主题覆盖 → 4. 跨域检查
    → 5. 标记检测 → 6. 禁止词检查 → 7. 启发式评分 → 8. 可选 LLM 评判
    → 9. 综合评分

    Returns:
        (ValidationResult, summary_obj, answer_text)
    """
    summary_obj, answer_text = extract_summary_payload(summary_step)
    v = ValidationResult(task_id=task["task_id"])

    # 基础维度
    v.embedding_relevance = embedding_relevance(
        task["description"], answer_text, encode_fn
    )
    v.topic_coverage, v.topics_hit, v.topics_missed = topic_coverage(
        answer_text, task.get("expected_topics", [])
    )
    domain = task.get("domain", task.get("tags", ["unknown"])[0])
    v.domain_consistent, v.cross_domain_hits = check_cross_domain(
        answer_text, domain, cross_domain_markers
    )
    v.has_error_flag, v.has_fallback_flag = detect_summary_flags(summary_obj)
    v.answer_nonempty = len(answer_text) > 40

    # 汇总问题
    if v.has_error_flag:
        v.issues.append("summary_has_error_flag")
    if v.has_fallback_flag:
        v.issues.append("summary_used_evidence_fallback")
    if not v.answer_nonempty:
        v.issues.append("answer_too_short_or_empty")
    if not v.domain_consistent:
        v.issues.append(f"possible_cross_domain: {v.cross_domain_hits}")
    if v.topic_coverage < 0.4:
        v.issues.append(f"low_topic_coverage: {v.topics_missed}")

    # 禁止词检查
    forbidden = task.get("forbidden_answer_topics") or []
    forbidden_hits = [
        term for term in forbidden
        if term.lower() in answer_text.lower()
    ]
    if forbidden_hits:
        v.issues.append(f"forbidden_topic_leak: {forbidden_hits}")
        v.domain_consistent = False

    # 启发式评分
    v.heuristic_score = round(heuristic_score(v), 3)
    v.heuristic_pass = v.heuristic_score >= pass_threshold

    # 可选 LLM 评判
    if use_llm_judge and llm is not None and answer_text:
        judge = llm_judge_answer(llm, task["description"], answer_text, summary_obj)
        v.llm_on_topic = judge.get("on_topic")
        v.llm_score = judge.get("score")
        v.llm_reason = judge.get("reason", "")
        v.llm_issues = judge.get("issues", [])
        if not v.llm_on_topic:
            v.issues.append(f"llm_judge_off_topic: {v.llm_reason[:80]}")
        v.issues.extend(v.llm_issues)

    # 综合评分
    v.composite_score = round(composite_score(v, use_llm=use_llm_judge), 3)
    v.composite_pass = v.composite_score >= pass_threshold and not v.has_error_flag

    return v, summary_obj, answer_text


# ═══════════════════════════════════════════════════════════════════════════════
# LLM 评判
# ═══════════════════════════════════════════════════════════════════════════════

def llm_judge_answer(
    llm, task_description: str, answer_text: str, summary_obj: Dict[str, Any]
) -> Dict[str, Any]:
    """使用 LLM 评判答案质量。

    评判维度：
    - on_topic: 是否切题
    - score: 1-5 分（5=优秀，1=完全错误领域）
    - reason: 一句话理由
    - issues: 具体问题列表
    """
    findings = ""
    if isinstance(summary_obj, dict):
        findings = json.dumps(
            {
                "key_findings": summary_obj.get("key_findings", [])[:5],
                "conclusion": (summary_obj.get("conclusion") or "")[:300],
            },
            ensure_ascii=False,
        )
    system = (
        "You evaluate whether an AI assistant's summary correctly answers the user's task. "
        "Reply with JSON only: "
        '{"on_topic":true|false,"score":1-5,"reason":"one sentence","issues":["..."]}'
    )
    user = (
        f"Task:\n{task_description[:600]}\n\n"
        f"Answer:\n{answer_text[:1200]}\n\n"
        f"Structured:\n{findings[:800]}\n\n"
        "Does the answer address the task domain and requirements? score 5=excellent, 1=wrong domain."
    )
    try:
        raw = llm.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=200,
            temperature=0.0,
        )
        from ..agents.base import _extract_json

        data = _extract_json(raw)
        on_topic = bool(data.get("on_topic", False))
        score = int(data.get("score", 1))
        score = max(1, min(5, score))
        return {
            "on_topic": on_topic,
            "score": score,
            "reason": str(data.get("reason", ""))[:300],
            "issues": list(data.get("issues", []))[:5],
        }
    except Exception as e:
        return {
            "on_topic": None,
            "score": None,
            "reason": f"judge_failed: {e}",
            "issues": ["llm_judge_error"],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Markdown 格式化输出
# ═══════════════════════════════════════════════════════════════════════════════

def format_answer_markdown(summary_obj: Dict[str, Any], answer_text: str) -> str:
    """将答案格式化为人类可读的 Markdown 块。"""
    if isinstance(summary_obj, dict) and summary_obj.get("key_findings"):
        lines = ["**要点**:"]
        for i, f in enumerate(summary_obj.get("key_findings", [])[:8], 1):
            lines.append(f"{i}. {f}")
        if summary_obj.get("conclusion"):
            lines.append(f"\n**结论**: {summary_obj['conclusion']}")
        if summary_obj.get("_error"):
            lines.append("\n⚠️ `_error`")
        if summary_obj.get("_fallback"):
            lines.append("\n⚠️ `_fallback`（证据拼装，非 LLM 生成）")
        return "\n".join(lines)
    return answer_text or "*(无输出)*"


def format_validation_markdown(v: ValidationResult) -> str:
    """将验证结果格式化为 Markdown 表格。"""
    status = "✅ PASS" if v.composite_pass else "❌ FAIL"
    lines = [
        f"**验证结果**: {status}（综合 {v.composite_score:.3f}，启发式 {v.heuristic_score:.3f}）",
        "",
        "| 指标 | 值 |",
        "|------|-----|",
        f"| 嵌入相关性 | {v.embedding_relevance:.3f} |",
        f"| 主题覆盖率 | {v.topic_coverage:.1%} ({len(v.topics_hit)}/{len(v.topics_hit)+len(v.topics_missed)}) |",
        f"| 域一致性 | {'是' if v.domain_consistent else '否'} |",
        f"| 非空回答 | {'是' if v.answer_nonempty else '否'} |",
    ]
    if v.llm_score is not None:
        lines.append(f"| LLM 评判 | {v.llm_score}/5，切题={'是' if v.llm_on_topic else '否'} |")
        if v.llm_reason:
            lines.append(f"| 评判说明 | {v.llm_reason} |")
    if v.topics_missed:
        lines.append(f"| 未覆盖主题词 | {', '.join(v.topics_missed)} |")
    if v.issues:
        lines.append(f"| 问题 | {'; '.join(v.issues)} |")
    return "\n".join(lines)
