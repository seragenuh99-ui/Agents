"""共享记忆的数据模型 — MemoryUnit 定义。

每条记忆记录任务执行过程中的中间结果、总结、证据链、结论或策略。
MemoryUnit 是贯穿整个系统的基础数据结构，被 MemoryStore、Orchestrator
和所有 Agent 共同使用。

记忆分类（memory_type）：
- result:   任务执行结果
- evidence: 支持性证据
- strategy: 领域策略/计划模板
- fact:     事实性知识
- error:    错误记录

抽象层级（abstraction_level）：
- 0: 具体实例（某次任务执行的具体结果）
- 1: 领域模板（从多个具体实例中抽象出的可复用模式）
- 2: 通用原则（更高层次的元知识）
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MemoryUnit:
    """共享记忆中的单条记忆单元。

    每条记忆记录：来源 Agent、任务主题、摘要、详细内容、标签、
    证据链（支持该结论的其他记忆 ID）、嵌入向量、访问统计、
    置信度、记忆类型、抽象层级，以及 SafeSieve-lite 模板填充统计。
    """

    # ---- 基础标识 ----
    memory_id: str = field(default_factory=lambda: str(uuid.uuid4()))  # 记忆唯一 ID
    source_agent: str = ""           # 创建该记忆的 Agent ID（planner/retriever/executor/summarizer）
    created_at: float = field(default_factory=time.time)  # 创建时间戳
    task_topic: str = ""             # 所属任务主题
    task_id: str = ""                # 所属任务 ID

    # ---- 内容 ----
    summary: str = ""                # 记忆摘要（用于快速浏览和搜索匹配）
    content: str = ""                # 完整内容（详细结果/报告/计划）
    tags: List[str] = field(default_factory=list)          # 标签列表（用于关键词/标签搜索）
    evidence_chain: List[str] = field(default_factory=list)  # 支持性证据的记忆 ID 链

    # ---- 向量 ----
    embedding: Optional[List[float]] = None  # 384 维语义嵌入向量（用于 FAISS 相似度搜索）

    # ---- 统计 ----
    access_count: int = 0            # 被访问次数（SQL 层自动递增）
    last_accessed: float = 0.0       # 最后访问时间戳
    confidence: float = 1.0          # 置信度 0-1（当前始终为 1.0，预留质量评估接口）

    # ---- 分类 ----
    memory_type: str = "result"      # 记忆类型：result / evidence / strategy / fact / error
    abstraction_level: int = 0       # 抽象层级：0=具体实例, 1=领域模板, 2=通用原则

    # ---- SafeSieve-lite 统计 ----
    fill_attempts: int = 0           # 模板填充尝试次数（仅 strategy 类型有意义）
    fill_successes: int = 0          # 模板填充成功次数

    # ---- 序列化 ----

    def to_dict(self) -> Dict[str, Any]:
        """转为字典（不含嵌入向量，用于 SQLite 存储）。"""
        return {
            "memory_id": self.memory_id,
            "source_agent": self.source_agent,
            "created_at": self.created_at,
            "task_topic": self.task_topic,
            "task_id": self.task_id,
            "summary": self.summary,
            "content": self.content,
            "tags": self.tags,
            "evidence_chain": self.evidence_chain,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "confidence": self.confidence,
            "memory_type": self.memory_type,
            "abstraction_level": self.abstraction_level,
            "fill_attempts": self.fill_attempts,
            "fill_successes": self.fill_successes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryUnit":
        """从字典恢复 MemoryUnit（不含嵌入向量）。"""
        return cls(
            memory_id=data.get("memory_id", str(uuid.uuid4())),
            source_agent=data.get("source_agent", ""),
            created_at=data.get("created_at", time.time()),
            task_topic=data.get("task_topic", ""),
            task_id=data.get("task_id", ""),
            summary=data.get("summary", ""),
            content=data.get("content", ""),
            tags=data.get("tags", []),
            evidence_chain=data.get("evidence_chain", []),
            access_count=data.get("access_count", 0),
            last_accessed=data.get("last_accessed", 0.0),
            confidence=data.get("confidence", 1.0),
            memory_type=data.get("memory_type", "result"),
            abstraction_level=data.get("abstraction_level", 0),
            fill_attempts=data.get("fill_attempts", 0),
            fill_successes=data.get("fill_successes", 0),
        )

    def to_search_text(self) -> str:
        """合并所有可搜索字段，用于全文关键词匹配的 relevance re-rank。"""
        return f"{self.task_topic} {self.summary} {' '.join(self.tags)} {self.content}"
