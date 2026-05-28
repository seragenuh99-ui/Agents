"""Memory data models for shared memory storage."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MemoryUnit:
    """A single memory unit stored in shared memory.

    Each memory records intermediate results, summaries, evidence chains,
    conclusions, or strategies from task execution.
    """
    memory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_agent: str = ""
    created_at: float = field(default_factory=time.time)
    task_topic: str = ""
    task_id: str = ""
    summary: str = ""
    content: str = ""
    tags: List[str] = field(default_factory=list)
    evidence_chain: List[str] = field(default_factory=list)  # IDs of supporting memories
    embedding: Optional[List[float]] = None
    access_count: int = 0
    last_accessed: float = 0.0
    confidence: float = 1.0  # 0-1 confidence score
    memory_type: str = "result"  # result, evidence, strategy, fact, error
    abstraction_level: int = 0  # 0=concrete instance, 1=domain template, 2=principle
    fill_attempts: int = 0  # SafeSieve-lite: template fill trials
    fill_successes: int = 0  # successful fills (good summary, no _error)

    def to_dict(self) -> Dict[str, Any]:
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
        """Combine fields for full-text search."""
        return f"{self.task_topic} {self.summary} {' '.join(self.tags)} {self.content}"
