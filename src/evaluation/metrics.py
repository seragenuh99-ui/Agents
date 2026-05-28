"""Performance metrics collection for multi-agent collaboration.

Tracks: message count, token/char overhead, non-text state transfers,
task latency, memory hit rate, and overall performance improvement.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TaskMetrics:
    """Metrics for a single task execution."""
    task_id: str = ""
    task_description: str = ""
    mode: str = "structured"  # "structured" or "text"
    start_time: float = 0.0
    end_time: float = 0.0

    # Communication metrics
    messages_sent: int = 0
    messages_received: int = 0
    structured_token_count: int = 0
    text_equivalent_token_count: int = 0
    communication_char_count: int = 0

    # State transfer metrics
    state_transfers: int = 0
    state_data_bytes: int = 0
    state_generation_ms: float = 0.0

    # Memory metrics
    memory_queries: int = 0
    memory_hits: int = 0
    memory_hit_rate: float = 0.0
    memories_stored: int = 0
    cross_task_memories_used: int = 0

    # Timing
    total_elapsed_ms: float = 0.0
    llm_call_count: int = 0
    llm_total_ms: float = 0.0

    # Real LLM API usage (from API response)
    llm_prompt_tokens: int = 0
    llm_completion_tokens: int = 0
    llm_cached_tokens: int = 0

    @property
    def llm_cache_hit_pct(self) -> float:
        if self.llm_prompt_tokens == 0:
            return 0.0
        return self.llm_cached_tokens / self.llm_prompt_tokens * 100

    @property
    def token_savings_pct(self) -> float:
        if self.text_equivalent_token_count == 0:
            return 0.0
        return (1 - self.structured_token_count / self.text_equivalent_token_count) * 100

    @property
    def avg_message_latency_ms(self) -> float:
        if self.messages_sent == 0:
            return 0.0
        return self.total_elapsed_ms / self.messages_sent


@dataclass
class ComparisonReport:
    """Comparison between structured protocol mode and text mode."""
    task_group: str = ""

    # Structured mode
    structured_messages: int = 0
    structured_tokens: int = 0
    structured_latency_ms: float = 0.0
    structured_state_bytes: int = 0

    # Text mode
    text_messages: int = 0
    text_tokens: int = 0
    text_latency_ms: float = 0.0
    text_state_bytes: int = 0

    @property
    def token_reduction_pct(self) -> float:
        if self.text_tokens == 0:
            return 0.0
        return (1 - self.structured_tokens / self.text_tokens) * 100

    @property
    def latency_reduction_pct(self) -> float:
        if self.text_latency_ms == 0:
            return 0.0
        return (1 - self.structured_latency_ms / self.text_latency_ms) * 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_group": self.task_group,
            "token_reduction_pct": round(self.token_reduction_pct, 2),
            "latency_reduction_pct": round(self.latency_reduction_pct, 2),
            "structured_mode": {
                "messages": self.structured_messages,
                "tokens": self.structured_tokens,
                "latency_ms": round(self.structured_latency_ms, 2),
                "state_bytes": self.structured_state_bytes,
            },
            "text_mode": {
                "messages": self.text_messages,
                "tokens": self.text_tokens,
                "latency_ms": round(self.text_latency_ms, 2),
                "state_bytes": self.text_state_bytes,
            },
        }


class MetricsCollector:
    """Collects and aggregates metrics during task execution."""

    def __init__(self):
        self._current_task: Optional[TaskMetrics] = None
        self._task_history: List[TaskMetrics] = []
        self._comparisons: List[ComparisonReport] = []

    def start_task(self, task_id: str, description: str, mode: str = "structured") -> None:
        self._current_task = TaskMetrics(
            task_id=task_id,
            task_description=description,
            mode=mode,
            start_time=time.time(),
        )

    def end_task(self) -> Optional[TaskMetrics]:
        if self._current_task:
            self._current_task.end_time = time.time()
            self._current_task.total_elapsed_ms = (
                self._current_task.end_time - self._current_task.start_time
            ) * 1000
            if self._current_task.memory_queries > 0:
                self._current_task.memory_hit_rate = (
                    self._current_task.memory_hits / self._current_task.memory_queries
                )
            self._task_history.append(self._current_task)
            task = self._current_task
            self._current_task = None
            return task
        return None

    def record_message(self, token_count: int, text_token_count: int, char_count: int) -> None:
        if self._current_task:
            self._current_task.messages_sent += 1
            self._current_task.structured_token_count += token_count
            self._current_task.text_equivalent_token_count += text_token_count
            self._current_task.communication_char_count += char_count

    def record_state_transfer(self, data_bytes: int, generation_ms: float) -> None:
        if self._current_task:
            self._current_task.state_transfers += 1
            self._current_task.state_data_bytes += data_bytes
            self._current_task.state_generation_ms += generation_ms

    def record_memory_query(self, hits: int, cross_task: int = 0) -> None:
        if self._current_task:
            self._current_task.memory_queries += 1
            self._current_task.memory_hits += hits
            self._current_task.cross_task_memories_used += cross_task

    def record_memory_store(self) -> None:
        if self._current_task:
            self._current_task.memories_stored += 1

    def record_llm_call(self, elapsed_ms: float) -> None:
        if self._current_task:
            self._current_task.llm_call_count += 1
            self._current_task.llm_total_ms += elapsed_ms

    def record_llm_usage(self, prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0) -> None:
        """Record real token usage from LLM API response."""
        if self._current_task:
            self._current_task.llm_prompt_tokens += prompt_tokens
            self._current_task.llm_completion_tokens += completion_tokens
            self._current_task.llm_cached_tokens += cached_tokens

    def add_comparison(self, report: ComparisonReport) -> None:
        self._comparisons.append(report)

    def get_aggregate_metrics(self) -> Dict[str, Any]:
        """Get aggregate metrics across all tasks."""
        if not self._task_history:
            return {}

        total_tasks = len(self._task_history)
        total_messages = sum(t.messages_sent for t in self._task_history)
        total_structured_tokens = sum(t.structured_token_count for t in self._task_history)
        total_text_tokens = sum(t.text_equivalent_token_count for t in self._task_history)
        total_latency = sum(t.total_elapsed_ms for t in self._task_history)
        total_state_transfers = sum(t.state_transfers for t in self._task_history)
        total_state_bytes = sum(t.state_data_bytes for t in self._task_history)
        total_memory_queries = sum(t.memory_queries for t in self._task_history)
        total_memory_hits = sum(t.memory_hits for t in self._task_history)

        structured_tasks = [t for t in self._task_history if t.mode == "structured"]
        text_tasks = [t for t in self._task_history if t.mode == "text"]

        def avg(lst, attr):
            vals = [getattr(t, attr) for t in lst]
            return sum(vals) / len(vals) if vals else 0

        return {
            "total_tasks": total_tasks,
            "total_messages": total_messages,
            "total_structured_tokens": total_structured_tokens,
            "total_text_equivalent_tokens": total_text_tokens,
            "overall_token_savings_pct": (
                (1 - total_structured_tokens / max(total_text_tokens, 1)) * 100
            ),
            "total_latency_ms": round(total_latency, 2),
            "avg_task_latency_ms": round(total_latency / max(total_tasks, 1), 2),
            "total_state_transfers": total_state_transfers,
            "total_state_data_bytes": total_state_bytes,
            "total_memory_queries": total_memory_queries,
            "total_memory_hits": total_memory_hits,
            "memory_hit_rate": (
                total_memory_hits / max(total_memory_queries, 1)
            ),
            "cross_task_memories_used": sum(t.cross_task_memories_used for t in self._task_history),
            "llm_usage": {
                "total_prompt_tokens": sum(t.llm_prompt_tokens for t in self._task_history),
                "total_completion_tokens": sum(t.llm_completion_tokens for t in self._task_history),
                "total_cached_tokens": sum(t.llm_cached_tokens for t in self._task_history),
                "cache_hit_pct": (
                    sum(t.llm_cached_tokens for t in self._task_history)
                    / max(sum(t.llm_prompt_tokens for t in self._task_history), 1) * 100
                ),
            },
            "mode_comparison": {
                "structured": {
                    "task_count": len(structured_tasks),
                    "avg_latency_ms": round(avg(structured_tasks, "total_elapsed_ms"), 2),
                    "avg_tokens": round(avg(structured_tasks, "structured_token_count"), 2),
                    "avg_messages": round(avg(structured_tasks, "messages_sent"), 2),
                },
                "text": {
                    "task_count": len(text_tasks),
                    "avg_latency_ms": round(avg(text_tasks, "total_elapsed_ms"), 2),
                    "avg_tokens": round(avg(text_tasks, "text_equivalent_token_count"), 2),
                    "avg_messages": round(avg(text_tasks, "messages_sent"), 2),
                },
            },
        }

    def get_task_details(self) -> List[Dict[str, Any]]:
        return [
            {
                "task_id": t.task_id,
                "description": t.task_description[:100],
                "mode": t.mode,
                "messages": t.messages_sent,
                "structured_tokens": t.structured_token_count,
                "text_tokens": t.text_equivalent_token_count,
                "token_savings_pct": round(t.token_savings_pct, 2),
                "state_transfers": t.state_transfers,
                "memory_hits": t.memory_hits,
                "memory_hit_rate": round(t.memory_hit_rate, 3),
                "cross_task_memories": t.cross_task_memories_used,
                "latency_ms": round(t.total_elapsed_ms, 2),
            }
            for t in self._task_history
        ]

    def clear(self) -> None:
        self._current_task = None
        self._task_history.clear()
        self._comparisons.clear()
