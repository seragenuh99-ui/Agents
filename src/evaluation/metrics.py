"""性能指标收集 — 多智能体协作效率度量。

追踪维度：
- 通信：消息数、结构化/文本 token 对比、通信字符数
- 状态传递：非文本状态传输次数、数据量、生成耗时
- 记忆：查询次数、命中率、跨任务记忆复用
- LLM 用量：API 调用次数、token 消耗、缓存命中率
- 延迟：端到端耗时、平均消息延迟

核心数据结构：
- TaskMetrics：单任务指标
- ComparisonReport：结构化 vs 文本模式对比
- MetricsCollector：指标收集与聚合
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ═══════════════════════════════════════════════════════════════════════════════
# TaskMetrics — 单任务执行指标
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TaskMetrics:
    """单次任务执行的完整指标。

    分为五个维度：
    - 基础信息：task_id, description, mode, 时间戳
    - 通信指标：消息数、token 数、字符数
    - 状态传递：传输次数、数据量
    - 记忆指标：查询/命中/存储
    - LLM 用量：API token 消耗、缓存命中
    """

    task_id: str = ""
    task_description: str = ""
    mode: str = "structured"  # "structured" 或 "text"
    start_time: float = 0.0
    end_time: float = 0.0

    # ── 通信指标 ──
    messages_sent: int = 0
    messages_received: int = 0
    structured_token_count: int = 0       # 结构化协议 token 数
    text_equivalent_token_count: int = 0  # 等效文本协议 token 数
    communication_char_count: int = 0

    # ── 状态传递指标 ──
    state_transfers: int = 0
    state_data_bytes: int = 0
    state_generation_ms: float = 0.0

    # ── 记忆指标 ──
    memory_queries: int = 0
    memory_hits: int = 0
    memory_hit_rate: float = 0.0
    memories_stored: int = 0
    cross_task_memories_used: int = 0

    # ── 时间与 LLM 调用 ──
    total_elapsed_ms: float = 0.0
    llm_call_count: int = 0
    llm_total_ms: float = 0.0

    # ── 真实 LLM API 用量（来自 API 响应）──
    llm_prompt_tokens: int = 0
    llm_completion_tokens: int = 0
    llm_cached_tokens: int = 0          # DeepSeek KV-cache 命中 token 数

    # ── 计算属性 ──

    @property
    def llm_cache_hit_pct(self) -> float:
        """LLM 提示缓存命中率（DeepSeek 前缀缓存）。"""
        if self.llm_prompt_tokens == 0:
            return 0.0
        return self.llm_cached_tokens / self.llm_prompt_tokens * 100

    @property
    def token_savings_pct(self) -> float:
        """结构化协议相比文本协议的 token 节省率。"""
        if self.text_equivalent_token_count == 0:
            return 0.0
        return (1 - self.structured_token_count / self.text_equivalent_token_count) * 100

    @property
    def avg_message_latency_ms(self) -> float:
        """平均每条消息的延迟。"""
        if self.messages_sent == 0:
            return 0.0
        return self.total_elapsed_ms / self.messages_sent


# ═══════════════════════════════════════════════════════════════════════════════
# ComparisonReport — 模式对比报告
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ComparisonReport:
    """结构化协议模式 vs 文本模式的对比报告。

    对比维度：消息数、token 数、延迟、状态数据量。
    计算属性给出节省百分比。
    """

    task_group: str = ""

    # 结构化模式
    structured_messages: int = 0
    structured_tokens: int = 0
    structured_latency_ms: float = 0.0
    structured_state_bytes: int = 0

    # 文本模式
    text_messages: int = 0
    text_tokens: int = 0
    text_latency_ms: float = 0.0
    text_state_bytes: int = 0

    @property
    def token_reduction_pct(self) -> float:
        """结构化协议 token 减少百分比。"""
        if self.text_tokens == 0:
            return 0.0
        return (1 - self.structured_tokens / self.text_tokens) * 100

    @property
    def latency_reduction_pct(self) -> float:
        """结构化协议延迟减少百分比。"""
        if self.text_latency_ms == 0:
            return 0.0
        return (1 - self.structured_latency_ms / self.text_latency_ms) * 100

    def to_dict(self) -> Dict[str, Any]:
        """导出为字典，用于 JSON 序列化。"""
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


# ═══════════════════════════════════════════════════════════════════════════════
# MetricsCollector — 指标收集器
# ═══════════════════════════════════════════════════════════════════════════════

class MetricsCollector:
    """任务执行期间的指标收集与聚合。

    生命周期：
    1. start_task() → 开始记录单任务指标
    2. record_*() → 执行期间递增各类计数
    3. end_task() → 结算并归档到 _task_history
    4. get_aggregate_metrics() → 跨任务聚合统计
    """

    def __init__(self):
        self._current_task: Optional[TaskMetrics] = None
        self._task_history: List[TaskMetrics] = []
        self._comparisons: List[ComparisonReport] = []

    # ── 任务生命周期 ──

    def start_task(self, task_id: str, description: str, mode: str = "structured") -> None:
        """开始追踪一个新任务。"""
        self._current_task = TaskMetrics(
            task_id=task_id,
            task_description=description,
            mode=mode,
            start_time=time.time(),
        )

    def end_task(self) -> Optional[TaskMetrics]:
        """结束当前任务追踪，计算延迟和命中率，归档到历史。"""
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

    # ── 增量记录方法 ──

    def record_message(self, token_count: int, text_token_count: int, char_count: int) -> None:
        """记录一条消息的通信开销。"""
        if self._current_task:
            self._current_task.messages_sent += 1
            self._current_task.structured_token_count += token_count
            self._current_task.text_equivalent_token_count += text_token_count
            self._current_task.communication_char_count += char_count

    def record_state_transfer(self, data_bytes: int, generation_ms: float) -> None:
        """记录一次非文本状态传递。"""
        if self._current_task:
            self._current_task.state_transfers += 1
            self._current_task.state_data_bytes += data_bytes
            self._current_task.state_generation_ms += generation_ms

    def record_memory_query(self, hits: int, cross_task: int = 0) -> None:
        """记录一次记忆查询及命中数。"""
        if self._current_task:
            self._current_task.memory_queries += 1
            self._current_task.memory_hits += hits
            self._current_task.cross_task_memories_used += cross_task

    def record_memory_store(self) -> None:
        """记录一次记忆存储。"""
        if self._current_task:
            self._current_task.memories_stored += 1

    def record_llm_call(self, elapsed_ms: float) -> None:
        """记录一次 LLM 调用及其耗时。"""
        if self._current_task:
            self._current_task.llm_call_count += 1
            self._current_task.llm_total_ms += elapsed_ms

    def record_llm_usage(self, prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0) -> None:
        """记录 LLM API 返回的真实 token 用量（含缓存命中 token）。"""
        if self._current_task:
            self._current_task.llm_prompt_tokens += prompt_tokens
            self._current_task.llm_completion_tokens += completion_tokens
            self._current_task.llm_cached_tokens += cached_tokens

    def add_comparison(self, report: ComparisonReport) -> None:
        """添加一条模式对比报告。"""
        self._comparisons.append(report)

    # ── 聚合查询 ──

    def get_aggregate_metrics(self) -> Dict[str, Any]:
        """跨所有任务计算聚合指标。

        Returns:
            包含总量、均值、模式对比、LLM 用量等维度的字典。
            如无历史任务则返回空字典。
        """
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
        """获取每个任务的详细信息列表。"""
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
        """清空所有指标（用于测试复位）。"""
        self._current_task = None
        self._task_history.clear()
        self._comparisons.clear()
