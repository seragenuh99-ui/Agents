"""状态交换总线 — Agent 间直接传递嵌入向量而非文本。

本模块实现了非文本状态传递的"传输层"：
- StatePacket: 状态数据包，包含嵌入向量 + 元信息（来源、目标、大小、耗时）
- StateExchangeBus: 状态交换总线，管理数据包的发送、接收、消费、统计

核心流程：
  upstream Agent 完成任务
    → transfer(text_or_state, source, target)
    → EmbeddingEngine 编码为 384 维向量
    → StatePacket 存入 _transfer_history

  downstream Agent 执行前
    → get_packets_for(self.agent_id)
    → 获取上游发来的 StatePacket 列表
    → 使用 packet.embedding 直接做 FAISS 搜索
    → mark_consumed(packet_ids) 清理已用数据包

相比纯文本传输的压缩收益：384 × 4 = 1536 bytes vs 原始文本数千 bytes
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .embeddings import EmbeddingEngine


# ============================================================
# StatePacket — 非文本状态数据包
# ============================================================

@dataclass
class StatePacket:
    """一次非文本状态传递的数据包。

    包含嵌入向量及其生成元信息。每个数据包由上游 Agent 创建，
    指定目标 Agent，下游 Agent 通过 StateExchangeBus 拉取。
    """
    packet_id: str = ""                         # 数据包唯一标识（sp_ + uuid4 前 12 位）
    source_agent: str = ""                      # 发送方 Agent ID
    target_agent: str = ""                      # 接收方 Agent ID
    timestamp: float = field(default_factory=time.time)  # 创建时间戳
    embedding: List[float] = field(default_factory=list)  # 384 维语义向量
    dimension: int = 0                          # 向量维度
    generation_method: str = "sentence_transformer"  # 编码方法
    source_context: str = ""                    # 状态内容的人类可读描述
    data_size_bytes: int = 0                    # 嵌入向量的字节大小（≈1536）
    generation_time_ms: float = 0.0             # 编码耗时（毫秒）

    # ---- 工厂方法 ----

    @classmethod
    def from_text(
        cls,
        text: str,
        source_agent: str,
        target_agent: str,
        engine: EmbeddingEngine,
        packet_id: str = "",
        context: str = "",
    ) -> "StatePacket":
        """从文本创建状态数据包（编码计时）。"""
        t0 = time.time()
        embedding = engine.encode(text)
        gen_time = (time.time() - t0) * 1000

        return cls(
            packet_id=packet_id or f"sp_{uuid.uuid4().hex[:12]}",
            source_agent=source_agent,
            target_agent=target_agent,
            embedding=embedding,
            dimension=len(embedding),
            source_context=context or f"Encoded from: {text[:100]}...",
            data_size_bytes=len(embedding) * 4,  # 每维 4 bytes（float32）
            generation_time_ms=gen_time,
        )

    @classmethod
    def from_state_dict(
        cls,
        state: Dict[str, Any],
        source_agent: str,
        target_agent: str,
        engine: EmbeddingEngine,
        packet_id: str = "",
        context: str = "",
    ) -> "StatePacket":
        """从 Agent 状态字典创建数据包（走 encode_state 路径）。"""
        t0 = time.time()
        embedding = engine.encode_state(state)
        gen_time = (time.time() - t0) * 1000

        return cls(
            packet_id=packet_id or f"sp_{uuid.uuid4().hex[:12]}",
            source_agent=source_agent,
            target_agent=target_agent,
            embedding=embedding,
            dimension=len(embedding),
            generation_method="state_encoder",
            source_context=context or f"Encoded agent state with {len(state)} keys",
            data_size_bytes=len(embedding) * 4,
            generation_time_ms=gen_time,
        )

    # ---- 工具方法 ----

    def cosine_similarity(self, other: "StatePacket") -> float:
        """计算两个状态数据包的余弦相似度。"""
        a = np.array(self.embedding, dtype=np.float64)
        b = np.array(other.embedding, dtype=np.float64)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

    def to_dict(self) -> Dict[str, Any]:
        """转为字典（不含嵌入向量本身，用于日志/调试）。"""
        return {
            "packet_id": self.packet_id,
            "source_agent": self.source_agent,
            "target_agent": self.target_agent,
            "timestamp": self.timestamp,
            "dimension": self.dimension,
            "generation_method": self.generation_method,
            "source_context": self.source_context,
            "data_size_bytes": self.data_size_bytes,
            "generation_time_ms": self.generation_time_ms,
        }


# ============================================================
# StateExchangeBus — 状态交换总线
# ============================================================

class StateExchangeBus:
    """管理 Agent 间非文本状态数据包的发送、接收与消费。

    核心机制：
    - transfer(): 上游 Agent 编码状态 → StatePacket → 存入 _transfer_history
    - get_packets_for(target): 下游 Agent 按目标 ID 拉取数据包（最新优先）
    - mark_consumed(ids): 消费后移除，避免跨任务重复使用

    统计指标：
    - total_transfers: 累计传输次数（单调递增，用于计算增量）
    - total_data_bytes: 累计嵌入向量字节数
    - total_text_saved_bytes: 相比文本传输节省的字节数
    """

    def __init__(self, engine: EmbeddingEngine):
        self.engine = engine
        self._transfer_history: List[StatePacket] = []  # 传输历史
        self._stats = {
            "total_transfers": 0,           # 累计传输次数（单调递增）
            "total_data_bytes": 0,          # 累计嵌入字节数
            "total_generation_ms": 0.0,     # 累计编码耗时
            "total_text_saved_bytes": 0,    # 累计节省的文本字节数
        }

    # ---- 发送 ----

    def transfer(
        self,
        text_or_state: Any,
        source_agent: str,
        target_agent: str,
        context: str = "",
    ) -> StatePacket:
        """将状态编码为嵌入向量并传输给目标 Agent。

        Args:
            text_or_state: 文本字符串或状态字典（dict → encode_state，str → encode）
            source_agent: 发送方 Agent ID
            target_agent: 接收方 Agent ID
            context: 人类可读的状态描述

        Returns:
            创建的 StatePacket
        """
        # 根据类型选择编码路径
        if isinstance(text_or_state, dict):
            packet = StatePacket.from_state_dict(
                text_or_state, source_agent, target_agent, self.engine, context=context
            )
        else:
            packet = StatePacket.from_text(
                str(text_or_state), source_agent, target_agent, self.engine, context=context
            )

        # 存入历史并更新统计
        self._transfer_history.append(packet)
        self._stats["total_transfers"] += 1
        self._stats["total_data_bytes"] += packet.data_size_bytes
        self._stats["total_generation_ms"] += packet.generation_time_ms

        # 估算文本节省量：原始文本字节数 - 嵌入向量字节数
        if isinstance(text_or_state, str):
            text_bytes = len(text_or_state.encode("utf-8"))
        else:
            text_bytes = len(str(text_or_state).encode("utf-8"))
        self._stats["total_text_saved_bytes"] += max(0, text_bytes - packet.data_size_bytes)

        return packet

    # ---- 接收 ----

    def get_packets_for(
        self, target_agent: str, since: float = 0, limit: int = 0
    ) -> List[StatePacket]:
        """获取发送给指定 Agent 的状态数据包。

        Args:
            target_agent: 目标 Agent ID
            since: 仅返回此时间戳之后的数据包（0 = 全部）
            limit: 最大返回数（0 = 无限制），按时间降序（最新优先）

        Returns:
            匹配的 StatePacket 列表
        """
        matches = [
            p for p in self._transfer_history
            if p.target_agent == target_agent and p.timestamp >= since
        ]
        matches.sort(key=lambda p: p.timestamp, reverse=True)
        return matches[:limit] if limit else matches

    def mark_consumed(self, packet_ids: List[str]) -> None:
        """标记数据包为已消费，从传输历史中移除。

        确保数据包不会被跨任务重复使用。
        """
        consumed = set(packet_ids)
        self._transfer_history = [
            p for p in self._transfer_history if p.packet_id not in consumed
        ]

    # ---- 调试与分析 ----

    def receive_and_compare(
        self, packet: StatePacket, reference_text: str
    ) -> Dict[str, Any]:
        """模拟接收数据包并与参考文本比较（调试/评估用）。

        Returns:
            包含相似度、压缩比等指标的字典
        """
        ref_embedding = self.engine.encode(reference_text)
        ref_packet = StatePacket(
            embedding=ref_embedding,
            source_agent="reference",
            target_agent="evaluator",
            dimension=len(ref_embedding),
        )
        similarity = packet.cosine_similarity(ref_packet)

        return {
            "similarity": similarity,
            "packet_dimension": packet.dimension,
            "packet_size_bytes": packet.data_size_bytes,
            "reference_text_length": len(reference_text),
            "compression_ratio": (
                packet.data_size_bytes / max(len(reference_text.encode("utf-8")), 1)
            ),
        }

    # ---- 统计与清理 ----

    def get_stats(self) -> Dict[str, Any]:
        """获取状态交换统计信息。

        Returns:
            包含传输次数、平均包大小、平均耗时、压缩节省量等指标。
        """
        return {
            **self._stats,
            "transfer_count": len(self._transfer_history),
            "avg_packet_size_bytes": (
                self._stats["total_data_bytes"] / max(self._stats["total_transfers"], 1)
            ),
            "avg_generation_ms": (
                self._stats["total_generation_ms"] / max(self._stats["total_transfers"], 1)
            ),
            "total_compression_saved_bytes": self._stats["total_text_saved_bytes"],
        }

    def clear(self) -> None:
        """清空所有传输历史和统计（测试用）。"""
        self._transfer_history.clear()
        self._stats = {
            "total_transfers": 0,
            "total_data_bytes": 0,
            "total_generation_ms": 0.0,
            "total_text_saved_bytes": 0,
        }
