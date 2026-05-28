"""State exchange module for direct embedding transfer between agents.

Allows agents to pass semantic state vectors directly to each other,
bypassing the "internal state → text → parsing → internal state" cycle.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .embeddings import EmbeddingEngine


@dataclass
class StatePacket:
    """A non-text state transfer packet.

    Contains the embedding vector plus metadata about its generation
    and intended use. This is the unit of non-text state exchange.
    """
    packet_id: str = ""
    source_agent: str = ""
    target_agent: str = ""
    timestamp: float = field(default_factory=time.time)
    embedding: List[float] = field(default_factory=list)
    dimension: int = 0
    generation_method: str = "sentence_transformer"
    source_context: str = ""  # description of what this state represents
    data_size_bytes: int = 0
    generation_time_ms: float = 0.0

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
        t0 = time.time()
        embedding = engine.encode(text)
        gen_time = (time.time() - t0) * 1000

        return cls(
            packet_id=packet_id or f"sp_{int(time.time()*1000)}",
            source_agent=source_agent,
            target_agent=target_agent,
            embedding=embedding,
            dimension=len(embedding),
            source_context=context or f"Encoded from: {text[:100]}...",
            data_size_bytes=len(embedding) * 4,  # float32
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
        t0 = time.time()
        embedding = engine.encode_state(state)
        gen_time = (time.time() - t0) * 1000

        return cls(
            packet_id=packet_id or f"sp_{int(time.time()*1000)}",
            source_agent=source_agent,
            target_agent=target_agent,
            embedding=embedding,
            dimension=len(embedding),
            generation_method="state_encoder",
            source_context=context or f"Encoded agent state with {len(state)} keys",
            data_size_bytes=len(embedding) * 4,
            generation_time_ms=gen_time,
        )

    def cosine_similarity(self, other: "StatePacket") -> float:
        """Compute similarity between two state packets."""
        a = np.array(self.embedding, dtype=np.float64)
        b = np.array(other.embedding, dtype=np.float64)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

    def to_dict(self) -> Dict[str, Any]:
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


class StateExchangeBus:
    """Manages non-text state packet exchange between agents."""

    def __init__(self, engine: EmbeddingEngine):
        self.engine = engine
        self._transfer_history: List[StatePacket] = []
        self._stats = {
            "total_transfers": 0,
            "total_data_bytes": 0,
            "total_generation_ms": 0.0,
            "total_text_saved_bytes": 0,
        }

    def transfer(
        self,
        text_or_state: Any,
        source_agent: str,
        target_agent: str,
        context: str = "",
    ) -> StatePacket:
        """Transfer state from one agent to another as embedding.

        Args:
            text_or_state: Either a text string or a state dict to encode
            source_agent: ID of sending agent
            target_agent: ID of receiving agent
            context: Human-readable description of what's being transferred
        """
        if isinstance(text_or_state, dict):
            packet = StatePacket.from_state_dict(
                text_or_state, source_agent, target_agent, self.engine, context=context
            )
        else:
            packet = StatePacket.from_text(
                str(text_or_state), source_agent, target_agent, self.engine, context=context
            )

        self._transfer_history.append(packet)
        self._stats["total_transfers"] += 1
        self._stats["total_data_bytes"] += packet.data_size_bytes
        self._stats["total_generation_ms"] += packet.generation_time_ms

        # Estimate text savings: compare embedding size vs text serialization
        if isinstance(text_or_state, str):
            text_bytes = len(text_or_state.encode("utf-8"))
        else:
            text_bytes = len(str(text_or_state).encode("utf-8"))
        self._stats["total_text_saved_bytes"] += max(0, text_bytes - packet.data_size_bytes)

        return packet

    def receive_and_compare(
        self, packet: StatePacket, reference_text: str
    ) -> Dict[str, Any]:
        """Simulate receiving a state packet and comparing with reference."""
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

    def get_stats(self) -> Dict[str, Any]:
        """Get state exchange statistics."""
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
        self._transfer_history.clear()
        self._stats = {
            "total_transfers": 0,
            "total_data_bytes": 0,
            "total_generation_ms": 0.0,
            "total_text_saved_bytes": 0,
        }
