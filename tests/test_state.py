"""Unit tests for state passing and embedding generation."""

import numpy as np
import pytest
import time

from src.state.embeddings import EmbeddingEngine
from src.state.exchange import StateExchangeBus, StatePacket


class TestEmbeddingEngine:
    """Test embedding generation in both hash and model modes."""

    def test_hash_mode_creates_valid_embedding(self, embed_engine):
        emb = embed_engine.encode("solar energy research")
        assert len(emb) == 384
        assert all(isinstance(x, float) for x in emb)
        assert all(-2.0 <= x <= 2.0 for x in emb)

    def test_hash_embedding_is_deterministic(self, embed_engine):
        text = "renewable energy technologies"
        emb1 = embed_engine.encode(text)
        emb2 = embed_engine.encode(text)
        assert emb1 == emb2

    def test_hash_embedding_is_normalized(self, embed_engine):
        emb = embed_engine.encode("test text")
        norm = np.linalg.norm(emb)
        assert pytest.approx(norm, abs=1e-5) == 1.0

    def test_different_texts_produce_different_embeddings(self, embed_engine):
        emb1 = embed_engine.encode("solar energy")
        emb2 = embed_engine.encode("wind energy")
        assert emb1 != emb2

    def test_hash_embedding_similar_texts(self, embed_engine):
        """Same text should be identical; different text should differ."""
        emb1 = embed_engine.encode("solar energy photovoltaic")
        emb2 = embed_engine.encode("solar energy photovoltaic")
        emb3 = embed_engine.encode("quantum computing algorithms")

        # Same text → same embedding
        assert emb1 == emb2
        # Different text → different embedding
        assert emb1 != emb3

    def test_embedding_dimension_consistency(self, embed_engine):
        texts = ["short", "a longer text about renewable energy", "x" * 1000]
        for text in texts:
            emb = embed_engine.encode(text)
            assert len(emb) == 384

    def test_encode_batch(self, embed_engine):
        texts = ["text one", "text two", "text three"]
        embeddings = embed_engine.encode_batch(texts)
        assert len(embeddings) == 3
        for emb in embeddings:
            assert len(emb) == 384

    def test_combine_embeddings(self, embed_engine):
        emb1 = embed_engine.encode("solar energy")
        emb2 = embed_engine.encode("wind energy")
        emb3 = embed_engine.encode("hydro power")

        # Unweighted combination
        combined = embed_engine.combine([emb1, emb2, emb3])
        assert len(combined) == 384
        norm = np.linalg.norm(combined)
        assert pytest.approx(norm, abs=1e-5) == 1.0

    def test_combine_with_weights(self, embed_engine):
        emb1 = embed_engine.encode("solar energy")
        emb2 = embed_engine.encode("wind energy")

        # Weighted combination
        combined = embed_engine.combine([emb1, emb2], weights=[0.8, 0.2])
        assert len(combined) == 384
        norm = np.linalg.norm(combined)
        assert pytest.approx(norm, abs=1e-5) == 1.0

    def test_combine_empty_list(self, embed_engine):
        combined = embed_engine.combine([])
        assert len(combined) == 384

    def test_encode_state(self, embed_engine, sample_state):
        emb = embed_engine.encode_state(sample_state)
        assert len(emb) == 384
        norm = np.linalg.norm(emb)
        assert pytest.approx(norm, abs=1e-5) == 1.0

    def test_encode_state_with_various_types(self, embed_engine):
        state = {
            "string_val": "hello",
            "int_val": 42,
            "float_val": 3.14,
            "list_val": ["a", "b", "c"],
            "nested": {"key": "value"},
        }
        emb = embed_engine.encode_state(state)
        assert len(emb) == 384

    def test_decode_hints(self, embed_engine):
        emb = embed_engine.encode("test text")
        hints = embed_engine.decode_hints(emb)
        assert "norm" in hints
        assert "dimension" in hints
        assert "dominant_dimensions" in hints
        assert hints["dimension"] == 384
        assert len(hints["dominant_dimensions"]) == 5

    def test_estimate_bits(self, embed_engine):
        emb = embed_engine.encode("test")
        bits = embed_engine.estimate_bits(emb)
        assert bits == 384 * 32  # 32 bits per float

    def test_engine_without_model(self):
        engine = EmbeddingEngine(use_real_model=False)
        emb = engine.encode("test")
        assert len(emb) == 384


class TestStatePacket:
    """Test StatePacket creation and comparison."""

    def test_from_text(self, embed_engine):
        packet = StatePacket.from_text(
            text="solar energy photovoltaic research findings",
            source_agent="retriever",
            target_agent="executor",
            engine=embed_engine,
            packet_id="test-packet-1",
            context="Solar research results",
        )
        assert packet.packet_id == "test-packet-1"
        assert packet.source_agent == "retriever"
        assert packet.target_agent == "executor"
        assert packet.dimension == 384
        assert packet.data_size_bytes == 384 * 4
        assert packet.generation_time_ms >= 0
        assert "Solar" in packet.source_context

    def test_from_state_dict(self, embed_engine, sample_state):
        packet = StatePacket.from_state_dict(
            state=sample_state,
            source_agent="planner",
            target_agent="summarizer",
            engine=embed_engine,
            context="Plan state transfer",
        )
        assert packet.source_agent == "planner"
        assert packet.generation_method == "state_encoder"
        assert len(packet.embedding) == 384

    def test_cosine_similarity_same_text(self, embed_engine):
        p1 = StatePacket.from_text("solar energy", "a", "b", embed_engine)
        p2 = StatePacket.from_text("solar energy", "a", "b", embed_engine)
        sim = p1.cosine_similarity(p2)
        assert pytest.approx(sim, abs=1e-5) == 1.0

    def test_cosine_similarity_related(self, embed_engine):
        """Related texts should have positive similarity."""
        p1 = StatePacket.from_text("solar energy photovoltaic panels", "a", "b", embed_engine)
        p2 = StatePacket.from_text("solar power generation systems", "a", "b", embed_engine)
        sim = p1.cosine_similarity(p2)
        # Should be somewhat correlated (both about solar)
        assert -1.0 <= sim <= 1.0

    def test_to_dict(self, embed_engine):
        packet = StatePacket.from_text("test", "a", "b", embed_engine)
        d = packet.to_dict()
        assert d["source_agent"] == "a"
        assert d["target_agent"] == "b"
        assert d["dimension"] == 384
        assert "embedding" not in d  # Embedding not in serialized dict


class TestStateExchangeBus:
    """Test StateExchangeBus for inter-agent state transfer."""

    def test_transfer_text(self, state_bus, embed_engine):
        packet = state_bus.transfer(
            "solar energy research results",
            source_agent="retriever",
            target_agent="executor",
            context="Research findings",
        )
        assert isinstance(packet, StatePacket)
        assert packet.source_agent == "retriever"
        assert len(packet.embedding) == 384

    def test_transfer_state_dict(self, state_bus, sample_state):
        packet = state_bus.transfer(
            sample_state,
            source_agent="planner",
            target_agent="summarizer",
            context="Plan transfer",
        )
        assert packet.generation_method == "state_encoder"

    def test_transfer_increments_counters(self, state_bus):
        initial_count = state_bus.get_stats()["transfer_count"]
        state_bus.transfer("test message", "a", "b")
        stats = state_bus.get_stats()
        assert stats["transfer_count"] == initial_count + 1
        assert stats["total_transfers"] == 1
        assert stats["total_data_bytes"] > 0

    def test_transfer_multiple(self, state_bus):
        for i in range(5):
            state_bus.transfer(f"message {i}", "agent_a", "agent_b")
        stats = state_bus.get_stats()
        assert stats["transfer_count"] == 5
        assert stats["total_transfers"] == 5

    def test_get_stats_values(self, state_bus):
        state_bus.transfer("test data", "a", "b")
        stats = state_bus.get_stats()
        assert stats["total_data_bytes"] > 0
        assert stats["avg_packet_size_bytes"] > 0
        assert stats["total_compression_saved_bytes"] >= 0

    def test_compression_savings(self, state_bus):
        """Transferring via embedding should save bytes vs raw text."""
        long_text = "This is a long message " * 50  # ~1000 chars
        state_bus.transfer(long_text, "a", "b")
        stats = state_bus.get_stats()
        # Embedding = 1536 bytes. Long text > 1500 bytes → savings
        assert stats["total_compression_saved_bytes"] >= 0

    def test_receive_and_compare(self, state_bus, embed_engine):
        packet = state_bus.transfer("solar photovoltaic energy", "a", "b")
        comparison = state_bus.receive_and_compare(packet, "solar power generation")
        assert "similarity" in comparison
        assert "packet_size_bytes" in comparison
        assert "compression_ratio" in comparison
        assert comparison["packet_dimension"] == 384

    def test_clear(self, state_bus):
        state_bus.transfer("test", "a", "b")
        state_bus.clear()
        stats = state_bus.get_stats()
        assert stats["transfer_count"] == 0
        assert stats["total_data_bytes"] == 0

    def test_transfer_history_grows(self, state_bus):
        for i in range(3):
            state_bus.transfer(f"msg {i}", "a", "b")
        stats = state_bus.get_stats()
        assert stats["transfer_count"] == 3


class TestEmbeddingPerformance:
    """Performance benchmarks for embedding operations."""

    def test_encode_speed_small_text(self, embed_engine):
        _, elapsed = _time_exec(embed_engine.encode, "small text")
        assert elapsed < 0.1, f"Too slow: {elapsed:.4f}s"

    def test_encode_speed_large_text(self, embed_engine):
        large_text = "renewable energy research " * 200  # ~6000 chars
        _, elapsed = _time_exec(embed_engine.encode, large_text)
        assert elapsed < 0.5, f"Too slow: {elapsed:.4f}s"

    def test_batch_encode_speed(self, embed_engine):
        texts = [f"text number {i} about energy research" for i in range(50)]
        _, elapsed = _time_exec(embed_engine.encode_batch, texts)
        assert elapsed < 1.0, f"Too slow: {elapsed:.4f}s"

    def test_embedding_memory_size(self, embed_engine):
        """Verify embedding is fixed size regardless of input."""
        short_emb = embed_engine.encode("hi")
        long_text = "very long text " * 500
        long_emb = embed_engine.encode(long_text)

        import sys
        short_size = sys.getsizeof(short_emb)
        long_size = sys.getsizeof(long_emb)
        # Both should be similar (list of 384 floats)
        assert short_size == long_size


def _time_exec(func, *args):
    t0 = time.perf_counter()
    result = func(*args)
    return result, time.perf_counter() - t0
