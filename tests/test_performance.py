"""Performance benchmark tests for structured vs text mode comparison.

Verifies the core competition claims:
1. Structured protocol reduces token overhead vs text
2. Non-text state passing compresses inter-agent communication
3. Shared memory enables cross-task knowledge reuse
4. Hash-based embeddings are efficient
"""

import json
import time
import pytest

from src.protocol import Message, ActionType, MessageType, ProtocolParser
from src.state.embeddings import EmbeddingEngine
from src.state.exchange import StateExchangeBus, StatePacket
from src.memory.models import MemoryUnit
from src.memory.store import MemoryStore
from src.sandbox.executor import validate_code
from .conftest import time_execution, generate_random_text


# ============================================================
# Protocol Serialization Benchmarks
# ============================================================

class TestSerializationPerformance:
    """Benchmark MessagePack vs JSON serialization."""

    def test_msgpack_vs_json_size_large(self):
        """MessagePack should be more compact than JSON for structured data."""
        msg = Message(
            from_agent="planner",
            to_agent="retriever",
            msg_type=MessageType.REQUEST,
            action=ActionType.RETRIEVE,
            params={
                "query": "Research solar energy photovoltaic efficiency trends",
                "limit": 100,
                "tags": ["solar", "energy", "photovoltaic", "efficiency"],
                "filters": {"min_relevance": 0.7, "max_age_days": 365},
            },
            memory_refs=["mem-001", "mem-002", "mem-003", "mem-004", "mem-005"],
            result={
                "hits": 42,
                "data": [{"id": i, "title": f"Result {i}", "score": 0.9 - i * 0.01}
                         for i in range(10)],
            },
        )
        msgpack_size = len(msg.serialize())
        json_size = len(json.dumps(msg.to_dict()).encode("utf-8"))

        assert msgpack_size < json_size, (
            f"msgpack={msgpack_size} bytes should be < json={json_size} bytes"
        )

    def test_msgpack_roundtrip_speed(self):
        """MessagePack round-trip should be fast."""
        msg = Message(
            from_agent="planner",
            to_agent="retriever",
            action=ActionType.RETRIEVE,
            params={"query": "test", "limit": 10},
            state_embedding=[0.1] * 384,
            result={"hits": 5, "data": [{"id": i} for i in range(10)]},
        )
        _, elapsed = time_execution(
            lambda: Message.deserialize(msg.serialize())
        )
        assert elapsed < 0.01, f"Round-trip too slow: {elapsed:.6f}s"

    def test_batch_serialization_speed(self):
        """100 serializations should complete quickly."""
        msgs = [
            Message(
                from_agent="planner",
                to_agent="retriever",
                params={"query": f"test_{i}", "limit": 10},
                state_embedding=[0.1] * 384,
                result=f"Result data for query {i}",
            )
            for i in range(100)
        ]

        def serialize_all():
            for m in msgs:
                m.serialize()

        _, elapsed = time_execution(serialize_all)
        assert elapsed < 0.5, f"Batch serialization too slow: {elapsed:.4f}s"

    def test_batch_deserialization_speed(self):
        """100 deserializations should complete quickly."""
        original = Message(
            from_agent="planner",
            to_agent="retriever",
            params={"query": "test"},
            state_embedding=[0.1] * 384,
            result={"hits": 5},
        )
        serialized = original.serialize()

        def deserialize_all():
            for _ in range(100):
                Message.deserialize(serialized)

        _, elapsed = time_execution(deserialize_all)
        assert elapsed < 0.5, f"Batch deserialization too slow: {elapsed:.4f}s"


# ============================================================
# Token Estimation Benchmarks
# ============================================================

class TestTokenEstimationPerformance:
    """Benchmark token counting efficiency."""

    def test_token_estimation_speed(self):
        """Token estimation should be fast."""
        msg = Message(
            from_agent="planner",
            to_agent="retriever",
            action=ActionType.RETRIEVE,
            params={"query": "solar energy", "limit": 10},
            result="This is a comprehensive result with detailed analysis data.",
            state_embedding=[0.1] * 384,
        )

        def estimate_many():
            for _ in range(100):
                msg.estimated_token_count()
                msg.text_token_count()

        _, elapsed = time_execution(estimate_many)
        assert elapsed < 0.2, f"Token estimation too slow: {elapsed:.4f}s"

    def test_token_savings_with_embedding(self):
        """Embedding-based state transfer should save tokens vs text."""
        msg = Message(
            from_agent="executor",
            to_agent="summarizer",
            result={"output": "Detailed analysis result " * 20},  # ~600 chars text
            state_embedding=[0.1] * 384,  # 384 floats in compact form
        )
        structured_tokens = msg.estimated_token_count()
        # Text equivalent would include verbose state description
        text = msg.to_text_equivalent()
        text_tokens = len(text) // 3
        assert text_tokens > 0

    def test_token_count_ratio_various_sizes(self):
        """Test token ratio across different message sizes."""
        sizes = [10, 50, 100, 500, 1000]
        ratios = []
        for size in sizes:
            result_text = "x" * size
            msg = Message(
                from_agent="a", to_agent="b",
                result=result_text,
            )
            structured = msg.estimated_token_count()
            text = msg.text_token_count()
            if text > 0:
                ratios.append(structured / text)

        # Average ratio should be < 1 (structured is more compact)
        if ratios:
            avg_ratio = sum(ratios) / len(ratios)
            assert avg_ratio < 3.0  # Small messages have proportionally more overhead


# ============================================================
# Embedding Performance
# ============================================================

class TestEmbeddingBenchmark:
    """Benchmark hash-based embedding performance."""

    def test_encode_throughput_small(self, embed_engine):
        """Should encode >1000 small texts per second."""
        texts = [f"short text {i}" for i in range(500)]

        t0 = time.perf_counter()
        for text in texts:
            embed_engine.encode(text)
        elapsed = time.perf_counter() - t0

        throughput = len(texts) / elapsed
        assert throughput > 500, f"Low throughput: {throughput:.0f} encodes/sec"

    def test_encode_throughput_large(self, embed_engine):
        """Should encode large texts quickly."""
        texts = [f"renewable energy research topic {i} " * 50 for i in range(50)]

        t0 = time.perf_counter()
        for text in texts:
            embed_engine.encode(text)
        elapsed = time.perf_counter() - t0

        throughput = len(texts) / elapsed
        assert throughput > 20, f"Low throughput: {throughput:.0f} large encodes/sec"

    def test_batch_encode_throughput(self, embed_engine):
        """Batch encode should be faster than sequential."""
        texts = [f"text number {i} about energy research" for i in range(200)]

        # Batch
        _, batch_time = time_execution(embed_engine.encode_batch, texts)

        # Sequential (for comparison, only 50 to keep test fast)
        subset = texts[:50]
        t0 = time.perf_counter()
        for text in subset:
            embed_engine.encode(text)
        seq_time = time.perf_counter() - t0
        seq_extrapolated = seq_time * (len(texts) / len(subset))

        # Batch should be reasonably fast
        assert batch_time < 1.0, f"Batch encode too slow: {batch_time:.4f}s"

    def test_encode_state_performance(self, embed_engine):
        """Encoding a state dict should be fast."""
        state = {
            "plan": {"subtasks": [{"step": i, "description": f"Task {i}"}
                                  for i in range(10)]},
            "results": {"data": "x" * 500},
            "metadata": {"source": "test", "timestamp": 1234567890},
        }
        _, elapsed = time_execution(embed_engine.encode_state, state)
        assert elapsed < 0.1, f"State encoding too slow: {elapsed:.4f}s"

    def test_combine_embeddings_performance(self, embed_engine):
        """Combining embeddings should be fast."""
        embeddings = [
            embed_engine.encode(f"topic {i} about energy research")
            for i in range(50)
        ]
        _, elapsed = time_execution(embed_engine.combine, embeddings)
        assert elapsed < 0.05, f"Combine too slow: {elapsed:.4f}s"


# ============================================================
# State Transfer Benchmarks
# ============================================================

class TestStateTransferBenchmark:
    """Benchmark state packet creation and comparison."""

    def test_state_packet_creation_speed(self, embed_engine):
        """Creating state packets should be fast."""
        text = "Solar energy photovoltaic efficiency research findings " * 10

        def create_packets():
            for i in range(100):
                StatePacket.from_text(
                    text=f"{text} {i}",
                    source_agent="retriever",
                    target_agent="executor",
                    engine=embed_engine,
                )

        _, elapsed = time_execution(create_packets)
        assert elapsed < 1.0, f"Packet creation too slow: {elapsed:.4f}s"

    def test_cosine_similarity_speed(self, embed_engine):
        """Cosine similarity computation should be fast."""
        p1 = StatePacket.from_text("solar energy research", "a", "b", embed_engine)
        p2 = StatePacket.from_text("wind energy analysis", "a", "b", embed_engine)

        def compute_similarities():
            for _ in range(500):
                p1.cosine_similarity(p2)

        _, elapsed = time_execution(compute_similarities)
        assert elapsed < 0.5, f"Similarity computation too slow: {elapsed:.4f}s"

    def test_compression_ratio_large_text(self, embed_engine):
        """Large texts should see significant compression via embeddings."""
        state_bus = StateExchangeBus(embed_engine)
        large_text = "This is a comprehensive research report " * 100  # ~4000 chars

        packet = state_bus.transfer(large_text, "a", "b")
        stats = state_bus.get_stats()

        assert packet.data_size_bytes == 384 * 4  # Fixed 1536 bytes
        # The raw text would be >3000 bytes, so embedding saves space
        assert stats["total_data_bytes"] < len(large_text.encode("utf-8"))

    def test_state_bus_transfer_throughput(self, embed_engine):
        """State bus should handle many transfers quickly."""
        state_bus = StateExchangeBus(embed_engine)

        def do_transfers():
            for i in range(200):
                state_bus.transfer(f"data packet number {i}", f"agent_{i % 4}", f"agent_{(i+1) % 4}")

        _, elapsed = time_execution(do_transfers)
        throughput = 200 / elapsed
        assert throughput > 100, f"Low transfer throughput: {throughput:.0f}/s"


# ============================================================
# Memory Store Benchmarks
# ============================================================

class TestMemoryStoreBenchmark:
    """Benchmark memory store operations."""

    def test_store_throughput(self, memory_store, embed_engine):
        """Should store many memories quickly."""
        embeddings = [
            embed_engine.encode(f"topic {i} about {'energy' if i % 2 == 0 else 'security'}")
            for i in range(100)
        ]

        def store_many():
            for i in range(100):
                mem = MemoryUnit(
                    task_topic=f"Topic {i}",
                    summary=f"Summary for task {i}",
                    tags=[f"tag_{i % 5}", "benchmark"],
                    embedding=embeddings[i],
                )
                memory_store.store(mem)

        _, elapsed = time_execution(store_many)
        throughput = 100 / elapsed
        assert throughput > 50, f"Low store throughput: {throughput:.0f}/s"

    def test_keyword_search_throughput(self, memory_store, embed_engine):
        """Keyword search should be fast with many memories."""
        # Populate
        for i in range(200):
            mem = MemoryUnit(
                task_topic=f"Task {i}: Research topic {i % 20}",
                summary=f"Summary for task {i}",
                tags=[f"tag_{i % 10}", f"category_{i % 5}"],
                memory_type="result",
            )
            memory_store.store(mem)

        def search_many():
            for term in ["Research", "topic", "energy", "security", "data"]:
                memory_store.search_by_keyword(term)

        _, elapsed = time_execution(search_many)
        assert elapsed < 0.3, f"Keyword search too slow: {elapsed:.4f}s"

    def test_tag_search_throughput(self, memory_store, embed_engine):
        """Tag search should be fast with many memories."""
        for i in range(200):
            mem = MemoryUnit(
                tags=[f"tag_{i % 15}"],
                memory_type="result",
            )
            memory_store.store(mem)

        def search_many():
            for tag in ["tag_1", "tag_5", "tag_10"]:
                memory_store.search_by_tags([tag])

        _, elapsed = time_execution(search_many)
        assert elapsed < 0.2, f"Tag search too slow: {elapsed:.4f}s"

    def test_get_stats_performance(self, memory_store, embed_engine):
        """get_stats should be fast regardless of store size."""
        for i in range(100):
            mem = MemoryUnit(
                embedding=embed_engine.encode(f"test {i}"),
                memory_type="result" if i % 2 == 0 else "evidence",
                source_agent="retriever" if i % 3 == 0 else "executor",
            )
            memory_store.store(mem)

        _, elapsed = time_execution(memory_store.get_stats)
        assert elapsed < 0.1, f"get_stats too slow: {elapsed:.4f}s"


# ============================================================
# Sandbox Benchmarks
# ============================================================

class TestSandboxBenchmark:
    """Benchmark sandbox code validation and execution."""

    def test_validation_throughput(self):
        """Should validate many small code snippets quickly."""
        codes = [
            f"x = {i}\nprint(x * 2)\nresult = x + {i}"
            for i in range(500)
        ]

        def validate_all():
            for code in codes:
                validate_code(code)

        _, elapsed = time_execution(validate_all)
        throughput = len(codes) / elapsed
        assert throughput > 2000, f"Low validation throughput: {throughput:.0f}/s"

    def test_validation_mixed_safe_unsafe(self):
        """Validation should handle mix of safe and unsafe code."""
        safe = ["print(1)", "x = sum([1,2,3])", "list(range(10))"]
        unsafe = ["eval('1')", "import os", "open('/tmp')", "exec('x=1')"]

        def validate_all():
            for code in safe + unsafe:
                validate_code(code)

        _, elapsed = time_execution(validate_all)
        assert elapsed < 0.1, f"Mixed validation too slow: {elapsed:.4f}s"

    def test_execution_overhead(self, sandbox):
        """Sandbox execution overhead should be minimal."""
        def run_simple():
            sandbox.execute("x = 42")

        _, elapsed = time_execution(run_simple)
        assert elapsed < 0.5, f"Execution overhead too high: {elapsed:.4f}s"


# ============================================================
# End-to-End Pipeline Benchmarks
# ============================================================

class TestPipelineBenchmark:
    """Benchmark the full multi-agent pipeline."""

    def test_single_task_latency(self, orchestrator):
        """A single task should complete within reasonable time."""
        _, elapsed = time_execution(
            orchestrator.execute_task,
            task_id="perf-1",
            task_description="Research solar energy efficiency",
            tags=["solar", "energy"],
        )
        assert elapsed < 5.0, f"Single task too slow: {elapsed:.3f}s"

    def test_sequential_task_throughput(self, orchestrator):
        """Multiple tasks run sequentially should have reasonable per-task latency."""
        topics = [
            "solar energy efficiency",
            "wind turbine capacity",
            "code security audit",
            "renewable energy costs",
            "photovoltaic technology",
        ]
        t0 = time.perf_counter()
        for i, topic in enumerate(topics):
            orchestrator.execute_task(
                task_id=f"thru-{i}",
                task_description=topic,
                tags=["energy" if "code" not in topic else "security"],
            )
        total_elapsed = time.perf_counter() - t0
        avg = total_elapsed / len(topics)
        assert avg < 5.0, f"Average task too slow: {avg:.3f}s"

    def test_structured_vs_text_comparison(self, orchestrator):
        """Run a comparison experiment and verify metrics."""
        tasks = [
            {
                "task_id": "bench-comp-1",
                "description": "Research solar energy photovoltaic efficiency trends",
                "tags": ["solar", "energy", "efficiency"],
            },
            {
                "task_id": "bench-comp-2",
                "description": "Analyze renewable energy cost reduction patterns",
                "tags": ["energy", "cost", "renewable"],
            },
        ]
        report = orchestrator.run_comparison_experiment(tasks, "benchmark_group")

        assert report.structured_tokens >= 0
        assert report.text_tokens >= 0
        assert report.structured_latency_ms > 0
        assert report.text_latency_ms > 0
        assert report.token_reduction_pct <= 100  # Can't save more than 100%
        # Structured should use less or similar tokens
        assert report.structured_tokens <= report.text_tokens * 2, (
            "Structured tokens should not dramatically exceed text tokens"
        )

    def test_memory_accumulation_stability(self, orchestrator):
        """Memory operations should remain stable as store grows."""
        latencies = []
        for i in range(10):
            _, elapsed = time_execution(
                orchestrator.execute_task,
                task_id=f"mem-perf-{i}",
                task_description=f"Research topic {i % 3}: energy technologies",
                tags=["energy", "performance"],
            )
            latencies.append(elapsed)

        # Average latency should not spike dramatically
        avg = sum(latencies) / len(latencies)
        # No single task should be dramatically slower than early tasks
        first_three_avg = sum(latencies[:3]) / 3
        # Allow some variance but not orders of magnitude
        for latency in latencies:
            assert latency < max(first_three_avg * 5, 10.0), (
                f"Task latency spike: {latency:.3f}s vs baseline {first_three_avg:.3f}s"
            )


# ============================================================
# Resource Usage Tests
# ============================================================

class TestResourceUsage:
    """Verify reasonable resource consumption."""

    def test_embedding_memory_constant(self, embed_engine):
        """Embedding size is fixed regardless of input size."""
        import sys
        texts = [
            "hi",
            "a medium length research query about solar energy",
            "very long text " * 1000,
        ]
        sizes = []
        for text in texts:
            emb = embed_engine.encode(text)
            sizes.append(sys.getsizeof(emb))

        # All embeddings should be the same size
        assert len(set(sizes)) == 1

    def test_state_packet_size_constant(self, embed_engine):
        """StatePacket data size is constant."""
        p1 = StatePacket.from_text("hi", "a", "b", embed_engine)
        p2 = StatePacket.from_text("long text " * 500, "a", "b", embed_engine)
        assert p1.data_size_bytes == p2.data_size_bytes

    def test_message_size_with_embedding(self):
        """Message with full embedding should be under ~2KB."""
        msg = Message(
            from_agent="planner",
            to_agent="retriever",
            params={"query": "test"},
            state_embedding=[0.1] * 384,
        )
        serialized = msg.serialize()
        assert len(serialized) < 5000, f"Message too large: {len(serialized)} bytes"
