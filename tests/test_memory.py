"""Unit tests for shared memory storage and retrieval."""

import json
import time
import pytest
import numpy as np

from src.memory.models import MemoryUnit
from src.memory.store import MemoryStore, HAS_FAISS


class TestMemoryUnit:
    """Test MemoryUnit data model."""

    def test_default_creation(self):
        mem = MemoryUnit()
        assert mem.memory_id
        assert len(mem.memory_id) == 36
        assert mem.source_agent == ""
        assert mem.tags == []
        assert mem.evidence_chain == []
        assert mem.access_count == 0
        assert mem.confidence == 1.0
        assert mem.memory_type == "result"

    def test_full_creation(self, sample_memory):
        assert sample_memory.source_agent == "retriever"
        assert sample_memory.task_topic == "Solar Energy Research"
        assert "solar" in sample_memory.tags
        assert sample_memory.memory_type == "result"
        assert sample_memory.embedding is not None
        assert len(sample_memory.embedding) == 384

    def test_to_dict(self, sample_memory):
        d = sample_memory.to_dict()
        assert d["memory_id"] == sample_memory.memory_id
        assert d["source_agent"] == "retriever"
        assert d["task_topic"] == "Solar Energy Research"
        assert "embedding" not in d  # Embedding excluded from dict

    def test_from_dict(self, sample_memory):
        d = sample_memory.to_dict()
        restored = MemoryUnit.from_dict(d)
        assert restored.memory_id == sample_memory.memory_id
        assert restored.source_agent == sample_memory.source_agent
        assert restored.tags == sample_memory.tags

    def test_to_search_text(self, sample_memory):
        text = sample_memory.to_search_text()
        assert "Solar Energy Research" in text
        assert "photovoltaic" in text.lower()
        assert "solar" in text

    def test_memory_types(self):
        for mtype in ["result", "evidence", "strategy", "fact", "error"]:
            mem = MemoryUnit(memory_type=mtype)
            assert mem.memory_type == mtype

    def test_evidence_chain(self):
        mem = MemoryUnit(
            evidence_chain=["mem-001", "mem-002", "mem-003"]
        )
        assert len(mem.evidence_chain) == 3
        assert "mem-001" in mem.evidence_chain

    def test_confidence_range(self):
        mem = MemoryUnit(confidence=0.85)
        assert mem.confidence == 0.85
        mem2 = MemoryUnit(confidence=0.0)
        assert mem2.confidence == 0.0


class TestMemoryStore:
    """Test MemoryStore CRUD and search operations."""

    def test_store_and_get(self, memory_store, sample_memory):
        mem_id = memory_store.store(sample_memory)
        assert mem_id == sample_memory.memory_id

        retrieved = memory_store.get(mem_id)
        assert retrieved is not None
        assert retrieved.source_agent == "retriever"
        assert retrieved.task_topic == "Solar Energy Research"
        assert "solar" in retrieved.tags

    def test_get_nonexistent(self, memory_store):
        assert memory_store.get("nonexistent-id") is None

    def test_store_updates_existing(self, memory_store, sample_memory):
        mem_id = memory_store.store(sample_memory)
        sample_memory.summary = "Updated summary"
        memory_store.store(sample_memory)

        retrieved = memory_store.get(mem_id)
        assert retrieved.summary == "Updated summary"

    def test_keyword_search(self, memory_store, sample_memory):
        memory_store.store(sample_memory)

        # Search by topic keyword
        results = memory_store.search_by_keyword("Solar")
        assert len(results) >= 1
        assert any("Solar" in r.task_topic for r in results)

        # Search by summary keyword
        results = memory_store.search_by_keyword("efficiency")
        assert len(results) >= 1
        assert any("efficiency" in r.summary.lower() for r in results)

        # Search by content keyword
        results = memory_store.search_by_keyword("monocrystalline")
        assert len(results) >= 1

        # Search by tag keyword
        results = memory_store.search_by_keyword("photovoltaic")
        assert len(results) >= 1

        # No match
        results = memory_store.search_by_keyword("zzzz_nonexistent_zzzz")
        assert len(results) == 0

    def test_keyword_search_case_insensitive(self, memory_store, sample_memory):
        memory_store.store(sample_memory)
        r1 = memory_store.search_by_keyword("SOLAR")
        r2 = memory_store.search_by_keyword("solar")
        assert len(r1) == len(r2)

    def test_tag_search(self, memory_store, sample_memory):
        memory_store.store(sample_memory)

        results = memory_store.search_by_tags(["solar"])
        assert len(results) >= 1

        results = memory_store.search_by_tags(["energy"])
        assert len(results) >= 1

        # Multiple tags - OR semantics
        results = memory_store.search_by_tags(["solar", "nonexistent"])
        assert len(results) >= 1  # Should still find solar matches

        # No match
        results = memory_store.search_by_tags(["zzzz_nonexistent_zzzz"])
        assert len(results) == 0

    def test_tag_search_multiple_memories(self, memory_store, embed_engine):
        """Store memories with different tags and verify tag search."""
        mem1 = MemoryUnit(tags=["solar", "energy"], memory_type="result",
                         embedding=embed_engine.encode("solar energy"))
        mem2 = MemoryUnit(tags=["wind", "energy"], memory_type="result",
                         embedding=embed_engine.encode("wind energy"))
        memory_store.store(mem1)
        memory_store.store(mem2)

        assert len(memory_store.search_by_tags(["solar"])) >= 1
        assert len(memory_store.search_by_tags(["wind"])) >= 1
        assert len(memory_store.search_by_tags(["energy"])) == 2

    def test_semantic_search(self, memory_store, embed_engine):
        """Test FAISS-based semantic similarity search."""
        mem1 = MemoryUnit(
            task_topic="Solar Energy",
            summary="Photovoltaic solar panels convert sunlight to electricity",
            tags=["solar", "energy"],
            embedding=embed_engine.encode("solar photovoltaic energy panels"),
        )
        mem2 = MemoryUnit(
            task_topic="Wind Energy",
            summary="Wind turbines generate electricity from kinetic wind energy",
            tags=["wind", "energy"],
            embedding=embed_engine.encode("wind turbine kinetic energy"),
        )
        mem3 = MemoryUnit(
            task_topic="Security Analysis",
            summary="SQL injection vulnerability patterns in web applications",
            tags=["security", "sql"],
            embedding=embed_engine.encode("sql injection security vulnerability"),
        )
        memory_store.store(mem1)
        memory_store.store(mem2)
        memory_store.store(mem3)

        # Search for solar-related content
        query = embed_engine.encode("solar power generation")
        results = memory_store.search_by_similarity(query, limit=3)

        if HAS_FAISS:
            assert len(results) > 0
            # Solar should be ranked higher than security for solar query
            if len(results) >= 2:
                first_topic = results[0][0].task_topic
                assert "Solar" in first_topic or "Wind" in first_topic

    def test_semantic_search_ranking(self, memory_store, embed_engine):
        """Verify semantic search ranks relevant results higher."""
        mem_solar = MemoryUnit(
            task_topic="Solar Energy Research",
            summary="Solar photovoltaic efficiency data",
            embedding=embed_engine.encode("solar photovoltaic efficiency panels energy"),
        )
        mem_security = MemoryUnit(
            task_topic="Code Security Audit",
            summary="SQL injection vulnerability detection",
            embedding=embed_engine.encode("sql injection vulnerability security code audit"),
        )
        memory_store.store(mem_solar)
        memory_store.store(mem_security)

        # Solar query
        solar_query = embed_engine.encode("solar power renewable energy photovoltaics")
        solar_results = memory_store.search_by_similarity(solar_query, limit=2)

        # Security query
        security_query = embed_engine.encode("sql injection security code vulnerability")
        security_results = memory_store.search_by_similarity(security_query, limit=2)

        if HAS_FAISS and len(solar_results) >= 2:
            # Solar query should rank solar memory higher
            assert solar_results[0][0].task_topic == "Solar Energy Research"
            # Security query should rank security memory higher
            assert security_results[0][0].task_topic == "Code Security Audit"

    def test_record_access(self, memory_store, sample_memory):
        mem_id = memory_store.store(sample_memory)
        memory_store.record_access(mem_id, "planner", "task-2")

        retrieved = memory_store.get(mem_id)
        assert retrieved.access_count == 1
        assert retrieved.last_accessed > 0

    def test_multiple_accesses(self, memory_store, sample_memory):
        mem_id = memory_store.store(sample_memory)
        for i in range(5):
            memory_store.record_access(mem_id, f"agent_{i}", f"task_{i}")

        retrieved = memory_store.get(mem_id)
        assert retrieved.access_count == 5

    def test_get_stats(self, memory_store, sample_memory):
        memory_store.store(sample_memory)
        stats = memory_store.get_stats()

        assert stats["total_memories"] == 1
        assert stats["total_accesses"] == 0
        assert "by_type" in stats
        assert "by_agent" in stats
        assert "by_topic" in stats
        assert "vector_index_size" in stats
        if HAS_FAISS:
            assert stats["vector_index_size"] == 1

    def test_stats_by_type(self, memory_store, embed_engine):
        for mtype in ["result", "evidence", "strategy", "result", "evidence"]:
            mem = MemoryUnit(
                memory_type=mtype,
                embedding=embed_engine.encode(f"test {mtype}"),
            )
            memory_store.store(mem, dedup=False)

        stats = memory_store.get_stats()
        assert stats["by_type"]["result"] == 2
        assert stats["by_type"]["evidence"] == 2
        assert stats["by_type"]["strategy"] == 1

    def test_stats_by_agent(self, memory_store, embed_engine):
        mem1 = MemoryUnit(source_agent="retriever",
                         embedding=embed_engine.encode("test"))
        mem2 = MemoryUnit(source_agent="executor",
                         embedding=embed_engine.encode("test2"))
        mem3 = MemoryUnit(source_agent="retriever",
                         embedding=embed_engine.encode("test3"))
        memory_store.store(mem1)
        memory_store.store(mem2)
        memory_store.store(mem3)

        stats = memory_store.get_stats()
        assert stats["by_agent"]["retriever"] == 2
        assert stats["by_agent"]["executor"] == 1

    def test_clear(self, memory_store, sample_memory):
        memory_store.store(sample_memory)
        assert memory_store.get_stats()["total_memories"] == 1

        memory_store.clear()
        assert memory_store.get_stats()["total_memories"] == 0
        assert memory_store.get(sample_memory.memory_id) is None


class TestPopulatedMemory:
    """Tests with pre-populated memory store."""

    def test_all_memories_stored(self, populated_memory):
        stats = populated_memory.get_stats()
        assert stats["total_memories"] == 4

    def test_cross_topic_search(self, populated_memory):
        """Search should find memories across different task topics."""
        solar_results = populated_memory.search_by_keyword("solar")
        assert len(solar_results) >= 2  # Solar and Cost memories reference solar

    def test_tag_filtering(self, populated_memory):
        energy_results = populated_memory.search_by_tags(["solar", "wind"])
        assert len(energy_results) >= 2  # Both energy topics

        security_results = populated_memory.search_by_tags(["security"])
        assert len(security_results) == 1

    def test_semantic_search_populated(self, populated_memory, embed_engine):
        if not HAS_FAISS:
            pytest.skip("FAISS not available")

        # Search for energy-related content
        query = embed_engine.encode("renewable energy generation")
        results = populated_memory.search_by_similarity(query, limit=4)
        assert len(results) > 0

        # Top results should be energy-related
        top_topics = [r[0].task_topic for r in results[:2]]
        assert any("Energy" in t or "Cost" in t for t in top_topics)

    def test_access_tracking(self, populated_memory, embed_engine):
        # Record some accesses
        populated_memory.record_access(
            list(populated_memory._id_to_idx.keys())[0],
            "planner", "task-5"
        )
        stats = populated_memory.get_stats()
        assert stats["total_accesses"] >= 1


class TestMemorySearchPerformance:
    """Performance tests for memory search operations."""

    def test_keyword_search_speed(self, memory_store):
        """Keyword search should be fast even with many memories."""
        # Populate with 100 memories
        for i in range(100):
            mem = MemoryUnit(
                task_topic=f"Task {i}: Research topic {i % 10}",
                summary=f"Summary for task {i}",
                tags=[f"tag_{i % 5}", f"category_{i % 3}"],
                memory_type="result",
            )
            memory_store.store(mem)

        import time
        t0 = time.perf_counter()
        results = memory_store.search_by_keyword("Research")
        elapsed = time.perf_counter() - t0

        assert len(results) > 0
        assert elapsed < 0.2, f"Keyword search too slow: {elapsed:.4f}s"

    def test_tag_search_speed(self, memory_store):
        """Tag search should be fast."""
        for i in range(100):
            mem = MemoryUnit(
                tags=[f"tag_{i % 10}"],
                memory_type="result",
            )
            memory_store.store(mem)

        import time
        t0 = time.perf_counter()
        results = memory_store.search_by_tags(["tag_1"])
        elapsed = time.perf_counter() - t0

        assert elapsed < 0.2, f"Tag search too slow: {elapsed:.4f}s"

    def test_semantic_search_speed(self, memory_store, embed_engine):
        """Semantic search should be fast with FAISS."""
        if not HAS_FAISS:
            pytest.skip("FAISS not available")

        for i in range(100):
            mem = MemoryUnit(
                task_topic=f"Topic {i}",
                summary=f"Summary {i} about {'energy' if i % 2 == 0 else 'security'}",
                embedding=embed_engine.encode(
                    f"topic {i} about {'renewable energy' if i % 2 == 0 else 'code security'}"
                ),
            )
            memory_store.store(mem)

        import time
        query = embed_engine.encode("renewable energy research")
        t0 = time.perf_counter()
        results = memory_store.search_by_similarity(query, limit=10)
        elapsed = time.perf_counter() - t0

        assert len(results) > 0
        assert elapsed < 0.1, f"Semantic search too slow: {elapsed:.4f}s"
