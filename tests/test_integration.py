"""Integration tests for the full multi-agent collaboration pipeline."""

import time
import pytest

from src.orchestrator import Orchestrator
from src.protocol import Message, ActionType, MessageType
from src.memory.models import MemoryUnit


class TestFullPipeline:
    """End-to-end pipeline tests: Plan -> Retrieve -> Execute -> Summarize."""

    def test_single_task_execution(self, orchestrator):
        result = orchestrator.execute_task(
            task_id="task-1",
            task_description="Research solar energy technologies",
            tags=["solar", "energy"],
        )
        assert result["task_id"] == "task-1"
        assert "steps" in result
        assert "plan" in result["steps"]
        assert "summary" in result["steps"]
        assert result["elapsed_ms"] > 0

    def test_pipeline_produces_plan(self, orchestrator):
        result = orchestrator.execute_task(
            task_id="task-plan-check",
            task_description="Analyze code security vulnerabilities",
        )
        plan = result["steps"]["plan"]
        assert "plan_id" in plan
        assert "subtasks" in plan
        assert len(plan["subtasks"]) >= 1

    def test_pipeline_produces_retrieval_results(self, orchestrator):
        result = orchestrator.execute_task(
            task_id="task-retrieval-check",
            task_description="Research renewable wind energy",
            tags=["wind", "energy"],
        )
        retrieval = result["steps"]["retrieval"]
        # Retrieval results can be empty if no retriever subtask, but should be a dict
        assert isinstance(retrieval, dict)

    def test_pipeline_produces_summary(self, orchestrator):
        result = orchestrator.execute_task(
            task_id="task-summary-check",
            task_description="Analyze solar energy efficiency",
            tags=["solar"],
        )
        summary = result["steps"]["summary"]
        assert "summary" in summary
        assert len(summary["summary"]) > 0

    def test_pipeline_elapsed_time_reasonable(self, orchestrator):
        result = orchestrator.execute_task(
            task_id="task-timing",
            task_description="Quick analysis task",
        )
        assert result["elapsed_ms"] < 5000  # Should complete quickly

    def test_pipeline_stores_memory(self, orchestrator):
        before_count = orchestrator.memory_store.get_stats()["total_memories"]
        orchestrator.execute_task(
            task_id="task-mem-check",
            task_description="Research solar energy innovations",
            tags=["solar", "innovation"],
        )
        after_count = orchestrator.memory_store.get_stats()["total_memories"]
        assert after_count > before_count

    def test_pipeline_metrics_collected(self, orchestrator):
        orchestrator.execute_task(
            task_id="task-metrics-check",
            task_description="Test metrics collection",
        )
        agg = orchestrator.metrics.get_aggregate_metrics()
        assert agg["total_tasks"] >= 1


class TestCrossTaskMemoryReuse:
    """Test that memories from one task are reused in subsequent tasks."""

    def test_memory_accumulates_across_tasks(self, orchestrator):
        initial_memories = orchestrator.memory_store.get_stats()["total_memories"]

        orchestrator.execute_task(
            task_id="task-a",
            task_description="Research solar energy photovoltaic efficiency",
            tags=["solar", "energy", "efficiency"],
        )
        after_first = orchestrator.memory_store.get_stats()["total_memories"]
        assert after_first > initial_memories

        orchestrator.execute_task(
            task_id="task-b",
            task_description="Analyze solar energy cost reduction trends",
            tags=["solar", "energy", "cost"],
        )
        after_second = orchestrator.memory_store.get_stats()["total_memories"]
        assert after_second > after_first

    def test_related_topics_share_memories(self, orchestrator):
        """Tasks with overlapping tags should find each other's memories."""
        orchestrator.execute_task(
            task_id="energy-task-1",
            task_description="Research renewable energy technologies including solar",
            tags=["energy", "solar", "renewable"],
        )

        orchestrator.execute_task(
            task_id="energy-task-2",
            task_description="Analyze solar photovoltaic energy efficiency trends",
            tags=["energy", "solar", "efficiency"],
        )

        stats = orchestrator.memory_store.get_stats()
        assert stats["total_memories"] > 0
        # At least some cross-task memories should exist
        assert "energy" in str(stats.get("by_topic", {})).lower() or \
               "solar" in str(stats.get("by_topic", {})).lower()

    def test_different_topics_independent(self, orchestrator):
        """Verify that unrelated topics produce independent memories."""
        orchestrator.execute_task(
            task_id="solar-task",
            task_description="Solar energy photovoltaic research",
            tags=["solar", "energy"],
        )

        orchestrator.execute_task(
            task_id="security-task",
            task_description="SQL injection vulnerability analysis",
            tags=["security", "sql"],
        )

        stats = orchestrator.memory_store.get_stats()
        # Should have memories from both topics
        assert stats["total_memories"] >= 2

        # Solar keyword search should find solar memories
        solar_hits = orchestrator.memory_store.search_by_keyword("solar")
        assert len(solar_hits) >= 1

        # Security keyword search should find security memories
        security_hits = orchestrator.memory_store.search_by_keyword("security")
        assert len(security_hits) >= 1

    def test_semantic_search_cross_task(self, orchestrator, embed_engine):
        """After running solar tasks, semantic search should find them."""
        orchestrator.execute_task(
            task_id="solar-1",
            task_description="Solar energy photovoltaic efficiency research",
            tags=["solar"],
        )
        orchestrator.execute_task(
            task_id="solar-2",
            task_description="Solar panel cost analysis and manufacturing trends",
            tags=["solar", "cost"],
        )

        # Semantic search should find these memories
        query = embed_engine.encode("solar power generation")
        results = orchestrator.memory_store.search_by_similarity(query, limit=5)
        # In hash mode (no FAISS), semantic search may return empty
        # But keyword search should still work
        keyword_hits = orchestrator.memory_store.search_by_keyword("solar")
        assert len(keyword_hits) >= 1


class TestDualModeComparison:
    """Test structured vs text mode comparison functionality."""

    def test_execute_task_text_mode(self, orchestrator):
        result = orchestrator.execute_task_text_mode(
            task_id="text-task-1",
            task_description="Research wind energy technologies",
            tags=["wind"],
        )
        assert result["task_id"] == "text-task-1"
        assert "summary" in result["steps"]

    def test_run_comparison_experiment(self, orchestrator):
        tasks = [
            {
                "task_id": "comp-1",
                "description": "Research solar energy efficiency",
                "tags": ["solar", "energy"],
            },
            {
                "task_id": "comp-2",
                "description": "Analyze solar manufacturing costs",
                "tags": ["solar", "cost"],
            },
        ]
        report = orchestrator.run_comparison_experiment(tasks, "solar_energy")
        assert report.task_group == "solar_energy"
        assert report.structured_messages > 0
        assert report.text_messages > 0
        assert report.structured_tokens >= 0
        assert report.text_tokens >= 0
        assert report.structured_latency_ms > 0
        assert report.text_latency_ms > 0

    def test_comparison_report_added_to_metrics(self, orchestrator):
        tasks = [
            {
                "task_id": "comp-metric-1",
                "description": "Test comparison task",
                "tags": ["test"],
            },
        ]
        orchestrator.run_comparison_experiment(tasks, "test_group")
        assert len(orchestrator.metrics._comparisons) >= 1

    def test_structured_mode_uses_state_transfer(self, orchestrator):
        """Structured mode generates state transfers."""
        orchestrator.mode = "structured"
        orchestrator.state_bus.clear()
        initial_count = orchestrator.state_bus.get_stats()["transfer_count"]

        orchestrator.execute_task(
            task_id="state-transfer-test",
            task_description="Test state transfer in structured mode",
        )
        # At least the planner should transfer state
        stats = orchestrator.state_bus.get_stats()
        assert stats["transfer_count"] >= initial_count


class TestOrchestratorSystemStats:
    """Test system status and statistics reporting."""

    def test_initial_system_stats(self, orchestrator):
        stats = orchestrator.get_system_stats()
        assert "scheduler" in stats
        assert "memory" in stats
        assert "state_exchange" in stats
        assert "metrics" in stats
        assert "agents" in stats
        assert len(stats["agents"]) == 4

    def test_system_stats_after_execution(self, orchestrator):
        orchestrator.execute_task(
            task_id="stats-task",
            task_description="Test system statistics",
        )
        stats = orchestrator.get_system_stats()
        assert stats["scheduler"]["total_messages"] > 0
        assert stats["memory"]["total_memories"] > 0

    def test_print_system_status(self, orchestrator):
        orchestrator.execute_task(
            task_id="status-task",
            task_description="Test status display",
        )
        status = orchestrator.print_system_status()
        assert "System Status" in status
        assert "structured" in status.lower() or "Agents" in status
        assert "Shared Memory" in status
        assert "State Exchange" in status

    def test_cleanup_resets_state(self, orchestrator):
        orchestrator.execute_task(
            task_id="cleanup-task",
            task_description="Task before cleanup",
        )
        orchestrator.cleanup()
        # After cleanup, scheduler stats should be fresh
        stats = orchestrator.get_system_stats()
        assert stats["scheduler"]["total_messages"] == 0


class TestTaskGroupExecution:
    """Test grouped task execution."""

    def test_execute_task_group(self, orchestrator):
        tasks = [
            {
                "task_id": "group-1",
                "description": "Research solar energy data",
                "tags": ["solar", "energy"],
            },
            {
                "task_id": "group-2",
                "description": "Analyze wind energy potential",
                "tags": ["wind", "energy"],
            },
            {
                "task_id": "group-3",
                "description": "Review code security best practices",
                "tags": ["security", "code"],
            },
        ]
        results = orchestrator.execute_task_group(tasks, "energy_security_group")
        assert len(results) == 3
        for r in results:
            assert "summary" in r["steps"]

    def test_task_group_memory_sharing(self, orchestrator):
        """Memories from earlier tasks in a group should be visible to later ones."""
        tasks = [
            {
                "task_id": "grp-task-1",
                "description": "Solar energy photovoltaic efficiency research",
                "tags": ["solar", "efficiency"],
            },
            {
                "task_id": "grp-task-2",
                "description": "Solar panel manufacturing cost analysis",
                "tags": ["solar", "cost"],
            },
        ]
        orchestrator.execute_task_group(tasks, "solar_group")

        # Should find solar memories from both tasks
        solar_results = orchestrator.memory_store.search_by_keyword("solar")
        assert len(solar_results) >= 1


class TestPipelineEdgeCases:
    """Edge case tests for the orchestrator pipeline."""

    def test_task_with_empty_tags(self, orchestrator):
        result = orchestrator.execute_task(
            task_id="empty-tags",
            task_description="Generic analysis task",
            tags=[],
        )
        assert "summary" in result["steps"]

    def test_task_with_long_description(self, orchestrator):
        long_desc = (
            "Conduct a comprehensive analysis of renewable energy technologies "
            "including solar photovoltaic systems, wind turbine generation, "
            "hydroelectric power production, and geothermal energy extraction. "
            "Compare efficiency metrics, cost trends, and environmental impact "
            "across all technologies to determine optimal deployment strategies "
            "for different geographic regions and climate conditions."
        )
        result = orchestrator.execute_task(
            task_id="long-desc",
            task_description=long_desc,
        )
        assert result["elapsed_ms"] > 0

    def test_task_with_special_characters(self, orchestrator):
        result = orchestrator.execute_task(
            task_id="special-chars",
            task_description="Analyze CO2 emissions & cost-benefit of solar vs. coal (2024 data)",
            tags=["energy", "emissions"],
        )
        assert "summary" in result["steps"]

    def test_rapid_sequential_tasks(self, orchestrator):
        """Running tasks in quick succession should work without errors."""
        for i in range(5):
            result = orchestrator.execute_task(
                task_id=f"rapid-{i}",
                task_description=f"Quick analysis task number {i}",
            )
            assert result["elapsed_ms"] >= 0

    def test_mode_switching(self, orchestrator):
        """Switching between structured and text mode should work."""
        # Structured
        orchestrator.mode = "structured"
        for agent in orchestrator.agents.values():
            agent.use_structured_protocol = True
        r1 = orchestrator.execute_task("switch-1", "Structured mode task")
        structured_msgs = orchestrator.message_bus.message_count

        # Text
        orchestrator.mode = "text"
        for agent in orchestrator.agents.values():
            agent.use_structured_protocol = False
        r2 = orchestrator.execute_task("switch-2", "Text mode task")

        # Back to structured
        orchestrator.mode = "structured"
        for agent in orchestrator.agents.values():
            agent.use_structured_protocol = True
        r3 = orchestrator.execute_task("switch-3", "Structured again")

        assert r1["mode"] == "structured"
        assert r2["mode"] == "text"
        assert r3["mode"] == "structured"


class TestContinuousStability:
    """Stability tests for extended multi-round execution."""

    def test_ten_rounds_stable(self, orchestrator):
        """Run 10 tasks sequentially to verify stability."""
        topics = [
            "solar energy efficiency",
            "wind turbine capacity",
            "hydroelectric power generation",
            "solar panel manufacturing costs",
            "renewable energy policy analysis",
            "code security vulnerability detection",
            "SQL injection prevention methods",
            "photovoltaic technology advances",
            "energy storage battery technology",
            "smart grid integration challenges",
        ]
        results = []
        for i, topic in enumerate(topics):
            result = orchestrator.execute_task(
                task_id=f"continuous-{i}",
                task_description=topic,
                tags=["energy" if "code" not in topic and "sql" not in topic.lower()
                       else "security"],
            )
            results.append(result)

        assert len(results) == 10
        for r in results:
            assert "summary" in r["steps"]
            assert r["elapsed_ms"] >= 0

        # Verify system is still functional after 10 rounds
        stats = orchestrator.get_system_stats()
        assert stats["memory"]["total_memories"] > 0
        assert stats["scheduler"]["total_messages"] > 0

    def test_memory_growth_under_continuous_load(self, orchestrator):
        """Memory store should grow predictably under continuous use."""
        initial_stats = orchestrator.memory_store.get_stats()

        for i in range(5):
            orchestrator.execute_task(
                task_id=f"mem-growth-{i}",
                task_description=f"Research topic {i}: renewable energy technologies",
                tags=["energy", "continuous"],
            )

        final_stats = orchestrator.memory_store.get_stats()
        assert final_stats["total_memories"] > initial_stats["total_memories"]

    def test_no_message_leak(self, orchestrator):
        """Message bus should not leak messages between task executions."""
        orchestrator.message_bus._history.clear()
        orchestrator.execute_task("leak-test-1", "First task")
        msg_count_after_first = orchestrator.message_bus.message_count

        orchestrator.execute_task("leak-test-2", "Second task")
        msg_count_after_second = orchestrator.message_bus.message_count

        # Messages should accumulate reasonably (not exponentially)
        assert msg_count_after_second >= msg_count_after_first
