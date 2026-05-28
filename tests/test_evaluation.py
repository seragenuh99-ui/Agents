"""Unit tests for evaluation metrics, comparison reports, and reporting."""

import json
import os
import tempfile
import time
import pytest

from src.evaluation.metrics import (
    TaskMetrics, ComparisonReport, MetricsCollector,
)
from src.evaluation.reporter import Reporter


class TestTaskMetrics:
    """Test TaskMetrics dataclass and computed properties."""

    def test_default_creation(self):
        tm = TaskMetrics()
        assert tm.task_id == ""
        assert tm.mode == "structured"
        assert tm.messages_sent == 0
        assert tm.structured_token_count == 0
        assert tm.text_equivalent_token_count == 0

    def test_full_creation(self):
        tm = TaskMetrics(
            task_id="task-1",
            task_description="Test task",
            mode="structured",
            messages_sent=10,
            structured_token_count=120,
            text_equivalent_token_count=200,
            state_transfers=3,
            state_data_bytes=4608,
            memory_queries=4,
            memory_hits=2,
            memories_stored=1,
            total_elapsed_ms=500.0,
            llm_call_count=3,
            llm_total_ms=300.0,
        )
        assert tm.task_id == "task-1"
        assert tm.structured_token_count == 120
        assert tm.text_equivalent_token_count == 200

    def test_token_savings_pct_positive(self):
        tm = TaskMetrics(
            structured_token_count=75,
            text_equivalent_token_count=100,
        )
        assert tm.token_savings_pct == pytest.approx(25.0)

    def test_token_savings_pct_negative(self):
        """If structured uses more tokens, savings is negative."""
        tm = TaskMetrics(
            structured_token_count=150,
            text_equivalent_token_count=100,
        )
        assert tm.token_savings_pct == pytest.approx(-50.0)

    def test_token_savings_pct_zero_denominator(self):
        tm = TaskMetrics(
            structured_token_count=100,
            text_equivalent_token_count=0,
        )
        assert tm.token_savings_pct == 0.0

    def test_token_savings_pct_no_savings(self):
        tm = TaskMetrics(
            structured_token_count=100,
            text_equivalent_token_count=100,
        )
        assert tm.token_savings_pct == pytest.approx(0.0)

    def test_avg_message_latency(self):
        tm = TaskMetrics(
            messages_sent=5,
            total_elapsed_ms=500.0,
        )
        assert tm.avg_message_latency_ms == pytest.approx(100.0)

    def test_avg_message_latency_zero_messages(self):
        tm = TaskMetrics(messages_sent=0, total_elapsed_ms=100.0)
        assert tm.avg_message_latency_ms == 0.0

    def test_memory_hit_rate_internal(self):
        """memory_hit_rate is computed in end_task if > 0."""
        tm = TaskMetrics(memory_queries=5, memory_hits=3)
        assert tm.memory_hit_rate == 0.0  # Not computed by property


class TestComparisonReport:
    """Test ComparisonReport dataclass and properties."""

    def test_default_creation(self):
        cr = ComparisonReport()
        assert cr.task_group == ""
        assert cr.token_reduction_pct == 0.0

    def test_token_reduction(self):
        cr = ComparisonReport(
            structured_tokens=80,
            text_tokens=100,
        )
        assert cr.token_reduction_pct == pytest.approx(20.0)

    def test_latency_reduction(self):
        cr = ComparisonReport(
            structured_latency_ms=60.0,
            text_latency_ms=100.0,
        )
        assert cr.latency_reduction_pct == pytest.approx(40.0)

    def test_token_reduction_zero_text(self):
        cr = ComparisonReport(structured_tokens=10, text_tokens=0)
        assert cr.token_reduction_pct == 0.0

    def test_latency_reduction_zero_text(self):
        cr = ComparisonReport(structured_latency_ms=10, text_latency_ms=0)
        assert cr.latency_reduction_pct == 0.0

    def test_negative_reduction(self):
        """Structured must be worse than text for negative reduction."""
        cr = ComparisonReport(
            structured_tokens=120,
            text_tokens=100,
        )
        assert cr.token_reduction_pct < 0

    def test_to_dict(self):
        cr = ComparisonReport(
            task_group="energy_research",
            structured_messages=8,
            structured_tokens=75,
            structured_latency_ms=300.0,
            structured_state_bytes=4608,
            text_messages=8,
            text_tokens=100,
            text_latency_ms=500.0,
            text_state_bytes=0,
        )
        d = cr.to_dict()
        assert d["task_group"] == "energy_research"
        assert d["token_reduction_pct"] == 25.0
        assert d["latency_reduction_pct"] == 40.0
        assert d["structured_mode"]["messages"] == 8
        assert d["text_mode"]["tokens"] == 100


class TestMetricsCollector:
    """Test MetricsCollector for task-level and aggregate metrics."""

    def test_initial_state(self, metrics):
        assert metrics._current_task is None
        assert metrics._task_history == []
        assert metrics._comparisons == []

    def test_start_and_end_task(self, metrics):
        metrics.start_task("task-1", "Test task", "structured")
        assert metrics._current_task is not None
        assert metrics._current_task.task_id == "task-1"

        task = metrics.end_task()
        assert task is not None
        assert task.task_id == "task-1"
        assert task.total_elapsed_ms >= 0
        assert len(metrics._task_history) == 1

    def test_end_task_without_start(self, metrics):
        assert metrics.end_task() is None

    def test_record_message(self, metrics):
        metrics.start_task("task-1", "Test", "structured")
        metrics.record_message(
            token_count=15,
            text_token_count=25,
            char_count=45,
        )
        metrics.record_message(
            token_count=10,
            text_token_count=20,
            char_count=30,
        )
        assert metrics._current_task.messages_sent == 2
        assert metrics._current_task.structured_token_count == 25
        assert metrics._current_task.text_equivalent_token_count == 45
        assert metrics._current_task.communication_char_count == 75

    def test_record_message_no_task(self, metrics):
        metrics.record_message(10, 20, 30)  # Should not raise

    def test_record_state_transfer(self, metrics):
        metrics.start_task("task-1", "Test", "structured")
        metrics.record_state_transfer(data_bytes=1536, generation_ms=2.5)
        metrics.record_state_transfer(data_bytes=1536, generation_ms=3.0)
        assert metrics._current_task.state_transfers == 2
        assert metrics._current_task.state_data_bytes == 3072
        assert metrics._current_task.state_generation_ms == 5.5

    def test_record_memory_query(self, metrics):
        metrics.start_task("task-1", "Test", "structured")
        metrics.record_memory_query(hits=3, cross_task=1)
        metrics.record_memory_query(hits=0, cross_task=0)
        assert metrics._current_task.memory_queries == 2
        assert metrics._current_task.memory_hits == 3
        assert metrics._current_task.cross_task_memories_used == 1

    def test_record_memory_query_default_cross_task(self, metrics):
        metrics.start_task("task-1", "Test")
        metrics.record_memory_query(hits=2)  # cross_task defaults to 0
        assert metrics._current_task.cross_task_memories_used == 0

    def test_record_memory_store(self, metrics):
        metrics.start_task("task-1", "Test", "structured")
        metrics.record_memory_store()
        metrics.record_memory_store()
        assert metrics._current_task.memories_stored == 2

    def test_record_llm_call(self, metrics):
        metrics.start_task("task-1", "Test", "structured")
        metrics.record_llm_call(150.0)
        metrics.record_llm_call(200.0)
        assert metrics._current_task.llm_call_count == 2
        assert metrics._current_task.llm_total_ms == 350.0

    def test_memory_hit_rate_computed_on_end(self, metrics):
        metrics.start_task("task-1", "Test", "structured")
        metrics.record_memory_query(hits=3)
        metrics.record_memory_query(hits=1)
        task = metrics.end_task()
        # 4 hits from 2 queries
        assert task.memory_hit_rate == pytest.approx(2.0)

    def test_memory_hit_rate_zero_queries(self, metrics):
        metrics.start_task("task-1", "Test")
        task = metrics.end_task()
        assert task.memory_hit_rate == 0.0

    def test_multiple_tasks(self, metrics):
        for i in range(3):
            metrics.start_task(f"task-{i}", f"Test {i}", "structured")
            metrics.record_message(10, 15, 30)
            metrics.end_task()

        assert len(metrics._task_history) == 3

    def test_add_comparison(self, metrics):
        cr = ComparisonReport(task_group="test", structured_tokens=80, text_tokens=100)
        metrics.add_comparison(cr)
        assert len(metrics._comparisons) == 1

    def test_clear(self, metrics):
        metrics.start_task("task-1", "Test", "structured")
        metrics.record_message(10, 20, 30)
        metrics.end_task()
        metrics.add_comparison(ComparisonReport(task_group="test"))

        metrics.clear()
        assert metrics._current_task is None
        assert metrics._task_history == []
        assert metrics._comparisons == []

    def test_get_aggregate_metrics_empty(self, metrics):
        assert metrics.get_aggregate_metrics() == {}

    def test_get_aggregate_metrics(self, metrics):
        metrics.start_task("task-1", "Test A", "structured")
        metrics.record_message(10, 20, 30)
        metrics.record_state_transfer(1536, 2.0)
        metrics.record_memory_query(hits=2, cross_task=2)
        metrics.end_task()

        metrics.start_task("task-2", "Test B", "text")
        metrics.record_message(12, 22, 36)
        metrics.record_memory_query(hits=0)
        metrics.end_task()

        agg = metrics.get_aggregate_metrics()
        assert agg["total_tasks"] == 2
        assert agg["total_messages"] == 2
        assert agg["total_structured_tokens"] == 22
        assert agg["total_text_equivalent_tokens"] == 42
        assert agg["total_state_transfers"] == 1
        assert agg["total_memory_queries"] == 2
        assert agg["total_memory_hits"] == 2
        assert agg["cross_task_memories_used"] == 2

    def test_get_aggregate_token_savings(self, metrics):
        metrics.start_task("task-1", "Test", "structured")
        metrics.record_message(10, 20, 30)
        metrics.end_task()

        agg = metrics.get_aggregate_metrics()
        assert agg["overall_token_savings_pct"] == pytest.approx(50.0)

    def test_get_aggregate_latency_by_mode(self, metrics):
        metrics.start_task("task-1", "S", "structured")
        metrics.end_task()

        metrics.start_task("task-2", "T", "text")
        metrics.end_task()

        agg = metrics.get_aggregate_metrics()
        mc = agg["mode_comparison"]
        assert mc["structured"]["task_count"] == 1
        assert mc["text"]["task_count"] == 1

    def test_get_task_details(self, metrics):
        metrics.start_task("task-1", "Research solar energy technologies", "structured")
        metrics.record_message(10, 20, 30)
        metrics.record_state_transfer(1536, 2.0)
        metrics.record_memory_query(hits=3, cross_task=1)
        metrics.end_task()

        details = metrics.get_task_details()
        assert len(details) == 1
        d = details[0]
        assert d["task_id"] == "task-1"
        assert d["mode"] == "structured"
        assert d["messages"] == 1
        assert d["structured_tokens"] == 10
        assert d["text_tokens"] == 20
        assert d["state_transfers"] == 1
        assert d["memory_hits"] == 3

    def test_get_task_details_truncates_description(self, metrics):
        long_desc = "Research " * 50  # > 100 chars
        metrics.start_task("task-1", long_desc)
        metrics.end_task()

        details = metrics.get_task_details()
        assert len(details[0]["description"]) <= 100

    def test_concurrent_task_replacement(self, metrics):
        """Starting a new task should cancel the previous one."""
        metrics.start_task("task-1", "First", "structured")
        metrics.start_task("task-2", "Second", "text")
        metrics.record_message(1, 2, 3)
        task = metrics.end_task()
        assert task.task_id == "task-2"
        assert len(metrics._task_history) == 1


class TestReporter:
    """Test Reporter for summary, JSON export, and comparison tables."""

    def test_print_summary_empty(self, metrics):
        reporter = Reporter(metrics)
        output = reporter.print_summary()
        assert "No metrics" in output

    def test_print_summary_with_data(self, metrics):
        metrics.start_task("task-1", "Solar energy research", "structured")
        metrics.record_message(10, 20, 30)
        metrics.record_state_transfer(1536, 2.0)
        metrics.record_memory_query(hits=3, cross_task=1)
        metrics.end_task()

        metrics.start_task("task-2", "Wind energy analysis", "text")
        metrics.record_message(15, 25, 45)
        metrics.end_task()

        reporter = Reporter(metrics)
        output = reporter.print_summary()

        assert "Total Tasks Executed" in output
        assert "2" in output
        assert "Token Savings" in output
        assert "State Transfer" in output
        assert "Shared Memory" in output
        assert "Mode Comparison" in output
        assert "[structured]" in output
        assert "[text]" in output

    def test_print_summary_with_comparisons(self, metrics):
        metrics.start_task("task-1", "Test", "structured")
        metrics.record_message(10, 20, 30)
        metrics.end_task()
        metrics.add_comparison(ComparisonReport(
            task_group="energy",
            structured_tokens=75,
            text_tokens=100,
            structured_latency_ms=300.0,
            text_latency_ms=500.0,
        ))
        reporter = Reporter(metrics)
        output = reporter.print_summary()
        assert "Comparison Reports" in output
        assert "energy" in output

    def test_export_json(self, metrics):
        metrics.start_task("task-1", "Test", "structured")
        metrics.record_message(10, 20, 30)
        metrics.end_task()
        metrics.add_comparison(ComparisonReport(
            task_group="test",
            structured_tokens=10, text_tokens=20,
            structured_latency_ms=100.0, text_latency_ms=200.0,
        ))

        reporter = Reporter(metrics)
        filepath = tempfile.mktemp(suffix=".json")
        try:
            result_path = reporter.export_json(filepath)
            assert result_path == filepath
            assert os.path.exists(filepath)

            with open(filepath) as f:
                data = json.load(f)
            assert "generated_at" in data
            assert "aggregate" in data
            assert "tasks" in data
            assert "comparisons" in data
            assert len(data["tasks"]) == 1
            assert len(data["comparisons"]) == 1
        finally:
            try:
                os.unlink(filepath)
            except OSError:
                pass

    def test_generate_comparison_table_empty(self, metrics):
        reporter = Reporter(metrics)
        output = reporter.generate_comparison_table()
        assert "No comparison data" in output

    def test_generate_comparison_table_with_data(self, metrics):
        metrics.add_comparison(ComparisonReport(
            task_group="energy_research",
            structured_messages=8, structured_tokens=75,
            structured_latency_ms=300.0, structured_state_bytes=4608,
            text_messages=8, text_tokens=100,
            text_latency_ms=500.0, text_state_bytes=0,
        ))
        reporter = Reporter(metrics)
        output = reporter.generate_comparison_table()

        assert "Task Group" in output
        assert "energy_research" in output
        assert "Structured" in output
        assert "Text" in output
        assert "Improvement" in output


class TestMetricsBatch:
    """Batch and edge-case tests for metrics collection."""

    def test_record_message_before_start(self, metrics):
        """Should not raise when recording before starting a task."""
        metrics.record_message(10, 20, 30)
        metrics.record_state_transfer(100, 1.0)
        metrics.record_memory_query(hits=3)
        metrics.record_memory_store()
        metrics.record_llm_call(100.0)
        # No exception = pass

    def test_many_tasks_aggregate(self, metrics):
        for i in range(20):
            metrics.start_task(f"task-{i}", f"Test {i}",
                             "structured" if i % 2 == 0 else "text")
            metrics.record_message(i + 1, (i + 1) * 2, (i + 1) * 3)
            metrics.record_state_transfer(1536, 1.0)
            metrics.record_memory_query(hits=i % 3)
            metrics.end_task()

        agg = metrics.get_aggregate_metrics()
        assert agg["total_tasks"] == 20
        assert agg["total_messages"] == 20
        details = metrics.get_task_details()
        assert len(details) == 20

    def test_zero_values_everywhere(self, metrics):
        """Task with all zeros should not cause division errors."""
        metrics.start_task("empty", "Empty task", "structured")
        metrics.end_task()

        details = metrics.get_task_details()
        assert details[0]["memory_hit_rate"] == 0.0
        assert details[0]["token_savings_pct"] == 0.0
