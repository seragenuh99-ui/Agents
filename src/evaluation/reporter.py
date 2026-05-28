"""Performance report generation and visualization."""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from .metrics import MetricsCollector, ComparisonReport, TaskMetrics


class Reporter:
    """Generates performance reports from collected metrics."""

    def __init__(self, metrics: MetricsCollector):
        self.metrics = metrics

    def print_summary(self) -> str:
        """Generate a human-readable summary report."""
        agg = self.metrics.get_aggregate_metrics()
        if not agg:
            return "No metrics collected yet."

        lines = [
            "=" * 70,
            "  Multi-Agent Collaboration Performance Report",
            "=" * 70,
            "",
            f"  Total Tasks Executed:        {agg['total_tasks']}",
            f"  Total Messages:              {agg['total_messages']}",
            f"  Total Latency:               {agg['total_latency_ms']:.2f} ms",
            f"  Avg Task Latency:            {agg['avg_task_latency_ms']:.2f} ms",
            "",
            "  --- Communication Efficiency ---",
            f"  Structured Protocol Tokens:  {agg['total_structured_tokens']}",
            f"  Text Equivalent Tokens:      {agg['total_text_equivalent_tokens']}",
            f"  Token Savings:               {agg['overall_token_savings_pct']:.1f}%",
            "",
            "  --- State Transfer ---",
            f"  Non-Text State Transfers:    {agg['total_state_transfers']}",
            f"  State Data Transferred:      {agg['total_state_data_bytes']} bytes",
            "",
            "  --- Shared Memory ---",
            f"  Memory Queries:              {agg['total_memory_queries']}",
            f"  Memory Hits:                 {agg['total_memory_hits']}",
            f"  Memory Hit Rate:             {agg['memory_hit_rate']:.2%}",
            f"  Cross-Task Memories Used:    {agg['cross_task_memories_used']}",
            "",
            "  --- LLM Real Usage (API) ---",
        ]

        llm_usage = agg.get("llm_usage", {})
        if llm_usage.get("total_prompt_tokens", 0) > 0:
            lines.extend([
                f"  LLM Call Count:              {self.metrics._task_history[-1].llm_call_count if self.metrics._task_history else 'N/A'}",
                f"  Prompt Tokens:               {llm_usage['total_prompt_tokens']}",
                f"  Completion Tokens:           {llm_usage['total_completion_tokens']}",
                f"  Cached Tokens:               {llm_usage['total_cached_tokens']}",
                f"  Cache Hit Rate:              {llm_usage['cache_hit_pct']:.1f}%",
            ])

        lines.append("")
        lines.append("  --- Mode Comparison ---")

        mc = agg.get("mode_comparison", {})
        for mode, stats in mc.items():
            lines.append(f"  [{mode}]")
            lines.append(f"    Tasks: {stats['task_count']}, "
                         f"Avg Latency: {stats['avg_latency_ms']} ms, "
                         f"Avg Tokens: {stats['avg_tokens']}, "
                         f"Avg Messages: {stats['avg_messages']}")

        lines.append("")
        lines.append("  --- Task Details ---")
        for task in self.metrics.get_task_details():
            lines.append(
                f"  [{task['mode']}] {task['task_id']}: "
                f"{task['messages']} msgs, "
                f"{task['token_savings_pct']:.1f}% token savings, "
                f"{task['memory_hits']} mem hits, "
                f"{task['latency_ms']:.1f} ms"
            )

        if self.metrics._comparisons:
            lines.append("")
            lines.append("  --- Comparison Reports ---")
            for comp in self.metrics._comparisons:
                d = comp.to_dict()
                lines.append(
                    f"  {d['task_group']}: "
                    f"Token Reduction: {d['token_reduction_pct']:.1f}%, "
                    f"Latency Reduction: {d['latency_reduction_pct']:.1f}%"
                )

        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)

    def export_json(self, filepath: str = "metrics_report.json") -> str:
        """Export metrics as JSON file."""
        report = {
            "generated_at": time.time(),
            "aggregate": self.metrics.get_aggregate_metrics(),
            "tasks": self.metrics.get_task_details(),
            "comparisons": [c.to_dict() for c in self.metrics._comparisons],
        }

        with open(filepath, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return filepath

    def generate_comparison_table(self) -> str:
        """Generate a markdown comparison table."""
        comparisons = self.metrics._comparisons
        if not comparisons:
            return "No comparison data available."

        rows = [
            "| Task Group | Mode | Messages | Tokens | Latency (ms) | State Bytes |",
            "|------------|------|----------|--------|--------------|-------------|",
        ]

        for comp in comparisons:
            d = comp.to_dict()
            rows.append(
                f"| {d['task_group']} | Structured | {d['structured_mode']['messages']} | "
                f"{d['structured_mode']['tokens']} | {d['structured_mode']['latency_ms']} | "
                f"{d['structured_mode']['state_bytes']} |"
            )
            rows.append(
                f"| {d['task_group']} | Text | {d['text_mode']['messages']} | "
                f"{d['text_mode']['tokens']} | {d['text_mode']['latency_ms']} | "
                f"{d['text_mode']['state_bytes']} |"
            )
            rows.append(
                f"| {d['task_group']} | **Improvement** | - | "
                f"**{d['token_reduction_pct']:.1f}%** | "
                f"**{d['latency_reduction_pct']:.1f}%** | - |"
            )

        return "\n".join(rows)
