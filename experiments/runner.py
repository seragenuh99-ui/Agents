"""Experiment runner for comparing structured vs text communication modes."""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.orchestrator import Orchestrator
from src.agents.base import LLMBackend
from src.evaluation.metrics import ComparisonReport
from src.evaluation.reporter import Reporter
from experiments.tasks import ALL_TASK_GROUPS


def run_full_experiment(
    llm: Optional[LLMBackend] = None,
    use_real_embeddings: bool = True,
    sandbox_enabled: bool = True,
    task_groups: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run the complete experiment suite.

    For each task group:
    1. Execute in structured protocol mode
    2. Execute in text communication mode
    3. Compare results
    """
    if task_groups is None:
        task_groups = ["energy_research", "code_security"]

    print("=" * 70)
    print("  Multi-Agent Collaboration Experiment Suite")
    print("=" * 70)
    print(f"  Task Groups: {task_groups}")
    print(f"  Real Embeddings: {use_real_embeddings}")
    print(f"  Sandbox: {sandbox_enabled}")
    print("=" * 70)
    print()

    orchestrator = Orchestrator(
        llm=llm,
        mode="structured",
        use_real_embeddings=use_real_embeddings,
        sandbox_enabled=sandbox_enabled,
    )

    reporter = Reporter(orchestrator.metrics)
    all_results: Dict[str, Any] = {}
    total_start = time.time()

    for group_name in task_groups:
        if group_name not in ALL_TASK_GROUPS:
            print(f"  [SKIP] Unknown task group: {group_name}")
            continue

        tasks = ALL_TASK_GROUPS[group_name]
        print(f"\n{'='*60}")
        print(f"  Task Group: {group_name} ({len(tasks)} tasks)")
        print(f"{'='*60}")

        # Step 1: Structured protocol mode
        print(f"\n  [1/2] Running in STRUCTURED protocol mode...")
        orchestrator.mode = "structured"
        for agent in orchestrator.agents.values():
            agent.use_structured_protocol = True
        orchestrator.message_bus._history.clear()
        orchestrator.state_bus.clear()
        orchestrator.metrics.clear()

        structured_start = time.time()
        structured_results = orchestrator.execute_task_group(tasks, group_name)
        structured_elapsed = time.time() - structured_start

        structured_msg_count = orchestrator.message_bus.message_count
        structured_tokens = orchestrator.message_bus.structured_token_count
        state_stats = orchestrator.state_bus.get_stats()

        print(f"    Messages: {structured_msg_count}")
        print(f"    Structured Tokens: {structured_tokens}")
        print(f"    State Transfers: {state_stats['transfer_count']}")
        print(f"    State Data: {state_stats['total_data_bytes']} bytes")
        print(f"    Latency: {structured_elapsed:.2f}s")

        for i, r in enumerate(structured_results):
            summary = r.get("steps", {}).get("summary", {}).get("summary", "N/A")
            print(f"    Task {i+1} result: {summary[:120]}...")

        # Step 2: Text communication mode
        print(f"\n  [2/2] Running in TEXT communication mode...")
        orchestrator.mode = "text"
        for agent in orchestrator.agents.values():
            agent.use_structured_protocol = False
        orchestrator.message_bus._history.clear()
        orchestrator.state_bus.clear()

        text_start = time.time()
        text_results = orchestrator.execute_task_group(tasks, group_name)
        text_elapsed = time.time() - text_start

        text_msg_count = orchestrator.message_bus.message_count
        text_tokens = orchestrator.message_bus.text_token_count

        print(f"    Messages: {text_msg_count}")
        print(f"    Text Tokens: {text_tokens}")
        print(f"    Latency: {text_elapsed:.2f}s")

        for i, r in enumerate(text_results):
            summary = r.get("steps", {}).get("summary", {}).get("summary", "N/A")
            print(f"    Task {i+1} result: {summary[:120]}...")

        # Step 3: Comparison
        report = ComparisonReport(
            task_group=group_name,
            structured_messages=structured_msg_count,
            structured_tokens=structured_tokens,
            structured_latency_ms=structured_elapsed * 1000,
            structured_state_bytes=state_stats.get("total_data_bytes", 0),
            text_messages=text_msg_count,
            text_tokens=text_tokens,
            text_latency_ms=text_elapsed * 1000,
        )
        orchestrator.metrics.add_comparison(report)

        print(f"\n  --- Comparison: {group_name} ---")
        print(f"  Token Reduction:   {report.token_reduction_pct:.1f}%")
        print(f"  Latency Reduction: {report.latency_reduction_pct:.1f}%")
        print(f"  Structured: {structured_msg_count} msgs, {structured_tokens} tokens, {structured_elapsed:.2f}s")
        print(f"  Text:       {text_msg_count} msgs, {text_tokens} tokens, {text_elapsed:.2f}s")

        all_results[group_name] = {
            "structured": {
                "results": structured_results,
                "messages": structured_msg_count,
                "tokens": structured_tokens,
                "latency_s": structured_elapsed,
            },
            "text": {
                "results": text_results,
                "messages": text_msg_count,
                "tokens": text_tokens,
                "latency_s": text_elapsed,
            },
            "comparison": report.to_dict(),
        }

        # Restore structured mode
        orchestrator.mode = "structured"
        for agent in orchestrator.agents.values():
            agent.use_structured_protocol = True

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  Experiment Complete")
    print(f"  Total Time: {total_elapsed:.2f}s")
    print(f"{'='*60}")

    # Print final summary
    reporter = Reporter(orchestrator.metrics)
    print("\n" + reporter.print_summary())

    # Memory stats
    mem_stats = orchestrator.memory_store.get_stats()
    print(f"\n  Shared Memory: {mem_stats['total_memories']} memories stored")
    print(f"  Vector Index: {mem_stats['vector_index_size']} vectors indexed")

    # Export metrics
    json_path = reporter.export_json("metrics_report.json")
    print(f"\n  Metrics exported to: {json_path}")

    return {
        "task_results": all_results,
        "memory_stats": mem_stats,
        "system_stats": orchestrator.get_system_stats(),
        "total_elapsed_s": total_elapsed,
    }


def run_continuous_tasks(
    num_tasks: int = 10,
    llm: Optional[LLMBackend] = None,
    use_real_embeddings: bool = True,
) -> Dict[str, Any]:
    """Run continuous tasks to validate system stability over 10+ rounds."""
    print("=" * 60)
    print(f"  Continuous Task Execution ({num_tasks} rounds)")
    print("=" * 60)

    orchestrator = Orchestrator(
        llm=llm,
        mode="structured",
        use_real_embeddings=use_real_embeddings,
    )

    tasks = ALL_TASK_GROUPS["continuous_10"][:num_tasks]
    results = []

    for i, task in enumerate(tasks):
        print(f"\n  Round {i+1}/{num_tasks}: {task['description'][:80]}...")
        result = orchestrator.execute_task(
            task_id=task["task_id"],
            task_description=task["description"],
            tags=task.get("tags", []),
        )
        results.append(result)

        mem_stats = orchestrator.memory_store.get_stats()
        print(f"    Memories: {mem_stats['total_memories']}, "
              f"Vector Index: {mem_stats['vector_index_size']}")
        summary = result.get("steps", {}).get("summary", {}).get("summary", "")
        if summary:
            print(f"    Result: {summary[:120]}...")

    reporter = Reporter(orchestrator.metrics)
    print("\n" + reporter.print_summary())

    json_path = reporter.export_json("continuous_metrics.json")
    print(f"\n  Metrics exported to: {json_path}")

    return {
        "results": results,
        "memory_stats": orchestrator.memory_store.get_stats(),
        "metrics": orchestrator.metrics.get_aggregate_metrics(),
    }
