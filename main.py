#!/usr/bin/env python3
"""Main entry point for the Multi-Agent Collaboration System.

Supports three execution modes:
  - experiment: Run full comparison experiment (structured vs text)
  - continuous: Run continuous tasks (10+ rounds) for stability testing
  - single: Run a single task with detailed output

Usage:
  python main.py experiment                    # Full comparison experiment
  python main.py experiment --groups energy_research code_security
  python main.py continuous --num-tasks 10     # Continuous 10-round test
  python main.py single "Research solar energy" # Single task
  python main.py chat                          # Interactive multi-agent chat
  python main.py stats                         # Show memory/system stats
"""

from __future__ import annotations

import argparse
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.agents.base import LLMBackend
from src.orchestrator import Orchestrator
from src.evaluation.reporter import Reporter
from src.paths import DEFAULT_CHAT_SESSION_DB, DEFAULT_SHARED_MEMORY_DB, result_path
from experiments.runner import run_full_experiment, run_continuous_tasks
from experiments.tasks import ALL_TASK_GROUPS


def _build_llm(args) -> LLMBackend:
    """Build LLMBackend from CLI arguments."""
    return LLMBackend(
        base_url=getattr(args, "api_base", None),
        api_key=getattr(args, "api_key", None),
        model=getattr(args, "model", None),
        provider=getattr(args, "provider", "custom"),
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Collaboration System with Structured Communication"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Experiment command
    exp_parser = subparsers.add_parser("experiment", help="Run comparison experiment")
    exp_parser.add_argument(
        "--groups", nargs="+",
        default=["energy_research", "code_security"],
        choices=list(ALL_TASK_GROUPS.keys()),
        help="Task groups to run",
    )
    exp_parser.add_argument(
        "--no-real-embeddings", action="store_true",
        help="Use hash-based embeddings instead of sentence-transformers",
    )
    exp_parser.add_argument(
        "--no-sandbox", action="store_true",
        help="Disable code sandbox",
    )
    exp_parser.add_argument(
        "--model", default=None,
        help="LLM model name (OpenAI-compatible API)",
    )
    exp_parser.add_argument(
        "--api-base", default=None,
        help="LLM API base URL",
    )
    exp_parser.add_argument(
        "--provider", default="custom",
        choices=["openai", "deepseek", "custom"],
        help="LLM provider (openai/deepseek/custom)",
    )
    exp_parser.add_argument(
        "--api-key", default=None,
        help="LLM API key",
    )
    exp_parser.add_argument(
        "--mock", action="store_true",
        help="Use mock LLM for offline testing",
    )

    # Continuous command
    cont_parser = subparsers.add_parser("continuous", help="Run continuous tasks")
    cont_parser.add_argument(
        "--num-tasks", type=int, default=10,
        help="Number of continuous tasks to run",
    )
    cont_parser.add_argument(
        "--no-real-embeddings", action="store_true",
        help="Use hash-based embeddings",
    )
    cont_parser.add_argument(
        "--provider", default="custom",
        choices=["openai", "deepseek", "custom"],
        help="LLM provider",
    )
    cont_parser.add_argument(
        "--api-key", default=None,
        help="LLM API key",
    )
    cont_parser.add_argument(
        "--model", default=None,
        help="LLM model name",
    )
    cont_parser.add_argument(
        "--api-base", default=None,
        help="LLM API base URL",
    )

    # Single task command
    single_parser = subparsers.add_parser("single", help="Run a single task")
    single_parser.add_argument("task", help="Task description")
    single_parser.add_argument(
        "--mode", choices=["structured", "text"], default="structured",
        help="Communication mode",
    )
    single_parser.add_argument(
        "--tags", nargs="+", default=[],
        help="Tags for memory indexing",
    )
    single_parser.add_argument(
        "--provider", default="custom",
        choices=["openai", "deepseek", "custom"],
        help="LLM provider",
    )
    single_parser.add_argument(
        "--api-key", default=None,
        help="LLM API key",
    )
    single_parser.add_argument(
        "--model", default=None,
        help="LLM model name",
    )
    single_parser.add_argument(
        "--api-base", default=None,
        help="LLM API base URL",
    )

    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Interactive multi-agent chat")
    chat_parser.add_argument("--db", dest="chat_db", default=DEFAULT_CHAT_SESSION_DB)
    chat_parser.add_argument(
        "--mode", choices=["structured", "text"], default="structured",
    )
    chat_parser.add_argument(
        "--provider", default="deepseek", choices=["openai", "deepseek", "custom"],
    )
    chat_parser.add_argument("--model", default=None)

    # Stats command
    subparsers.add_parser("stats", help="Show system stats from last run")

    # Demo command (quick demo without LLM)
    demo_parser = subparsers.add_parser("demo", help="Run quick demo with mock data")
    demo_parser.add_argument(
        "--groups", nargs="+",
        default=["energy_research"],
        choices=list(ALL_TASK_GROUPS.keys()),
    )
    demo_parser.add_argument(
        "--provider", default="custom",
        choices=["openai", "deepseek", "custom"],
        help="LLM provider (use real LLM instead of mock)",
    )
    demo_parser.add_argument(
        "--api-key", default=None,
        help="LLM API key (enables real LLM mode)",
    )
    demo_parser.add_argument(
        "--model", default=None,
        help="LLM model name",
    )
    demo_parser.add_argument(
        "--api-base", default=None,
        help="LLM API base URL",
    )

    return parser.parse_args()


def cmd_experiment(args):
    """Run full comparison experiment."""
    if args.mock:
        # Use MockLLM for offline testing
        class MockLLM:
            def __init__(self):
                self.call_count = 0
                self.last_elapsed_ms = 0.0
            def chat(self, messages, **kwargs):
                self.call_count += 1
                return f"[Mock #{self.call_count}] Processing: {messages[-1]['content'][:100]}..."
            def chat_structured(self, messages, output_format, **kwargs):
                self.call_count += 1
                return {
                    "plan_id": f"plan_{self.call_count}",
                    "task_description": messages[-1]['content'][:200],
                    "subtasks": [
                        {"step": 1, "description": "Search for relevant information",
                         "agent_role": "retriever", "action": "retrieve",
                         "params": {"query": messages[-1]['content'][:100]}, "depends_on": []},
                        {"step": 2, "description": "Process and analyze data",
                         "agent_role": "executor", "action": "execute",
                         "params": {"input": "Process retrieved data"}, "depends_on": [1]},
                        {"step": 3, "description": "Generate summary report",
                         "agent_role": "summarizer", "action": "summarize",
                         "params": {"input": "Summarize all findings"}, "depends_on": [2]},
                    ],
                    "expected_outcome": f"Completed: {messages[-1]['content'][:100]}",
                }
            def get_usage_stats(self):
                return {"call_count": self.call_count, "total_prompt_tokens": 0,
                        "total_completion_tokens": 0, "total_tokens": 0,
                        "total_cached_tokens": 0, "cache_hit_rate": 0, "last_usage": {}}
        llm = MockLLM()
    else:
        llm = _build_llm(args)

    results = run_full_experiment(
        llm=llm,
        use_real_embeddings=not args.no_real_embeddings,
        sandbox_enabled=not args.no_sandbox,
        task_groups=args.groups,
    )

    print("\nExperiment complete!")
    print(f"Results saved to {result_path('metrics_report.json')}")

    # Print comparison table
    orchestrator = Orchestrator(
        llm=llm,
        use_real_embeddings=not args.no_real_embeddings,
    )
    reporter = Reporter(orchestrator.metrics)
    print("\n" + reporter.generate_comparison_table())
    orchestrator.cleanup()
    return 0


def cmd_continuous(args):
    """Run continuous tasks for stability validation."""
    llm = _build_llm(args)

    results = run_continuous_tasks(
        num_tasks=args.num_tasks,
        llm=llm,
        use_real_embeddings=not args.no_real_embeddings,
    )

    print("\nContinuous test complete!")
    print(f"Ran {args.num_tasks} tasks successfully")
    memory_stats = results.get("memory_stats", {})
    print(f"Total memories accumulated: {memory_stats.get('total_memories', 0)}")
    return 0


def cmd_single(args):
    """Run a single task."""
    llm = _build_llm(args)
    orchestrator = Orchestrator(llm=llm, mode=args.mode)

    if args.mode == "text":
        result = orchestrator.execute_task_text_mode(
            task_id=f"single_{int(__import__('time').time())}",
            task_description=args.task,
            tags=args.tags,
        )
    else:
        result = orchestrator.execute_task(
            task_id=f"single_{int(__import__('time').time())}",
            task_description=args.task,
            tags=args.tags,
        )

    print("\n" + "=" * 60)
    print("  Task Result")
    print("=" * 60)
    print(f"  Mode: {args.mode}")
    print(f"  Elapsed: {result.get('elapsed_ms', 0):.1f} ms")
    print()

    steps = result.get("steps", {})
    if "plan" in steps:
        plan = steps["plan"]
        print(f"  Plan: {plan.get('expected_outcome', 'N/A')}")
        for st in plan.get("subtasks", []):
            print(f"    Step {st['step']}: [{st['agent_role']}] {st['description'][:80]}")

    if "summary" in steps:
        summary = steps["summary"].get("summary", "N/A")
        print(f"\n  Final Summary:\n  {summary}")

    print(f"\n  Memory refs: {steps.get('summary', {}).get('evidence_refs', [])}")

    # Show stats
    reporter = Reporter(orchestrator.metrics)
    print("\n" + orchestrator.print_system_status())
    print("\n" + reporter.generate_comparison_table())

    json_path = reporter.export_json(result_path("single_task_metrics.json"))
    print(f"\n  Metrics saved to: {json_path}")
    orchestrator.cleanup()
    return 0


def cmd_stats(args):
    """Show system statistics."""
    orchestrator = Orchestrator()
    print(orchestrator.print_system_status())

    mem_stats = orchestrator.memory_store.get_stats()
    print(f"\n  Memory by Type: {mem_stats.get('by_type', {})}")
    print(f"  Memory by Topic: {mem_stats.get('by_topic', {})}")
    return 0


def cmd_demo(args):
    """Run a quick demo.

    Without --api-key: uses mock LLM and deterministic embeddings.
    With --api-key: uses real LLM for full multi-agent experience.
    """
    use_real_llm = bool(args.api_key or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY"))

    print("=" * 70)
    print("  DEMO: Multi-Agent Collaboration System")
    if use_real_llm:
        print(f"  LLM: {args.provider} / {args.model or 'default'}")
    else:
        print("  LLM: Mock (offline mode)")
    print("  Mode: Structured Protocol + Non-Text State + Shared Memory")
    print("=" * 70)

    if use_real_llm:
        llm = _build_llm(args)
        use_real_emb = True
    else:
        # Use mock LLM (returns deterministic responses)
        class MockLLM:
            def __init__(self):
                self.call_count = 0
                self.last_elapsed_ms = 0.0

            def chat(self, messages, **kwargs):
                self.call_count += 1
                return f"[Mock LLM Response #{self.call_count}] Processing: {messages[-1]['content'][:100]}..."

            def chat_structured(self, messages, output_format, **kwargs):
                self.call_count += 1
                return {
                    "plan_id": f"plan_demo_{self.call_count}",
                    "task_description": messages[-1]['content'][:200],
                    "subtasks": [
                        {
                            "step": 1,
                            "description": "Search for relevant information",
                            "agent_role": "retriever",
                            "action": "retrieve",
                            "params": {"query": messages[-1]['content'][:100]},
                            "depends_on": [],
                        },
                        {
                            "step": 2,
                            "description": "Process and analyze data",
                            "agent_role": "executor",
                            "action": "execute",
                            "params": {"input": "Process retrieved data"},
                            "depends_on": [1],
                        },
                        {
                            "step": 3,
                            "description": "Generate summary report",
                            "agent_role": "summarizer",
                            "action": "summarize",
                            "params": {"input": "Summarize all findings"},
                            "depends_on": [2],
                        },
                    ],
                    "expected_outcome": f"Completed analysis for: {messages[-1]['content'][:100]}",
                }

            def get_usage_stats(self):
                return {"call_count": self.call_count, "total_prompt_tokens": 0,
                        "total_completion_tokens": 0, "total_tokens": 0,
                        "total_cached_tokens": 0, "cache_hit_rate": 0, "last_usage": {}}

        llm = MockLLM()
        use_real_emb = False

    orchestrator = Orchestrator(
        llm=llm,
        mode="structured",
        use_real_embeddings=use_real_emb,
        sandbox_enabled=True,
    )

    for group_name in args.groups:
        tasks = ALL_TASK_GROUPS[group_name]
        print(f"\n{'='*60}")
        print(f"  Task Group: {group_name} ({len(tasks)} tasks)")
        print(f"{'='*60}")

        # Run in structured mode
        print("\n  [Structured Protocol Mode]")
        for i, task in enumerate(tasks):
            print(f"\n  --- Task {i+1}: {task['description'][:80]}...")
            result = orchestrator.execute_task(
                task_id=task["task_id"],
                task_description=task["description"],
                tags=task.get("tags", []),
            )

            plan = result.get("steps", {}).get("plan", {})
            summary = result.get("steps", {}).get("summary", {})

            print(f"    Plan: {plan.get('expected_outcome', 'N/A')[:100]}")
            print(f"    Subtasks: {len(plan.get('subtasks', []))}")

            # Show memory reuse
            mem_refs = plan.get("memory_refs", [])
            if mem_refs:
                print(f"    Memory references used: {len(mem_refs)}")
                for ref in mem_refs[:3]:
                    mem = orchestrator.memory_store.get(ref)
                    if mem:
                        print(f"      - [{mem.memory_type}] {mem.summary[:80]}")

            if summary:
                inner = summary.get('summary', summary)
                if isinstance(inner, dict):
                    text = inner.get('conclusion', '') or '; '.join(
                        inner.get('key_findings', [])[:2]
                    ) or str(inner)
                else:
                    text = str(inner)
                print(f"    Summary: {text[:150]}")

            print(f"    Latency: {result.get('elapsed_ms', 0):.1f} ms")

        # Show system stats
        print("\n" + orchestrator.print_system_status())

    # Export
    reporter = Reporter(orchestrator.metrics)
    json_path = reporter.export_json(result_path("demo_metrics.json"))
    print(f"\n  Demo metrics exported to: {json_path}")
    print(f"  Memory database: {DEFAULT_SHARED_MEMORY_DB}")

    return 0


def cmd_chat(args):
    """Launch interactive chat (same as python chat.py)."""
    import chat as chat_module

    argv = ["chat.py", "--db", args.chat_db, "--mode", args.mode, "--provider", args.provider]
    if args.model:
        argv.extend(["--model", args.model])
    old = sys.argv
    sys.argv = argv
    try:
        return chat_module.main()
    finally:
        sys.argv = old


def main():
    args = parse_args()

    commands = {
        "experiment": cmd_experiment,
        "continuous": cmd_continuous,
        "single": cmd_single,
        "chat": cmd_chat,
        "stats": cmd_stats,
        "demo": cmd_demo,
    }

    if args.command in commands:
        return commands[args.command](args)
    else:
        print("Please specify a command. Use --help for usage.")
        print("Quick start: python main.py demo")
        return 1


if __name__ == "__main__":
    sys.exit(main())
