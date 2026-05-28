"""P1: Hierarchical memory experiment.

Validates:
1. After 3+ concrete strategy memories accumulate in a domain, a template is created
2. The template is stored with abstraction_level=1
3. Planner uses the template preferentially for new tasks in the same domain
4. Template fill is cheaper than full LLM generation

Since E2E cache often hits before P1 can demonstrate value, this experiment
disables E2E cache by using tasks with cos < 0.50 to ensure plan generation
proceeds, so we can observe template accumulation and promotion.
"""

import json
import os
import sys
import time
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.base import LLMBackend
from src.agents.planner import PlannerAgent
from src.protocol.scheduler import Scheduler, AgentRegistry, MessageBus
from src.state.embeddings import EmbeddingEngine
from src.state.exchange import StateExchangeBus
from src.memory.store import MemoryStore
from src.orchestrator import Orchestrator


class LightMockLLM:
    """Minimal mock that returns deterministic structured responses."""

    def __init__(self):
        self.chat_calls = 0
        self.structured_calls = 0
        self.prompt_est = 0
        self.compl_est = 0

    def _est(self, text):
        return len(text) // 3

    def chat(self, messages, **kwargs):
        self.chat_calls += 1
        content = messages[-1].get("content", "")
        self.prompt_est += self._est(content)
        if "YES/NO" in content:
            # Similarity judge
            resp = "YES" if any(
                kw in content.lower()
                for kw in ["solar", "energy", "security", "code", "python"]
            ) else "NO"
            self.compl_est += 3
            return resp
        # General chat
        resp = f"[Mock #{self.chat_calls}] Response"
        self.compl_est += self._est(resp)
        return resp

    def chat_structured(self, messages, output_format, **kwargs):
        self.structured_calls += 1
        content = messages[-1].get("content", "")
        self.prompt_est += self._est(content)

        # Check for template creation prompt
        if "Domain template→JSON" in messages[0].get("content", ""):
            resp = {
                "template_for": "solar_energy_research",
                "common_tags": ["solar", "energy"],
                "subtask_pattern": [
                    {
                        "step": 1,
                        "role": "retriever",
                        "action": "retrieve",
                        "params_template": {"query": "{topic} technology details"},
                    },
                    {
                        "step": 2,
                        "role": "executor",
                        "action": "execute",
                        "params_template": {"input": "Process {topic} data"},
                    },
                    {
                        "step": 3,
                        "role": "summarizer",
                        "action": "summarize",
                        "params_template": {"input": "Summarize {topic} findings"},
                    },
                ],
                "notes": "Standard research pipeline for energy technologies",
            }
        elif "Adapt plan template" in messages[0].get("content", ""):
            # Template fill — LLM adapts template
            resp = {
                "plan_id": f"plan_template_fill_{self.structured_calls}",
                "task_description": content[:200],
                "subtasks": [
                    {
                        "step": 1,
                        "description": f"Search for: {content[:60]}",
                        "agent_role": "retriever",
                        "action": "retrieve",
                        "params": {"query": content[:80]},
                        "depends_on": [],
                    },
                    {
                        "step": 2,
                        "description": "Process retrieved data",
                        "agent_role": "executor",
                        "action": "execute",
                        "params": {"input": "Analyze data"},
                        "depends_on": [1],
                    },
                    {
                        "step": 3,
                        "description": "Generate summary",
                        "agent_role": "summarizer",
                        "action": "summarize",
                        "params": {"input": "Summarize"},
                        "depends_on": [2],
                    },
                ],
                "expected_outcome": f"Report: {content[:80]}",
            }
        else:
            # Full plan generation
            resp = {
                "plan_id": f"plan_full_{self.structured_calls}",
                "task_description": content[:200],
                "subtasks": [
                    {
                        "step": 1,
                        "description": f"Search for information about: {content[:80]}",
                        "agent_role": "retriever",
                        "action": "retrieve",
                        "params": {"query": content[:100]},
                        "depends_on": [],
                    },
                    {
                        "step": 2,
                        "description": "Process and analyze retrieved information",
                        "agent_role": "executor",
                        "action": "execute",
                        "params": {"input": "Process retrieved data"},
                        "depends_on": [1],
                    },
                    {
                        "step": 3,
                        "description": "Generate summary report from processed data",
                        "agent_role": "summarizer",
                        "action": "summarize",
                        "params": {"input": "Create report from results"},
                        "depends_on": [2],
                    },
                ],
                "expected_outcome": f"Completed: {content[:80]}",
            }
        self.compl_est += len(json.dumps(resp)) // 2
        return resp

    def get_usage_stats(self):
        return {
            "call_count": self.chat_calls + self.structured_calls,
            "total_prompt_tokens": self.prompt_est,
            "total_completion_tokens": self.compl_est,
            "total_tokens": self.prompt_est + self.compl_est,
            "total_cached_tokens": 0,
            "cache_hit_rate": 0,
            "last_usage": {},
        }


def clear_db():
    for path in ["_p1_test.db", "_p1_test.db-shm", "_p1_test.db-wal"]:
        try:
            os.unlink(path)
        except OSError:
            pass


def create_standalone_planner(db_path, llm_mock, embed_engine):
    """Create a standalone PlannerAgent for testing template promotion."""
    registry = AgentRegistry()
    bus = MessageBus()
    scheduler = Scheduler(registry, bus)
    state_bus = StateExchangeBus(embed_engine)
    store = MemoryStore(db_path=db_path)

    planner = PlannerAgent(
        scheduler=scheduler,
        llm=llm_mock,
        embedding_engine=embed_engine,
        state_bus=state_bus,
        memory_store=store,
        use_structured_protocol=True,
    )
    return planner, store


def run_p1_experiment():
    clear_db()
    print("=" * 70)
    print("  P1: Hierarchical Memory Experiment")
    print("  Domain Templates from Concrete Memory Accumulation")
    print("=" * 70)

    embed_engine = EmbeddingEngine(use_real_model=True)
    llm = LightMockLLM()

    # Phase 1: Accumulate 3 solar-energy strategies
    print("\n" + "─" * 50)
    print("  Phase 1: Accumulate concrete strategy memories")
    print("─" * 50)

    solar_tasks = [
        {
            "task_id": "solar-1",
            "description": "Research photovoltaic cell types, manufacturing processes, and efficiency comparisons",
            "tags": ["solar", "energy", "research", "photovoltaic"],
        },
        {
            "task_id": "solar-2",
            "description": "Analyze solar energy cost trends from 2010 to 2025 including panel prices, installation, and LCOE",
            "tags": ["solar", "energy", "cost", "analysis"],
        },
        {
            "task_id": "solar-3",
            "description": "Evaluate environmental impacts of large-scale solar farms including land use and biodiversity",
            "tags": ["solar", "energy", "environmental", "impact"],
        },
    ]

    planner, store = create_standalone_planner("_p1_test.db", llm, embed_engine)

    for i, task in enumerate(solar_tasks):
        print(f"\n  Task {i+1}: {task['description'][:70]}...")
        result = planner.execute_task({
            "task_description": task["description"],
            "task_id": task["task_id"],
            "tags": task["tags"],
        })
        reuse = "reuse" if result.get("llm_skipped") else "gen"
        templ = f", template={result.get('_from_template', False)}" if result.get("_from_template") else ""
        promoted = f", promoted={result.get('_promoted_template', '')[:20]}" if result.get("_promoted_template") else ""
        print(f"    → {reuse}{templ}{promoted}")
        print(f"    Memory count: {store.get_stats()['total_memories']}")
        print(f"    Abstraction levels: {store.get_stats().get('by_type', {})}")

    # Check if template was promoted
    stats = store.get_stats()
    print(f"\n  After Phase 1:")
    print(f"    Total memories: {stats['total_memories']}")
    print(f"    By type: {stats['by_type']}")

    # Check specifically for templates
    templates = store.get_templates(["solar", "energy"], memory_type="strategy")
    print(f"    Templates found: {len(templates)}")
    for tmem in templates:
        print(f"      - {tmem.task_topic} (level={tmem.abstraction_level})")
        try:
            content = json.loads(tmem.content)
            print(f"        template_for: {content.get('template_for', 'N/A')}")
            print(f"        subtask_pattern: {len(content.get('subtask_pattern', []))} steps")
        except Exception:
            pass

    # Phase 2: Run a 4th solar task — should use template
    print("\n" + "─" * 50)
    print("  Phase 2: New solar task — should use template")
    print("─" * 50)

    llm_before = llm.structured_calls + llm.chat_calls
    task4 = {
        "task_id": "solar-4",
        "description": "Research solar energy policy incentives and their impact on residential photovoltaic adoption rates",
        "tags": ["solar", "energy", "policy", "incentives"],
    }
    print(f"\n  Task 4: {task4['description'][:70]}...")
    result4 = planner.execute_task({
        "task_description": task4["description"],
        "task_id": task4["task_id"],
        "tags": task4["tags"],
    })
    llm_after = llm.structured_calls + llm.chat_calls

    from_template = result4.get("_from_template", False)
    template_id = result4.get("_template_from", "")
    print(f"    _from_template: {from_template}")
    print(f"    _template_from: {template_id}")
    print(f"    template_filled: {result4.get('_template_filled', False)}")
    print(f"    llm_skipped: {result4.get('llm_skipped', False)}")
    print(f"    LLM calls this task: {llm_after - llm_before}")

    # Phase 3: Run a cross-domain task — should NOT use solar template
    print("\n" + "─" * 50)
    print("  Phase 3: Cross-domain task — should NOT use solar template")
    print("─" * 50)

    task5 = {
        "task_id": "security-1",
        "description": "Audit Python code for SQL injection and XSS vulnerabilities",
        "tags": ["security", "code", "python", "audit"],
    }
    print(f"\n  Task 5: {task5['description'][:70]}...")
    result5 = planner.execute_task({
        "task_description": task5["description"],
        "task_id": task5["task_id"],
        "tags": task5["tags"],
    })
    print(f"    _from_template: {result5.get('_from_template', False)}")
    print(f"    llm_skipped: {result5.get('llm_skipped', False)}")

    # Summary
    print("\n" + "=" * 70)
    print("  P1 EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"\n  Template promoted after 3 solar tasks: {'YES' if len(templates) > 0 else 'NO'}")
    print(f"  Templates found: {len(templates)}")
    print(f"  Task 4 used template: {from_template}")
    print(f"  Task 5 (cross-domain) used template: {result5.get('_from_template', False)}")
    print(f"\n  LLM stats: {llm.chat_calls} chat + {llm.structured_calls} structured")
    print(f"  Token estimate: {llm.prompt_est} prompt + {llm.compl_est} compl = {llm.prompt_est + llm.compl_est}")

    final_stats = store.get_stats()
    print(f"\n  Final memory state:")
    print(f"    Total: {final_stats['total_memories']}")
    print(f"    By type: {final_stats['by_type']}")

    # Check all memories' abstraction levels
    from collections import Counter
    level_counts = Counter()
    if store._index is not None:
        for mid in store._idx_to_id.values():
            mem = store.get(mid)
            if mem:
                level_counts[mem.abstraction_level] += 1
    print(f"    By abstraction_level: {dict(level_counts)}")


if __name__ == "__main__":
    run_p1_experiment()
