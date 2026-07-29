"""Comprehensive benchmark: all optimizations (P0+P1+P3), bge embedding, tiktoken.

Measures:
1. E2E cache hit rate (exact + domain)
2. LLM call distribution per task
3. Accurate token counts via tiktoken (cl100k_base)
4. Latency per task
5. Cross-task memory reuse
"""

import json
import os
import sys
import time
import tiktoken

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.orchestrator import Orchestrator

# tiktoken encoder (cl100k_base compatible with DeepSeek/GPT-4)
ENCODER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(ENCODER.encode(text))


# 5-task benchmark
BENCHMARK_TASKS = [
    {
        "task_id": "b1_solar_research",
        "description": "Research and summarize solar energy technologies including photovoltaic types, "
                       "efficiency metrics, costs, and environmental impact. Create a comprehensive report.",
        "tags": ["solar", "renewable", "energy", "research"],
    },
    {
        "task_id": "b2_solar_variant",
        "description": "Research solar energy technologies focusing on photovoltaic efficiency improvements, "
                       "cost trends, and environmental considerations. Produce a detailed summary.",
        "tags": ["solar", "renewable", "energy", "photovoltaic", "research"],
    },
    {
        "task_id": "b3_wind_vs_solar",
        "description": "Research wind energy technologies and compare them with solar energy. "
                       "Analyze which technology is better suited for different geographic regions and use cases. "
                       "Include cost comparison, capacity factors, and environmental trade-offs.",
        "tags": ["wind", "solar", "renewable", "comparison", "energy"],
    },
    {
        "task_id": "b4_code_security",
        "description": "Analyze Python code for common security vulnerabilities. Search for SQL injection, "
                       "command injection, hardcoded credentials, path traversal, and insecure deserialization. "
                       "Document detection methods and recommended fixes.",
        "tags": ["security", "code", "python", "vulnerability"],
    },
    {
        "task_id": "b5_code_security_variant",
        "description": "Review Python applications for security vulnerabilities including SQL injection, "
                       "command injection, credential exposure, path traversal, and deserialization issues. "
                       "Provide detection methods and remediation guidance.",
        "tags": ["security", "code", "python", "vulnerability", "audit"],
    },
]


class TiktokenMockLLM:
    """Mock LLM with tiktoken-based accurate token counting."""

    def __init__(self):
        self.chat_calls = 0
        self.structured_calls = 0
        self.judge_calls = 0
        self.judge_yes = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def _count(self, text: str) -> int:
        return count_tokens(text)

    def chat(self, messages, **kwargs):
        self.chat_calls += 1
        content_text = messages[-1].get("content", "")
        self.prompt_tokens += self._count(content_text)

        # Judge detection
        if "YES/NO" in content_text:
            self.judge_calls += 1
            # Extract task A and task B from prompt for fair comparison
            task_a = ""
            task_b = ""
            if "Task A:" in content_text and "Task B:" in content_text:
                try:
                    task_a = content_text.split("Task A:")[1].split("\n")[0].strip()
                    task_b = content_text.split("Task B:")[1].split("\n")[0].strip()
                except Exception:
                    pass
            # Compare keywords across BOTH tasks (both must contain the kw)
            domain_kw = {
                "solar", "wind", "energy", "photovoltaic", "renewable",  # energy domain
                "security", "code", "python", "vulnerability", "sql", "injection", "audit",  # security domain
            }
            a_kw = set(task_a.lower().split())
            b_kw = set(task_b.lower().split())
            domain_a = a_kw & domain_kw
            domain_b = b_kw & domain_kw
            yes = len(domain_a) > 0 and len(domain_b) > 0 and len(domain_a & domain_b) > 0
            self.judge_yes += 1 if yes else 0
            resp = "YES" if yes else "NO"
            self.completion_tokens += self._count(resp)
            return resp

        resp = f"[Mock Chat #{self.chat_calls}] Processed request."
        self.completion_tokens += self._count(resp)
        return resp

    def chat_structured(self, messages, output_format, **kwargs):
        self.structured_calls += 1
        content_text = messages[-1].get("content", "")
        self.prompt_tokens += self._count(content_text)

        # Check for summarizer or planner
        is_summarizer = any(
            "key_findings" in m.get("content", "") and "Summarize" in m.get("content", "")
            for m in messages
        )
        is_planner = "subtasks" in str(output_format) or "Plan" in messages[0].get("content", "")

        if is_summarizer:
            resp = {
                "key_findings": [
                    "Renewable energy technologies continue to advance rapidly",
                    "Cost reductions have made solar and wind competitive with fossil fuels",
                ],
                "facts": ["Solar PV efficiency ranges 15-22%", "Wind capacity factors reach 25-55%"],
                "conclusion": "Both solar and wind energy are viable for large-scale deployment",
            }
        else:
            resp = {
                "plan_id": f"plan_{self.structured_calls}",
                "task_description": content_text[:200],
                "subtasks": [
                    {"step": 1, "description": "Search for relevant information",
                     "agent_role": "retriever", "action": "retrieve",
                     "params": {"query": content_text[:100]}, "depends_on": []},
                    {"step": 2, "description": "Process and analyze retrieved data",
                     "agent_role": "executor", "action": "execute",
                     "params": {"input": "Process retrieved information"}, "depends_on": [1]},
                    {"step": 3, "description": "Generate final summary report",
                     "agent_role": "summarizer", "action": "summarize",
                     "params": {"input": "Summarize all findings"}, "depends_on": [2]},
                ],
                "expected_outcome": f"Comprehensive report on: {content_text[:80]}",
            }

        resp_json = json.dumps(resp, ensure_ascii=False)
        self.completion_tokens += self._count(resp_json)
        return resp

    def get_usage_stats(self):
        return {
            "call_count": self.chat_calls + self.structured_calls,
            "total_prompt_tokens": self.prompt_tokens,
            "total_completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "total_cached_tokens": 0,
            "cache_hit_rate": 0,
            "last_usage": {},
        }

    def reset_counts(self):
        self.chat_calls = 0
        self.structured_calls = 0
        self.judge_calls = 0
        self.judge_yes = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0


def clear_db():
    for path in ["_benchmark.db", "_benchmark.db-shm", "_benchmark.db-wal"]:
        try:
            os.unlink(path)
        except OSError:
            pass


def run_benchmark():
    clear_db()
    print("=" * 80)
    print("  BENCHMARK: P0+P1+P3 + BGE Embedding + tiktoken")
    print("=" * 80)

    llm = TiktokenMockLLM()
    orch = Orchestrator(
        llm=llm,
        mode="structured",
        use_real_embeddings=True,
        sandbox_enabled=True,
        memory_db_path="_benchmark.db",
    )

    results = []
    total_start = time.time()

    for i, task in enumerate(BENCHMARK_TASKS):
        desc_short = task["description"][:85]
        print(f"\n  [{i+1}/5] {task['task_id']}: {desc_short}...")

        llm.reset_counts()
        t0 = time.time()
        result = orch.execute_task(
            task_id=task["task_id"],
            task_description=task["description"],
            tags=task.get("tags", []),
        )
        elapsed = time.time() - t0

        e2e_cached = result.get("_e2e_cached", False)
        e2e_match = result.get("_e2e_match", "")
        e2e_domain = result.get("_e2e_domain_match", False)
        plan = result.get("steps", {}).get("plan", {})
        from_template = plan.get("_from_template", False)
        template_filled = plan.get("_template_filled", False)
        llm_skipped = plan.get("llm_skipped", False)

        # Determine strategy
        if e2e_cached and e2e_match == "exact":
            strategy = "E2E EXACT"
            detail = "full reuse, 0 agent calls"
        elif e2e_domain:
            strategy = "E2E DOMAIN"
            detail = f"plan template, cos={result.get('_e2e_domain_score', 0):.3f}"
        elif from_template:
            strategy = "P1 TEMPLATE"
            detail = f"from={plan.get('_template_from', '')[:20]}"
        elif template_filled:
            strategy = "TEMPLATE FILL"
            detail = "LLM adapted template"
        elif llm_skipped:
            strategy = "PLAN REUSED"
            detail = ""
        else:
            strategy = "FULL GENERATION"
            detail = ""

        agent_calls = llm.chat_calls + llm.structured_calls
        tokens = llm.prompt_tokens + llm.completion_tokens

        print(f"    → {strategy} | agent={agent_calls} judge={llm.judge_calls} | "
              f"prompt={llm.prompt_tokens} compl={llm.completion_tokens} total={tokens} | "
              f"{elapsed*1000:.0f}ms")
        if detail:
            print(f"      {detail}")

        results.append({
            "task_id": task["task_id"],
            "strategy": strategy,
            "detail": detail,
            "llm_calls": agent_calls,
            "judge_calls": llm.judge_calls,
            "prompt_tokens": llm.prompt_tokens,
            "completion_tokens": llm.completion_tokens,
            "total_tokens": tokens,
            "elapsed_ms": elapsed * 1000,
            "e2e_cached": e2e_cached,
            "e2e_match": e2e_match,
            "e2e_domain": e2e_domain,
            "from_template": from_template,
            "template_filled": template_filled,
        })

        # Memory stats snapshot
        mems = orch.memory_store.get_stats()["total_memories"]
        print(f"      memory store: {mems} total")

    total_elapsed = time.time() - total_start

    # Summary
    print("\n" + "=" * 80)
    print("  BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"\n  {'Task':<28} {'Strategy':<22} {'LLM':>4} {'Judge':>6} {'Prompt':>8} {'Compl':>8} {'Total':>8} {'Lat':>8}")
    print(f"  {'─'*28} {'─'*22} {'─'*4} {'─'*6} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    for r in results:
        print(f"  {r['task_id']:<28} {r['strategy']:<22} {r['llm_calls']:>4} {r['judge_calls']:>6} "
              f"{r['prompt_tokens']:>8} {r['completion_tokens']:>8} {r['total_tokens']:>8} {r['elapsed_ms']:>7.0f}ms")

    total_llm = sum(r["llm_calls"] for r in results)
    total_judge = sum(r["judge_calls"] for r in results)
    total_prompt = sum(r["prompt_tokens"] for r in results)
    total_compl = sum(r["completion_tokens"] for r in results)
    total_tok = sum(r["total_tokens"] for r in results)
    total_lat = sum(r["elapsed_ms"] for r in results)

    print(f"  {'─'*28} {'─'*22} {'─'*4} {'─'*6} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    print(f"  {'TOTAL':<28} {'':<22} {total_llm:>4} {total_judge:>6} "
          f"{total_prompt:>8} {total_compl:>8} {total_tok:>8} {total_lat:>7.0f}ms")

    # Strategy distribution
    from collections import Counter
    strat_counts = Counter(r["strategy"] for r in results)
    print(f"\n  Strategy distribution: {dict(strat_counts)}")
    exact_hits = sum(1 for r in results if r["e2e_match"] == "exact")
    domain_hits = sum(1 for r in results if r["e2e_domain"])
    template_hits = sum(1 for r in results if r["from_template"])
    full_gens = sum(1 for r in results if r["strategy"] == "FULL GENERATION")
    print(f"  E2E exact hits:  {exact_hits}/5")
    print(f"  E2E domain hits: {domain_hits}/5")
    print(f"  P1 template hits:{template_hits}/5")
    print(f"  Full generations:{full_gens}/5")

    # Token efficiency estimate
    # Estimate text-mode equivalent: each task ~3000 tokens (agent text communication)
    text_equiv = 5 * 3000  # rough estimate
    print(f"\n  Token estimate vs text equivalent:")
    print(f"    Structured (tiktoken): {total_tok}")
    print(f"    Text equivalent (est): {text_equiv}")
    if text_equiv > 0:
        print(f"    Savings: {(1 - total_tok/text_equiv)*100:.1f}%")

    # Export
    out = {
        "benchmark": "P0+P1+P3 + BGE + tiktoken",
        "results": results,
        "totals": {
            "llm_calls": total_llm,
            "judge_calls": total_judge,
            "prompt_tokens": total_prompt,
            "completion_tokens": total_compl,
            "total_tokens": total_tok,
            "elapsed_ms": total_lat,
        },
        "strategy_distribution": dict(strat_counts),
    }
    with open("benchmark_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results exported to: benchmark_results.json")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
