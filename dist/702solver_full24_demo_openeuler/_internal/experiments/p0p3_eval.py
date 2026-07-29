"""P0+P3 comprehensive evaluation experiment.

Compares the current system (P0: LLM judge + P3: Summarizer semantic cache)
against the Round 4 baseline, with detailed per-task tracing.

5 tasks — same structure as Round 4 experiment:
  t1: Solar energy research (cold start)
  t2: Solar energy research variant → should hit E2E cache (cos>0.90)
  t3: Wind vs solar comparison → should hit E2E cache via LLM judge (P0)
  t4: Code security audit (new domain, full generation)
  t5: Code security audit variant → should hit E2E cache (cos>0.90)
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.conftest import CountingMockLLM
from src.orchestrator import Orchestrator
from src.agents.base import LLMBackend


EVAL_TASKS = [
    {
        "task_id": "t1_solar_research",
        "description": "Research and summarize solar energy technologies including photovoltaic types, "
                       "efficiency metrics, costs, and environmental impact. Create a comprehensive report.",
        "tags": ["solar", "renewable", "energy", "research"],
    },
    {
        "task_id": "t2_solar_variant",
        "description": "Research solar energy technologies focusing on photovoltaic efficiency improvements, "
                       "cost trends, and environmental considerations. Produce a detailed summary.",
        "tags": ["solar", "renewable", "energy", "photovoltaic", "research"],
    },
    {
        "task_id": "t3_wind_vs_solar",
        "description": "Research wind energy technologies and compare them with solar energy. "
                       "Analyze which technology is better suited for different geographic regions and use cases. "
                       "Include cost comparison, capacity factors, and environmental trade-offs.",
        "tags": ["wind", "solar", "renewable", "comparison", "energy"],
    },
    {
        "task_id": "t4_code_security",
        "description": "Analyze Python code for common security vulnerabilities. Search for SQL injection, "
                       "command injection, hardcoded credentials, path traversal, and insecure deserialization. "
                       "Document detection methods and recommended fixes.",
        "tags": ["security", "code", "python", "vulnerability"],
    },
    {
        "task_id": "t5_code_security_variant",
        "description": "Review Python applications for security vulnerabilities including SQL injection, "
                       "command injection, credential exposure, path traversal, and deserialization issues. "
                       "Provide detection methods and remediation guidance.",
        "tags": ["security", "code", "python", "vulnerability", "audit"],
    },
]


# ---------------------------------------------------------------------------
# Token estimation helpers
# ---------------------------------------------------------------------------

def estimate_prompt_tokens(messages):
    """Estimate prompt tokens from a list of messages."""
    total = 0
    for m in messages:
        for key in ("content",):
            total += len(m.get(key, "")) // 3
    return total


def estimate_completion_tokens(response):
    """Estimate completion tokens from response."""
    if isinstance(response, dict):
        return len(json.dumps(response)) // 2
    return len(str(response)) // 2


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

class TokenCountingMockLLM:
    """Mock LLM that counts tokens and produces deterministic responses
    that exercise the full agent pipeline."""

    def __init__(self):
        self.chat_calls = 0
        self.structured_calls = 0
        self.total_prompt_est = 0
        self.total_completion_est = 0
        self.judge_calls = 0
        self.judge_yes_count = 0

    def _estimate_prompt(self, messages):
        t = 0
        for m in messages:
            t += len(m.get("content", "")) // 3
        return max(t, 1)

    def chat(self, messages, **kwargs):
        self.chat_calls += 1
        prompt_est = self._estimate_prompt(messages)
        self.total_prompt_est += prompt_est

        # Detect judge calls
        if any("YES/NO" in m.get("content", "") for m in messages):
            self.judge_calls += 1
            # Simulate LLM similarity judgment:
            # Look at the two task descriptions to decide
            task_a = ""
            task_b = ""
            for m in messages:
                content = m.get("content", "")
                if "Task A:" in content:
                    task_a = content.split("Task A:")[1].split("\n")[0].strip()
                if "Task B:" in content:
                    task_b = content.split("Task B:")[1].split("\n")[0].strip()
            # Compare domain keywords across both tasks (both must share at least one)
            domain_kw = {"solar", "wind", "energy", "photovoltaic", "renewable",
                         "security", "code", "python", "vulnerability", "sql", "injection", "audit"}
            a_set = set(task_a.lower().split())
            b_set = set(task_b.lower().split())
            shared = (a_set & domain_kw) & (b_set & domain_kw)
            yes = len(shared) > 0
            self.judge_yes_count += 1 if yes else 0
            self.total_completion_est += 3
            return "YES" if yes else "NO"

        # Regular chat
        resp = f"[Mock Chat #{self.chat_calls}] Analysis of: {messages[-1].get('content','')[:100]}"
        self.total_completion_est += len(resp) // 3
        return resp

    def chat_structured(self, messages, output_format, **kwargs):
        self.structured_calls += 1
        prompt_est = self._estimate_prompt(messages)
        self.total_prompt_est += prompt_est

        last_content = messages[-1].get("content", "")[:200]

        # Check if this is a summarizer call
        is_summarizer = False
        for m in messages:
            if "Summarize" in m.get("content", "") and "key_findings" in m.get("content", ""):
                is_summarizer = True
                break

        if is_summarizer:
            resp = {
                "key_findings": [
                    "Renewable energy technologies continue to advance rapidly",
                    "Cost reductions have made solar and wind competitive with fossil fuels",
                ],
                "facts": [
                    "Solar PV efficiency ranges 15-22%",
                    "Wind capacity factors reach 25-55%",
                ],
                "conclusion": "Both solar and wind energy are viable for large-scale deployment",
            }
        else:
            resp = {
                "plan_id": f"plan_{self.structured_calls}",
                "task_description": last_content[:200],
                "subtasks": [
                    {
                        "step": 1,
                        "description": "Search for relevant information",
                        "agent_role": "retriever",
                        "action": "retrieve",
                        "params": {"query": last_content[:100]},
                        "depends_on": [],
                    },
                    {
                        "step": 2,
                        "description": "Process and analyze retrieved data",
                        "agent_role": "executor",
                        "action": "execute",
                        "params": {"input": "Process retrieved information"},
                        "depends_on": [1],
                    },
                    {
                        "step": 3,
                        "description": "Generate final summary report",
                        "agent_role": "summarizer",
                        "action": "summarize",
                        "params": {"input": "Summarize all findings"},
                        "depends_on": [2],
                    },
                ],
                "expected_outcome": f"Comprehensive report on: {last_content[:80]}",
            }

        self.total_completion_est += len(json.dumps(resp)) // 2
        return resp

    def get_usage_stats(self):
        return {
            "call_count": self.chat_calls + self.structured_calls,
            "total_prompt_tokens": self.total_prompt_est,
            "total_completion_tokens": self.total_completion_est,
            "total_tokens": self.total_prompt_est + self.total_completion_est,
            "total_cached_tokens": 0,
            "cache_hit_rate": 0,
            "last_usage": {},
        }


def clear_memory_db():
    """Remove the memory database for a clean start."""
    for path in ["shared_memory.db", "shared_memory.db-shm", "shared_memory.db-wal"]:
        try:
            os.unlink(path)
        except OSError:
            pass


def run():
    clear_memory_db()

    print("=" * 80)
    print("  P0 + P3 Comprehensive Evaluation Experiment")
    print("  P0: FAISS coarse recall → LLM fine-grained task similarity judgment")
    print("  P3: Summarizer cache with semantic matching (replaces Jaccard)")
    print("=" * 80)
    print()

    llm = TokenCountingMockLLM()
    orch = Orchestrator(
        llm=llm,
        mode="structured",
        use_real_embeddings=True,
        sandbox_enabled=True,
    )

    results = []
    total_start = time.time()

    for i, task in enumerate(EVAL_TASKS):
        print(f"\n{'─' * 70}")
        print(f"  Task {i+1}/{len(EVAL_TASKS)}: {task['task_id']}")
        print(f"  Description: {task['description'][:90]}...")
        print(f"{'─' * 70}")

        llm_before_chat = llm.chat_calls
        llm_before_struct = llm.structured_calls
        llm_before_judge = llm.judge_calls
        prompt_before = llm.total_prompt_est
        compl_before = llm.total_completion_est

        t0 = time.time()
        result = orch.execute_task(
            task_id=task["task_id"],
            task_description=task["description"],
            tags=task.get("tags", []),
        )
        elapsed = time.time() - t0

        # Determine what happened
        e2e_cached = result.get("_e2e_cached", False)
        e2e_score = result.get("_e2e_score", 0)
        e2e_judge = result.get("_e2e_judge", "")
        steps = result.get("steps", {})
        plan_info = steps.get("plan", {})
        summary_info = steps.get("summary", {})
        plan_reused = plan_info.get("_reused_from", "")
        plan_template = plan_info.get("_template_filled", False)
        plan_template_from = plan_info.get("_template_from", "")
        plan_llm_judged = plan_info.get("_llm_judged", False)
        llm_skipped = plan_info.get("llm_skipped", False)

        # Summarizer cache
        summ_cached = summary_info.get("_summ_cached_from", "")

        # Token deltas for this task
        delta_chat = llm.chat_calls - llm_before_chat
        delta_struct = llm.structured_calls - llm_before_struct
        delta_judge = llm.judge_calls - llm_before_judge
        delta_prompt = llm.total_prompt_est - prompt_before
        delta_compl = llm.total_completion_est - compl_before

        # What happened
        if e2e_cached:
            mode = "E2E CACHE"
            detail = f"(score={e2e_score:.3f}, via={e2e_judge})"
        elif plan_template:
            mode = "TEMPLATE FILL"
            detail = f"(from={plan_template_from[:16]}, score={plan_info.get('_template_score', 0):.3f}"
            if plan_llm_judged:
                detail += ", llm_judged=YES"
            detail += ")"
        elif plan_reused:
            mode = "PLAN REUSED"
            detail = f"(from={plan_reused[:16]})"
        else:
            mode = "FULL GENERATION"
            detail = ""

        if summ_cached:
            mode += " + SUMM_CACHED"

        print(f"  Result: {mode} {detail}")
        print(f"  Latency: {elapsed*1000:.1f}ms")
        print(f"  LLM calls this task: {delta_chat} chat + {delta_struct} struct "
              f"+ {delta_judge} judge = {delta_chat + delta_struct} total")
        print(f"  Tokens this task: {delta_prompt} prompt + {delta_compl} completion "
              f"= {delta_prompt + delta_compl}")
        print(f"  Plan llm_skipped={llm_skipped}, template_filled={plan_template}")

        mem_stats = orch.memory_store.get_stats()
        print(f"  Memory store: {mem_stats['total_memories']} total, "
              f"vector index: {mem_stats['vector_index_size']}")

        results.append({
            "task_id": task["task_id"],
            "mode": mode,
            "detail": detail,
            "e2e_cached": e2e_cached,
            "e2e_score": e2e_score,
            "e2e_judge": e2e_judge,
            "plan_reused": bool(plan_reused),
            "plan_template": plan_template,
            "llm_judged": plan_llm_judged,
            "summ_cached": bool(summ_cached),
            "llm_chat_calls": delta_chat,
            "llm_struct_calls": delta_struct,
            "judge_calls": delta_judge,
            "prompt_tokens": delta_prompt,
            "completion_tokens": delta_compl,
            "total_tokens": delta_prompt + delta_compl,
            "elapsed_ms": elapsed * 1000,
        })

    total_elapsed = time.time() - total_start

    # ---------- Summary ----------
    print()
    print("=" * 80)
    print("  EXPERIMENT SUMMARY")
    print("=" * 80)

    total_chat = sum(r["llm_chat_calls"] for r in results)
    total_struct = sum(r["llm_struct_calls"] for r in results)
    total_judge = sum(r["judge_calls"] for r in results)
    total_agent_llm = total_chat + total_struct
    total_prompt = sum(r["prompt_tokens"] for r in results)
    total_compl = sum(r["completion_tokens"] for r in results)
    total_tokens = total_prompt + total_compl
    e2e_hits = sum(1 for r in results if r["e2e_cached"])
    template_hits = sum(1 for r in results if r["plan_template"])
    full_gens = sum(1 for r in results if r["mode"].startswith("FULL GENERATION"))
    judge_hits = sum(1 for r in results if r["e2e_judge"] == "llm")
    summ_cache_hits = sum(1 for r in results if r["summ_cached"])

    print(f"\n  Task Results:")
    print(f"  {'Task':<28} {'Strategy':<35} {'LLM':>5} {'Prompt':>8} {'Compl':>7} {'Total':>8} {'Lat':>8}")
    print(f"  {'─'*28} {'─'*35} {'─'*5} {'─'*8} {'─'*7} {'─'*8} {'─'*8}")
    for r in results:
        strategy = r["mode"]
        print(f"  {r['task_id']:<28} {strategy:<35} {r['llm_chat_calls']+r['llm_struct_calls']:>5} "
              f"{r['prompt_tokens']:>8} {r['completion_tokens']:>7} {r['total_tokens']:>8} "
              f"{r['elapsed_ms']:>7.0f}ms")

    print(f"\n  {'─'*28} {'─'*35} {'─'*5} {'─'*8} {'─'*7} {'─'*8} {'─'*8}")
    print(f"  {'TOTAL':<28} {'':<35} {total_agent_llm:>5} {total_prompt:>8} {total_compl:>7} "
          f"{total_tokens:>8} {total_elapsed*1000:>7.0f}ms")

    print(f"\n  Cache Summary:")
    print(f"    E2E cache hits:    {e2e_hits}/{len(results)} "
          f"({e2e_hits/len(results)*100:.0f}%)")
    print(f"    LLM judge hits:    {judge_hits}")
    print(f"    Template fills:    {template_hits}")
    print(f"    Summarizer caches: {summ_cache_hits}")
    print(f"    Full generations:  {full_gens}")

    print(f"\n  LLM Usage:")
    print(f"    Total LLM calls:  {total_agent_llm} (agent) + {total_judge} (judge) "
          f"= {total_agent_llm + total_judge}")
    print(f"    Judge calls:      {total_judge} (YES: {llm.judge_yes_count})")
    print(f"    Total tokens:     {total_tokens} (prompt: {total_prompt}, completion: {total_compl})")

    # Comparison with Round 4 baseline
    print(f"\n  Round 4 Baseline Comparison:")
    round4_tokens = 4234  # From optimized_round4_research.md
    round4_calls = 7      # From optimized_round4_research.md
    if total_tokens > 0:
        delta = round4_tokens - total_tokens
        pct = delta / round4_tokens * 100
        direction = "better" if delta > 0 else "worse"
        print(f"    Round 4 tokens:   {round4_tokens}")
        print(f"    P0+P3 tokens:     {total_tokens}")
        print(f"    Delta:            {delta:+d} ({pct:+.1f}%) → {direction}")
        print(f"    Round 4 calls:    {round4_calls}")
        print(f"    P0+P3 calls:      {total_agent_llm + total_judge}")
        print()
        print(f"    NOTE: P0 adds judge cost (~50 prompt + 3 completion tokens per call)")
        print(f"    Judge tokens:     ~{total_judge * 53} (estimate)")
        print(f"    Net token delta (excluding judge overhead): "
              f"{delta + total_judge * 53:+d}")

    # Export
    summary = {
        "experiment": "P0+P3 evaluation",
        "results": results,
        "totals": {
            "llm_chat_calls": total_chat,
            "llm_struct_calls": total_struct,
            "judge_calls": total_judge,
            "total_llm_calls": total_agent_llm + total_judge,
            "prompt_tokens": total_prompt,
            "completion_tokens": total_compl,
            "total_tokens": total_tokens,
            "elapsed_ms": total_elapsed * 1000,
        },
        "cache_summary": {
            "e2e_hits": e2e_hits,
            "judge_hits": judge_hits,
            "template_fills": template_hits,
            "summarizer_cache_hits": summ_cache_hits,
            "full_generations": full_gens,
        },
        "memory_stats": orch.memory_store.get_stats(),
    }
    with open("p0p3_eval_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Full results exported to: p0p3_eval_results.json")

    print("=" * 80)


if __name__ == "__main__":
    run()
