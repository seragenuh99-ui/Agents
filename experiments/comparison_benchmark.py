"""Cache vs No-cache comparison: measures what caching and refs actually save.

Structured (cache+refs): full system with E2E cache, planner cache, summarizer P3 cache
No-cache (baseline): all caching disabled, every task forces full LLM generation

This measures the real token savings from the caching/memory system.
"""

import json
import os
import sys
import time
import tiktoken

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.paths import result_path

env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

from src.agents.base import LLMBackend
from src.orchestrator import Orchestrator

ENCODER = tiktoken.get_encoding("cl100k_base")

ALL_TASKS = [
    {"task_id": "e1_solar_basics", "description": "Research and summarize solar energy technologies: photovoltaic types (monocrystalline, polycrystalline, thin-film), their efficiency ranges, manufacturing costs, and environmental impact. Provide a structured report.", "tags": ["solar", "energy", "photovoltaic", "research"]},
    {"task_id": "e2_solar_advanced", "description": "Investigate cutting-edge solar technology advancements: perovskite tandem cells, bifacial panels, concentrated solar power (CSP), and building-integrated photovoltaics (BIPV). Compare their efficiency potential and commercialization timelines.", "tags": ["solar", "energy", "advanced", "technology"]},
    {"task_id": "e3_wind_energy", "description": "Research wind energy technologies comprehensively: onshore vs offshore turbines, capacity factors (25-55%), blade aerodynamics, generator types (DFIG, PMSG), and global installation trends. Provide technical analysis.", "tags": ["wind", "energy", "turbines", "research"]},
    {"task_id": "e4_renewable_comparison", "description": "Compare solar and wind energy across multiple dimensions: levelized cost of energy (LCOE), land use requirements, capacity factors, environmental impacts, grid integration challenges, and geographic suitability. Recommend optimal deployment strategies for different regions.", "tags": ["solar", "wind", "energy", "comparison", "analysis"]},
    {"task_id": "s1_python_vuln", "description": "Analyze common Python security vulnerabilities: SQL injection via string formatting, command injection via os.system(), hardcoded credentials, path traversal in file operations, insecure deserialization via pickle, and XSS in web frameworks. Document detection methods and code-level remediation for each vulnerability type.", "tags": ["security", "python", "vulnerability", "analysis"]},
    {"task_id": "s2_web_security", "description": "Perform a comprehensive web application security audit covering: CSRF protection mechanisms, JWT token validation best practices, CORS misconfiguration risks, rate limiting strategies, OAuth 2.0 implementation pitfalls, and SQL injection prevention in ORM-based applications. Provide severity ratings for each vulnerability class.", "tags": ["security", "web", "audit", "python"]},
    {"task_id": "s3_python_vuln_v2", "description": "Study Python application security flaws including SQL injection, command injection, credential exposure in source code, path traversal attacks, insecure pickle usage, and cross-site scripting. Provide detection approaches and fix recommendations for each vulnerability.", "tags": ["security", "python", "vulnerability", "study"]},
    {"task_id": "s4_code_review", "description": "Design a systematic code review checklist for Python web applications: input validation patterns, authentication flow security, session management, database query parameterization, file upload handling, error information leakage prevention, and dependency vulnerability scanning. Include example code snippets for each check.", "tags": ["security", "code", "review", "python"]},
    {"task_id": "d1_query_optimization", "description": "Research SQL query optimization techniques: index selection strategies (B-tree vs Hash vs GiST), EXPLAIN ANALYZE interpretation, common subquery optimizations, JOIN algorithm selection (nested loop vs hash join vs merge join), and materialized view usage patterns. Provide practical optimization guidelines.", "tags": ["database", "sql", "optimization", "performance"]},
    {"task_id": "d2_nosql_comparison", "description": "Compare NoSQL database types for different workload patterns: key-value stores (Redis), document databases (MongoDB), column-family stores (Cassandra), and graph databases (Neo4j). Analyze their consistency models, scaling characteristics, query capabilities, and appropriate use cases.", "tags": ["database", "nosql", "comparison", "architecture"]},
    {"task_id": "d3_db_performance", "description": "Investigate database performance tuning: connection pooling strategies, query cache optimization, buffer pool sizing, WAL configuration, replication lag reduction, partitioning strategies, and sharding approaches. Include specific configuration recommendations for PostgreSQL and MySQL.", "tags": ["database", "performance", "tuning", "sql"]},
    {"task_id": "d4_data_modeling", "description": "Research data modeling best practices: normalization vs denormalization trade-offs, star schema vs snowflake schema for analytics, entity-relationship modeling patterns, temporal data handling, hierarchical data representation (adjacency list vs nested sets vs materialized path), and polyglot persistence strategies.", "tags": ["database", "modeling", "design", "architecture"]},
]


def clear_db(path):
    for p in [path, path + "-shm", path + "-wal"]:
        try:
            os.unlink(p)
        except OSError:
            pass


def run_benchmark(llm, db_path, disable_cache=False):
    """Run all 12 tasks. If disable_cache=True, monkey-patch to force full generation."""
    clear_db(db_path)

    orch = Orchestrator(
        llm=llm, mode="structured",
        use_real_embeddings=True,
        sandbox_enabled=True,
        memory_db_path=db_path,
    )

    if disable_cache:
        # Disable all caching by overriding the memory store's FAISS index
        # With no index, all embedding-based cache checks will skip
        orch.memory_store._index = None
        orch.memory_store._id_to_idx.clear()
        orch.memory_store._idx_to_id.clear()

        # Also disable template promotion (needs accumulated memories)
        original_promote = orch.planner._maybe_promote_to_template
        orch.planner._maybe_promote_to_template = lambda *a, **kw: None
        original_promote_summ = orch.summarizer._maybe_promote_to_template
        orch.summarizer._maybe_promote_to_template = lambda *a, **kw: None

    results = []
    total_start = time.time()

    for i, task in enumerate(ALL_TASKS):
        desc_short = task["description"][:60]
        strategy_label = "FULL" if disable_cache else "?"
        print(f"  [{i+1:2d}/12] {task['task_id']}: {desc_short}...", end=" ", flush=True)

        t0 = time.time()
        result = orch.execute_task(
            task_id=task["task_id"],
            task_description=task["description"],
            tags=task.get("tags", []),
        )
        elapsed = time.time() - t0

        llm_stats = llm.get_usage_stats()

        summary_step = result.get("steps", {}).get("summary", {})
        summary_text = summary_step.get("summary", "")
        if isinstance(summary_text, dict):
            summary_text = summary_text.get("conclusion", "") or json.dumps(summary_text)
        summary_text = str(summary_text)

        # Relevance
        try:
            import numpy as np
            task_emb = orch.embedding_engine.encode(task["description"])
            summ_emb = orch.embedding_engine.encode(summary_text[:500])
            a, b = np.array(task_emb), np.array(summ_emb)
            relevance = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        except Exception:
            relevance = 0.0

        plan = result.get("steps", {}).get("plan", {})
        e2e = result.get("_e2e_cached", False)

        if not disable_cache:
            if e2e and result.get("_e2e_match") == "exact":
                strategy = "E2E"
            elif plan.get("_from_template"):
                strategy = "TMPL"
            elif plan.get("llm_skipped"):
                strategy = "REUS"
            elif plan.get("_template_filled"):
                strategy = "FILL"
            else:
                strategy = "FULL"
        else:
            strategy = "FULL"

        prompt_t = llm_stats["total_prompt_tokens"]
        compl_t = llm_stats["total_completion_tokens"]
        total_t = prompt_t + compl_t
        calls = llm_stats["call_count"]

        print(f"{strategy} calls={calls} tok={total_t} rel={relevance:.3f} {elapsed*1000:.0f}ms")

        results.append({
            "task_id": task["task_id"],
            "domain": task["tags"][0],
            "strategy": strategy,
            "api_calls": calls,
            "prompt_tokens": prompt_t,
            "completion_tokens": compl_t,
            "total_tokens": total_t,
            "relevance": round(relevance, 3),
            "elapsed_ms": elapsed * 1000,
            "summary_preview": summary_text[:200],
        })

        llm.reset_stats()

        # Show memory growth
        if not disable_cache and i % 3 == 2:
            mems = orch.memory_store.get_stats()
            print(f"      memory: {mems['total_memories']} total")

    total_elapsed = time.time() - total_start

    totals = {
        "total_api_calls": sum(r["api_calls"] for r in results),
        "total_prompt_tokens": sum(r["prompt_tokens"] for r in results),
        "total_completion_tokens": sum(r["completion_tokens"] for r in results),
        "total_tokens": sum(r["total_tokens"] for r in results),
        "total_elapsed_ms": total_elapsed * 1000,
        "avg_relevance": sum(r["relevance"] for r in results) / len(results),
    }

    return results, totals


def main():
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set")
        return 1

    print("=" * 70)
    print("  Cache vs No-Cache Comparison — DeepSeek API")
    print("  12 tasks, 3 domains")
    print("=" * 70)

    # ---- With Cache ----
    print("\n" + "=" * 70)
    print("  RUN 1: With Caching (E2E + Planner + P3 + Refs)")
    print("=" * 70)

    llm_cached = LLMBackend(provider="deepseek", api_key=api_key, model="deepseek-chat")
    cached_results, cached_totals = run_benchmark(llm_cached, "_cmp_cached.db", disable_cache=False)

    # ---- Without Cache ----
    print("\n" + "=" * 70)
    print("  RUN 2: No Cache (force full generation every task)")
    print("=" * 70)

    llm_nocache = LLMBackend(provider="deepseek", api_key=api_key, model="deepseek-chat")
    nocache_results, nocache_totals = run_benchmark(llm_nocache, "_cmp_nocache.db", disable_cache=True)

    # ---- Comparison ----
    s_tok = cached_totals["total_tokens"]
    n_tok = nocache_totals["total_tokens"]
    s_call = cached_totals["total_api_calls"]
    n_call = nocache_totals["total_api_calls"]
    s_lat = cached_totals["total_elapsed_ms"]
    n_lat = nocache_totals["total_elapsed_ms"]

    print("\n\n" + "=" * 70)
    print("  FINAL COMPARISON: With Cache vs Without Cache")
    print("=" * 70)

    print(f"\n  {'Metric':<30} {'With Cache':>12} {'No Cache':>12} {'Savings':>12}")
    print(f"  {'─'*30} {'─'*12} {'─'*12} {'─'*12}")

    token_save = (1 - s_tok / n_tok) * 100 if n_tok > 0 else 0
    call_save = (1 - s_call / n_call) * 100 if n_call > 0 else 0
    lat_save = (1 - s_lat / n_lat) * 100 if n_lat > 0 else 0

    print(f"  {'Total LLM Tokens':<30} {s_tok:>12,} {n_tok:>12,} {token_save:>11.1f}%")
    print(f"  {'Total API Calls':<30} {s_call:>12} {n_call:>12} {call_save:>11.1f}%")
    print(f"  {'Total Latency (ms)':<30} {s_lat:>12,.0f} {n_lat:>12,.0f} {lat_save:>11.1f}%")
    print(f"  {'Avg Relevance':<30} {cached_totals['avg_relevance']:>12.3f} {nocache_totals['avg_relevance']:>12.3f} {'':>12}")

    print(f"\n  {'─'*30} {'─'*12} {'─'*12} {'─'*12}")
    print(f"  {'Prompt Tokens':<30} {cached_totals['total_prompt_tokens']:>12,} {nocache_totals['total_prompt_tokens']:>12,}")
    print(f"  {'Completion Tokens':<30} {cached_totals['total_completion_tokens']:>12,} {nocache_totals['total_completion_tokens']:>12,}")

    # Per-task table
    print(f"\n  Per-task Token Comparison:")
    print(f"  {'Task':<22} {'Domain':<10} {'Cached':>8} {'NoCache':>8} {'Saved':>8}")
    print(f"  {'─'*22} {'─'*10} {'─'*8} {'─'*8} {'─'*8}")
    for cr, nr in zip(cached_results, nocache_results):
        saved = nr["total_tokens"] - cr["total_tokens"]
        print(f"  {cr['task_id']:<22} {cr['domain']:<10} {cr['total_tokens']:>8,} {nr['total_tokens']:>8,} {saved:>8,}")

    # Export
    output = {
        "config": {"model": "deepseek-chat", "tasks": 12},
        "with_cache": {"results": cached_results, "totals": cached_totals},
        "no_cache": {"results": nocache_results, "totals": nocache_totals},
        "savings": {
            "token_pct": round(token_save, 1),
            "call_pct": round(call_save, 1),
            "latency_pct": round(lat_save, 1),
        },
    }
    with open(result_path("cache_comparison_results.json"), "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Results exported to: cache_comparison_results.json")

    print(f"\n  === SUMMARY ===")
    print(f"  Token savings from caching: {token_save:.1f}%")
    print(f"  ({s_tok:,} with cache vs {n_tok:,} without cache)")
    print(f"  API call savings: {call_save:.1f}% ({s_call} vs {n_call})")
    print(f"  Latency savings: {lat_save:.1f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
