"""Real LLM benchmark with DeepSeek API, BGE embeddings, tiktoken counting.

Phase 1 (verify): 3 tasks — verify API connectivity, structured output, caching
Phase 2 (full): 12 tasks across 3 domains — comprehensive metrics
"""

import json
import os
import sys
import time
import tiktoken

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load .env
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
from src.state.embeddings import EmbeddingEngine

ENCODER = tiktoken.get_encoding("cl100k_base")

# ============================================================
# Task definitions
# ============================================================

DOMAIN_ENERGY = [
    {
        "task_id": "e1_solar_basics",
        "description": "Research and summarize solar energy technologies: photovoltaic types (monocrystalline, polycrystalline, thin-film), their efficiency ranges, manufacturing costs, and environmental impact. Provide a structured report.",
        "tags": ["solar", "energy", "photovoltaic", "research"],
    },
    {
        "task_id": "e2_solar_advanced",
        "description": "Investigate cutting-edge solar technology advancements: perovskite tandem cells, bifacial panels, concentrated solar power (CSP), and building-integrated photovoltaics (BIPV). Compare their efficiency potential and commercialization timelines.",
        "tags": ["solar", "energy", "advanced", "technology"],
    },
    {
        "task_id": "e3_wind_energy",
        "description": "Research wind energy technologies comprehensively: onshore vs offshore turbines, capacity factors (25-55%), blade aerodynamics, generator types (DFIG, PMSG), and global installation trends. Provide technical analysis.",
        "tags": ["wind", "energy", "turbines", "research"],
    },
    {
        "task_id": "e4_renewable_comparison",
        "description": "Compare solar and wind energy across multiple dimensions: levelized cost of energy (LCOE), land use requirements, capacity factors, environmental impacts, grid integration challenges, and geographic suitability. Recommend optimal deployment strategies for different regions.",
        "tags": ["solar", "wind", "energy", "comparison", "analysis"],
    },
]

DOMAIN_SECURITY = [
    {
        "task_id": "s1_python_vuln",
        "description": "Analyze common Python security vulnerabilities: SQL injection via string formatting, command injection via os.system(), hardcoded credentials, path traversal in file operations, insecure deserialization via pickle, and XSS in web frameworks. Document detection methods and code-level remediation for each vulnerability type.",
        "tags": ["security", "python", "vulnerability", "analysis"],
    },
    {
        "task_id": "s2_web_security",
        "description": "Perform a comprehensive web application security audit covering: CSRF protection mechanisms, JWT token validation best practices, CORS misconfiguration risks, rate limiting strategies, OAuth 2.0 implementation pitfalls, and SQL injection prevention in ORM-based applications. Provide severity ratings for each vulnerability class.",
        "tags": ["security", "web", "audit", "python"],
    },
    {
        "task_id": "s3_python_vuln_v2",
        "description": "Study Python application security flaws including SQL injection, command injection, credential exposure in source code, path traversal attacks, insecure pickle usage, and cross-site scripting. Provide detection approaches and fix recommendations for each vulnerability.",
        "tags": ["security", "python", "vulnerability", "study"],
    },
    {
        "task_id": "s4_code_review",
        "description": "Design a systematic code review checklist for Python web applications: input validation patterns, authentication flow security, session management, database query parameterization, file upload handling, error information leakage prevention, and dependency vulnerability scanning. Include example code snippets for each check.",
        "tags": ["security", "code", "review", "python"],
    },
]

DOMAIN_DATABASE = [
    {
        "task_id": "d1_query_optimization",
        "description": "Research SQL query optimization techniques: index selection strategies (B-tree vs Hash vs GiST), EXPLAIN ANALYZE interpretation, common subquery optimizations, JOIN algorithm selection (nested loop vs hash join vs merge join), and materialized view usage patterns. Provide practical optimization guidelines.",
        "tags": ["database", "sql", "optimization", "performance"],
    },
    {
        "task_id": "d2_nosql_comparison",
        "description": "Compare NoSQL database types for different workload patterns: key-value stores (Redis), document databases (MongoDB), column-family stores (Cassandra), and graph databases (Neo4j). Analyze their consistency models, scaling characteristics, query capabilities, and appropriate use cases.",
        "tags": ["database", "nosql", "comparison", "architecture"],
    },
    {
        "task_id": "d3_db_performance",
        "description": "Investigate database performance tuning: connection pooling strategies, query cache optimization, buffer pool sizing, WAL configuration, replication lag reduction, partitioning strategies, and sharding approaches. Include specific configuration recommendations for PostgreSQL and MySQL.",
        "tags": ["database", "performance", "tuning", "sql"],
    },
    {
        "task_id": "d4_data_modeling",
        "description": "Research data modeling best practices: normalization vs denormalization trade-offs, star schema vs snowflake schema for analytics, entity-relationship modeling patterns, temporal data handling, hierarchical data representation (adjacency list vs nested sets vs materialized path), and polyglot persistence strategies.",
        "tags": ["database", "modeling", "design", "architecture"],
    },
]

ALL_TASKS = DOMAIN_ENERGY + DOMAIN_SECURITY + DOMAIN_DATABASE


def count_tokens(text: str) -> int:
    return len(ENCODER.encode(text))


def output_similarity_score(orchestrator, task_description: str, summary_text: str) -> float:
    """Validate output by computing embedding similarity between task and summary."""
    if not summary_text:
        return 0.0
    task_emb = orchestrator.embedding_engine.encode(task_description)
    summ_emb = orchestrator.embedding_engine.encode(summary_text[:500])
    import numpy as np
    a, b = np.array(task_emb), np.array(summ_emb)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def clear_db():
    for path in ["_real_bench.db", "_real_bench.db-shm", "_real_bench.db-wal"]:
        try:
            os.unlink(path)
        except OSError:
            pass


def run_phase1_verify(llm):
    """3-task verification: test API connectivity and structured output."""
    print("=" * 70)
    print("  Phase 1: API Verification (3 tasks)")
    print("=" * 70)

    clear_db()
    orch = Orchestrator(llm=llm, mode="structured", use_real_embeddings=True,
                        sandbox_enabled=True, memory_db_path="_real_bench.db")

    verify_tasks = [
        {
            "task_id": "v1",
            "description": "Briefly explain what solar photovoltaic technology is and list its main types.",
            "tags": ["solar", "energy"],
        },
        {
            "task_id": "v2",
            "description": "Explain solar PV technology types including monocrystalline, polycrystalline, and thin-film panels. What are their characteristics?",
            "tags": ["solar", "energy", "photovoltaic"],
        },
        {
            "task_id": "v3",
            "description": "What is SQL injection? List common prevention methods in Python applications.",
            "tags": ["security", "sql", "python"],
        },
    ]

    results = []
    for i, task in enumerate(verify_tasks):
        desc_short = task["description"][:80]
        print(f"\n  [{i+1}/3] {task['task_id']}: {desc_short}...")

        t0 = time.time()
        result = orch.execute_task(
            task_id=task["task_id"],
            task_description=task["description"],
            tags=task.get("tags", []),
        )
        elapsed = time.time() - t0

        e2e = result.get("_e2e_cached", False)
        e2e_match = result.get("_e2e_match", "")
        e2e_domain = result.get("_e2e_domain_match", False)
        plan = result.get("steps", {}).get("plan", {})
        summary_step = result.get("steps", {}).get("summary", {})
        summary_text = summary_step.get("summary", "")

        if isinstance(summary_text, dict):
            summary_text = summary_text.get("conclusion", "") or str(summary_text)
        summary_text = str(summary_text)[:500]

        # Output relevance check
        relevance = output_similarity_score(orch, task["description"], summary_text)

        # Strategy
        if e2e and e2e_match == "exact":
            strategy = "E2E EXACT"
        elif e2e_domain:
            strategy = f"E2E DOMAIN (cos={result.get('_e2e_domain_score', 0):.3f})"
        elif plan.get("_from_template"):
            strategy = f"P1 TEMPLATE ({plan.get('_template_from', '')[:16]})"
        elif plan.get("llm_skipped"):
            strategy = "PLAN REUSED"
        else:
            strategy = "FULL GENERATION"

        llm_stats = llm.get_usage_stats()
        print(f"    Strategy: {strategy}")
        print(f"    LLM calls: {llm_stats['call_count']}")
        print(f"    API tokens: prompt={llm_stats['total_prompt_tokens']} "
              f"compl={llm_stats['total_completion_tokens']} "
              f"total={llm_stats['total_tokens']}")
        print(f"    Output relevance: {relevance:.3f} (cosine with task)")
        print(f"    Summary: {summary_text[:150]}...")
        print(f"    Latency: {elapsed*1000:.0f}ms")

        results.append({
            "task_id": task["task_id"],
            "strategy": strategy,
            "e2e_cached": e2e,
            "e2e_match": e2e_match,
            "e2e_domain": e2e_domain,
            "output_relevance": round(relevance, 3),
            "prompt_tokens": llm_stats["total_prompt_tokens"],
            "completion_tokens": llm_stats["total_completion_tokens"],
            "total_tokens": llm_stats["total_tokens"],
            "elapsed_ms": elapsed * 1000,
            "summary_preview": summary_text[:200],
        })

        llm.reset_stats()
        mems = orch.memory_store.get_stats()
        print(f"    Memory: {mems['total_memories']} total, by type: {mems['by_type']}")

    return results


def run_phase2_full(llm):
    """12-task full benchmark across 3 domains."""
    print("\n\n" + "=" * 70)
    print("  Phase 2: Full Benchmark (12 tasks, 3 domains)")
    print("=" * 70)

    clear_db()
    orch = Orchestrator(llm=llm, mode="structured", use_real_embeddings=True,
                        sandbox_enabled=True, memory_db_path="_real_bench.db")

    results = []
    total_start = time.time()
    total_llm_tokens = 0
    total_llm_calls = 0

    for i, task in enumerate(ALL_TASKS):
        desc_short = task["description"][:80]
        domain = task["tags"][0] if task["tags"] else "?"
        print(f"\n  [{i+1:2d}/12] {task['task_id']} [{domain}]: {desc_short}...")

        t0 = time.time()
        result = orch.execute_task(
            task_id=task["task_id"],
            task_description=task["description"],
            tags=task.get("tags", []),
        )
        elapsed = time.time() - t0

        e2e = result.get("_e2e_cached", False)
        e2e_match = result.get("_e2e_match", "")
        e2e_domain = result.get("_e2e_domain_match", False)
        domain_score = result.get("_e2e_domain_score", 0)
        plan = result.get("steps", {}).get("plan", {})
        summary_step = result.get("steps", {}).get("summary", {})
        summary_text = summary_step.get("summary", "")
        if isinstance(summary_text, dict):
            summary_text = summary_text.get("conclusion", "") or json.dumps(summary_text)
        summary_text = str(summary_text)

        relevance = output_similarity_score(orch, task["description"], summary_text)

        if e2e and e2e_match == "exact":
            strategy = "E2E EXACT"
        elif e2e_domain:
            strategy = f"E2E DOMAIN"
        elif plan.get("_from_template"):
            strategy = "P1 TEMPLATE"
        elif plan.get("llm_skipped"):
            strategy = "PLAN REUSED"
        elif plan.get("_template_filled"):
            strategy = "TEMPLATE FILL"
        else:
            strategy = "FULL GEN"

        llm_stats = llm.get_usage_stats()
        agent_tokens = llm_stats["total_tokens"]
        agent_calls = llm_stats["call_count"]

        print(f"    → {strategy} | API calls={agent_calls} | "
              f"tokens=prompt({llm_stats['total_prompt_tokens']})/"
              f"compl({llm_stats['total_completion_tokens']})/"
              f"total({agent_tokens}) | "
              f"relevance={relevance:.3f} | {elapsed*1000:.0f}ms")

        if e2e_domain:
            print(f"      domain cos={domain_score:.3f}")
        if plan.get("_from_template"):
            print(f"      template_from={plan.get('_template_from', '')[:30]}")
        if plan.get("_promoted_template"):
            print(f"      promoted={plan['_promoted_template'][:20]}")

        results.append({
            "task_id": task["task_id"],
            "domain": domain,
            "strategy": strategy,
            "e2e_cached": e2e,
            "e2e_match": e2e_match,
            "e2e_domain": e2e_domain,
            "domain_cos": round(domain_score, 3) if e2e_domain else 0,
            "from_template": plan.get("_from_template", False),
            "template_filled": plan.get("_template_filled", False),
            "output_relevance": round(relevance, 3),
            "prompt_tokens": llm_stats["total_prompt_tokens"],
            "completion_tokens": llm_stats["total_completion_tokens"],
            "total_tokens": agent_tokens,
            "api_calls": agent_calls,
            "elapsed_ms": elapsed * 1000,
            "summary_preview": summary_text[:200],
        })

        total_llm_tokens += agent_tokens
        total_llm_calls += agent_calls
        llm.reset_stats()

        mems = orch.memory_store.get_stats()
        if i % 4 == 3:
            print(f"      memory: {mems['total_memories']} total, "
                  f"by type: {mems['by_type']}")

    total_elapsed = time.time() - total_start

    # Summary
    print("\n\n" + "=" * 70)
    print("  PHASE 2 SUMMARY")
    print("=" * 70)

    print(f"\n  {'Task':<22} {'Domain':<10} {'Strategy':<16} {'Calls':>5} {'Tokens':>7} {'Rel':>6} {'Lat':>7}")
    print(f"  {'─'*22} {'─'*10} {'─'*16} {'─'*5} {'─'*7} {'─'*6} {'─'*7}")
    for r in results:
        print(f"  {r['task_id']:<22} {r['domain']:<10} {r['strategy']:<16} "
              f"{r['api_calls']:>5} {r['total_tokens']:>7} {r['output_relevance']:>6.3f} {r['elapsed_ms']:>6.0f}ms")

    total_api_calls = sum(r["api_calls"] for r in results)
    total_api_tokens = sum(r["total_tokens"] for r in results)
    total_lat = sum(r["elapsed_ms"] for r in results)

    print(f"  {'─'*22} {'─'*10} {'─'*16} {'─'*5} {'─'*7} {'─'*6} {'─'*7}")
    print(f"  {'TOTAL':<22} {'':<10} {'':<16} {total_api_calls:>5} {total_api_tokens:>7} "
          f"{'':>6} {total_lat:>6.0f}ms")

    # Strategy distribution
    from collections import Counter
    domain_strats = Counter()
    for r in results:
        domain_strats[f"{r['domain']}/{r['strategy']}"] += 1

    print(f"\n  Strategy distribution:")
    for k, v in sorted(domain_strats.items()):
        print(f"    {k}: {v}")

    # Output quality
    avg_rel = sum(r["output_relevance"] for r in results) / len(results)
    low_rel = [r for r in results if r["output_relevance"] < 0.3]
    print(f"\n  Output quality:")
    print(f"    Avg relevance: {avg_rel:.3f}")
    print(f"    Low relevance (<0.3): {len(low_rel)}/{len(results)}")

    # Token efficiency
    text_equiv_est = len(ALL_TASKS) * 3000
    print(f"\n  Token efficiency:")
    print(f"    Real API tokens: {total_api_tokens}")
    print(f"    Text equiv (est): {text_equiv_est}")
    print(f"    Savings: {(1 - total_api_tokens/text_equiv_est)*100:.1f}%")

    # Memory summary
    mems = orch.memory_store.get_stats()
    print(f"\n  Memory state:")
    print(f"    Total: {mems['total_memories']}")
    print(f"    By type: {mems['by_type']}")

    # Check for templates
    for mtype in ["strategy", "result"]:
        templates = orch.memory_store.get_templates(["solar", "energy", "security", "database"], mtype)
        print(f"    {mtype} templates (level>=1): {len(templates)}")
        for t in templates:
            print(f"      - {t.task_topic} (level={t.abstraction_level})")

    return results, orch


def main():
    # Load API key
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set. Check .env file.")
        return 1

    print("=" * 70)
    print("  Real LLM Benchmark — DeepSeek API + BGE + tiktoken")
    print(f"  Model: deepseek-chat")
    print(f"  Embedding: BAAI/bge-small-en-v1.5 (384d)")
    print(f"  Tokenizer: tiktoken cl100k_base")
    print("=" * 70)

    llm = LLMBackend(
        provider="deepseek",
        api_key=api_key,
        model="deepseek-chat",
    )

    # Phase 1: Verify
    p1_results = run_phase1_verify(llm)
    p1_failures = [r for r in p1_results if r["total_tokens"] == 0 and not r["e2e_cached"]]
    if p1_failures:
        print(f"\n  ERROR: Phase 1 verification failed — {len(p1_failures)} tasks got 0 tokens without E2E cache")
        return 1

    # Phase 2: Full benchmark
    p2_results, orch = run_phase2_full(llm)

    # Export
    output = {
        "config": {
            "model": "deepseek-chat",
            "embedding": "BAAI/bge-small-en-v1.5",
            "tokenizer": "tiktoken cl100k_base",
            "tasks": len(ALL_TASKS),
        },
        "phase1_verify": p1_results,
        "phase2_results": p2_results,
    }
    with open("real_llm_benchmark_results.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Full results exported to: real_llm_benchmark_results.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
