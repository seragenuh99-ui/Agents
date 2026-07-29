"""Benchmark task suites — core, extended, adversarial (v2 experiment design)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from experiments.benchmark_tasks import BENCHMARK_TASKS, CROSS_DOMAIN_MARKERS
from experiments.tasks import (
    ALL_TASK_GROUPS,
    TASK_GROUP_3_CONTINUOUS,
)

# --- v2 extended tasks (6): broader coverage, same domains ---
EXTENDED_TASKS: List[Dict[str, Any]] = [
    {
        "task_id": "e5_hydro_storage",
        "description": "Analyze hydropower and pumped-storage hydro for grid balancing: capacity ranges, round-trip efficiency (70-85%), environmental trade-offs, and pairing with variable solar/wind. Provide quantitative comparisons.",
        "tags": ["hydro", "energy"],
        "expected_topics": ["hydro", "pumped", "storage", "grid", "efficiency"],
        "domain": "energy",
        "suite": "extended",
    },
    {
        "task_id": "e6_grid_integration",
        "description": "Research grid integration challenges for high renewable penetration: curtailment, frequency regulation, inverter requirements, and battery storage sizing. Focus on engineering solutions.",
        "tags": ["grid", "energy"],
        "expected_topics": ["grid", "curtail", "frequency", "storage", "renewable"],
        "domain": "energy",
        "suite": "extended",
    },
    {
        "task_id": "s5_secrets_management",
        "description": "Explain secrets management for Python services: environment variables, vault integration, key rotation, and preventing credential leakage in CI/CD logs. Do not focus on web CSRF/JWT.",
        "tags": ["security", "secrets"],
        "expected_topics": ["secret", "vault", "credential", "rotation", "environment"],
        "domain": "security",
        "suite": "extended",
    },
    {
        "task_id": "s6_supply_chain",
        "description": "Describe software supply chain security for Python projects: dependency pinning, SBOM, typosquatting, and signed packages. Exclude general web application pentesting topics.",
        "tags": ["security", "supply"],
        "expected_topics": ["dependency", "sbom", "supply", "package", "sign"],
        "domain": "security",
        "suite": "extended",
    },
    {
        "task_id": "d5_transactions",
        "description": "Explain database transaction isolation levels (READ UNCOMMITTED through SERIALIZABLE): anomalies prevented, performance impact, and when to use each in PostgreSQL/MySQL.",
        "tags": ["database", "transaction"],
        "expected_topics": ["transaction", "isolation", "serializ", "anomaly", "acid"],
        "domain": "database",
        "suite": "extended",
    },
    {
        "task_id": "d6_backup_recovery",
        "description": "Research backup and disaster recovery for relational databases: PITR, WAL archiving, RPO/RTO targets, and failover topologies. Do not compare NoSQL product families.",
        "tags": ["database", "backup"],
        "expected_topics": ["backup", "recovery", "pitr", "wal", "failover", "rpo"],
        "domain": "database",
        "suite": "extended",
    },
]

# --- Adversarial (6): wording similar to core tasks but different intent; must NOT E2E-reuse ---
ADVERSARIAL_TASKS: List[Dict[str, Any]] = [
    {
        "task_id": "x1_solar_policy_only",
        "description": "Write a policy brief on solar deployment incentives ONLY: feed-in tariffs, tax credits, net metering rules, and permitting timelines. Do not discuss photovoltaic cell physics or manufacturing processes.",
        "tags": ["solar", "energy", "adversarial", "no-e2e-cache"],
        "expected_topics": ["policy", "tariff", "incentive", "permit", "net metering"],
        "forbidden_answer_topics": ["monocrystalline", "perovskite", "photovoltaic efficiency"],
        "domain": "energy",
        "suite": "adversarial",
        "must_not_e2e": True,
    },
    {
        "task_id": "x2_wind_economics_only",
        "description": "Analyze offshore wind project economics ONLY: CAPEX breakdown, O&M costs, contract for difference (CfD), and investor risk. Do not explain DFIG vs PMSG generator technical details.",
        "tags": ["wind", "energy", "adversarial", "no-e2e-cache"],
        "expected_topics": ["capex", "o&m", "cfd", "economics", "invest"],
        "forbidden_answer_topics": ["dfig", "pmsg", "blade aerodynamics"],
        "domain": "energy",
        "suite": "adversarial",
        "must_not_e2e": True,
    },
    {
        "task_id": "x3_security_compliance_only",
        "description": "Outline SOC2 and ISO27001 control mapping for a Python SaaS backend: access control, logging, change management. Do not list code-level vulnerability patterns like SQL injection fixes.",
        "tags": ["security", "compliance", "adversarial", "no-e2e-cache"],
        "expected_topics": ["soc2", "iso", "control", "compliance", "audit"],
        "forbidden_answer_topics": ["sql injection", "os.system", "pickle.loads"],
        "domain": "security",
        "suite": "adversarial",
        "must_not_e2e": True,
    },
    {
        "task_id": "x4_web_tls_only",
        "description": "Document TLS configuration hardening for web servers ONLY: cipher suites, certificate rotation, HSTS, and mTLS patterns. Do not discuss CSRF tokens, JWT claims, or OAuth flows.",
        "tags": ["security", "web", "adversarial", "no-e2e-cache"],
        "expected_topics": ["tls", "cipher", "certificate", "hsts", "mtls"],
        "forbidden_answer_topics": ["csrf", "jwt", "oauth", "cors"],
        "domain": "security",
        "suite": "adversarial",
        "must_not_e2e": True,
    },
    {
        "task_id": "x5_db_migration_only",
        "description": "Describe zero-downtime schema migration strategies ONLY: expand-contract, dual-write, backfill jobs, and rollback plans. Do not explain B-tree vs hash index selection.",
        "tags": ["database", "migration", "adversarial", "no-e2e-cache"],
        "expected_topics": ["migration", "schema", "backfill", "rollback", "zero-downtime"],
        "forbidden_answer_topics": ["b-tree", "hash index", "explain analyze"],
        "domain": "database",
        "suite": "adversarial",
        "must_not_e2e": True,
    },
    {
        "task_id": "x6_db_cap_only",
        "description": "Explain CAP theorem trade-offs and eventual consistency patterns ONLY: quorum reads, conflict resolution, and PACELC framing. Do not compare Redis vs MongoDB product features.",
        "tags": ["database", "distributed", "adversarial", "no-e2e-cache"],
        "expected_topics": ["cap", "consistency", "quorum", "eventual", "pacelc"],
        "forbidden_answer_topics": ["redis", "mongodb", "cassandra", "neo4j"],
        "domain": "database",
        "suite": "adversarial",
        "must_not_e2e": True,
    },
]

# Human fact-check sample: 2 per domain from core (diverse strategies)
HUMAN_FACTCHECK_TASK_IDS = [
    "e2_solar_advanced",
    "e4_renewable_comparison",
    "s1_python_vuln",
    "s2_web_security",
    "d1_query_optimization",
    "d4_data_modeling",
]

FULL_SUITE_24: List[Dict[str, Any]] = (
    BENCHMARK_TASKS + EXTENDED_TASKS + ADVERSARIAL_TASKS
)

SUITES: Dict[str, List[Dict[str, Any]]] = {
    "core12": BENCHMARK_TASKS,
    "extended6": EXTENDED_TASKS,
    "adversarial6": ADVERSARIAL_TASKS,
    "full24": FULL_SUITE_24,
    "energy4": [t for t in BENCHMARK_TASKS if t.get("domain") == "energy"],
    "factcheck6": [t for t in BENCHMARK_TASKS if t["task_id"] in HUMAN_FACTCHECK_TASK_IDS],
    "energy_research": ALL_TASK_GROUPS["energy_research"],
    "code_security": ALL_TASK_GROUPS["code_security"],
    "continuous12": TASK_GROUP_3_CONTINUOUS,
    "continuous_10": ALL_TASK_GROUPS["continuous_10"],
}


def get_suite(name: str) -> List[Dict[str, Any]]:
    if name not in SUITES:
        raise KeyError(f"Unknown suite {name!r}; choose from {list(SUITES)}")
    return list(SUITES[name])


def task_by_id(task_id: str) -> Dict[str, Any]:
    for suite_tasks in SUITES.values():
        for t in suite_tasks:
            if t["task_id"] == task_id:
                return dict(t)
    raise KeyError(task_id)


def resolve_tasks(
    *,
    suite: Optional[str] = None,
    task_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Build task list from suite name and/or explicit task ids."""
    if task_ids:
        return [task_by_id(tid) for tid in task_ids]
    if suite:
        return get_suite(suite)
    raise ValueError("需要指定 suite 或 task_ids")
