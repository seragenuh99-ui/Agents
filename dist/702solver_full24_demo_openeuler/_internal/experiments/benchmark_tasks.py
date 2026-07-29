"""12-task benchmark suite with validation hints (topics + cross-domain guards)."""

from typing import Any, Dict, List

# Shared by optimized_benchmark.py, quality_benchmark.py, multi_run_benchmark.py
BENCHMARK_TASKS: List[Dict[str, Any]] = [
    {
        "task_id": "e1_solar_basics",
        "description": "Research and summarize solar energy technologies: photovoltaic types (monocrystalline, polycrystalline, thin-film), their efficiency ranges, manufacturing costs, and environmental impact. Provide a structured report.",
        "tags": ["solar", "energy"],
        "expected_topics": ["solar", "photovoltaic", "efficiency", "cost", "environment"],
        "domain": "energy",
    },
    {
        "task_id": "e2_solar_advanced",
        "description": "Investigate cutting-edge solar technology advancements: perovskite tandem cells, bifacial panels, concentrated solar power (CSP), and building-integrated photovoltaics (BIPV). Compare their efficiency potential and commercialization timelines.",
        "tags": ["solar", "energy"],
        "expected_topics": ["perovskite", "bifacial", "solar", "efficiency", "commercial"],
        "domain": "energy",
    },
    {
        "task_id": "e3_wind_energy",
        "description": "Research wind energy technologies comprehensively: onshore vs offshore turbines, capacity factors (25-55%), blade aerodynamics, generator types (DFIG, PMSG), and global installation trends. Provide technical analysis.",
        "tags": ["wind", "energy"],
        "expected_topics": ["wind", "turbine", "offshore", "onshore", "capacity"],
        "domain": "energy",
    },
    {
        "task_id": "e4_renewable_comparison",
        "description": "Compare solar and wind energy across multiple dimensions: levelized cost of energy (LCOE), land use requirements, capacity factors, environmental impacts, grid integration challenges, and geographic suitability. Recommend optimal deployment strategies for different regions.",
        "tags": ["solar", "wind"],
        "expected_topics": ["solar", "wind", "lcoe", "capacity", "compare"],
        "domain": "energy",
    },
    {
        "task_id": "s1_python_vuln",
        "description": "Analyze common Python security vulnerabilities: SQL injection via string formatting, command injection via os.system(), hardcoded credentials, path traversal in file operations, insecure deserialization via pickle, and XSS in web frameworks. Document detection methods and code-level remediation for each vulnerability type.",
        "tags": ["security", "python"],
        "expected_topics": ["sql", "injection", "security", "vulnerab", "python", "pickle"],
        "domain": "security",
    },
    {
        "task_id": "s2_web_security",
        "description": "Perform a comprehensive web application security audit covering: CSRF protection mechanisms, JWT token validation best practices, CORS misconfiguration risks, rate limiting strategies, OAuth 2.0 implementation pitfalls, and SQL injection prevention in ORM-based applications. Provide severity ratings for each vulnerability class.",
        "tags": ["security", "web"],
        "expected_topics": ["csrf", "jwt", "cors", "oauth", "security", "web"],
        "domain": "security",
    },
    {
        "task_id": "s3_python_vuln_v2",
        "description": "Study Python application security flaws including SQL injection, command injection, credential exposure in source code, path traversal attacks, insecure pickle usage, and cross-site scripting. Provide detection approaches and fix recommendations for each vulnerability.",
        "tags": ["security", "python"],
        "expected_topics": ["sql", "injection", "security", "python", "xss", "pickle"],
        "domain": "security",
    },
    {
        "task_id": "s4_code_review",
        "description": "Design a systematic code review checklist for Python web applications: input validation patterns, authentication flow security, session management, database query parameterization, file upload handling, error information leakage prevention, and dependency vulnerability scanning. Include example code snippets for each check.",
        "tags": ["security", "code"],
        "expected_topics": ["review", "validation", "authentication", "session", "security", "python"],
        "domain": "security",
    },
    {
        "task_id": "d1_query_optimization",
        "description": "Research SQL query optimization techniques: index selection strategies (B-tree vs Hash vs GiST), EXPLAIN ANALYZE interpretation, common subquery optimizations, JOIN algorithm selection (nested loop vs hash join vs merge join), and materialized view usage patterns. Provide practical optimization guidelines.",
        "tags": ["database", "sql"],
        "expected_topics": ["sql", "index", "query", "join", "explain", "optim"],
        "domain": "database",
    },
    {
        "task_id": "d2_nosql_comparison",
        "description": "Compare NoSQL database types for different workload patterns: key-value stores (Redis), document databases (MongoDB), column-family stores (Cassandra), and graph databases (Neo4j). Analyze their consistency models, scaling characteristics, query capabilities, and appropriate use cases.",
        "tags": ["database", "nosql"],
        "expected_topics": ["redis", "mongodb", "cassandra", "neo4j", "nosql", "database"],
        "domain": "database",
    },
    {
        "task_id": "d3_db_performance",
        "description": "Investigate database performance tuning: connection pooling strategies, query cache optimization, buffer pool sizing, WAL configuration, replication lag reduction, partitioning strategies, and sharding approaches. Include specific configuration recommendations for PostgreSQL and MySQL.",
        "tags": ["database", "performance"],
        "expected_topics": ["database", "pool", "buffer", "replication", "postgresql", "mysql"],
        "domain": "database",
    },
    {
        "task_id": "d4_data_modeling",
        "description": "Research data modeling best practices: normalization vs denormalization trade-offs, star schema vs snowflake schema for analytics, entity-relationship modeling patterns, temporal data handling, hierarchical data representation (adjacency list vs nested sets vs materialized path), and polyglot persistence strategies.",
        "tags": ["database", "modeling"],
        "expected_topics": ["normalization", "schema", "model", "entity", "data", "database"],
        "domain": "database",
    },
]

# Terms that strongly suggest wrong-domain leakage when dominant in another domain's answer
CROSS_DOMAIN_MARKERS: Dict[str, List[str]] = {
    "security": ["photovoltaic", "wind turbine", "lcoe", "perovskite", "bifacial panel"],
    "database": ["csrf token", "xss attack vector", "os.system(", "pickle.loads"],
    "energy": ["sql injection via string", "jwt validation", "normalization 3nf checklist only"],
}
