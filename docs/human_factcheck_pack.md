# 人工事实核查包（6 题样本）

请评审员对每题勾选：**事实正确** / **部分正确** / **错误**，并备注。
自动指标（composite、主题覆盖）不能替代本表。

## e2_solar_advanced

**领域**: energy

### 题目

Investigate cutting-edge solar technology advancements: perovskite tandem cells, bifacial panels, concentrated solar power (CSP), and building-integrated photovoltaics (BIPV). Compare their efficiency potential and commercialization timelines.

### 系统回答（自动）

**要点**:
1. Perovskite tandem cells achieve >30% efficiency in lab, potential for >40% with silicon.
2. Bifacial panels boost energy yield by 10-30% with 20-22% efficiency, commercial now.
3. CSP with storage offers 15-25% efficiency, dispatchable power, but higher cost.
4. BIPV integrates into buildings, efficiency 10-20%, niche market with aesthetic value.
5. Perovskite tandems and bifacial panels are closest to commercialization; CSP and BIPV face cost and integration hurdles.

**结论**: Perovskite tandem cells and bifacial panels offer the highest efficiency potential and near-term commercialization, while CSP and BIPV have specialized roles with longer timelines.

### 自动评分

- composite: 0.977
- topic_coverage: 1.0
- strategy: TEMPLATE_FILL_DROP

### 人工核查

| 项目 | 勾选 |
|------|------|
| 事实正确 | [ ] |
| 部分正确 | [ ] |
| 错误 | [ ] |

备注：

---

## e4_renewable_comparison

**领域**: energy

### 题目

Compare solar and wind energy across multiple dimensions: levelized cost of energy (LCOE), land use requirements, capacity factors, environmental impacts, grid integration challenges, and geographic suitability. Recommend optimal deployment strategies for different regions.

### 系统回答（自动）

**要点**:
1. Solar LCOE has dropped significantly, often below wind, but varies by region.
2. Wind has higher capacity factors, especially offshore (40-55%) vs solar (15-25%).
3. Solar requires less land per MWh than wind, but wind can share land with agriculture.
4. Both have low operational emissions; solar has manufacturing waste, wind impacts birds.
5. Grid integration challenges include intermittency; solar is more predictable diurnally.

**结论**: Optimal deployment: solar in sun-rich, land-constrained areas; wind in windy, offshore or rural regions; hybrid systems for grid stability.

### 自动评分

- composite: 0.974
- topic_coverage: 1.0
- strategy: P1_TEMPLATE_DROP

### 人工核查

| 项目 | 勾选 |
|------|------|
| 事实正确 | [ ] |
| 部分正确 | [ ] |
| 错误 | [ ] |

备注：

---

## s1_python_vuln

**领域**: security

### 题目

Analyze common Python security vulnerabilities: SQL injection via string formatting, command injection via os.system(), hardcoded credentials, path traversal in file operations, insecure deserialization via pickle, and XSS in web frameworks. Document detection methods and code-level remediation for each vulnerability type.

### 系统回答（自动）

**要点**:
1. SQL injection via string formatting: Use parameterized queries or ORM to prevent.
2. Command injection via os.system(): Use subprocess with shell=False or shlex.quote().
3. Hardcoded credentials: Store in environment variables or secret management services.
4. Path traversal: Validate and sanitize file paths using os.path.realpath and allowlist.
5. Insecure deserialization via pickle: Avoid pickle; use safer alternatives like JSON.

**结论**: Common Python vulnerabilities can be mitigated by avoiding dangerous functions, using safe APIs, and validating inputs.

### 自动评分

- composite: 0.979
- topic_coverage: 1.0
- strategy: FULL_GEN

### 人工核查

| 项目 | 勾选 |
|------|------|
| 事实正确 | [ ] |
| 部分正确 | [ ] |
| 错误 | [ ] |

备注：

---

## s2_web_security

**领域**: security

### 题目

Perform a comprehensive web application security audit covering: CSRF protection mechanisms, JWT token validation best practices, CORS misconfiguration risks, rate limiting strategies, OAuth 2.0 implementation pitfalls, and SQL injection prevention in ORM-based applications. Provide severity ratings for each vulnerability class.

### 系统回答（自动）

**要点**:
1. CSRF: Missing anti-CSRF tokens in state-changing requests (Severity: High).
2. JWT: Weak signature algorithm (e.g., 'none') or missing expiration (Severity: Critical).
3. CORS: Overly permissive Access-Control-Allow-Origin: * (Severity: Medium).
4. Rate Limiting: No rate limiting on login endpoints (Severity: High).
5. SQL Injection: ORM used but raw queries via string formatting (Severity: Critical).

**结论**: The audit reveals critical vulnerabilities in JWT validation and SQL injection, requiring immediate remediation to prevent data breaches and unauthorized access.

### 自动评分

- composite: 0.943
- topic_coverage: 0.8333333333333334
- strategy: TEMPLATE_FILL

### 人工核查

| 项目 | 勾选 |
|------|------|
| 事实正确 | [ ] |
| 部分正确 | [ ] |
| 错误 | [ ] |

备注：

---

## d1_query_optimization

**领域**: database

### 题目

Research SQL query optimization techniques: index selection strategies (B-tree vs Hash vs GiST), EXPLAIN ANALYZE interpretation, common subquery optimizations, JOIN algorithm selection (nested loop vs hash join vs merge join), and materialized view usage patterns. Provide practical optimization guidelines.

### 系统回答（自动）

**要点**:
1. B-tree indexes are best for equality and range queries; Hash indexes for exact matches; GiST for complex data types and full-text search.
2. EXPLAIN ANALYZE shows actual execution times and row counts, helping identify sequential scans, high-cost nodes, and misestimated rows.
3. Common subquery optimizations: rewrite as JOINs, use EXISTS instead of IN for large sets, and apply subquery flattening where possible.
4. Nested loop joins are efficient for small datasets; hash joins for large, unsorted data; merge joins for sorted inputs or when sorting is cheap.
5. Materialized views improve query performance by precomputing and storing expensive aggregations or joins, refreshed on schedule or on-demand.

**结论**: Effective SQL optimization requires matching index types to query patterns, interpreting EXPLAIN ANALYZE output to identify bottlenecks, rewriting subqueries efficiently, choosing appropriate join algorithms based on data size and sort order, and leveraging materialized views for costly aggregations with controlled refresh strategies.

### 自动评分

- composite: 0.984
- topic_coverage: 1.0
- strategy: FULL_GEN

### 人工核查

| 项目 | 勾选 |
|------|------|
| 事实正确 | [ ] |
| 部分正确 | [ ] |
| 错误 | [ ] |

备注：

---

## d4_data_modeling

**领域**: database

### 题目

Research data modeling best practices: normalization vs denormalization trade-offs, star schema vs snowflake schema for analytics, entity-relationship modeling patterns, temporal data handling, hierarchical data representation (adjacency list vs nested sets vs materialized path), and polyglot persistence strategies.

### 系统回答（自动）

**要点**:
1. Normalization reduces redundancy but increases joins; denormalization improves read performance at the cost of update anomalies.
2. Star schema is simpler and faster for analytics; snowflake schema normalizes dimensions for storage efficiency.
3. Adjacency list is simple for trees but recursive; nested sets allow fast subtree queries but slow updates; materialized path balances both.
4. Temporal data handling requires valid time and transaction time; bitemporal modeling captures both.
5. Polyglot persistence uses multiple databases optimized for different data models within one system.

**结论**: Data modeling best practices involve balancing normalization and denormalization based on workload, choosing appropriate schema for analytics, selecting hierarchical representation method per use case, implementing temporal data handling for historical accuracy, and adopting polyglot persistence for diverse data needs.

### 自动评分

- composite: 0.915
- topic_coverage: 1.0
- strategy: TEMPLATE_FILL_DROP

### 人工核查

| 项目 | 勾选 |
|------|------|
| 事实正确 | [ ] |
| 部分正确 | [ ] |
| 错误 | [ ] |

备注：

---
