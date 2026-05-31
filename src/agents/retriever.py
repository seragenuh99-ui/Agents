"""Retriever Agent — 多源信息检索：共享记忆 + 知识库 + LLM 研究。

RetrieverAgent 负责为任务搜集相关证据，支持三种检索路径：
1. 共享记忆搜索（跨任务复用）：通过 query_memory() 三路搜索
2. 模拟知识库搜索（KNOWLEDGE_BASE）：关键词 + 标签路由匹配
3. LLM 开放题研究（_llm_open_qa_research）：当 demo 知识库不适用时

检索结果缓存：
  对 cos > 0.88 的相似查询，复用已有的 evidence 类型检索结果，
  跳过冗余搜索。

标签路由优化：
  _TAG_KB_CATEGORY 将标签映射到知识库类别，减少跨领域噪音。
  例如 "solar" 标签只搜索 "renewable energy" 类别。
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from .base import BaseAgent, LLMBackend, LLMError
from ..task_intent import is_open_qa
from ..protocol import Message, MessageType, ActionType
from ..protocol.scheduler import Scheduler
from ..state.embeddings import EmbeddingEngine
from ..state.exchange import StateExchangeBus
from ..memory.store import MemoryStore
from ..memory.models import MemoryUnit


# ============================================================
# 模拟知识库 — 用于确定性测试
# ============================================================

KNOWLEDGE_BASE = {
    "renewable energy": {
        "solar": {
            "summary": "Solar energy converts sunlight into electricity using photovoltaic cells or concentrated solar power. Modern solar panels achieve 15-22% efficiency. Global installed capacity exceeded 1 TW in 2022. Costs have dropped 90% since 2010. Key challenges include intermittency and energy storage.",
            "facts": [
                "Photovoltaic effect discovered by Edmond Becquerel in 1839",
                "First practical silicon solar cell: Bell Labs, 1954 (6% efficiency)",
                "China leads global solar production with 80%+ market share in panel manufacturing",
                "Perovskite solar cells show promise for >30% efficiency in tandem configurations",
                "Average solar panel lifespan: 25-30 years with 0.5%/year degradation",
            ],
            "technologies": ["Monocrystalline Silicon", "Polycrystalline Silicon", "Thin-Film", "Perovskite", "Bifacial"],
            "key_metrics": {"avg_efficiency": "18-22%", "cost_per_watt": "$0.20-0.30", "energy_payback_time": "1-4 years", "carbon_footprint": "40-50 gCO2/kWh"},
            "tags": ["solar", "photovoltaic", "renewable", "energy", "sun"],
        },
        "wind": {
            "summary": "Wind energy harnesses kinetic energy from wind using turbines to generate electricity. Onshore and offshore wind farms are major renewable energy sources. Turbines convert 35-45% of wind kinetic energy. Global capacity reached 900+ GW in 2023. Offshore wind is growing rapidly with larger turbines.",
            "facts": [
                "First electricity-generating wind turbine: Charles Brush, 1888, Cleveland Ohio",
                "Modern turbines can exceed 15 MW capacity with 240m+ rotor diameter",
                "Denmark generates 50%+ of its electricity from wind power",
                "Offshore wind capacity factor reaches 40-55% vs 25-35% for onshore",
                "Floating offshore wind enables deployment in deep waters (>60m)",
            ],
            "technologies": ["Horizontal Axis", "Vertical Axis", "Offshore Fixed", "Floating Offshore", "Small/Distributed"],
            "key_metrics": {"avg_capacity_factor": "25-55%", "cost_per_mwh": "$30-60", "turbine_lifespan": "20-25 years", "carbon_footprint": "10-12 gCO2/kWh"},
            "tags": ["wind", "turbine", "renewable", "energy", "offshore"],
        },
        "hydro": {
            "summary": "Hydropower uses flowing or falling water to generate electricity. It's the largest renewable electricity source globally, providing 16% of world electricity. Includes conventional dams, run-of-river, and pumped storage for grid balancing.",
            "facts": [
                "Hydropower provides ~16% of global electricity generation",
                "Three Gorges Dam (China): 22.5 GW, world's largest power station",
                "Pumped storage represents 95%+ of global energy storage capacity",
                "Small hydro (<10 MW) has lower environmental impact",
                "Norway generates 95%+ of its electricity from hydropower",
            ],
            "technologies": ["Conventional Dam", "Run-of-River", "Pumped Storage", "Small Hydro", "Tidal"],
            "key_metrics": {"efficiency": "90%+ (turbine efficiency)", "capacity_factor": "30-60%", "lifespan": "50-100 years", "carbon_footprint": "24 gCO2/kWh (reservoir)"},
            "tags": ["hydro", "water", "renewable", "energy", "dam"],
        },
    },
    "code analysis": {
        "security_patterns": {
            "summary": "Common security vulnerability patterns in Python code: SQL injection, XSS, command injection, insecure deserialization, hardcoded credentials, path traversal, and improper access control.",
            "patterns": [
                {"name": "SQL Injection", "detection": "String formatting in SQL queries: f-strings, .format(), % operator with user input", "severity": "critical", "fix": "Use parameterized queries with ? placeholders"},
                {"name": "Command Injection", "detection": "os.system(), subprocess with shell=True and user input", "severity": "critical", "fix": "Use subprocess.run with list args and shell=False"},
                {"name": "Path Traversal", "detection": "File operations with unsanitized user input paths", "severity": "high", "fix": "Use os.path.realpath and validate against allowed directories"},
                {"name": "Hardcoded Secrets", "detection": "API keys, passwords, tokens directly in source code", "severity": "high", "fix": "Use environment variables or secret management services"},
                {"name": "Insecure Deserialization", "detection": "pickle.loads with untrusted input", "severity": "critical", "fix": "Use JSON or other safe serialization formats"},
            ],
            "tags": ["security", "vulnerability", "code", "python", "pattern"],
        },
        "best_practices": {
            "summary": "Python coding best practices: use type hints, follow PEP 8, write docstrings, use context managers, prefer list comprehensions over loops for simple cases, use generators for large datasets.",
            "practices": ["Use pathlib instead of os.path for file operations", "Use dataclasses for simple data containers", "Use f-strings for string formatting (Python 3.6+)", "Handle exceptions at the appropriate level", "Write tests with pytest"],
            "tags": ["best-practices", "python", "coding", "style"],
        },
    },
    "database systems": {
        "sql_optimization": {"summary": "SQL optimization: choose B-tree indexes for range/equality, Hash for equality-only, GiST for full-text/geo. Use EXPLAIN ANALYZE for actual row counts and timing. Prefer hash join for large equi-joins, nested loop for small tables, merge join for sorted inputs. Materialized views trade storage for read speed.", "tags": ["database", "sql", "index", "query", "optimization", "explain"]},
        "nosql_comparison": {"summary": "Redis: in-memory key-value, low latency, eventual consistency in cluster mode. MongoDB: document model, flexible schema, secondary indexes. Cassandra: wide-column, linear write scaling, tunable consistency. Neo4j: graph traversals, Cypher queries, relationship-heavy workloads.", "tags": ["database", "nosql", "redis", "mongodb", "cassandra", "neo4j"]},
        "performance_tuning": {"summary": "Connection pooling (PgBouncer, HikariCP) reduces handshake overhead. PostgreSQL: shared_buffers ~25% RAM, effective_cache_size ~75% RAM, tune wal_buffers and checkpoint_timeout. MySQL: innodb_buffer_pool_size dominant. Replication lag: synchronous replicas vs async, parallel apply.", "tags": ["database", "performance", "postgresql", "mysql", "pool", "replication"]},
        "data_modeling": {"summary": "Normalization reduces redundancy; denormalization speeds reads. Star schema: fact table + dimension tables for OLAP; snowflake normalizes dimensions further. Temporal data: valid-time vs transaction-time columns. Hierarchies: adjacency list (simple), nested sets (read-heavy), materialized path (flexible queries).", "tags": ["database", "modeling", "schema", "normalization", "star", "entity"]},
    },
}

# 标签 → KB 类别路由表（减少跨领域噪音）
_TAG_KB_CATEGORY: Dict[str, str] = {}
for _cat, _topics in KNOWLEDGE_BASE.items():
    for _data in _topics.values():
        for _tag in _data.get("tags", []):
            _TAG_KB_CATEGORY[_tag] = _cat


# ============================================================
# RetrieverAgent — 多源信息检索
# ============================================================

class RetrieverAgent(BaseAgent):
    """信息检索 Agent：搜索内部记忆和外部知识源。

    核心能力：retrieve（检索）, search（搜索）, query_memory（记忆查询）, store_memory（记忆存储）

    检索流程：
    1. 合并 Orchestrator 的主动建议记忆
    2. 缓存检查：cos > 0.88 时复用已有检索结果
    3. 共享记忆搜索（三路：关键词 + 标签 + 语义）
    4. 知识库搜索（标签路由 + 关键词匹配，开放题跳过）
    5. 开放题 LLM 知识补充
    6. 合并去重 + 全局字符预算（AgentPrune）
    7. 存储检索结果 + 状态传递
    """

    def __init__(
        self,
        scheduler: Scheduler,
        llm: LLMBackend,
        embedding_engine: EmbeddingEngine,
        state_bus: StateExchangeBus,
        memory_store: MemoryStore,
        use_structured_protocol: bool = True,
    ):
        super().__init__(
            agent_id="retriever",
            role="retriever",
            capabilities=["retrieve", "search", "query_memory", "store_memory"],
            scheduler=scheduler,
            llm=llm,
            embedding_engine=embedding_engine,
            state_bus=state_bus,
            memory_store=memory_store,
            use_structured_protocol=use_structured_protocol,
        )
        self.run_options = None

    # ---- 消息处理 ----

    def handle_message(self, message: Message) -> Optional[Message]:
        if message.msg_type == MessageType.REQUEST:
            if message.action == ActionType.RETRIEVE:
                return self._handle_retrieve_request(message)
        return None

    def _handle_retrieve_request(self, message: Message) -> Optional[Message]:
        result = self.execute_task(message.params)
        return self.scheduler.send_response(
            message,
            result=result,
            embedding=self.embedding_engine.encode_state(result),
            memory_refs=result.get("memory_refs", []),
        )

    # ---- 核心任务执行 ----

    def execute_task(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """多策略信息检索。

        Args:
            task_input: 包含 query, tags, task_id, 可选 suggested_memories

        Returns:
            包含 memory_hits, knowledge_base_hits, combined_results 的字典
        """
        t0 = time.time()
        query = task_input.get("query", "")
        tags = task_input.get("tags", [])
        task_id = task_input.get("task_id", "unknown")
        self._context["task_id"] = task_id
        self._emit_status(f"② 检索：查询「{query[:40]}…」")

        results: Dict[str, Any] = {
            "query": query,
            "memory_hits": [],
            "knowledge_base_hits": [],
            "combined_results": "",
            "memory_refs": [],
        }

        # ---- 合并 Orchestrator 的主动建议记忆 ----
        suggested = task_input.get("suggested_memories", [])
        for s in suggested:
            results["memory_hits"].append({
                "memory_id": s["memory_id"],
                "summary": s["summary"],
                "type": s.get("type", "evidence"),
                "source": "proactive_suggestion",
                "relevance": s.get("relevance", 0),
            })
            results["memory_refs"].append(s["memory_id"])

        # ---- 检索结果缓存：cos > 阈值时复用已有结果 ----
        cache_th = 0.88
        if getattr(self, "run_options", None) is not None:
            cache_th = self.run_options.retriever_cache_threshold
        if self.memory_store._index is not None and self.memory_store._index.ntotal > 0:
            try:
                query_emb = self.embedding_engine.encode(query)
                similar = self.memory_store.search_by_similarity(query_emb, limit=3)
                for mem, score in similar:
                    if score > cache_th and mem.memory_type == "evidence":
                        cached_tags = mem.tags if hasattr(mem, "tags") else []
                        if tags and cached_tags and not any(t in cached_tags for t in tags):
                            continue
                        try:
                            cached = json.loads(mem.content)
                            if isinstance(cached, dict) and "memory_hits" in cached:
                                results = cached.copy()
                                results["_cached_from"] = mem.memory_id
                                results["query"] = query
                                results["elapsed_ms"] = (time.time() - t0) * 1000
                                break
                        except (json.JSONDecodeError, TypeError):
                            pass
            except Exception:
                pass

        # 无缓存命中时执行全新检索
        if not results.get("_cached_from"):
            # 1. 共享记忆搜索
            memory_results = self.query_memory(query=query, tags=tags, limit=5)
            for mem in memory_results:
                results["memory_hits"].append({
                    "memory_id": mem.memory_id,
                    "summary": mem.summary,
                    "content": mem.content[:300],
                    "topic": mem.task_topic,
                    "source": mem.source_agent,
                    "type": mem.memory_type,
                })
                results["memory_refs"].append(mem.memory_id)

            # 2. 知识库搜索（开放题跳过 — demo KB 仅覆盖能源/安全/数据库）
            if is_open_qa(query, tags):
                kb_hits = []
            else:
                kb_hits = self._search_knowledge_base(query, tags=tags)
            results["knowledge_base_hits"] = kb_hits

            # 2b. 开放题 LLM 知识补充
            if is_open_qa(query, tags):
                self._emit_status("② 检索：开放题，调用模型补充相关知识…")
                llm_notes = self._llm_open_qa_research(query)
                if llm_notes:
                    results["llm_research"] = llm_notes
                    results["memory_hits"].append({
                        "memory_id": "llm_research",
                        "summary": llm_notes[:200],
                        "content": llm_notes,
                        "type": "llm_research",
                        "source": "llm",
                    })

            # 3. 合并结果（去重 + 全局字符预算 — AgentPrune）
            max_ev = 750
            if getattr(self, "run_options", None) is not None:
                max_ev = self.run_options.evidence_max_chars
            combined = self._build_combined_evidence(
                results["memory_hits"], results["knowledge_base_hits"], max_chars=max_ev,
            )
            if results.get("llm_research"):
                extra = f"\n=== Model Research ===\n{results['llm_research']}"
                combined = (combined + extra) if combined else extra.strip()
                if len(combined) > max_ev:
                    combined = combined[:max_ev] + "\n...[truncated]"
            results["combined_results"] = combined
            results["hit_count"] = len(results["memory_hits"]) + len(results["knowledge_base_hits"])
        else:
            results["hit_count"] = len(results.get("memory_hits", [])) + len(results.get("knowledge_base_hits", []))
        results["memory_hit_count"] = len(results["memory_hits"])
        results["elapsed_ms"] = (time.time() - t0) * 1000

        # ---- 存储检索结果到共享记忆 ----
        if results["hit_count"] > 0:
            memory_id = self.store_memory(
                topic=f"Retrieval: {query[:80]}",
                summary=f"Found {results['hit_count']} results for: {query[:150]}",
                content=json.dumps(results, ensure_ascii=False, indent=2),
                tags=["retrieval", "search"] + tags,
                memory_type="evidence",
                embedding_text=query,
            )
            results["memory_refs"].append(memory_id)

        # ---- 状态传递：将检索嵌入发送给 Executor ----
        self.transfer_state(
            target_agent="executor",
            state_data={"retrieval_results": results, "query": query},
            context=f"Retrieval results for: {query[:100]}",
        )

        self._task_history.append({"task_id": task_id, "query": query, "hit_count": results["hit_count"]})
        return results

    # ---- 证据合并 ----

    @staticmethod
    def _build_combined_evidence(
        memory_hits: List[Dict[str, Any]],
        kb_hits: List[Dict[str, Any]],
        max_chars: int = 800,
    ) -> str:
        """合并检索命中结果，去重并应用字符预算上限。"""
        seen: set = set()
        lines: List[str] = []

        if memory_hits:
            lines.append("=== From Shared Memory ===")
            for hit in memory_hits:
                key = hit.get("memory_id") or (hit.get("summary") or "")[:80]
                if key in seen:
                    continue
                seen.add(key)
                lines.append(f"- [{hit.get('type', 'evidence')}] {hit.get('summary', '')[:180]}")

        if kb_hits:
            lines.append("=== From Knowledge Base ===")
            for hit in kb_hits:
                key = f"{hit.get('category')}:{hit.get('topic')}"
                if key in seen:
                    continue
                seen.add(key)
                summary = hit.get("summary", str(hit)[:200])
                lines.append(f"- {summary[:200]}")

        combined = "\n".join(lines)
        if len(combined) > max_chars:
            combined = combined[:max_chars] + "\n...[truncated]"
        return combined

    # ---- 知识库搜索 ----

    def _resolve_kb_categories(self, tags: List[str]) -> List[str]:
        """将任务标签映射到知识库类别；无标签时搜索全部类别。"""
        if tags and is_open_qa("", tags):
            return []
        if not tags:
            return list(KNOWLEDGE_BASE.keys())
        cats = []
        for tag in tags:
            cat = _TAG_KB_CATEGORY.get(tag.lower())
            if cat and cat not in cats:
                cats.append(cat)
        return cats or list(KNOWLEDGE_BASE.keys())

    def _search_knowledge_base(
        self, query: str, tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """搜索模拟知识库：标签路由 + 关键词匹配。

        标签路由减少跨领域噪音（例如 "solar" 标签只搜索 "renewable energy"）。
        """
        hits = []
        query_lower = query.lower()
        query_words = set(query_lower.split())
        categories = self._resolve_kb_categories(tags or [])

        for category in categories:
            topics = KNOWLEDGE_BASE.get(category, {})
            cat_words = set(category.split())
            for topic_name, topic_data in topics.items():
                topic_tags = set(t.lower() for t in topic_data.get("tags", []))
                tag_overlap = bool(tags and topic_tags & set(t.lower() for t in tags))
                topic_text = json.dumps(topic_data).lower()
                keyword_hit = (
                    query_words & cat_words
                    or any(word in topic_text for word in query_words)
                )
                if tag_overlap or keyword_hit:
                    hits.append({
                        "category": category,
                        "topic": topic_name,
                        "summary": topic_data.get("summary", ""),
                        "tags": topic_data.get("tags", []),
                        "source": "knowledge_base",
                    })

        return hits[:6]

    # ---- 开放题 LLM 研究 ----

    def _llm_open_qa_research(self, query: str) -> str:
        """当 demo 知识库不适用时，使用 LLM 获取背景知识。

        针对中文问题使用中文提示词，要求准确、实用、不分条列项。
        """
        cjk = any("一" <= c <= "鿿" for c in query)
        if cjk:
            system = (
                "你是知识检索助手。根据问题列出准确、实用的要点（地名、特色、建议），"
                "用中文，分条列出，不要编造不存在的具体店铺名，不确定可概括说明。"
            )
        else:
            system = (
                "You are a research assistant. List accurate, practical bullet points "
                "for the user's question. Same language as the question. No fabrication."
            )
        try:
            return self._call_llm(
                system,
                query[:600],
                max_tokens=520,
                temperature=0.35,
                status_hint="② 检索：开放题知识补充",
            )
        except LLMError:
            return ""
