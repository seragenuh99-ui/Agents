"""共享记忆存储 — SQLite 元数据 + FAISS 向量索引。

本模块实现了混合存储架构：
- SQLite：存储记忆的结构化元数据（来源、主题、标签、摘要、内容等）
- FAISS IndexFlatIP：存储 384 维嵌入向量，支持 L2 归一化后的内积相似度搜索

支持的搜索模式：
1. search_by_keyword(): SQL LIKE 全文搜索 → to_search_text() relevance re-rank
2. search_by_tags(): 标签 OR 匹配
3. search_by_similarity(): FAISS 语义相似度搜索 → P2-lite 效用重排

去重机制：
- store() 在插入前检查语义重复（cosine > 0.95 + 同 memory_type）
- 若重复则复用已有记忆，仅记录访问日志

SafeSieve-lite 模板评估：
- template_success_rate(): 计算模板历史填充成功率
- effective_similarity(): 降低填充成功率低的策略的相似度排名
- record_template_outcome(): 记录每次模板使用结果
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

from .models import MemoryUnit


class MemoryStore:
    """持久化共享记忆存储：SQLite 管理元数据，FAISS 管理向量索引。

    线程安全（threading.Lock），支持：
    - 关键词搜索、标签搜索、语义相似度搜索
    - 跨任务记忆复用
    - 语义去重
    - 访问日志与命中率统计
    - 域模板检索（用于 Planner 的 template-fill 优化）
    """

    HNSW_THRESHOLD = 100_000  # 切换到 HNSW 索引的向量数量阈值（当前数据量下不会触发）

    def __init__(self, db_path: str = "shared_memory.db", embedding_dim: int = 384):
        self.db_path = db_path
        self.embedding_dim = embedding_dim
        self._lock = threading.Lock()
        self._index = None                       # FAISS 索引实例
        self._id_to_idx: Dict[str, int] = {}     # memory_id → FAISS 索引位置
        self._idx_to_id: Dict[int, str] = {}     # FAISS 索引位置 → memory_id

        self._init_db()

    # ============================================================
    # 数据库初始化
    # ============================================================

    def _init_db(self) -> None:
        """初始化 SQLite 表结构和索引，并加载已有嵌入向量到 FAISS。"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id TEXT PRIMARY KEY,
                    source_agent TEXT,
                    created_at REAL,
                    task_topic TEXT,
                    task_id TEXT,
                    summary TEXT,
                    content TEXT,
                    tags TEXT,
                    evidence_chain TEXT,
                    embedding BLOB,
                    access_count INTEGER DEFAULT 0,
                    last_accessed REAL DEFAULT 0,
                    confidence REAL DEFAULT 1.0,
                    memory_type TEXT DEFAULT 'result',
                    abstraction_level INTEGER DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_task_topic ON memories(task_topic)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_tags ON memories(tags)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memory_type ON memories(memory_type)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_abstraction_level ON memories(abstraction_level)
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS access_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id TEXT,
                    agent_id TEXT,
                    accessed_at REAL,
                    task_id TEXT
                )
            """)
            # 迁移：为已有数据库添加新列
            try:
                conn.execute("ALTER TABLE memories ADD COLUMN abstraction_level INTEGER DEFAULT 0")
            except sqlite3.OperationalError:
                pass
            for col, typedef in (
                ("fill_attempts", "INTEGER DEFAULT 0"),
                ("fill_successes", "INTEGER DEFAULT 0"),
            ):
                try:
                    conn.execute(f"ALTER TABLE memories ADD COLUMN {col} {typedef}")
                except sqlite3.OperationalError:
                    pass
            conn.commit()

        self._load_embeddings()

    # ============================================================
    # FAISS 索引管理
    # ============================================================

    def _load_embeddings(self) -> None:
        """启动时从 SQLite 加载已有嵌入向量到 FAISS 索引。"""
        if not HAS_FAISS:
            self._index = None
            return

        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT memory_id, embedding FROM memories WHERE embedding IS NOT NULL"
            ).fetchall()

        if not rows:
            self._index = self._create_index()
            return

        embeddings = []
        for i, (mem_id, emb_blob) in enumerate(rows):
            emb = np.frombuffer(emb_blob, dtype=np.float32)
            if len(emb) == self.embedding_dim:
                embeddings.append(emb)
                self._id_to_idx[mem_id] = i
                self._idx_to_id[i] = mem_id

        if embeddings:
            emb_matrix = np.stack(embeddings).astype(np.float32)
            self._index = self._create_index()
            faiss.normalize_L2(emb_matrix)
            self._index.add(emb_matrix)
        else:
            self._index = self._create_index()

    def _create_index(self):
        """创建 FAISS 索引。

        使用 IndexFlatIP（内积索引）进行精确搜索。
        内积等价于 L2 归一化后的余弦相似度。
        HNSW 升级路径保留但当前数据量不会触发（阈值：100K 向量）。
        """
        if not HAS_FAISS:
            return None
        return faiss.IndexFlatIP(self.embedding_dim)

    def _ensure_index(self) -> None:
        """检查是否需要将 FlatIP 升级为 HNSW 索引。

        当前数据规模下（< 100K 向量）FlatIP 足够快，此方法为未来扩展预留。
        """
        if not HAS_FAISS or self._index is None:
            return
        if isinstance(self._index, faiss.IndexHNSWFlat):
            return
        if self._index.ntotal < self.HNSW_THRESHOLD:
            return

        M = 32
        new_index = faiss.IndexHNSWFlat(self.embedding_dim, M)
        embs = []
        for i in range(self._index.ntotal):
            vec = np.zeros(self.embedding_dim, dtype=np.float32)
            self._index.reconstruct(i, vec)
            embs.append(vec)
        if embs:
            emb_matrix = np.stack(embs).astype(np.float32)
            faiss.normalize_L2(emb_matrix)
            new_index.add(emb_matrix)
        self._index = new_index

    # ============================================================
    # 去重
    # ============================================================

    def _find_duplicate(self, embedding: List[float], threshold: float = 0.95) -> Optional[str]:
        """检查是否存在近乎相同的嵌入向量（语义去重）。

        Args:
            embedding: 待检查的嵌入向量
            threshold: 余弦相似度阈值（默认 0.95）

        Returns:
            若存在则返回已有记忆 ID，否则返回 None
        """
        if self._index is None or self._index.ntotal == 0:
            return None
        emb = np.array(embedding, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(emb)
        scores, indices = self._index.search(emb, 1)
        if scores[0][0] > threshold:
            idx = int(indices[0][0])
            return self._idx_to_id.get(idx)
        return None

    # ============================================================
    # 存储
    # ============================================================

    def store(self, memory: MemoryUnit, dedup: bool = True) -> str:
        """存储一条记忆。返回 memory_id。

        去重逻辑（dedup=True 时）：
        - 仅对同 memory_type 进行去重（不同类型可能共享 embedding_text 但含义不同）
        - 若找到 cosine > 0.95 的同类型已有记忆，则复用并记录访问

        存储路径：
        - SQLite 写入/更新元数据（INSERT OR REPLACE）
        - FAISS 索引添加新向量
        """
        # 语义去重检查
        if dedup and memory.embedding is not None and self._index is not None and self._index.ntotal > 0:
            existing_id = self._find_duplicate(memory.embedding)
            if existing_id and existing_id != memory.memory_id:
                existing_mem = self.get(existing_id)
                if existing_mem and existing_mem.memory_type == memory.memory_type:
                    self.record_access(existing_id, memory.source_agent, memory.task_id)
                    return existing_id

        with self._lock:
            # SQLite 存储
            with sqlite3.connect(self.db_path) as conn:
                emb_blob = None
                if memory.embedding is not None:
                    emb_blob = np.array(memory.embedding, dtype=np.float32).tobytes()

                conn.execute(
                    """INSERT OR REPLACE INTO memories
                    (memory_id, source_agent, created_at, task_topic, task_id,
                     summary, content, tags, evidence_chain, embedding,
                     access_count, last_accessed, confidence, memory_type,
                     abstraction_level, fill_attempts, fill_successes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        memory.memory_id,
                        memory.source_agent,
                        memory.created_at,
                        memory.task_topic,
                        memory.task_id,
                        memory.summary,
                        memory.content,
                        json.dumps(memory.tags),
                        json.dumps(memory.evidence_chain),
                        emb_blob,
                        memory.access_count,
                        memory.last_accessed,
                        memory.confidence,
                        memory.memory_type,
                        memory.abstraction_level,
                        memory.fill_attempts,
                        memory.fill_successes,
                    ),
                )
                conn.commit()

            # FAISS 索引更新
            if memory.embedding is not None and self._index is not None and HAS_FAISS:
                emb = np.array(memory.embedding, dtype=np.float32).reshape(1, -1)
                faiss.normalize_L2(emb)
                self._id_to_idx[memory.memory_id] = self._index.ntotal
                self._idx_to_id[self._index.ntotal] = memory.memory_id
                self._index.add(emb)
                self._ensure_index()

            return memory.memory_id

    # ============================================================
    # 检索
    # ============================================================

    def get(self, memory_id: str) -> Optional[MemoryUnit]:
        """按 ID 检索单条记忆。"""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT * FROM memories WHERE memory_id = ?", (memory_id,)
                ).fetchone()

        if not row:
            return None

        return self._row_to_memory(row)

    def search_by_keyword(self, query: str, limit: int = 10) -> List[MemoryUnit]:
        """全文关键词搜索（SQL LIKE + relevance re-rank）。

        在 task_topic、summary、tags、content 四个字段中搜索，
        先按访问次数排序取候选，再用 to_search_text() 按词命中数重排。
        """
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """SELECT * FROM memories
                WHERE task_topic LIKE ? OR summary LIKE ? OR tags LIKE ? OR content LIKE ?
                ORDER BY access_count DESC, created_at DESC
                LIMIT ?""",
                (f"%{query}%", f"%{query}%", f"%{query}%", f"%{query}%", limit),
            ).fetchall()

        memories = [self._row_to_memory(row) for row in rows]
        # Relevance re-rank：统计 query 中每个词在 to_search_text() 中的命中次数
        query_lower = query.lower()
        scored = []
        for mem in memories:
            search_text = mem.to_search_text().lower()
            score = sum(1 for word in query_lower.split() if word in search_text)
            scored.append((score, mem))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [mem for _, mem in scored[:limit]]

    def search_by_tags(self, tags: List[str], limit: int = 10) -> List[MemoryUnit]:
        """标签搜索：匹配任意一个标签的记忆（OR 逻辑），按访问次数排序。"""
        results = []
        with sqlite3.connect(self.db_path) as conn:
            for tag in tags:
                rows = conn.execute(
                    """SELECT * FROM memories
                    WHERE tags LIKE ?
                    ORDER BY access_count DESC
                    LIMIT ?""",
                    (f"%{tag}%", limit),
                ).fetchall()
                for row in rows:
                    mem = self._row_to_memory(row)
                    if mem.memory_id not in {r.memory_id for r in results}:
                        results.append(mem)
        return results[:limit]

    def search_by_similarity(
        self, query_embedding: List[float], limit: int = 10,
        memory_type: Optional[str] = None,
    ) -> List[Tuple[MemoryUnit, float]]:
        """FAISS 语义相似度搜索。

        Args:
            query_embedding: 查询嵌入向量（384 维）
            limit: 返回结果数
            memory_type: 可选，限定记忆类型（如 "strategy"）

        Returns:
            (MemoryUnit, similarity_score) 列表，按综合效用降序排列

        P2-lite 效用重排：
            final_score = 0.75 × sim + 0.15 × utility + 0.10 × success_rate
            其中 utility = min(1.0, access_count/10), success_rate = 模板填充成功率
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        self._ensure_index()

        query = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(query)

        # 当限定类型时，扩大搜索池以避免稀有类型被挤出 top-k
        search_k = min(limit * 3, self._index.ntotal) if memory_type else limit
        scores, indices = self._index.search(query, search_k)

        candidates: List[Tuple[MemoryUnit, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                mem_id = self._idx_to_id.get(int(idx))
                if mem_id:
                    mem = self.get(mem_id)
                    if mem:
                        if memory_type and mem.memory_type != memory_type:
                            continue
                        candidates.append((mem, float(score)))

        # P2-lite 效用重排（PlugMem / MS：优先使用被频繁复用的记忆）
        def _utility_key(item: Tuple[MemoryUnit, float]) -> float:
            mem, sim = item
            utility = min(1.0, (mem.access_count or 0) / 10.0)
            success = self.template_success_rate(mem)
            return 0.75 * sim + 0.15 * utility + 0.10 * success

        candidates.sort(key=_utility_key, reverse=True)
        return candidates[:limit]

    # ============================================================
    # 模板与具体记忆检索
    # ============================================================

    def get_templates(
        self, tags: List[str], memory_type: str = "strategy", limit: int = 3,
    ) -> List[MemoryUnit]:
        """获取匹配标签的领域模板（abstraction_level >= 1）。

        用于 Planner 的 template-fill 优化：复用已有的成功计划模板，
        避免从零生成。
        """
        with sqlite3.connect(self.db_path) as conn:
            conditions = [
                "memory_type = ?",
                "abstraction_level >= 1",
                "(" + " OR ".join(["tags LIKE ?" for _ in tags]) + ")",
            ]
            params: list = [memory_type]
            for tag in tags:
                params.append(f"%{tag}%")
            sql = f"""SELECT * FROM memories
                WHERE {' AND '.join(conditions)}
                ORDER BY abstraction_level DESC, access_count DESC
                LIMIT ?"""
            params.append(limit)
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_memory(row) for row in rows]

    def get_concrete_memories(
        self, tags: List[str], memory_type: str, limit: int = 5,
    ) -> List[MemoryUnit]:
        """获取匹配标签的具体记忆（abstraction_level = 0）。

        用于 Promoter：收集 N>=2 个同类型具体记忆后，
        合成一个领域模板。
        """
        with sqlite3.connect(self.db_path) as conn:
            conditions = [
                "memory_type = ?",
                "abstraction_level = 0",
                "(" + " OR ".join(["tags LIKE ?" for _ in tags]) + ")",
            ]
            params: list = [memory_type]
            for tag in tags:
                params.append(f"%{tag}%")
            sql = f"""SELECT * FROM memories
                WHERE {' AND '.join(conditions)}
                ORDER BY created_at DESC
                LIMIT ?"""
            params.append(limit)
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_memory(row) for row in rows]

    def count_by_tags_and_type(
        self, tags: List[str], memory_type: str,
        abstraction_level: Optional[int] = None,
    ) -> int:
        """统计匹配标签和类型的记忆数量。

        用于 Promoter 判断是否达到合成模板的阈值（N>=2）。
        """
        with sqlite3.connect(self.db_path) as conn:
            conditions = ["memory_type = ?"]
            params: list = [memory_type]
            if abstraction_level is not None:
                conditions.append("abstraction_level = ?")
                params.append(abstraction_level)
            conditions.append("(" + " OR ".join(["tags LIKE ?" for _ in tags]) + ")")
            for tag in tags:
                params.append(f"%{tag}%")
            sql = f"SELECT COUNT(*) FROM memories WHERE {' AND '.join(conditions)}"
            return conn.execute(sql, params).fetchone()[0]

    # ============================================================
    # SafeSieve-lite 模板评估
    # ============================================================

    @staticmethod
    def template_success_rate(mem: MemoryUnit) -> float:
        """计算模板的历史填充成功率。

        无数据时返回 1.0（乐观初始化），有数据时返回实际比例。
        """
        if mem.fill_attempts <= 0:
            return 1.0
        return mem.fill_successes / mem.fill_attempts

    def effective_similarity(self, mem: MemoryUnit, raw_score: float) -> float:
        """根据模板历史成功率调整相似度分数。

        - 成功率 >= 25%：轻微折扣（× 0.65~1.0）
        - 尝试 >= 2 次且成功率 < 25%：大幅降权（× 0.45）
        防止低质量模板被反复选中。
        """
        rate = self.template_success_rate(mem)
        if mem.fill_attempts >= 2 and rate < 0.25:
            return raw_score * 0.45
        return raw_score * (0.65 + 0.35 * rate)

    def record_template_outcome(self, memory_id: str, success: bool) -> None:
        """记录模板填充的结果（成功/失败）。

        用于 SafeSieve-lite：追踪模板的实际使用效果。
        """
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """UPDATE memories SET
                       fill_attempts = fill_attempts + 1,
                       fill_successes = fill_successes + ?
                       WHERE memory_id = ?""",
                    (1 if success else 0, memory_id),
                )
                conn.commit()

    # ============================================================
    # 访问记录与统计
    # ============================================================

    def record_access(self, memory_id: str, agent_id: str, task_id: str) -> None:
        """记录一次记忆访问（更新 access_count 并写入 access_log）。"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE memories SET access_count = access_count + 1, last_accessed = ? WHERE memory_id = ?",
                (time.time(), memory_id),
            )
            conn.execute(
                "INSERT INTO access_log (memory_id, agent_id, accessed_at, task_id) VALUES (?, ?, ?, ?)",
                (memory_id, agent_id, time.time(), task_id),
            )
            conn.commit()

    def get_hit_rate(self) -> float:
        """计算记忆命中率：unique_accessed / total_accesses。"""
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM access_log").fetchone()[0]
            unique = conn.execute(
                "SELECT COUNT(DISTINCT memory_id) FROM access_log"
            ).fetchone()[0]
        return unique / max(total, 1) if total > 0 else 0.0

    def get_stats(self) -> Dict[str, Any]:
        """获取记忆系统的综合统计信息。

        Returns:
            包含总数、访问量、命中率、按类型/Agent/主题分布、向量索引大小。
        """
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            total_accesses = conn.execute(
                "SELECT SUM(access_count) FROM memories"
            ).fetchone()[0] or 0
            by_type = conn.execute(
                "SELECT memory_type, COUNT(*) FROM memories GROUP BY memory_type"
            ).fetchall()
            by_agent = conn.execute(
                "SELECT source_agent, COUNT(*) FROM memories GROUP BY source_agent"
            ).fetchall()
            by_topic = conn.execute(
                "SELECT task_topic, COUNT(*) FROM memories GROUP BY task_topic ORDER BY COUNT(*) DESC"
            ).fetchall()

        return {
            "total_memories": total,
            "total_accesses": total_accesses,
            "hit_rate": self.get_hit_rate(),
            "by_type": dict(by_type),
            "by_agent": dict(by_agent),
            "by_topic": dict(by_topic),
            "vector_index_size": self._index.ntotal if self._index else 0,
        }

    # ============================================================
    # 清理
    # ============================================================

    def clear(self) -> None:
        """清空所有记忆（测试用）。"""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM memories")
                conn.execute("DELETE FROM access_log")
                conn.commit()
            self._id_to_idx.clear()
            self._idx_to_id.clear()
            if self._index is not None and HAS_FAISS:
                self._index = self._create_index()

    # ============================================================
    # 内部辅助
    # ============================================================

    def _row_to_memory(self, row: Tuple) -> MemoryUnit:
        """将 SQLite 行元组转换为 MemoryUnit 对象。"""
        cols = [
            "memory_id", "source_agent", "created_at", "task_topic", "task_id",
            "summary", "content", "tags", "evidence_chain", "embedding",
            "access_count", "last_accessed", "confidence", "memory_type",
            "abstraction_level", "fill_attempts", "fill_successes",
        ]
        padded = list(row) + [0] * max(0, len(cols) - len(row))
        d = dict(zip(cols, padded))
        emb_blob = d.get("embedding")
        embedding = None
        if emb_blob is not None:
            embedding = np.frombuffer(emb_blob, dtype=np.float32).tolist()
        return MemoryUnit(
            memory_id=d["memory_id"],
            source_agent=d["source_agent"],
            created_at=d["created_at"],
            task_topic=d["task_topic"],
            task_id=d["task_id"],
            summary=d["summary"],
            content=d["content"],
            tags=json.loads(d["tags"]) if d["tags"] else [],
            evidence_chain=json.loads(d["evidence_chain"]) if d["evidence_chain"] else [],
            embedding=embedding,
            access_count=d["access_count"],
            last_accessed=d["last_accessed"],
            confidence=d["confidence"],
            memory_type=d["memory_type"],
            abstraction_level=d.get("abstraction_level", 0),
            fill_attempts=int(d.get("fill_attempts") or 0),
            fill_successes=int(d.get("fill_successes") or 0),
        )
