"""Shared memory storage with SQLite metadata + FAISS vector index."""

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
    """Persistent shared memory storage combining SQLite and FAISS.

    Supports: keyword search, tag search, semantic similarity search,
    and cross-task memory reuse.
    """

    def __init__(self, db_path: str = "shared_memory.db", embedding_dim: int = 384):
        self.db_path = db_path
        self.embedding_dim = embedding_dim
        self._lock = threading.Lock()
        self._index = None
        self._id_to_idx: Dict[str, int] = {}
        self._idx_to_id: Dict[int, str] = {}
        self._embeddings: List[np.ndarray] = []
        self._dirty = False

        self._init_db()

    def _init_db(self) -> None:
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
            # Migration: add abstraction_level to pre-existing databases
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

    def _load_embeddings(self) -> None:
        """Load stored embeddings into FAISS index on startup."""
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

    HNSW_THRESHOLD = 100_000  # Switch to HNSW when ntotal >= this (FlatIP is fast enough for 100K)

    def _create_index(self):
        """Create a FAISS index. Uses FlatIP for small datasets, HNSW for large.

        IndexFlatIP: O(N) brute force — fast for < 32 vectors
        IndexHNSWFlat: O(log N) graph-based — fast for 32+ vectors
        """
        if not HAS_FAISS:
            return None
        return faiss.IndexFlatIP(self.embedding_dim)

    def _ensure_index(self) -> None:
        """Upgrade index from FlatIP to HNSW when enough vectors accumulated."""
        if not HAS_FAISS or self._index is None:
            return
        if isinstance(self._index, faiss.IndexHNSWFlat):
            return  # Already HNSW
        if self._index.ntotal < self.HNSW_THRESHOLD:
            return  # Not enough vectors yet

        # Rebuild as HNSW
        M = 32
        new_index = faiss.IndexHNSWFlat(self.embedding_dim, M)
        # Transfer vectors
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

    def _rebuild_index(self) -> None:
        """Rebuild FAISS index from stored embeddings."""
        self._load_embeddings()
        self._dirty = False

    def _find_duplicate(self, embedding: List[float], threshold: float = 0.95) -> Optional[str]:
        """Check if a near-identical embedding already exists (dedup)."""
        if self._index is None or self._index.ntotal == 0:
            return None
        emb = np.array(embedding, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(emb)
        scores, indices = self._index.search(emb, 1)
        if scores[0][0] > threshold:
            idx = int(indices[0][0])
            return self._idx_to_id.get(idx)
        return None

    def store(self, memory: MemoryUnit, dedup: bool = True) -> str:
        """Store a memory unit. Returns memory_id.

        When dedup=True, checks semantic similarity before storing;
        if a near-duplicate exists (cosine sim > 0.95), reuses it.
        """
        # Dedup check: only dedup against same memory_type.
        # Different types (e.g. strategy vs result) may share the same
        # embedding_text for alignment but are semantically distinct.
        if dedup and memory.embedding is not None and self._index is not None and self._index.ntotal > 0:
            existing_id = self._find_duplicate(memory.embedding)
            if existing_id and existing_id != memory.memory_id:
                existing_mem = self.get(existing_id)
                if existing_mem and existing_mem.memory_type == memory.memory_type:
                    self.record_access(existing_id, memory.source_agent, memory.task_id)
                    return existing_id

        with self._lock:
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

            if memory.embedding is not None and self._index is not None and HAS_FAISS:
                emb = np.array(memory.embedding, dtype=np.float32).reshape(1, -1)
                faiss.normalize_L2(emb)
                self._id_to_idx[memory.memory_id] = self._index.ntotal
                self._idx_to_id[self._index.ntotal] = memory.memory_id
                self._index.add(emb)
                self._ensure_index()

            return memory.memory_id

    def get(self, memory_id: str) -> Optional[MemoryUnit]:
        """Retrieve a memory by ID."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT * FROM memories WHERE memory_id = ?", (memory_id,)
                ).fetchone()

        if not row:
            return None

        return self._row_to_memory(row)

    @staticmethod
    def template_success_rate(mem: MemoryUnit) -> float:
        """SafeSieve-lite: historical template-fill success (1.0 if no data)."""
        if mem.fill_attempts <= 0:
            return 1.0
        return mem.fill_successes / mem.fill_attempts

    def effective_similarity(self, mem: MemoryUnit, raw_score: float) -> float:
        """Down-rank strategies with poor fill history."""
        rate = self.template_success_rate(mem)
        if mem.fill_attempts >= 2 and rate < 0.25:
            return raw_score * 0.45
        return raw_score * (0.65 + 0.35 * rate)

    def record_template_outcome(self, memory_id: str, success: bool) -> None:
        """Record whether reusing this strategy memory led to a good outcome."""
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

    def _row_to_memory(self, row: Tuple) -> MemoryUnit:
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

    def search_by_keyword(self, query: str, limit: int = 10) -> List[MemoryUnit]:
        """Full-text keyword search across task_topic, summary, tags, and content."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """SELECT * FROM memories
                WHERE task_topic LIKE ? OR summary LIKE ? OR tags LIKE ? OR content LIKE ?
                ORDER BY access_count DESC, created_at DESC
                LIMIT ?""",
                (f"%{query}%", f"%{query}%", f"%{query}%", f"%{query}%", limit),
            ).fetchall()

        return [self._row_to_memory(row) for row in rows]

    def search_by_tags(self, tags: List[str], limit: int = 10) -> List[MemoryUnit]:
        """Search memories matching any of the given tags."""
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
        """Semantic similarity search using FAISS vector index.

        When memory_type is set, searches a larger pool (up to limit*3) and
        filters to the requested type, so that rare types aren't crowded out
        by abundant ones in the top-k.
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        self._ensure_index()

        query = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(query)

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

        # P2-lite utility rerank (PlugMem / MS: prefer frequently reused memories)
        def _utility_key(item: Tuple[MemoryUnit, float]) -> float:
            mem, sim = item
            utility = min(1.0, (mem.access_count or 0) / 10.0)
            success = self.template_success_rate(mem)
            return 0.75 * sim + 0.15 * utility + 0.10 * success

        candidates.sort(key=_utility_key, reverse=True)
        return candidates[:limit]

    def count_by_tags_and_type(
        self, tags: List[str], memory_type: str,
        abstraction_level: Optional[int] = None,
    ) -> int:
        """Count memories matching the given tags and type."""
        with sqlite3.connect(self.db_path) as conn:
            # Count memories with at least one matching tag
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

    def get_templates(
        self, tags: List[str], memory_type: str = "strategy", limit: int = 3,
    ) -> List[MemoryUnit]:
        """Get domain templates matching the given tags and type."""
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
        """Get concrete (abstraction_level=0) memories matching tags."""
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

    def record_access(self, memory_id: str, agent_id: str, task_id: str) -> None:
        """Log memory access for analytics."""
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

    def get_hit_rate(self, query_count: int, hit_count: int) -> float:
        """Calculate memory hit rate."""
        return hit_count / max(query_count, 1)

    def get_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
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
            "by_type": dict(by_type),
            "by_agent": dict(by_agent),
            "by_topic": dict(by_topic),
            "vector_index_size": self._index.ntotal if self._index else 0,
        }

    def clear(self) -> None:
        """Clear all memories (for testing)."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM memories")
                conn.execute("DELETE FROM access_log")
                conn.commit()
            self._id_to_idx.clear()
            self._idx_to_id.clear()
            self._embeddings.clear()
            if self._index is not None and HAS_FAISS:
                self._index = self._create_index()
