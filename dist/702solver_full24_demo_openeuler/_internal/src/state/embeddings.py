"""语义嵌入引擎 — 非文本状态传递的核心编码器。

本模块实现了 Agent 间"不通过文本序列化/反序列化"传递状态的关键机制：
将 Agent 的内部状态编码为 384 维稠密语义向量，下游 Agent 直接使用该向量
进行 FAISS 语义检索，跳过"内部状态 → 文本 → 解析 → 内部状态"的冗余环节。

模型选择：
- 首选：BAAI/bge-small-en-v1.5（通过 sentence-transformers 加载，HF 镜像）
- 降级：确定性 SHAKE-256 哈希嵌入（离线/测试环境，无需网络下载）

核心方法：
  encode(text) → 单条文本编码为 384 维归一化向量
  encode_state(state_dict) → Agent 状态字典编码为嵌入向量
  combine(embeddings, weights) → 多嵌入加权融合（用于证据整合）
  encode_batch(texts) → 批量编码（用于单次 FAISS 搜索）
"""

from __future__ import annotations

import hashlib
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np


class EmbeddingEngine:
    """语义嵌入引擎：生成和管理 Agent 状态的稠密向量表示。

    双模式运行：
    - 真实模型模式（use_real_model=True）：加载 BGE-small-en-v1.5，生成语义嵌入
    - 哈希降级模式（模型加载失败或 use_real_model=False）：使用 SHAKE-256 确定性哈希
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", use_real_model: bool = True):
        self.model_name = model_name
        self.model = None
        self.dimension = 384                         # BGE-small-en-v1.5 的输出维度
        self.use_real_model = use_real_model
        self._generation_count = 0                   # 已生成的嵌入向量总数

        if use_real_model:
            self._init_model()

    # ---- 模型初始化 ----

    def _init_model(self) -> None:
        """加载 sentence-transformers 模型。

        优先从本地缓存加载（local_files_only=True），失败后尝试从 HF 镜像下载。
        若全部失败，自动降级为哈希嵌入模式。
        """
        try:
            from sentence_transformers import SentenceTransformer

            # 设置 HF 镜像（国内网络加速）
            if "HF_ENDPOINT" not in os.environ:
                os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            try:
                self.model = SentenceTransformer(
                    self.model_name,
                    local_files_only=True,
                )
            except Exception:
                # 本地无缓存，从镜像下载
                self.model = SentenceTransformer(self.model_name)

            self.dimension = self.model.get_embedding_dimension()
        except Exception as e:
            print(f"[WARNING] Could not load sentence-transformers model: {e}")
            print("[WARNING] Falling back to hash-based embeddings.")
            self.model = None
            self.use_real_model = False

    # ---- 编码（单条） ----

    def encode(self, text: str) -> List[float]:
        """将文本编码为归一化嵌入向量。

        真实模型：调用 SentenceTransformer.encode(normalize_embeddings=True)
        降级模式：SHAKE-256 哈希 → 确定性伪随机向量 → L2 归一化
        """
        self._generation_count += 1
        if self.model is not None:
            embedding = self.model.encode(text, normalize_embeddings=True)
            return embedding.tolist()

        return self._hash_embed(text)

    # ---- 批量编码 ----

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """批量编码多条文本，单次模型调用。

        用于 Orchestrator._suggest_memories_batch() 单次 FAISS 搜索。
        """
        self._generation_count += len(texts)
        if self.model is not None:
            embeddings = self.model.encode(texts, normalize_embeddings=True)
            return embeddings.tolist()

        return [self._hash_embed(t) for t in texts]

    # ---- 哈希降级 ----

    def _hash_embed(self, text: str) -> List[float]:
        """确定性哈希嵌入（降级方案）。

        使用 SHAKE-256 生成固定维度的伪随机向量：
        1. 文本 → SHAKE-256 摘要 → 字节序列
        2. 字节 → uint32 数组
        3. uint32 → [-1, 1] 范围的 float64
        4. L2 归一化

        相同输入始终产生相同向量，确保离线测试可复现。
        """
        h = hashlib.shake_256(text.encode("utf-8"))
        raw = h.digest(max(self.dimension * 4, 32))
        uint_arr = np.frombuffer(raw, dtype=np.uint32).copy()
        if len(uint_arr) < self.dimension:
            uint_arr = np.resize(uint_arr, self.dimension)
        result = ((uint_arr[:self.dimension].astype(np.float64) / np.float64(2**32)) * 2 - 1)
        norm = np.linalg.norm(result)
        if norm > 0:
            result = result / norm
        return result.tolist()

    # ---- 嵌入融合 ----

    def combine(self, embeddings: List[List[float]], weights: Optional[List[float]] = None) -> List[float]:
        """加权融合多个嵌入向量（用于证据整合/多源状态合并）。

        Args:
            embeddings: 待融合的嵌入向量列表
            weights: 各向量的权重（默认等权），自动归一化

        Returns:
            加权平均后的归一化向量
        """
        if not embeddings:
            return self._hash_embed("empty")

        if weights is None:
            weights = [1.0] * len(embeddings)

        total_weight = sum(weights)
        combined = np.zeros(len(embeddings[0]), dtype=np.float64)
        for emb, w in zip(embeddings, weights):
            combined += np.array(emb, dtype=np.float64) * (w / total_weight)

        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm

        return combined.tolist()

    # ---- 状态编码（非文本传递核心） ----

    def encode_state(self, state: Dict[str, Any]) -> List[float]:
        """将 Agent 状态字典编码为紧凑嵌入向量。

        这是非文本状态传递的关键入口：
        不是将完整状态序列化为文本，而是产生一个捕获核心语义的稠密向量。
        下游 Agent 直接使用该向量进行语义搜索，无需重新从文本编码。

        编码策略：将状态字典展平为 "key: value | key: value | ..." 格式文本，
        然后调用 encode() 生成嵌入。列表值截取前 10 个元素。
        """
        text_components = []
        for key, value in state.items():
            if isinstance(value, str):
                text_components.append(f"{key}: {value}")
            elif isinstance(value, (int, float)):
                text_components.append(f"{key}: {value}")
            elif isinstance(value, list):
                text_components.append(f"{key}: {' '.join(str(v) for v in value[:10])}")

        combined_text = " | ".join(text_components)
        return self.encode(combined_text)

    # ---- 工具方法 ----

    def decode_hints(self, embedding: List[float], top_k: int = 5) -> Dict[str, Any]:
        """从嵌入向量中提取结构分析信息（调试/分析用）。

        注意：嵌入向量不可完全逆向还原，此方法仅提取可解释的结构信号。

        Returns:
            包含范数、均值、标准差、稀疏度、主导维度等分析指标的字典
        """
        arr = np.array(embedding)
        top_indices = np.argsort(np.abs(arr))[-top_k:][::-1]
        return {
            "norm": float(np.linalg.norm(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "sparsity": float(np.sum(np.abs(arr) < 0.01) / len(arr)),
            "dominant_dimensions": [(int(i), float(arr[i])) for i in top_indices],
            "dimension": len(embedding),
        }

    @property
    def generation_count(self) -> int:
        """已生成的嵌入向量总数（含批量编码）。"""
        return self._generation_count

    def estimate_bits(self, embedding: List[float]) -> int:
        """估算嵌入向量的信息量（bits）。

        粗略估计：32 bits/float × 维度数
        """
        arr = np.array(embedding)
        return len(embedding) * 32
