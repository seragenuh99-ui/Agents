"""Embedding generation and transformation for non-text state passing.

Generates semantic embeddings from agent state, task context, and intermediate
results. These embeddings can be passed directly between agents without
text serialization/deserialization overhead.
"""

from __future__ import annotations

import hashlib
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np


class EmbeddingEngine:
    """Generates and manages semantic embeddings for agent state.

    Uses sentence-transformers for semantic embeddings with fallback
    to a deterministic hash-based embedding for offline/reproducible testing.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", use_real_model: bool = True):
        self.model_name = model_name
        self.model = None
        self.dimension = 384  # BAAI/bge-small-en-v1.5
        self.use_real_model = use_real_model
        self._generation_count = 0

        if use_real_model:
            self._init_model()

    def _init_model(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            # Set mirror before loading (in case direct HF is blocked)
            if "HF_ENDPOINT" not in os.environ:
                os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            try:
                self.model = SentenceTransformer(
                    self.model_name,
                    local_files_only=True,
                )
            except Exception:
                # Download from mirror if not cached
                self.model = SentenceTransformer(self.model_name)

            self.dimension = self.model.get_embedding_dimension()
        except Exception as e:
            print(f"[WARNING] Could not load sentence-transformers model: {e}")
            print("[WARNING] Falling back to hash-based embeddings.")
            self.model = None
            self.use_real_model = False

    def encode(self, text: str) -> List[float]:
        """Generate embedding for a text string."""
        if self.model is not None:
            embedding = self.model.encode(text, normalize_embeddings=True)
            return embedding.tolist()

        # Fallback: deterministic hash-based embedding
        return self._hash_embed(text)

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if self.model is not None:
            embeddings = self.model.encode(texts, normalize_embeddings=True)
            return embeddings.tolist()

        return [self._hash_embed(t) for t in texts]

    def _hash_embed(self, text: str) -> List[float]:
        """Deterministic hash-based embedding as fallback.

        Uses SHAKE-256 to generate a fixed-dimensional vector that is
        deterministic and reproducible for the same input text.
        """
        h = hashlib.shake_256(text.encode("utf-8"))
        # Generate enough bytes for the embedding
        raw = h.digest(max(self.dimension * 4, 32))
        # Interpret as uint32 for valid float seed values
        uint_arr = np.frombuffer(raw, dtype=np.uint32).copy()
        # Map to [-1, 1] range via simple transformation
        if len(uint_arr) < self.dimension:
            uint_arr = np.resize(uint_arr, self.dimension)
        result = ((uint_arr[:self.dimension].astype(np.float64) / np.float64(2**32)) * 2 - 1)
        # Normalize
        norm = np.linalg.norm(result)
        if norm > 0:
            result = result / norm
        return result.tolist()

    def combine(self, embeddings: List[List[float]], weights: Optional[List[float]] = None) -> List[float]:
        """Combine multiple embeddings with optional weights (e.g., for evidence fusion)."""
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

    def encode_state(self, state: Dict[str, Any]) -> List[float]:
        """Encode agent state into a compact embedding vector.

        This is the key mechanism for non-text state passing:
        instead of serializing the full state as text, we produce
        a dense semantic vector that captures the essential information.
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

    def decode_hints(self, embedding: List[float], top_k: int = 5) -> Dict[str, Any]:
        """Decode an embedding back to structural hints.

        Since embeddings are not fully reversible, this extracts
        interpretable signals from the vector for debugging/analysis.

        Returns: structural analysis of the embedding (sparsity, dominant dims, etc.)
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
        return self._generation_count

    def estimate_bits(self, embedding: List[float]) -> int:
        """Estimate the information content of an embedding in bits."""
        arr = np.array(embedding)
        # Rough estimate: 32 bits per float * dimension
        return len(embedding) * 32
