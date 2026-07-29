#!/usr/bin/env python3
"""Compute pairwise cosine similarity for benchmark task descriptions (cache analysis)."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from experiments.benchmark_suites import get_suite
from src.paths import result_path
from src.state.embeddings import EmbeddingEngine


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="core12", help="core12 | full24 | adversarial6")
    args = parser.parse_args()

    engine = EmbeddingEngine(use_real_model=True)
    tasks = get_suite(args.suite)
    ids = [t["task_id"] for t in tasks]
    texts = [t["description"] for t in tasks]
    embs = [np.array(engine.encode(t)) for t in texts]

    n = len(ids)
    matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            a, b = embs[i], embs[j]
            row.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))
        matrix.append(row)

    out = {
        "suite": args.suite,
        "model": engine.model_name,
        "task_ids": ids,
        "cosine_matrix": matrix,
        "pairs_above_085": [
            {"a": ids[i], "b": ids[j], "cos": round(matrix[i][j], 3)}
            for i in range(n)
            for j in range(i + 1, n)
            if matrix[i][j] > 0.85
        ],
    }

    path = result_path("embedding_similarity_matrix.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"Saved {path}")
    print("Pairs cos>0.85 (E2E candidates):")
    for p in out["pairs_above_085"]:
        print(f"  {p['a']} <-> {p['b']}: {p['cos']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
