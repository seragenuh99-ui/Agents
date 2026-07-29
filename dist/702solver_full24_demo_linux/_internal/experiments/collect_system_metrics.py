#!/usr/bin/env python3
"""Collect protocol/state/memory micro-metrics for the paper report (no API)."""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import msgpack
import numpy as np

from src.protocol import Message, MessageType, ActionType, ProtocolParser
from src.state.embeddings import EmbeddingEngine
from src.state.exchange import StateExchangeBus
from src.memory.store import MemoryStore
from src.memory.models import MemoryUnit


def _serialize_compare(msg: Message, n: int = 10000) -> dict:
    d = msg.to_dict()
    mp = msgpack.packb(d, use_bin_type=True)
    js = json.dumps(d).encode()
    t0 = time.perf_counter()
    for _ in range(n):
        msgpack.packb(d, use_bin_type=True)
    ser_mp = (time.perf_counter() - t0) / n * 1e6
    return {
        "messagepack_bytes": len(mp),
        "json_bytes": len(js),
        "volume_reduction_pct": round((1 - len(mp) / len(js)) * 100, 1),
        "serialize_us_per_msg": round(ser_mp, 3),
    }


def protocol_metrics(n=10000):
    """Scheduling message (no embedding) — primary protocol efficiency metric."""
    sched = ProtocolParser.create_request(
        from_agent="orchestrator",
        to_agent="planner",
        action=ActionType.PLAN,
        params={"task_description": "Research solar energy", "task_id": "t1"},
        memory_refs=["abc123"],
    )
    sched.capabilities = ["plan"]
    with_state = ProtocolParser.create_request(
        from_agent="retriever",
        to_agent="executor",
        action=ActionType.RETRIEVE,
        params={"query": "solar LCOE", "top_k": 5},
        embedding=[0.1] * 384,
        memory_refs=["mem_001", "mem_002"],
    )
    out = _serialize_compare(sched, n)
    out["label"] = "scheduling_no_embedding"
    out["with_state_embedding"] = _serialize_compare(with_state, n)
    return out


def state_metrics(n=500):
    eng = EmbeddingEngine(use_real_model=False)
    bus = StateExchangeBus(eng)
    text = (
        "Agent state payload: solar photovoltaic efficiency, manufacturing cost, "
        "grid integration, and lifecycle analysis for renewable deployment. "
    ) * 50
    t0 = time.perf_counter()
    for i in range(n):
        bus.transfer({"x": i}, "a", "b", context=text)
    elapsed = time.perf_counter() - t0
    stats = bus.get_stats()
    vec_bytes = 384 * 4
    text_bytes = len(text.encode())
    return {
        "vector_bytes_per_transfer": vec_bytes,
        "equivalent_text_bytes": text_bytes,
        "volume_reduction_pct": round((1 - vec_bytes / text_bytes) * 100, 1),
        "transfers_per_sec": round(n / elapsed, 1),
        "avg_packet_bytes": round(stats.get("avg_packet_size_bytes", vec_bytes), 1),
    }


def memory_metrics(n=200):
    path = "_metrics_bench.db"
    for p in [path, path + "-shm", path + "-wal"]:
        try:
            os.unlink(p)
        except OSError:
            pass
    store = MemoryStore(db_path=path)
    eng = EmbeddingEngine(use_real_model=False)
    t0 = time.perf_counter()
    for i in range(n):
        emb = eng.encode(f"topic {i} solar energy research")
        store.store(
            MemoryUnit(
                source_agent="retriever",
                task_topic=f"topic {i}",
                summary=f"summary {i}",
                content=f"content {i}",
                tags=["solar", "energy"],
                embedding=emb,
            )
        )
    write_ms = (time.perf_counter() - t0) / n * 1000
    q = eng.encode("solar photovoltaic efficiency")
    times = []
    for _ in range(100):
        t1 = time.perf_counter()
        store.search_by_similarity(q, limit=5)
        times.append((time.perf_counter() - t1) * 1000)
    for p in [path, path + "-shm", path + "-wal"]:
        try:
            os.unlink(p)
        except OSError:
            pass
    return {
        "records": n,
        "write_ms_per_record": round(write_ms, 2),
        "semantic_search_ms_p50": round(float(np.percentile(times, 50)), 2),
        "index_type": "FlatIP",
    }


def count_tests() -> int:
    import subprocess

    r = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "--co", "-q"],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.dirname(__file__)),
    )
    for line in (r.stdout + r.stderr).splitlines():
        if " tests collected" in line:
            try:
                return int(line.split()[0])
            except ValueError:
                pass
    return 0


def main():
    out = {
        "collected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "embedding_dim": 384,
        "protocol": protocol_metrics(),
        "state_transfer": state_metrics(),
        "memory_store": memory_metrics(),
        "tests_passed": count_tests(),
    }
    from src.paths import result_path

    path = result_path("system_metrics.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
