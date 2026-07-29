"""Execute run.py sessions across modes and task sets."""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from experiments.benchmark_suites import get_suite, resolve_tasks
from experiments.experiment_common import BatchSession, clear_db, run_task_batch
from experiments.pure_text_baseline import run_text_baseline_task
from src.paths import result_path
from src.run_options import DEFAULT_OPTIONS, RunOptions
from src.state.embeddings import EmbeddingEngine

from .task_catalog import MODE_CATALOG


def nocache_options() -> RunOptions:
    return RunOptions(
        enable_memory_index=False,
        enable_e2e_cache=False,
        enable_planner_cache=False,
        enable_summarizer_cache=False,
        enable_executor_dropout=False,
    )


MODE_NUM = {1: "pure", 2: "structured", 3: "nocache"}

REPL_CHAT_MODES = frozenset({2, 3})


def mode_num_from_key(mode_key: str) -> int:
    for num, key in MODE_NUM.items():
        if key == mode_key:
            return num
    raise ValueError(f"unknown mode_key: {mode_key!r}")


@dataclass
class ReplChatSession:
    """Multi-turn REPL: one DB + one Orchestrator across questions."""

    modes_key: tuple
    mode_num: int
    db_path: str
    batch: BatchSession
    turn_count: int = 0


def repl_chat_db_path(mode_num: int) -> str:
    from src.paths import database_path, ensure_output_dirs

    ensure_output_dirs()
    name = "repl_chat_nocache.db" if mode_num == 3 else "repl_chat.db"
    return database_path(name)


def repl_supports_shared_memory(modes: List[int]) -> bool:
    return len(modes) == 1 and modes[0] in REPL_CHAT_MODES


def start_repl_chat_session(api_key: str, mode_num: int) -> ReplChatSession:
    db_path = repl_chat_db_path(mode_num)
    clear_db(db_path)
    print(
        "  [对话] 已开启新会话，共享记忆库："
        f"{db_path}",
        flush=True,
    )
    return ReplChatSession(
        modes_key=(mode_num,),
        mode_num=mode_num,
        db_path=db_path,
        batch=BatchSession(),
    )


def run_repl_chat_turn(
    api_key: str,
    chat: ReplChatSession,
    task: Dict[str, Any],
    *,
    run_quality: bool = True,
) -> Dict[str, Any]:
    """Run one question reusing chat session memory."""
    options = DEFAULT_OPTIONS if chat.mode_num == 2 else nocache_options()
    first = chat.batch.orch is None
    if first:
        print(
            "\n  [系统] 正在加载语义向量模型（仅首次，完成后可连续提问）…",
            flush=True,
        )
    results, _ = run_task_batch(
        api_key,
        [task],
        db_path=chat.db_path,
        options=options,
        run_quality=run_quality,
        verbose=True,
        session=chat.batch,
        clear_db_before=False,
        reset_llm_each_task=True,
    )
    chat.turn_count += 1
    if chat.batch.orch and chat.batch.orch.memory_store._index is not None:
        n_mem = chat.batch.orch.memory_store._index.ntotal
        print(f"  [对话] 共享记忆库现有约 {n_mem} 条向量索引（本对话累计）", flush=True)
    return results[0]


@dataclass
class RunConfig:
    modes: List[int]
    suite: Optional[str] = None
    task_ids: Optional[List[str]] = None
    ask_texts: Optional[List[str]] = None
    run_quality: bool = True
    json_path: Optional[str] = None
    db_prefix: str = "_run_session"
    save_json: bool = True


@dataclass
class ModeRunResult:
    mode_num: int
    mode_key: str
    mode_label: str
    results: List[Dict[str, Any]] = field(default_factory=list)
    totals: Dict[str, Any] = field(default_factory=dict)
    wall_clock_sec: float = 0.0


def build_tasks_from_asks(texts: List[str], tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    tasks = []
    for i, text in enumerate(texts):
        tasks.append(
            {
                "task_id": f"custom_{i + 1}_{uuid.uuid4().hex[:6]}",
                "description": text.strip(),
                "tags": tags or ["custom"],
                "domain": "custom",
            }
        )
    return tasks


def resolve_config_tasks(config: RunConfig) -> List[Dict[str, Any]]:
    if config.ask_texts:
        return build_tasks_from_asks(config.ask_texts)
    return resolve_tasks(suite=config.suite, task_ids=config.task_ids)


def _enrich_pure_row(row: Dict[str, Any], task: Dict[str, Any]) -> Dict[str, Any]:
    row = dict(row)
    row["description"] = task.get("description", "")
    row.update(
        {
            "messages_sent": 0,
            "structured_protocol_tokens": 0,
            "text_equivalent_tokens": row.get("total_tokens", 0),
            "communication_chars": 0,
            "state_transfers": 0,
            "state_data_bytes": 0,
            "memory_queries": 0,
            "memory_hits": 0,
            "memory_hit_rate": 0.0,
            "cross_task_memories": 0,
            "e2e_cached": False,
            "executor_dropped": False,
        }
    )
    return row


def run_pure_mode(
    api_key: str,
    tasks: List[Dict[str, Any]],
    *,
    run_quality: bool = True,
    on_task: Optional[Callable[[Dict[str, Any], int, int], None]] = None,
) -> ModeRunResult:
    embed = EmbeddingEngine(model_name="BAAI/bge-small-en-v1.5", use_real_model=True)
    encode_fn = embed.encode
    results: List[Dict[str, Any]] = []
    quality_pass = 0
    t0 = time.time()

    for i, task in enumerate(tasks):
        desc_short = (task.get("description") or "")[:40]
        print(
            f"  [{i + 1}/{len(tasks)}] {task['task_id']}: {desc_short}…",
            flush=True,
        )
        print(
            "        纯文本基线：3 次 API 调用中（约 30 秒～1 分钟）…",
            flush=True,
        )
        row = run_text_baseline_task(task, api_key, encode_fn)
        row = _enrich_pure_row(row, task)
        if row.get("composite_pass"):
            quality_pass += 1
        results.append(row)
        if on_task:
            on_task(row, i + 1, len(tasks))

    wall = time.time() - t0
    totals = {
        "total_tokens": sum(r["total_tokens"] for r in results),
        "total_elapsed_ms": sum(r["elapsed_ms"] for r in results),
        "total_api_calls": sum(r["api_calls"] for r in results),
        "avg_relevance": sum(r["relevance"] for r in results) / max(len(results), 1),
        "quality_pass_count": quality_pass if run_quality else None,
        "wall_clock_sec": round(wall, 1),
        "total_messages": 0,
        "memory_hit_rate": 0.0,
    }
    return ModeRunResult(
        mode_num=1,
        mode_key="pure",
        mode_label=MODE_CATALOG[1][1],
        results=results,
        totals=totals,
        wall_clock_sec=wall,
    )


def run_structured_mode(
    api_key: str,
    tasks: List[Dict[str, Any]],
    *,
    mode_num: int,
    options: RunOptions,
    run_quality: bool = True,
    db_path: str,
    on_task: Optional[Callable[[Dict[str, Any], int, int], None]] = None,
) -> ModeRunResult:
    clear_db(db_path)
    t0 = time.time()
    results, totals = run_task_batch(
        api_key,
        tasks,
        db_path=db_path,
        options=options,
        cold_db_per_task=False,
        run_quality=run_quality,
        verbose=True,
    )
    totals["wall_clock_sec"] = round(time.time() - t0, 1)
    for i, row in enumerate(results):
        task = tasks[i]
        row["description"] = task.get("description", "")
        if on_task:
            on_task(row, i + 1, len(tasks))

    return ModeRunResult(
        mode_num=mode_num,
        mode_key=MODE_NUM[mode_num],
        mode_label=MODE_CATALOG[mode_num][1],
        results=results,
        totals=totals,
        wall_clock_sec=totals["wall_clock_sec"],
    )


def execute_session(
    api_key: str,
    config: RunConfig,
    *,
    on_task: Optional[Callable[[Dict[str, Any], int, int, str], None]] = None,
) -> Dict[str, Any]:
    """Run all configured modes; return serializable session report."""
    tasks = resolve_config_tasks(config)
    suite_label = config.suite or "自定义题目"
    sessions: List[Dict[str, Any]] = []

    for mode_num in config.modes:
        mode_key = MODE_NUM[mode_num]
        print()
        print(f">>> 开始模式 {mode_num}：{MODE_CATALOG[mode_num][1]}（共 {len(tasks)} 题）")

        def _wrap(row, idx, total):
            if on_task:
                on_task(row, idx, total, mode_key)

        if mode_num == 1:
            mr = run_pure_mode(
                api_key, tasks, run_quality=config.run_quality, on_task=_wrap
            )
        elif mode_num == 2:
            db = f"{config.db_prefix}_m2.db"
            mr = run_structured_mode(
                api_key,
                tasks,
                mode_num=2,
                options=DEFAULT_OPTIONS,
                run_quality=config.run_quality,
                db_path=db,
                on_task=_wrap,
            )
        else:
            db = f"{config.db_prefix}_m3.db"
            mr = run_structured_mode(
                api_key,
                tasks,
                mode_num=3,
                options=nocache_options(),
                run_quality=config.run_quality,
                db_path=db,
                on_task=_wrap,
            )

        sessions.append(
            {
                "mode_num": mr.mode_num,
                "mode_key": mr.mode_key,
                "mode_label": mr.mode_label,
                "results": mr.results,
                "totals": mr.totals,
            }
        )

    report = {
        "suite": config.suite,
        "task_count": len(tasks),
        "modes": [MODE_NUM[m] for m in config.modes],
        "sessions": sessions,
        "tasks": [{"task_id": t["task_id"], "description": t.get("description", "")[:200]} for t in tasks],
    }
    if config.save_json:
        import json

        if config.json_path:
            os.makedirs(os.path.dirname(config.json_path) or ".", exist_ok=True)
            with open(config.json_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            report["json_path"] = config.json_path
        else:
            ts = time.strftime("%Y%m%d_%H%M%S")
            modes_s = "_".join(str(m) for m in config.modes)
            auto = result_path(f"run_{modes_s}_{config.suite or 'custom'}_{ts}.json")
            with open(auto, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            report["json_path"] = auto

    report["suite_label"] = suite_label
    return report
