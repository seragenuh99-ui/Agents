#!/usr/bin/env python3
"""Qt demo dashboard for the 702solver recording.

This is intentionally read-only: it visualizes existing experiment artifacts and
key source files without running LLM calls or changing databases.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt, QThread, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QFont, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from experiments.benchmark_suites import get_suite
from experiments.experiment_common import classify_strategy, clear_db
from src.agents.base import LLMBackend
from src.evaluation.quality_validator import (
    extract_summary_payload,
    format_answer_markdown,
    validate_task,
)
from src.orchestrator import Orchestrator
from src.paths import database_path, ensure_output_dirs
from src.run_options import DEFAULT_OPTIONS, RunOptions
from src.state.embeddings import EmbeddingEngine


ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
REPORT_PATH = ROOT / "output" / "results" / "full24_controlled_comparison.json"

KEY_FILES = [
    {
        "title": "Orchestrator",
        "path": ROOT / "src" / "orchestrator.py",
        "why": "系统总调度：检查 E2E 缓存，调用 Planner/Retriever/Executor/Summarizer，记录 token、状态传递和记忆命中。",
    },
    {
        "title": "Structured Message",
        "path": ROOT / "src" / "protocol" / "__init__.py",
        "why": "结构化通信协议：Message 携带 action、params、result、state_embedding、memory_refs，并支持 MessagePack 序列化。",
    },
    {
        "title": "State Exchange",
        "path": ROOT / "src" / "state" / "exchange.py",
        "why": "非文本状态传递：Agent 把中间状态编码为 384 维向量包，发送给下游 Agent 直接用于语义检索。",
    },
    {
        "title": "Memory Store",
        "path": ROOT / "src" / "memory" / "store.py",
        "why": "共享记忆存储：SQLite 保存元数据，FAISS 做向量相似度检索，让相似任务复用历史经验。",
    },
    {
        "title": "Run Entry",
        "path": ROOT / "run.py",
        "why": "中文实验入口：支持 pure/structured/nocache 三种模式，以及 core12/full24/continuous12 等题库。",
    },
    {
        "title": "Full24 Result",
        "path": ROOT / "docs" / "full24_controlled_comparison.md",
        "why": "正式实验报告：24 题三组对照，展示 token、API 调用、质量通过率和逐题策略。",
    },
]


COLORS = {
    "bg": "#f6f7f9",
    "panel": "#ffffff",
    "line": "#d9dde5",
    "text": "#1f2937",
    "muted": "#667085",
    "blue": "#2563eb",
    "green": "#16a34a",
    "amber": "#d97706",
    "red": "#dc2626",
    "slate": "#475569",
}


def load_env_file() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def nocache_options() -> RunOptions:
    return RunOptions(
        enable_memory_index=False,
        enable_e2e_cache=False,
        enable_planner_cache=False,
        enable_summarizer_cache=False,
        enable_executor_dropout=False,
    )


def summarize_structured_result(
    task: Dict[str, Any],
    result: Dict[str, Any],
    llm: LLMBackend,
    orch: Orchestrator,
    elapsed: float,
    stats_before: Dict[str, Any],
) -> Dict[str, Any]:
    stats_after = llm.get_usage_stats()
    prompt_t = stats_after["total_prompt_tokens"] - stats_before["total_prompt_tokens"]
    compl_t = stats_after["total_completion_tokens"] - stats_before["total_completion_tokens"]
    cached = stats_after["total_cached_tokens"] - stats_before["total_cached_tokens"]
    calls = stats_after["call_count"] - stats_before["call_count"]

    plan = result.get("steps", {}).get("plan", {})
    summary_step = result.get("steps", {}).get("summary", {})
    strategy = classify_strategy(result, plan if isinstance(plan, dict) else {})
    summary_obj: Dict[str, Any] = {}
    summary_text = ""
    summary_full = ""
    if isinstance(summary_step, dict):
        summary_obj, summary_text = extract_summary_payload(summary_step)
        summary_full = format_answer_markdown(summary_obj, summary_text)

    composite_score: Optional[float] = None
    composite_pass: Optional[bool] = None
    if summary_step:
        try:
            validation, _, _ = validate_task(
                task,
                summary_step,
                encode_fn=orch.embedding_engine.encode,
                pass_threshold=0.65,
                use_llm_judge=False,
            )
            composite_score = validation.composite_score
            composite_pass = validation.composite_pass
        except Exception:
            pass

    collab: Dict[str, Any] = {}
    if orch.metrics._task_history:
        tm = orch.metrics._task_history[-1]
        collab = {
            "messages_sent": tm.messages_sent,
            "structured_protocol_tokens": tm.structured_token_count,
            "text_equivalent_tokens": tm.text_equivalent_token_count,
            "state_transfers": tm.state_transfers,
            "state_data_bytes": tm.state_data_bytes,
            "memory_queries": tm.memory_queries,
            "memory_hits": tm.memory_hits,
            "memory_hit_rate": round(tm.memory_hit_rate, 4),
            "cross_task_memories": tm.cross_task_memories_used,
        }

    return {
        "task_id": task["task_id"],
        "variant": "",
        "strategy": strategy,
        "api_calls": calls,
        "prompt_tokens": prompt_t,
        "completion_tokens": compl_t,
        "cached_tokens": cached,
        "total_tokens": prompt_t + compl_t,
        "elapsed_ms": elapsed * 1000,
        "composite_score": composite_score,
        "composite_pass": composite_pass,
        "e2e_cached": bool(result.get("_e2e_cached")),
        "executor_dropped": bool(result.get("_executor_dropped")),
        "summary_full": summary_full or summary_text,
        "summary_preview": (summary_text or summary_full)[:260],
        **collab,
    }


class LiveRunWorker(QThread):
    progress = Signal(str)
    metrics = Signal(dict)
    row_ready = Signal(dict)
    finished_ok = Signal(list)
    failed = Signal(str)

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def run(self) -> None:
        try:
            self._run()
        except Exception as exc:
            self.failed.emit(str(exc))

    def _emit_usage(self, label: str, llm: LLMBackend) -> None:
        stats = llm.get_usage_stats()
        self.metrics.emit({
            "active_variant": label,
            "api_calls": stats["call_count"],
            "prompt_tokens": stats["total_prompt_tokens"],
            "completion_tokens": stats["total_completion_tokens"],
            "cached_tokens": stats["total_cached_tokens"],
            "total_tokens": stats["total_tokens"],
        })

    def _run(self) -> None:
        ensure_output_dirs()
        api_key = self.config["api_key"].strip()
        if not api_key:
            raise RuntimeError("缺少 DEEPSEEK_API_KEY。请在界面填写，或在 .env 里配置。")

        task_text = self.config["task_text"].strip()
        if not task_text:
            raise RuntimeError("请输入要演示的任务。")

        task = {
            "task_id": f"live_{uuid.uuid4().hex[:8]}",
            "description": task_text,
            "tags": [t.strip() for t in self.config["tags"].split(",") if t.strip()] or ["live-demo"],
            "domain": "live-demo",
            "expected_topics": [],
        }
        rows: List[Dict[str, Any]] = []

        db_path = database_path("live_dashboard_memory.db")
        if self.config["reset_memory"]:
            clear_db(db_path)
            self.progress.emit("已清空本次演示记忆库，便于展示冷启动。")

        if self.config["run_pure"]:
            row = self._run_pure(task, api_key)
            rows.append(row)
            self.row_ready.emit(row)
            if self._stop_requested:
                self.finished_ok.emit(rows)
                return

        shared_orch: Optional[Orchestrator] = None
        shared_llm: Optional[LLMBackend] = None
        for variant, options in [
            ("B 结构化无缓存", nocache_options()),
            ("C 结构化全功能", DEFAULT_OPTIONS),
        ]:
            if self._stop_requested:
                break
            if variant.startswith("B") and not self.config["run_nocache"]:
                continue
            if variant.startswith("C") and not self.config["run_full"]:
                continue
            row, shared_orch, shared_llm = self._run_structured(
                task, api_key, variant, options, db_path, shared_orch, shared_llm
            )
            rows.append(row)
            self.row_ready.emit(row)

        self.finished_ok.emit(rows)

    def _run_pure(self, task: Dict[str, Any], api_key: str) -> Dict[str, Any]:
        label = "A 纯文本基线"
        self.progress.emit(f"开始 {label}：同一个任务拆成 plan / retrieve / summarize 三次长文本调用。")
        llm = LLMBackend(provider="deepseek", api_key=api_key, model=self.config["model"])
        embed = EmbeddingEngine(use_real_model=not self.config["fast_embeddings"])
        t0 = time.time()
        pieces: List[str] = []

        calls = [
            (
                "plan",
                "You are a task planning agent. Decompose the task into clear subtasks.",
                f"Create a detailed execution plan for:\n\n{task['description']}",
            ),
            (
                "retrieve",
                "You are a research assistant. Provide relevant facts and context.",
                f"Research and provide information for:\n\n{task['description']}",
            ),
            (
                "summarize",
                "You are a report writer. Synthesize the final answer professionally.",
                "",
            ),
        ]
        for name, system, user in calls:
            if self._stop_requested:
                break
            if name == "summarize":
                user = (
                    f"TASK:\n{task['description']}\n\n"
                    f"PREVIOUS OUTPUT:\n{chr(10).join(pieces)[:3200]}\n\n"
                    "Write the final report."
                )
            self.progress.emit(f"{label} / {name}：等待 DeepSeek 返回...")
            text = llm.chat(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_tokens=900,
                timeout=(15, 120),
            )
            pieces.append(text)
            self._emit_usage(label, llm)
            self.progress.emit(
                f"{label} / {name} 完成：累计 token {llm.get_usage_stats()['total_tokens']}。"
            )

        elapsed = time.time() - t0
        stats = llm.get_usage_stats()
        summary_text = pieces[-1] if pieces else ""
        composite_score: Optional[float] = None
        composite_pass: Optional[bool] = None
        try:
            validation, _, _ = validate_task(
                task,
                {"summary": summary_text, "task_id": task["task_id"]},
                encode_fn=embed.encode,
                pass_threshold=0.65,
                use_llm_judge=False,
            )
            composite_score = validation.composite_score
            composite_pass = validation.composite_pass
        except Exception:
            pass
        return {
            "variant": label,
            "task_id": task["task_id"],
            "strategy": "PURE_TEXT_3CALL",
            "api_calls": stats["call_count"],
            "prompt_tokens": stats["total_prompt_tokens"],
            "completion_tokens": stats["total_completion_tokens"],
            "cached_tokens": stats["total_cached_tokens"],
            "total_tokens": stats["total_tokens"],
            "elapsed_ms": elapsed * 1000,
            "composite_score": composite_score,
            "composite_pass": composite_pass,
            "e2e_cached": False,
            "executor_dropped": False,
            "messages_sent": 0,
            "structured_protocol_tokens": 0,
            "text_equivalent_tokens": stats["total_tokens"],
            "state_transfers": 0,
            "state_data_bytes": 0,
            "memory_queries": 0,
            "memory_hits": 0,
            "memory_hit_rate": 0,
            "cross_task_memories": 0,
            "summary_full": summary_text,
            "summary_preview": summary_text[:260],
        }

    def _run_structured(
        self,
        task: Dict[str, Any],
        api_key: str,
        label: str,
        options: RunOptions,
        db_path: str,
        shared_orch: Optional[Orchestrator],
        shared_llm: Optional[LLMBackend],
    ) -> tuple[Dict[str, Any], Optional[Orchestrator], Optional[LLMBackend]]:
        self.progress.emit(f"开始 {label}：Planner → Retriever → Executor → Summarizer。")
        llm = shared_llm or LLMBackend(provider="deepseek", api_key=api_key, model=self.config["model"])
        llm.reset_stats()

        def on_status(message: str) -> None:
            clean = message.replace("鈥?", "...").replace("锛?", "：")
            self.progress.emit(f"{label}：{clean}")
            self._emit_usage(label, llm)

        llm.on_status = on_status

        if shared_orch is None or label.startswith("B") or not self.config["reuse_runtime"]:
            if label.startswith("B"):
                clear_db(database_path("live_dashboard_nocache.db"))
                db_for_variant = database_path("live_dashboard_nocache.db")
            else:
                db_for_variant = db_path
            orch = Orchestrator(
                llm=llm,
                mode="structured",
                use_real_embeddings=not self.config["fast_embeddings"],
                sandbox_enabled=True,
                memory_db_path=db_for_variant,
                run_options=options,
            )
        else:
            orch = shared_orch
            orch.run_options = options
            for agent in orch.agents.values():
                agent.run_options = options
            orch.llm = llm

        orch._progress_fn = on_status
        stats_before = llm.get_usage_stats()
        t0 = time.time()
        result = orch.execute_task(task["task_id"], task["description"], task["tags"])
        elapsed = time.time() - t0
        row = summarize_structured_result(task, result, llm, orch, elapsed, stats_before)
        row["variant"] = label
        self.progress.emit(
            f"{label} 完成：token {row['total_tokens']}，API {row['api_calls']} 次，策略 {row['strategy']}。"
        )
        return row, (orch if label.startswith("C") else shared_orch), llm


class Full24RunWorker(QThread):
    progress = Signal(str)
    task_done = Signal(dict)
    totals = Signal(dict)
    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def run(self) -> None:
        try:
            self._run()
        except Exception as exc:
            self.failed.emit(str(exc))

    def _run(self) -> None:
        ensure_output_dirs()
        api_key = self.config["api_key"].strip()
        if not api_key:
            raise RuntimeError("缺少 DEEPSEEK_API_KEY。请在界面填写，或在 .env 里配置。")

        mode = self.config["mode"]
        tasks = get_suite("full24")
        if self.config["reset_memory"] and mode == "C":
            clear_db(database_path("full24_dashboard_c_memory.db"))
            self.progress.emit("C 模式记忆库已清空，将从冷启动开始跑 24 题。")

        rows: List[Dict[str, Any]] = []
        if mode == "A":
            for idx, task in enumerate(tasks, start=1):
                if self._stop_requested:
                    break
                row = self._run_pure_task(task, idx, len(tasks), api_key)
                rows.append(row)
                self.task_done.emit(row)
                self.totals.emit(self._build_totals(mode, rows))
        else:
            rows = self._run_c_tasks(tasks, api_key)

        self.totals.emit(self._build_totals(mode, rows))
        self.finished_ok.emit(mode)

    def _run_pure_task(
        self, task: Dict[str, Any], idx: int, total: int, api_key: str
    ) -> Dict[str, Any]:
        label = "A"
        self.progress.emit(f"A 纯文本 [{idx}/{total}] {task['task_id']}：开始 3 次长文本调用。")
        llm = LLMBackend(provider="deepseek", api_key=api_key, model=self.config["model"])
        t0 = time.time()
        pieces: List[str] = []
        calls = [
            (
                "plan",
                "You are a task planning agent. Decompose the task into clear subtasks.",
                f"Create a detailed execution plan for:\n\n{task['description']}",
            ),
            (
                "retrieve",
                "You are a research assistant. Provide relevant facts and context.",
                f"Research and provide information for:\n\n{task['description']}",
            ),
            (
                "summarize",
                "You are a report writer. Synthesize the final answer professionally.",
                "",
            ),
        ]
        for name, system, user in calls:
            if self._stop_requested:
                break
            if name == "summarize":
                user = (
                    f"TASK:\n{task['description']}\n\n"
                    f"PREVIOUS OUTPUT:\n{chr(10).join(pieces)[:3200]}\n\n"
                    "Write the final report."
                )
            self.progress.emit(f"A 纯文本 [{idx}/{total}] {task['task_id']} / {name}：等待 DeepSeek...")
            text = llm.chat(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_tokens=900,
                timeout=(15, 120),
            )
            pieces.append(text)
            stats = llm.get_usage_stats()
            self.progress.emit(
                f"A 纯文本 [{idx}/{total}] {name} 完成：累计 token {stats['total_tokens']}。"
            )

        stats = llm.get_usage_stats()
        row = {
            "mode": label,
            "task_id": task["task_id"],
            "api_calls": stats["call_count"],
            "prompt_tokens": stats["total_prompt_tokens"],
            "completion_tokens": stats["total_completion_tokens"],
            "cached_tokens": stats["total_cached_tokens"],
            "total_tokens": stats["total_tokens"],
            "elapsed_ms": (time.time() - t0) * 1000,
            "strategy": "PURE_TEXT_3CALL",
            "memory_hits": 0,
            "summary_preview": (pieces[-1] if pieces else "")[:180],
        }
        self.progress.emit(
            f"A 纯文本 [{idx}/{total}] {task['task_id']} 完成：token {row['total_tokens']}，API {row['api_calls']}。"
        )
        return row

    def _run_c_tasks(self, tasks: List[Dict[str, Any]], api_key: str) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        llm = LLMBackend(provider="deepseek", api_key=api_key, model=self.config["model"])
        db_path = database_path("full24_dashboard_c_memory.db")
        orch = Orchestrator(
            llm=llm,
            mode="structured",
            use_real_embeddings=not self.config["fast_embeddings"],
            sandbox_enabled=True,
            memory_db_path=db_path,
            run_options=DEFAULT_OPTIONS,
        )

        for idx, task in enumerate(tasks, start=1):
            if self._stop_requested:
                break
            self.progress.emit(f"C 最终方案 [{idx}/{len(tasks)}] {task['task_id']}：开始多 Agent 流水线。")
            llm.reset_stats()
            stats_before = llm.get_usage_stats()

            def on_status(message: str) -> None:
                clean = message.replace("鈥?", "...").replace("锛?", "：")
                self.progress.emit(f"C [{idx}/{len(tasks)}] {task['task_id']}：{clean}")

            llm.on_status = on_status
            orch._progress_fn = on_status
            t0 = time.time()
            result = orch.execute_task(task["task_id"], task["description"], task.get("tags", []))
            row = summarize_structured_result(
                task, result, llm, orch, time.time() - t0, stats_before
            )
            row["mode"] = "C"
            rows.append(row)
            self.task_done.emit(row)
            self.totals.emit(self._build_totals("C", rows))
            self.progress.emit(
                f"C 最终方案 [{idx}/{len(tasks)}] {task['task_id']} 完成：token {row['total_tokens']}，"
                f"API {row['api_calls']}，策略 {row['strategy']}。"
            )
        try:
            orch.cleanup()
        except Exception:
            pass
        return rows

    @staticmethod
    def _build_totals(mode: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "mode": mode,
            "done": len(rows),
            "tokens": sum(r.get("total_tokens", 0) for r in rows),
            "api_calls": sum(r.get("api_calls", 0) for r in rows),
            "elapsed_ms": sum(r.get("elapsed_ms", 0) for r in rows),
            "memory_hits": sum(r.get("memory_hits", 0) for r in rows),
            "dropout_hits": sum(1 for r in rows if r.get("executor_dropped")),
        }


def read_report() -> Dict[str, Any]:
    if not REPORT_PATH.exists():
        return {
            "meta": {"suite": "missing", "task_count": 0, "model": "unknown"},
            "variants": {},
            "per_task": [],
        }
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def variant_label(key: str) -> str:
    if key.startswith("A_"):
        return "A 纯文本"
    if key.startswith("B_"):
        return "B 结构化无缓存"
    if key.startswith("C_"):
        return "C 结构化全功能"
    return key


def pct(part: float, whole: float) -> float:
    return 0.0 if whole <= 0 else part * 100.0 / whole


class Card(QFrame):
    def __init__(self, title: str, value: str, note: str, color: str = COLORS["blue"]):
        super().__init__()
        self.setObjectName("card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(6)

        title_label = QLabel(title)
        title_label.setObjectName("cardTitle")
        value_label = QLabel(value)
        value_label.setObjectName("cardValue")
        value_label.setStyleSheet(f"color: {color};")
        note_label = QLabel(note)
        note_label.setObjectName("cardNote")
        note_label.setWordWrap(True)

        layout.addWidget(title_label)
        layout.addWidget(value_label)
        layout.addWidget(note_label)


class BarRow(QWidget):
    def __init__(self, name: str, value: float, max_value: float, suffix: str, color: str):
        super().__init__()
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 3, 0, 3)
        layout.setColumnStretch(1, 1)

        name_label = QLabel(name)
        name_label.setMinimumWidth(118)
        name_label.setObjectName("barName")
        value_label = QLabel(f"{value:,.0f}{suffix}")
        value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        value_label.setMinimumWidth(96)
        value_label.setObjectName("barValue")

        track = QFrame()
        track.setObjectName("barTrack")
        track.setMinimumHeight(18)
        track_layout = QHBoxLayout(track)
        track_layout.setContentsMargins(0, 0, 0, 0)
        track_layout.setSpacing(0)

        fill = QFrame()
        fill.setStyleSheet(f"background: {color}; border-radius: 4px;")
        spacer = QWidget()
        ratio = 0 if max_value <= 0 else max(0.02, min(1.0, value / max_value))
        track_layout.addWidget(fill, int(ratio * 1000))
        track_layout.addWidget(spacer, int((1 - ratio) * 1000))

        layout.addWidget(name_label, 0, 0)
        layout.addWidget(track, 0, 1)
        layout.addWidget(value_label, 0, 2)


class Dashboard(QMainWindow):
    def __init__(self, report: Dict[str, Any]):
        super().__init__()
        self.report = report
        self.worker: Optional[LiveRunWorker] = None
        self.live_rows: List[Dict[str, Any]] = []
        self.setWindowTitle("702solver 演示控制台")
        self.resize(1380, 860)

        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(16, 14, 16, 16)
        outer.setSpacing(12)

        outer.addWidget(self._build_header())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_center_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([260, 690, 430])
        outer.addWidget(splitter, 1)

        self.setCentralWidget(root)
        self._load_file(0)

    def _build_header(self) -> QWidget:
        meta = self.report.get("meta", {})
        header = QFrame()
        header.setObjectName("header")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(18, 14, 18, 14)

        title_box = QVBoxLayout()
        title = QLabel("702solver 多 Agent 共享记忆诊断系统")
        title.setObjectName("title")
        subtitle = QLabel(
            f"结构化协议 + 384 维状态传递 + SQLite/FAISS 共享记忆 | "
            f"{meta.get('suite', 'full24')} / {meta.get('task_count', 24)} 题 / {meta.get('model', 'deepseek-chat')}"
        )
        subtitle.setObjectName("subtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)

        cmd = QLabel("录屏建议：先跑实时对比，再用历史 full24 报告做正式佐证")
        cmd.setObjectName("pill")
        cmd.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addLayout(title_box, 1)
        layout.addWidget(cmd)
        return header

    def _build_left_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("panel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        label = QLabel("关键文件")
        label.setObjectName("sectionTitle")
        self.file_list = QListWidget()
        self.file_list.setObjectName("fileList")
        for item in KEY_FILES:
            qitem = QListWidgetItem(item["title"])
            qitem.setData(Qt.ItemDataRole.UserRole, item)
            self.file_list.addItem(qitem)
        self.file_list.currentRowChanged.connect(self._load_file)

        flow = QLabel(
            "演示路线\n"
            "1. Orchestrator 说明四 Agent 流水线\n"
            "2. Message 说明结构化通信\n"
            "3. StatePacket 说明非文本传递\n"
            "4. MemoryStore 说明复用\n"
            "5. Full24 Result 说明效果"
        )
        flow.setObjectName("flow")
        flow.setWordWrap(True)

        self.open_btn = QPushButton("打开当前文件")
        self.open_btn.clicked.connect(self._open_current_file)

        layout.addWidget(label)
        layout.addWidget(self.file_list, 1)
        layout.addWidget(flow)
        layout.addWidget(self.open_btn)
        return panel

    def _build_center_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        layout.addWidget(self._build_live_panel())
        scroll.setWidget(content)
        return scroll

        layout.addWidget(self._build_cards())
        layout.addWidget(self._build_live_panel())
        layout.addWidget(self._build_bars("历史 full24 报告：Token 对比", "total_tokens", "", COLORS["blue"]))
        layout.addWidget(self._build_bars("历史 full24 报告：API 调用次数", "total_api_calls", " 次", COLORS["green"]))
        layout.addWidget(self._build_bars("历史 full24 报告：总耗时", "wall_clock_sec", " 秒", COLORS["amber"]))
        layout.addWidget(self._build_task_table(), 1)

        scroll.setWidget(content)
        return scroll

    def _build_single_task_panel(self) -> QWidget:
        box = QFrame()
        box.setObjectName("panel")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        title = QLabel("实时演示：同一个任务下对比 token 消耗（这里才是你刚刚现场跑出来的数据）")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        form = QGridLayout()
        form.setColumnStretch(1, 1)
        form.setColumnStretch(3, 1)

        self.api_key_input = QLineEdit(os.environ.get("DEEPSEEK_API_KEY", ""))
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_input.setPlaceholderText("DEEPSEEK_API_KEY")
        self.model_input = QLineEdit("deepseek-chat")
        self.tags_input = QLineEdit("energy,live-demo")
        self.task_input = QTextEdit()
        self.task_input.setMinimumHeight(78)
        self.task_input.setPlaceholderText("输入一个任务，例如：分析某城市建设风光储一体化能源系统的技术路线、成本因素和风险。")
        self.task_input.setPlainText("分析某城市建设风光储一体化能源系统的技术路线、成本因素和风险，并给出实施建议。")

        form.addWidget(QLabel("DeepSeek Key"), 0, 0)
        form.addWidget(self.api_key_input, 0, 1)
        form.addWidget(QLabel("模型"), 0, 2)
        form.addWidget(self.model_input, 0, 3)
        form.addWidget(QLabel("标签"), 1, 0)
        form.addWidget(self.tags_input, 1, 1, 1, 3)
        form.addWidget(QLabel("任务"), 2, 0)
        form.addWidget(self.task_input, 2, 1, 1, 3)
        layout.addLayout(form)

        toggles = QHBoxLayout()
        self.run_pure_cb = QCheckBox("A 纯文本")
        self.run_pure_cb.setChecked(True)
        self.run_nocache_cb = QCheckBox("B 结构化无缓存")
        self.run_nocache_cb.setChecked(True)
        self.run_full_cb = QCheckBox("C 结构化全功能")
        self.run_full_cb.setChecked(True)
        self.fast_embed_cb = QCheckBox("快速 embedding")
        self.fast_embed_cb.setChecked(True)
        self.reset_memory_cb = QCheckBox("运行前清空演示记忆")
        self.reset_memory_cb.setChecked(True)
        self.reuse_runtime_cb = QCheckBox("C 模式复用运行时记忆")
        self.reuse_runtime_cb.setChecked(True)
        for w in [
            self.run_pure_cb,
            self.run_nocache_cb,
            self.run_full_cb,
            self.fast_embed_cb,
            self.reset_memory_cb,
            self.reuse_runtime_cb,
        ]:
            toggles.addWidget(w)
        toggles.addStretch(1)
        layout.addLayout(toggles)

        actions = QHBoxLayout()
        self.run_btn = QPushButton("开始实时运行")
        self.run_btn.clicked.connect(self._start_live_run)
        self.stop_btn = QPushButton("请求停止")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_live_run)
        actions.addWidget(self.run_btn)
        actions.addWidget(self.stop_btn)
        actions.addStretch(1)
        layout.addLayout(actions)

        live_cards = QWidget()
        grid = QGridLayout(live_cards)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(10)
        self.live_variant = QLabel("-")
        self.live_tokens = QLabel("0")
        self.live_calls = QLabel("0")
        self.live_saved = QLabel("-")
        for label, value, col, color in [
            ("当前模式", self.live_variant, 0, COLORS["blue"]),
            ("实时 token", self.live_tokens, 1, COLORS["green"]),
            ("API 调用", self.live_calls, 2, COLORS["amber"]),
            ("C 相比 A", self.live_saved, 3, COLORS["green"]),
        ]:
            card = QFrame()
            card.setObjectName("card")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(14, 10, 14, 10)
            t = QLabel(label)
            t.setObjectName("cardTitle")
            value.setObjectName("cardValue")
            value.setStyleSheet(f"color: {color};")
            card_layout.addWidget(t)
            card_layout.addWidget(value)
            grid.addWidget(card, 0, col)
        layout.addWidget(live_cards)

        self.live_table = QTableWidget(0, 11)
        self.live_table.setHorizontalHeaderLabels([
            "模式", "策略", "API", "Prompt", "Output", "总 Token",
            "缓存", "耗时", "消息", "记忆命中", "质量",
        ])
        self.live_table.verticalHeader().setVisible(False)
        self.live_table.setAlternatingRowColors(True)
        self.live_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.live_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.live_table.setMinimumHeight(150)
        self.live_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.live_table)

        self.live_log = QTextEdit()
        self.live_log.setReadOnly(True)
        self.live_log.setMinimumHeight(150)
        self.live_log.setObjectName("log")
        layout.addWidget(self.live_log)
        return box

    def _build_live_panel(self) -> QWidget:
        box = QFrame()
        box.setObjectName("panel")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        title = QLabel("full24 实时实验台：先展示 24 题空表，再依次运行 A 纯文本和 C 最终方案")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        form = QGridLayout()
        form.setColumnStretch(1, 1)
        form.setColumnStretch(3, 1)
        self.api_key_input = QLineEdit(os.environ.get("DEEPSEEK_API_KEY", ""))
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_input.setPlaceholderText("DEEPSEEK_API_KEY")
        self.model_input = QLineEdit("deepseek-chat")
        self.fast_embed_cb = QCheckBox("快速 embedding（录屏推荐）")
        self.fast_embed_cb.setChecked(True)
        self.reset_memory_cb = QCheckBox("运行 C 前清空记忆")
        self.reset_memory_cb.setChecked(True)
        form.addWidget(QLabel("DeepSeek Key"), 0, 0)
        form.addWidget(self.api_key_input, 0, 1)
        form.addWidget(QLabel("模型"), 0, 2)
        form.addWidget(self.model_input, 0, 3)
        form.addWidget(self.fast_embed_cb, 1, 1)
        form.addWidget(self.reset_memory_cb, 1, 2)
        layout.addLayout(form)

        actions = QHBoxLayout()
        self.run_a_btn = QPushButton("1. 运行 A 纯文本 24题")
        self.run_a_btn.clicked.connect(lambda: self._start_full24_run("A"))
        self.run_c_btn = QPushButton("2. 运行 C 最终方案 24题")
        self.run_c_btn.clicked.connect(lambda: self._start_full24_run("C"))
        self.clear_full24_btn = QPushButton("清空表格指标")
        self.clear_full24_btn.clicked.connect(self._reset_full24_table)
        self.stop_btn = QPushButton("请求停止")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_live_run)
        actions.addWidget(self.run_a_btn)
        actions.addWidget(self.run_c_btn)
        actions.addWidget(self.clear_full24_btn)
        actions.addWidget(self.stop_btn)
        actions.addStretch(1)
        layout.addLayout(actions)

        live_cards = QWidget()
        grid = QGridLayout(live_cards)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(10)
        self.live_variant = QLabel("-")
        self.live_tokens = QLabel("0")
        self.live_calls = QLabel("0")
        self.live_saved = QLabel("-")
        for label, value, col, color in [
            ("当前阶段", self.live_variant, 0, COLORS["blue"]),
            ("累计 token", self.live_tokens, 1, COLORS["green"]),
            ("累计 API", self.live_calls, 2, COLORS["amber"]),
            ("C 相比 A", self.live_saved, 3, COLORS["green"]),
        ]:
            card = QFrame()
            card.setObjectName("card")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(14, 10, 14, 10)
            t = QLabel(label)
            t.setObjectName("cardTitle")
            value.setObjectName("cardValue")
            value.setStyleSheet(f"color: {color};")
            card_layout.addWidget(t)
            card_layout.addWidget(value)
            grid.addWidget(card, 0, col)
        layout.addWidget(live_cards)

        self.full24_tasks = get_suite("full24")
        self.full24_rows: Dict[str, Dict[str, Any]] = {}
        self.live_table = QTableWidget(len(self.full24_tasks), 13)
        self.live_table.setHorizontalHeaderLabels([
            "#", "任务ID", "领域", "题目摘要",
            "A Token", "A API", "A 耗时",
            "C Token", "C API", "C 耗时", "C 策略", "记忆命中", "节省",
        ])
        self.live_table.verticalHeader().setVisible(False)
        self.live_table.setAlternatingRowColors(True)
        self.live_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.live_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.live_table.setMinimumHeight(360)
        self.live_table.horizontalHeader().setStretchLastSection(True)
        self._reset_full24_table()
        layout.addWidget(self.live_table)

        self.live_log = QTextEdit()
        self.live_log.setReadOnly(True)
        self.live_log.setMinimumHeight(150)
        self.live_log.setObjectName("log")
        layout.addWidget(self.live_log)
        return box

    def _build_cards(self) -> QWidget:
        variants = self.report.get("variants", {})
        a = variants.get("A_pure_text_full24", {})
        b = variants.get("B_nocache_full24", {})
        c = variants.get("C_structured_full24", {})
        a_tokens = a.get("total_tokens", 0)
        b_tokens = b.get("total_tokens", 0)
        c_tokens = c.get("total_tokens", 0)

        cards = QWidget()
        grid = QGridLayout(cards)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(10)
        grid.addWidget(Card("主结论", f"{100 - pct(c_tokens, a_tokens):.1f}%", "C 相比纯文本 A 的 token 节省", COLORS["green"]), 0, 0)
        grid.addWidget(Card("质量通过", c.get("quality_rate", "24/24"), "结构化全功能在 full24 上通过", COLORS["blue"]), 0, 1)
        grid.addWidget(Card("模板/执行跳过", str(c.get("dropout_hits", 0)), "Executor dropout 命中次数", COLORS["amber"]), 0, 2)
        grid.addWidget(Card("记忆收益", f"{100 - pct(c_tokens, b_tokens):.1f}%", "C 相比无缓存 B 的 token 节省", COLORS["green"]), 0, 3)
        return cards

    def _build_bars(self, title: str, field: str, suffix: str, color: str) -> QWidget:
        variants = self.report.get("variants", {})
        values = [(variant_label(k), float(v.get(field, 0) or 0)) for k, v in variants.items()]
        max_value = max((v for _, v in values), default=1)

        box = QFrame()
        box.setObjectName("panel")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(14, 12, 14, 12)
        title_label = QLabel(title)
        title_label.setObjectName("sectionTitle")
        layout.addWidget(title_label)
        for name, value in values:
            row_color = color
            if name.startswith("A"):
                row_color = COLORS["slate"]
            elif name.startswith("B"):
                row_color = COLORS["amber"]
            elif name.startswith("C"):
                row_color = COLORS["green"]
            layout.addWidget(BarRow(name, value, max_value, suffix, row_color))
        return box

    def _build_task_table(self) -> QWidget:
        rows = self.report.get("per_task", [])
        box = QFrame()
        box.setObjectName("panel")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(14, 12, 14, 12)
        title = QLabel("历史 full24 报告：逐题效果、C 模式策略与节省率")
        title.setObjectName("sectionTitle")

        table = QTableWidget(len(rows), 5)
        table.setHorizontalHeaderLabels(["任务", "A Token", "C Token", "节省", "C 策略"])
        table.verticalHeader().setVisible(False)
        table.setAlternatingRowColors(True)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.setMinimumHeight(320)

        for r, row in enumerate(rows):
            values = [
                row.get("task_id", ""),
                f"{row.get('A_tokens', 0):,}",
                f"{row.get('C_tokens', 0):,}",
                f"{row.get('C_vs_A_save_pct', 0):.1f}%",
                row.get("C_strategy", ""),
            ]
            for c, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if c in (1, 2, 3):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                table.setItem(r, c, item)
        table.resizeColumnsToContents()
        table.horizontalHeader().setStretchLastSection(True)

        layout.addWidget(title)
        layout.addWidget(table)
        return box

    def _build_right_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("panel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        self.file_title = QLabel()
        self.file_title.setObjectName("sectionTitle")
        self.file_path = QLabel()
        self.file_path.setObjectName("path")
        self.file_path.setWordWrap(True)
        self.file_why = QLabel()
        self.file_why.setObjectName("why")
        self.file_why.setWordWrap(True)
        self.code_view = QTextEdit()
        self.code_view.setReadOnly(True)
        self.code_view.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.code_view.setFont(QFont("Consolas", 10))
        self.code_view.setObjectName("code")

        layout.addWidget(self.file_title)
        layout.addWidget(self.file_path)
        layout.addWidget(self.file_why)
        layout.addWidget(self.code_view, 1)
        return panel

    def _load_file(self, row: int) -> None:
        if row < 0:
            row = 0
        self.file_list.setCurrentRow(row)
        item = KEY_FILES[row]
        path = item["path"]
        self.file_title.setText(item["title"])
        self.file_path.setText(str(path))
        self.file_why.setText(item["why"])
        if not path.exists():
            self.code_view.setPlainText("文件不存在。")
            return
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        preview = "\n".join(f"{i + 1:>4}  {line}" for i, line in enumerate(lines[:260]))
        if len(lines) > 260:
            preview += f"\n\n... 仅显示前 260 行，共 {len(lines)} 行。点击“打开当前文件”查看完整文件。"
        self.code_view.setPlainText(preview)

    def _open_current_file(self) -> None:
        row = self.file_list.currentRow()
        if row < 0:
            return
        path = KEY_FILES[row]["path"]
        if not path.exists():
            QMessageBox.warning(self, "文件不存在", str(path))
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _reset_full24_table(self) -> None:
        self.full24_rows = {}
        for r, task in enumerate(self.full24_tasks):
            values = [
                str(r + 1),
                task.get("task_id", ""),
                task.get("domain", ""),
                task.get("description", "")[:92],
                "", "", "", "", "", "", "", "", "",
            ]
            for c, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if c in (0, 4, 5, 7, 8, 11, 12):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.live_table.setItem(r, c, item)
        self.live_table.resizeColumnsToContents()
        self.live_variant.setText("-")
        self.live_tokens.setText("0")
        self.live_calls.setText("0")
        self.live_saved.setText("-")
        if hasattr(self, "live_log"):
            self.live_log.clear()
            self._append_log("已载入 full24 共 24 题；当前没有任何实时指标。请先运行 A，再运行 C。")

    def _start_full24_run(self, mode: str) -> None:
        if self.worker and self.worker.isRunning():
            return
        config = {
            "mode": mode,
            "api_key": self.api_key_input.text(),
            "model": self.model_input.text().strip() or "deepseek-chat",
            "fast_embeddings": self.fast_embed_cb.isChecked(),
            "reset_memory": self.reset_memory_cb.isChecked(),
        }
        if mode == "C" and not any(
            row.get("A", {}).get("total_tokens") for row in self.full24_rows.values()
        ):
            self._append_log("提示：你还没跑 A。可以继续跑 C，但演示顺序建议先 A 后 C。")
        self.worker = Full24RunWorker(config)
        self.worker.progress.connect(self._append_log)
        self.worker.task_done.connect(self._update_full24_task_row)
        self.worker.totals.connect(self._update_full24_totals)
        self.worker.finished_ok.connect(self._full24_finished)
        self.worker.failed.connect(self._live_failed)
        self.run_a_btn.setEnabled(False)
        self.run_c_btn.setEnabled(False)
        self.clear_full24_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._append_log(f"开始运行 {mode} 模式 full24。24 题会逐题填入表格。")
        self.worker.start()

    def _update_full24_task_row(self, row: Dict[str, Any]) -> None:
        task_id = row.get("task_id", "")
        mode = row.get("mode", "")
        if not task_id or not mode:
            return
        bucket = self.full24_rows.setdefault(task_id, {})
        bucket[mode] = row
        index = next((i for i, t in enumerate(self.full24_tasks) if t["task_id"] == task_id), -1)
        if index < 0:
            return
        if mode == "A":
            updates = {
                4: f"{row.get('total_tokens', 0):,}",
                5: str(row.get("api_calls", 0)),
                6: f"{row.get('elapsed_ms', 0) / 1000:.1f}s",
            }
        else:
            updates = {
                7: f"{row.get('total_tokens', 0):,}",
                8: str(row.get("api_calls", 0)),
                9: f"{row.get('elapsed_ms', 0) / 1000:.1f}s",
                10: row.get("strategy", ""),
                11: str(row.get("memory_hits", 0)),
            }
            a = bucket.get("A", {})
            if a.get("total_tokens"):
                saved = (1 - row.get("total_tokens", 0) / max(a.get("total_tokens", 1), 1)) * 100
                updates[12] = f"{saved:.1f}%"
        for col, value in updates.items():
            item = QTableWidgetItem(str(value))
            if col in (4, 5, 7, 8, 11, 12):
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.live_table.setItem(index, col, item)
        self.live_table.scrollToItem(self.live_table.item(index, 0))
        self.live_table.resizeColumnsToContents()
        self._recompute_full24_savings()

    def _update_full24_totals(self, totals: Dict[str, Any]) -> None:
        mode = totals.get("mode", "-")
        self.live_variant.setText(f"{mode} ({totals.get('done', 0)}/24)")
        self.live_tokens.setText(f"{int(totals.get('tokens', 0)):,}")
        self.live_calls.setText(str(totals.get("api_calls", 0)))
        self._recompute_full24_savings()

    def _recompute_full24_savings(self) -> None:
        a_total = sum(row.get("A", {}).get("total_tokens", 0) for row in self.full24_rows.values())
        c_total = sum(row.get("C", {}).get("total_tokens", 0) for row in self.full24_rows.values())
        if a_total and c_total:
            self.live_saved.setText(f"{(1 - c_total / a_total) * 100:.1f}%")

    def _full24_finished(self, mode: str) -> None:
        self.run_a_btn.setEnabled(True)
        self.run_c_btn.setEnabled(True)
        self.clear_full24_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._append_log(f"{mode} 模式 full24 运行结束。")

    def _start_live_run(self) -> None:
        if self.worker and self.worker.isRunning():
            return
        self.live_rows = []
        self.live_table.setRowCount(0)
        self.live_log.clear()
        self.live_variant.setText("-")
        self.live_tokens.setText("0")
        self.live_calls.setText("0")
        self.live_saved.setText("-")

        config = {
            "api_key": self.api_key_input.text(),
            "model": self.model_input.text().strip() or "deepseek-chat",
            "task_text": self.task_input.toPlainText(),
            "tags": self.tags_input.text(),
            "run_pure": self.run_pure_cb.isChecked(),
            "run_nocache": self.run_nocache_cb.isChecked(),
            "run_full": self.run_full_cb.isChecked(),
            "fast_embeddings": self.fast_embed_cb.isChecked(),
            "reset_memory": self.reset_memory_cb.isChecked(),
            "reuse_runtime": self.reuse_runtime_cb.isChecked(),
        }
        if not any([config["run_pure"], config["run_nocache"], config["run_full"]]):
            QMessageBox.warning(self, "请选择模式", "至少选择一个要运行的模式。")
            return

        self._append_log("开始实时演示。DeepSeek token 会以 API 返回 usage 为准。")
        if config["fast_embeddings"]:
            self._append_log("快速 embedding 已启用：LLM token 真实，向量检索使用确定性 hash，适合录屏。")

        self.worker = LiveRunWorker(config)
        self.worker.progress.connect(self._append_log)
        self.worker.metrics.connect(self._update_live_metrics)
        self.worker.row_ready.connect(self._append_live_row)
        self.worker.finished_ok.connect(self._live_finished)
        self.worker.failed.connect(self._live_failed)
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.worker.start()

    def _stop_live_run(self) -> None:
        if self.worker and self.worker.isRunning():
            self.worker.request_stop()
            self._append_log("已请求停止：当前 API 调用结束后会停下。")

    def _append_log(self, text: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.live_log.append(f"[{timestamp}] {text}")
        self.live_log.verticalScrollBar().setValue(self.live_log.verticalScrollBar().maximum())

    def _update_live_metrics(self, stats: Dict[str, Any]) -> None:
        self.live_variant.setText(str(stats.get("active_variant", "-")))
        self.live_tokens.setText(f"{int(stats.get('total_tokens', 0)):,}")
        self.live_calls.setText(str(stats.get("api_calls", 0)))

    def _append_live_row(self, row: Dict[str, Any]) -> None:
        self.live_rows.append(row)
        r = self.live_table.rowCount()
        self.live_table.insertRow(r)
        values = [
            row.get("variant", ""),
            row.get("strategy", ""),
            row.get("api_calls", 0),
            row.get("prompt_tokens", 0),
            row.get("completion_tokens", 0),
            row.get("total_tokens", 0),
            row.get("cached_tokens", 0),
            f"{row.get('elapsed_ms', 0) / 1000:.1f}s",
            row.get("messages_sent", 0),
            row.get("memory_hits", 0),
            "PASS" if row.get("composite_pass") else ("-" if row.get("composite_pass") is None else "CHECK"),
        ]
        for c, value in enumerate(values):
            item = QTableWidgetItem(str(value))
            if c in (2, 3, 4, 5, 6, 8, 9):
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.live_table.setItem(r, c, item)
        self.live_table.resizeColumnsToContents()
        self._recompute_live_savings()
        preview = row.get("summary_preview") or ""
        if preview:
            self._append_log(f"{row.get('variant')} 摘要预览：{preview}")

    def _recompute_live_savings(self) -> None:
        a = next((r for r in self.live_rows if str(r.get("variant", "")).startswith("A")), None)
        c = next((r for r in self.live_rows if str(r.get("variant", "")).startswith("C")), None)
        if not a or not c or not a.get("total_tokens"):
            self.live_saved.setText("-")
            return
        saved = (1 - c.get("total_tokens", 0) / max(a.get("total_tokens", 1), 1)) * 100
        self.live_saved.setText(f"{saved:.1f}%")

    def _live_finished(self, rows: List[Dict[str, Any]]) -> None:
        if hasattr(self, "run_btn"):
            self.run_btn.setEnabled(True)
        if hasattr(self, "run_a_btn"):
            self.run_a_btn.setEnabled(True)
            self.run_c_btn.setEnabled(True)
            self.clear_full24_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._append_log(f"实时演示结束，共完成 {len(rows)} 个模式。")
        self._recompute_live_savings()

    def _live_failed(self, message: str) -> None:
        if hasattr(self, "run_btn"):
            self.run_btn.setEnabled(True)
        if hasattr(self, "run_a_btn"):
            self.run_a_btn.setEnabled(True)
            self.run_c_btn.setEnabled(True)
            self.clear_full24_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._append_log(f"运行失败：{message}")
        QMessageBox.critical(self, "运行失败", message)


def apply_theme(app: QApplication) -> None:
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(COLORS["bg"]))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(COLORS["text"]))
    app.setPalette(palette)
    app.setStyleSheet(
        f"""
        QWidget {{
            font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif;
            color: {COLORS["text"]};
            font-size: 13px;
        }}
        QMainWindow, QScrollArea {{
            background: {COLORS["bg"]};
        }}
        #header, #panel, #card {{
            background: {COLORS["panel"]};
            border: 1px solid {COLORS["line"]};
            border-radius: 8px;
        }}
        #title {{
            font-size: 24px;
            font-weight: 700;
        }}
        #subtitle, #cardNote, #path {{
            color: {COLORS["muted"]};
        }}
        #sectionTitle {{
            font-size: 15px;
            font-weight: 700;
        }}
        #cardTitle {{
            color: {COLORS["muted"]};
            font-size: 12px;
        }}
        #cardValue {{
            font-size: 30px;
            font-weight: 800;
        }}
        #pill {{
            background: #eef2ff;
            border: 1px solid #c7d2fe;
            border-radius: 8px;
            color: #3730a3;
            padding: 8px 12px;
        }}
        #flow, #why {{
            background: #f8fafc;
            border: 1px solid {COLORS["line"]};
            border-radius: 6px;
            padding: 10px;
            line-height: 1.4;
        }}
        #barTrack {{
            background: #edf0f5;
            border-radius: 4px;
        }}
        #barName {{
            color: {COLORS["text"]};
        }}
        #barValue {{
            color: {COLORS["muted"]};
            font-variant-numeric: tabular-nums;
        }}
        QListWidget, QTextEdit, QTableWidget {{
            background: #ffffff;
            border: 1px solid {COLORS["line"]};
            border-radius: 6px;
            selection-background-color: #dbeafe;
        }}
        QLineEdit {{
            background: #ffffff;
            border: 1px solid {COLORS["line"]};
            border-radius: 6px;
            padding: 8px;
        }}
        QListWidget::item {{
            padding: 10px;
            border-bottom: 1px solid #eef1f5;
        }}
        QListWidget::item:selected {{
            color: {COLORS["blue"]};
            font-weight: 700;
        }}
        QPushButton {{
            background: {COLORS["blue"]};
            color: white;
            border: 0;
            border-radius: 6px;
            padding: 9px 12px;
            font-weight: 700;
        }}
        QPushButton:hover {{
            background: #1d4ed8;
        }}
        QHeaderView::section {{
            background: #f1f5f9;
            border: 0;
            border-bottom: 1px solid {COLORS["line"]};
            padding: 7px;
            font-weight: 700;
        }}
        #code {{
            background: #0f172a;
            color: #e5e7eb;
            border-color: #1e293b;
            padding: 8px;
        }}
        #log {{
            background: #111827;
            color: #d1fae5;
            border-color: #1f2937;
            padding: 8px;
        }}
        """
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch the 702solver Qt demo dashboard.")
    parser.add_argument("--smoke", action="store_true", help="Create the window, then exit immediately.")
    args = parser.parse_args()

    load_env_file()
    app = QApplication(sys.argv)
    apply_theme(app)
    window = Dashboard(read_report())
    if args.smoke:
        return 0
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
