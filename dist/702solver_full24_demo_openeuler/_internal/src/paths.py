"""Project paths — runtime artifacts live under ``output/``."""

from __future__ import annotations

import os

# Repository root (parent of ``src/``)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUTPUT_DIR = os.path.join(ROOT, "output")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")
DATABASES_DIR = os.path.join(OUTPUT_DIR, "databases")
EXPERIMENT_MATRIX_DIR = os.path.join(OUTPUT_DIR, "experiment_matrix")
EXPERIMENT_MATRIX_V2_DIR = os.path.join(OUTPUT_DIR, "experiment_matrix_v2")

# Default SQLite paths
DEFAULT_SHARED_MEMORY_DB = os.path.join(DATABASES_DIR, "shared_memory.db")
DEFAULT_CHAT_SESSION_DB = os.path.join(DATABASES_DIR, "chat_session.db")


def ensure_output_dirs() -> None:
    for d in (
        OUTPUT_DIR,
        RESULTS_DIR,
        LOGS_DIR,
        DATABASES_DIR,
        EXPERIMENT_MATRIX_DIR,
        EXPERIMENT_MATRIX_V2_DIR,
    ):
        os.makedirs(d, exist_ok=True)


def result_path(filename: str) -> str:
    ensure_output_dirs()
    return os.path.join(RESULTS_DIR, filename)


def log_path(filename: str) -> str:
    ensure_output_dirs()
    return os.path.join(LOGS_DIR, filename)


def database_path(filename: str) -> str:
    ensure_output_dirs()
    return os.path.join(DATABASES_DIR, filename)


def resolve_path(filename: str, *, subdir: str = "results") -> str:
    """Prefer ``output/<subdir>/``; fall back to repo root for legacy files."""
    base = {
        "results": RESULTS_DIR,
        "logs": LOGS_DIR,
        "databases": DATABASES_DIR,
    }.get(subdir, RESULTS_DIR)
    new = os.path.join(base, filename)
    legacy = os.path.join(ROOT, filename)
    if os.path.exists(new):
        return new
    if os.path.exists(legacy):
        return legacy
    return new
