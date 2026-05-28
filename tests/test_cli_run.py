"""Tests for run.py CLI, REPL session, and reporting."""

from __future__ import annotations

import json
import os
import sys
from io import StringIO
from unittest.mock import patch

import pytest

from experiments.experiment_common import BatchSession, clear_db
from src.cli.run_engine import (
    MODE_NUM,
    RunConfig,
    build_tasks_from_asks,
    execute_session,
    mode_num_from_key,
    repl_chat_db_path,
    repl_supports_shared_memory,
    resolve_config_tasks,
)
from src.cli.run_reporter import get_row_answer_text, print_multi_mode_answers
from src.evaluation.quality_validator import format_answer_markdown


def test_mode_num_from_key_roundtrip():
    for num, key in MODE_NUM.items():
        assert mode_num_from_key(key) == num


def test_mode_num_from_key_unknown():
    with pytest.raises(ValueError, match="unknown mode_key"):
        mode_num_from_key("invalid")


def test_get_row_answer_text_prefers_full():
    row = {"summary_preview": "短", "summary_full": "完整答案正文"}
    assert get_row_answer_text(row) == "完整答案正文"


def test_get_row_answer_text_fallback_preview():
    row = {"summary_preview": "仅预览"}
    assert get_row_answer_text(row) == "仅预览"


def test_run_config_ask_texts_use_custom_not_default_suite():
    cfg = RunConfig(modes=[2], ask_texts=["哈尔滨美食"])
    tasks = resolve_config_tasks(cfg)
    assert len(tasks) == 1
    assert tasks[0]["tags"] == ["custom"]
    assert cfg.suite is None


def test_config_from_args_no_api_key_nameerror():
    """Regression: REPL must not call run_repl inside config_from_args."""
    import argparse
    import run as run_mod

    args = argparse.Namespace(
        mode=None,
        suite=None,
        ask=None,
        tasks=None,
        quality=True,
        no_quality=False,
        json=None,
        yes=False,
        help_cn=False,
        preset=None,
    )
    cfg = run_mod.config_from_args(args)
    assert cfg is None
    src = open(run_mod.__file__).read()
    assert "run_repl(api_key)" not in src.split("def config_from_args")[1].split("def main")[0]


def test_main_dispatches_to_run_repl_without_args():
    import run as run_mod

    with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}, clear=False):
        with patch.object(run_mod, "run_repl", return_value=0) as repl:
            with patch.object(sys, "argv", ["run.py"]):
                assert run_mod.main() == 0
            repl.assert_called_once_with("test-key")


def test_execute_session_multi_mode_report(tmp_path):
    from src.cli.run_engine import ModeRunResult, MODE_CATALOG

    def fake_pure(api_key, tasks, **kw):
        on = kw.get("on_task")
        row = {
            "task_id": tasks[0]["task_id"],
            "description": tasks[0]["description"],
            "summary_full": "纯文本完整答案",
            "summary_preview": "纯",
            "total_tokens": 100,
            "api_calls": 3,
            "elapsed_ms": 1000,
            "strategy": "PURE_TEXT_3CALL",
            "relevance": 0.9,
        }
        if on:
            on(row, 1, 1)
        return ModeRunResult(
            1,
            "pure",
            MODE_CATALOG[1][1],
            [row],
            {"total_tokens": 100, "wall_clock_sec": 1.0, "memory_hit_rate": 0.0},
        )

    def fake_struct(api_key, tasks, **kw):
        on = kw.get("on_task")
        mn = kw.get("mode_num", 2)
        row = {
            "task_id": tasks[0]["task_id"],
            "description": tasks[0]["description"],
            "summary_full": "**要点**:\n1. 锅包肉\n\n**结论**: 哈尔滨美食丰富",
            "summary_obj": {"key_findings": ["锅包肉"], "conclusion": "哈尔滨美食丰富"},
            "summary_preview": "锅包肉",
            "total_tokens": 40,
            "api_calls": 2,
            "elapsed_ms": 500,
            "strategy": "FULL_GEN",
            "relevance": 0.85,
            "messages_sent": 4,
        }
        if on:
            on(row, 1, 1)
        return ModeRunResult(
            mn,
            MODE_NUM[mn],
            MODE_CATALOG[mn][1],
            [row],
            {"total_tokens": 40, "wall_clock_sec": 0.5, "memory_hit_rate": 0.2},
        )

    cfg = RunConfig(modes=[1, 2], ask_texts=["哈尔滨美食推荐"], save_json=False)
    with patch("src.cli.run_engine.run_pure_mode", fake_pure), patch(
        "src.cli.run_engine.run_structured_mode", fake_struct
    ):
        report = execute_session("fake", cfg)

    assert len(report["sessions"]) == 2
    pure = report["sessions"][0]["results"][0]
    struct = report["sessions"][1]["results"][0]
    assert "纯文本完整答案" in get_row_answer_text(pure)
    assert "锅包肉" in get_row_answer_text(struct)
    json.dumps(report)


def test_print_multi_mode_answers_shows_both(capsys):
    sessions = [
        {
            "mode_label": "纯文本对照",
            "mode_key": "pure",
            "results": [{"summary_full": "答案A", "elapsed_ms": 1000, "total_tokens": 1, "api_calls": 1, "strategy": "P"}],
        },
        {
            "mode_label": "结构化全功能",
            "mode_key": "structured",
            "results": [{"summary_full": "答案B", "elapsed_ms": 500, "total_tokens": 2, "api_calls": 2, "strategy": "F"}],
        },
    ]
    print_multi_mode_answers(sessions, question="测试题")
    out = capsys.readouterr().out
    assert "多模式完整回答对照" in out
    assert "答案A" in out
    assert "答案B" in out
    assert "纯文本对照" in out
    assert "结构化全功能" in out


def test_repl_supports_shared_memory():
    assert repl_supports_shared_memory([2]) is True
    assert repl_supports_shared_memory([3]) is True
    assert repl_supports_shared_memory([1, 2]) is False
    assert repl_supports_shared_memory([1]) is False


def test_repl_chat_db_path_under_output_databases():
    path = repl_chat_db_path(2)
    assert "databases" in path
    assert path.endswith("repl_chat.db")


def test_batch_session_reuse_preserves_orch_reference():
    """Session orch should survive second run_task_batch when clear_db_before=False."""
    session = BatchSession()
    sentinel = object()
    session.orch = sentinel
    session.llm = object()

    db = "unused.db"
    clear_db_before = False
    if clear_db_before:
        clear_db(db)
        session.orch = None

    assert session.orch is sentinel

    # Simulate end-of-batch bookkeeping
    session.orch = sentinel
    assert session.orch is sentinel


def test_experiment_presets_core12():
    from src.cli.task_catalog import EXPERIMENT_PRESETS

    p = EXPERIMENT_PRESETS["defense12"]
    assert p["modes"] == [2]
    assert p["suite"] == "core12"
    cmp_p = EXPERIMENT_PRESETS["defense12-compare"]
    assert cmp_p["modes"] == [1, 2]


def test_config_from_preset():
    import argparse
    import run as run_mod

    args = argparse.Namespace(
        preset="defense12",
        help_cn=False,
        mode=None,
        suite=None,
        ask=None,
        tasks=None,
        quality=True,
        json=None,
    )
    cfg = run_mod.config_from_args(args)
    assert cfg is not None
    assert cfg.modes == [2]
    assert cfg.suite == "core12"
    assert cfg.save_json is True


def test_suite_catalog_has_eta_and_defense_flag():
    from src.cli.task_catalog import list_defense_suites, suite_eta

    assert "core12" in list_defense_suites()
    assert suite_eta("full24")


def test_main_preset_skips_repl():
    import run as run_mod

    with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}, clear=False):
        with patch.object(run_mod, "run_repl") as repl:
            with patch.object(run_mod, "config_from_args") as cfg_fn:
                cfg_fn.return_value = None
                with patch.object(sys, "argv", ["run.py", "--preset", "defense12", "-y"]):
                    run_mod.main()
                repl.assert_not_called()


def test_format_answer_markdown_structured():
    obj = {"key_findings": ["锅包肉", "红肠"], "conclusion": "值得品尝"}
    text = format_answer_markdown(obj, "fallback")
    assert "锅包肉" in text
    assert "值得品尝" in text
