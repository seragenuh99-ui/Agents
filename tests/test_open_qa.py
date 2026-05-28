"""Tests for open Q&A optimizations in structured mode."""

from src.agents.retriever import RetrieverAgent
from src.run_options import DEFAULT_OPTIONS, options_for_task
from src.task_intent import is_open_qa, requires_fresh_synthesis, blocks_executor_dropout


def test_is_open_qa_custom_tag():
    assert is_open_qa("anything", ["custom"])


def test_is_open_qa_chinese_food():
    assert is_open_qa("哈尔滨的美食有什么推荐", ["custom"])


def test_is_open_qa_not_energy_benchmark():
    assert not is_open_qa(
        "Compare solar and wind LCOE", ["energy", "research"]
    )


def test_requires_fresh_synthesis_open_qa():
    assert requires_fresh_synthesis("哈尔滨美食", ["custom"])


def test_blocks_executor_dropout_open_qa():
    assert blocks_executor_dropout("哈尔滨美食", ["custom"])


def test_options_for_task_open_qa_richer():
    opts = options_for_task(DEFAULT_OPTIONS, "哈尔滨美食推荐", ["custom"])
    assert opts.evidence_max_chars >= 1200
    assert opts.summarizer_max_tokens >= 1024
    assert opts.enable_planner_cache is False


def test_retriever_kb_categories_empty_for_custom():
    agent = RetrieverAgent.__new__(RetrieverAgent)
    assert agent._resolve_kb_categories(["custom"]) == []
