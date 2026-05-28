"""Adversarial task must not trigger E2E cache."""

import json

import pytest

from src.memory.models import MemoryUnit
from src.orchestrator import Orchestrator
from src.run_options import DEFAULT_OPTIONS
from tests.conftest import CountingMockLLM


@pytest.fixture
def orch_with_cached_summary(tmp_path):
    db = str(tmp_path / "test.db")
    llm = CountingMockLLM()
    orch = Orchestrator(
        llm=llm,
        mode="structured",
        use_real_embeddings=False,
        memory_db_path=db,
        run_options=DEFAULT_OPTIONS,
    )
    cached_summary = {
        "summary": {"key_findings": ["cached"], "facts": [], "conclusion": "ok"},
        "evidence_refs": [],
    }
    mem = MemoryUnit(
        source_agent="summarizer",
        task_topic="security",
        summary="cached",
        content=json.dumps(cached_summary),
        tags=["security", "summary"],
        memory_type="result",
        embedding=orch.embedding_engine.encode("python security vulnerabilities"),
    )
    orch.memory_store.store(mem, dedup=False)
    return orch


def test_no_e2e_tag_blocks_cache(orch_with_cached_summary):
    orch = orch_with_cached_summary
    result = orch.execute_task(
        task_id="x_test",
        task_description="SOC2 compliance mapping only, no code vulns",
        tags=["security", "adversarial", "no-e2e-cache"],
    )
    assert not result.get("_e2e_cached")


def test_benchmark_suites_adversarial_flags():
    from experiments.benchmark_suites import get_suite

    adv = get_suite("adversarial6")
    assert len(adv) == 6
    assert all(t.get("must_not_e2e") for t in adv)
    assert all("no-e2e-cache" in t["tags"] for t in adv)
