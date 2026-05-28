"""Regression: compliance tasks must not reuse vulnerability templates/summaries."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.agents.summarizer import SummarizerAgent
from src.memory.models import MemoryUnit
from src.run_options import RunOptions


@pytest.fixture
def summarizer_with_vuln_cache():
    scheduler = MagicMock()
    llm = MagicMock()
    embedding = MagicMock()
    state_bus = MagicMock()
    store = MagicMock()

    vuln_summary = {
        "key_findings": ["SQL injection via string formatting is detectable"],
        "facts": ["Command injection via os.system() is detectable"],
        "conclusion": "Use parameterized queries and avoid os.system().",
    }
    vuln_mem = MemoryUnit(
        memory_id="vuln-result-1",
        task_topic="Summary: Analyze Python security vulnerabilities",
        task_id="s1_python_vuln",
        summary="vuln findings",
        content=json.dumps({"summary": vuln_summary, "task_id": "s1_python_vuln"}),
        tags=["security", "python", "summary"],
        memory_type="result",
    )

    class FakeIndex:
        ntotal = 1

    store._index = FakeIndex()
    store.search_by_similarity.return_value = [(vuln_mem, 0.91)]
    store.get.return_value = None
    store.count_by_tags_and_type.return_value = 0
    store.get_templates.return_value = []
    store.store.return_value = "new-mem-id"

    embedding.encode.side_effect = lambda text: np.ones(384, dtype=np.float32)

    agent = SummarizerAgent(
        scheduler, llm, embedding, state_bus, store, use_structured_protocol=True
    )
    agent.run_options = RunOptions(enable_intent_cache_gate=True)
    llm.chat_structured.return_value = {
        "key_findings": ["SOC2 CC6 covers logical access controls"],
        "facts": ["ISO27001 A.12 covers operations security"],
        "conclusion": "Map controls to logging and change management.",
    }
    return agent, llm


def test_compliance_task_skips_vuln_summary_cache(summarizer_with_vuln_cache):
    agent, llm = summarizer_with_vuln_cache
    compliance_desc = (
        "Outline SOC2 and ISO27001 control mapping for a Python SaaS backend: "
        "access control, logging, change management."
    )
    result = agent.execute_task(
        {
            "task_id": "x3_security_compliance_only",
            "task_description": compliance_desc,
            "tags": ["security", "compliance", "adversarial", "no-e2e-cache"],
            "plan": {"task_description": compliance_desc, "subtasks": []},
            "retrieval_results": {
                1: {"combined_results": "SOC2 CC6. ISO27001 Annex A controls."}
            },
        }
    )
    assert llm.chat_structured.called
    summary = result["summary"]
    assert isinstance(summary, dict)
    joined = " ".join(summary.get("key_findings", []) + [summary.get("conclusion", "")])
    assert "SOC2" in joined or "ISO" in joined
    assert "sql injection" not in joined.lower()
