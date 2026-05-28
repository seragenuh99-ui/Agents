"""Regression: compare tasks must not reuse single-topic summary caches."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.agents.summarizer import SummarizerAgent
from src.memory.models import MemoryUnit
from src.run_options import RunOptions


@pytest.fixture
def summarizer_with_memory():
    """Summarizer backed by a fake memory store with one wind-energy result."""
    scheduler = MagicMock()
    llm = MagicMock()
    embedding = MagicMock()
    state_bus = MagicMock()
    store = MagicMock()

    wind_summary = {
        "key_findings": ["Onshore wind capacity factor 25-35%"],
        "facts": ["Offshore wind reaches 40-55%"],
        "conclusion": "Wind dominates offshore deployment.",
    }
    wind_mem = MemoryUnit(
        memory_id="wind-result-1",
        task_topic="Summary: Research wind energy technologies",
        task_id="e3_wind_energy",
        summary="wind findings",
        content=json.dumps({"summary": wind_summary, "task_id": "e3_wind_energy"}),
        tags=["wind", "energy", "summary"],
        memory_type="result",
    )

    class FakeIndex:
        ntotal = 1

    store._index = FakeIndex()
    store.search_by_similarity.return_value = [(wind_mem, 0.88)]
    store.get.return_value = None
    store.count_by_tags_and_type.return_value = 0
    store.get_templates.return_value = []
    store.store.return_value = "new-mem-id"

    def encode(text):
        return np.ones(384, dtype=np.float32)

    embedding.encode.side_effect = encode

    agent = SummarizerAgent(
        scheduler, llm, embedding, state_bus, store, use_structured_protocol=True
    )
    agent.run_options = RunOptions(enable_intent_cache_gate=True)
    llm.chat_structured.return_value = {
        "key_findings": ["Solar LCOE lower than wind in sunny regions"],
        "facts": ["Wind offshore CF 40-55%"],
        "conclusion": "Solar and wind suit different regions.",
    }
    return agent, llm


def test_compare_task_skips_wind_summary_cache(summarizer_with_memory):
    agent, llm = summarizer_with_memory
    compare_desc = (
        "Compare solar and wind energy across LCOE, land use, and capacity factors"
    )
    result = agent.execute_task(
        {
            "task_id": "e4_renewable_comparison",
            "task_description": compare_desc,
            "tags": ["solar", "wind", "energy", "comparison"],
            "plan": {"task_description": compare_desc, "subtasks": []},
            "retrieval_results": {
                1: {"combined_results": "Solar LCOE $30/MWh. Wind onshore $40/MWh."}
            },
        }
    )
    assert llm.chat_structured.called
    summary = result["summary"]
    assert isinstance(summary, dict)
    assert "Solar" in " ".join(summary.get("key_findings", []))
