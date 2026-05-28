"""Shared test fixtures and helpers."""

import os
import sys
import tempfile
import pytest
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.protocol import Message, MessageType, ActionType, ProtocolParser
from src.protocol.scheduler import Scheduler, AgentRegistry, MessageBus
from src.state.embeddings import EmbeddingEngine
from src.state.exchange import StateExchangeBus, StatePacket
from src.memory.models import MemoryUnit
from src.memory.store import MemoryStore
from src.evaluation.metrics import MetricsCollector, ComparisonReport, TaskMetrics
from src.evaluation.reporter import Reporter
from src.agents.base import LLMBackend, BaseAgent
from src.agents.planner import PlannerAgent
from src.agents.retriever import RetrieverAgent, KNOWLEDGE_BASE
from src.agents.executor import ExecutorAgent
from src.agents.summarizer import SummarizerAgent
from src.sandbox.executor import SandboxExecutor, validate_code, CodeValidator
from src.orchestrator import Orchestrator


class CountingMockLLM:
    """Mock LLM that counts calls and returns deterministic responses."""

    def __init__(self):
        self.chat_calls = 0
        self.structured_calls = 0
        self.chat_history: List[List[Dict]] = []
        self.structured_history: List[List[Dict]] = []

    def chat(self, messages, **kwargs):
        self.chat_calls += 1
        self.chat_history.append(messages)
        return f"[Mock Chat #{self.chat_calls}] Input length: {len(messages[-1]['content'])} chars"

    def chat_structured(self, messages, output_format, **kwargs):
        self.structured_calls += 1
        self.structured_history.append(messages)
        return {
            "plan_id": f"plan_mock_{self.structured_calls}",
            "task_description": messages[-1]["content"][:200],
            "subtasks": [
                {
                    "step": 1,
                    "description": "Search for information",
                    "agent_role": "retriever",
                    "action": "retrieve",
                    "params": {"query": messages[-1]["content"][:100]},
                    "depends_on": [],
                },
                {
                    "step": 2,
                    "description": "Process data",
                    "agent_role": "executor",
                    "action": "execute",
                    "params": {"input": "Process and analyze"},
                    "depends_on": [1],
                },
                {
                    "step": 3,
                    "description": "Generate summary",
                    "agent_role": "summarizer",
                    "action": "summarize",
                    "params": {"input": "Summarize findings"},
                    "depends_on": [2],
                },
            ],
            "expected_outcome": f"Completed: {messages[-1]['content'][:80]}",
        }

    def get_usage_stats(self):
        return {
            "call_count": self.chat_calls + self.structured_calls,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "total_cached_tokens": 0,
            "cache_hit_rate": 0,
            "last_usage": {},
        }


@pytest.fixture
def mock_llm():
    return CountingMockLLM()


@pytest.fixture
def embed_engine():
    """Embedding engine in hash mode (offline-safe)."""
    return EmbeddingEngine(use_real_model=False)


@pytest.fixture
def registry():
    return AgentRegistry()


@pytest.fixture
def message_bus():
    return MessageBus()


@pytest.fixture
def scheduler(registry, message_bus):
    return Scheduler(registry, message_bus)


@pytest.fixture
def state_bus(embed_engine):
    return StateExchangeBus(embed_engine)


@pytest.fixture
def memory_store():
    """In-memory store using temp file."""
    db_path = tempfile.mktemp(suffix=".db")
    store = MemoryStore(db_path=db_path)
    yield store
    # Cleanup
    try:
        os.unlink(db_path)
    except OSError:
        pass


@pytest.fixture
def metrics():
    return MetricsCollector()


@pytest.fixture
def sandbox():
    return SandboxExecutor(timeout_sec=5.0)


@pytest.fixture
def sample_message():
    return Message(
        from_agent="planner",
        to_agent="retriever",
        msg_type=MessageType.REQUEST,
        action=ActionType.RETRIEVE,
        params={"query": "solar energy technologies", "limit": 10},
        state_embedding=[0.1, -0.2, 0.3, 0.0, -0.1],
        memory_refs=["mem-001", "mem-002"],
        context={"task_id": "test-task-1"},
    )


@pytest.fixture
def sample_state():
    return {
        "plan": {"subtasks": [{"step": 1, "agent": "retriever"}]},
        "retrieval_results": {"hits": 5, "topic": "solar energy"},
        "task_id": "test-task-1",
    }


@pytest.fixture
def sample_memory(embed_engine):
    return MemoryUnit(
        source_agent="retriever",
        task_topic="Solar Energy Research",
        task_id="test-task-1",
        summary="Research findings on photovoltaic efficiency and cost trends",
        content="Detailed analysis: monocrystalline panels achieve 18-22% efficiency. "
                "Costs have decreased 90% since 2010. Key manufacturers in China dominate.",
        tags=["solar", "energy", "photovoltaic", "research"],
        memory_type="result",
        embedding=embed_engine.encode("solar energy photovoltaic research findings"),
    )


@pytest.fixture
def orchestrator(mock_llm, memory_store):
    """Full orchestrator with mock LLM."""
    import tempfile
    db_path = tempfile.mktemp(suffix=".db")
    orch = Orchestrator(
        llm=mock_llm,
        mode="structured",
        use_real_embeddings=False,
        sandbox_enabled=True,
        memory_db_path=db_path,
    )
    orch._test_db_path = db_path
    yield orch
    try:
        os.unlink(db_path)
    except OSError:
        pass


@pytest.fixture
def populated_memory(memory_store, embed_engine):
    """Memory store pre-populated with test data."""
    memories = [
        MemoryUnit(
            source_agent="retriever",
            task_topic="Solar Energy Research",
            summary="Solar PV efficiency ranges from 15-22%",
            content="Monocrystalline: 18-22%, Polycrystalline: 15-18%, Thin-film: 10-12%",
            tags=["solar", "efficiency", "photovoltaic"],
            memory_type="evidence",
            embedding=embed_engine.encode("solar photovoltaic efficiency data"),
        ),
        MemoryUnit(
            source_agent="executor",
            task_topic="Cost Analysis",
            summary="Solar costs dropped 90% since 2010",
            content="Cost per watt decreased from $2.00 to $0.20. China leads manufacturing.",
            tags=["solar", "cost", "economics"],
            memory_type="result",
            embedding=embed_engine.encode("solar energy cost reduction economics"),
        ),
        MemoryUnit(
            source_agent="retriever",
            task_topic="Wind Energy Research",
            summary="Wind turbine capacity factor 25-55%",
            content="Onshore: 25-35%, Offshore: 40-55%. Denmark leads with 50%+ wind electricity.",
            tags=["wind", "energy", "capacity"],
            memory_type="evidence",
            embedding=embed_engine.encode("wind energy turbine capacity factor statistics"),
        ),
        MemoryUnit(
            source_agent="summarizer",
            task_topic="Security Analysis",
            summary="SQL injection is a critical vulnerability pattern",
            content="String formatting in SQL queries is the most common injection vector.",
            tags=["security", "sql", "vulnerability"],
            memory_type="strategy",
            embedding=embed_engine.encode("sql injection security vulnerability detection"),
        ),
    ]
    for mem in memories:
        memory_store.store(mem)
    return memory_store


# ============================================================
# Performance test helpers
# ============================================================

def time_execution(func, *args, **kwargs):
    """Measure execution time of a function."""
    import time
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    return result, elapsed


def generate_random_text(length: int) -> str:
    """Generate random text of given length for benchmarking."""
    import random
    import string
    words = []
    for _ in range(length // 6):
        word_len = random.randint(3, 10)
        words.append(''.join(random.choices(string.ascii_lowercase, k=word_len)))
    return ' '.join(words)
