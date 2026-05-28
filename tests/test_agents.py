"""Unit tests for agent implementations."""

import pytest

from src.agents.planner import PlannerAgent
from src.agents.retriever import RetrieverAgent, KNOWLEDGE_BASE
from src.agents.executor import ExecutorAgent
from src.agents.summarizer import SummarizerAgent
from src.protocol import ActionType


class TestPlannerAgent:
    """Test Planner agent task decomposition."""

    def test_agent_initialization(self, scheduler, mock_llm, embed_engine,
                                   state_bus, memory_store):
        agent = PlannerAgent(
            scheduler, mock_llm, embed_engine, state_bus, memory_store,
        )
        assert agent.agent_id == "planner"
        assert agent.role == "planner"
        assert "plan" in agent.capabilities
        assert "decompose" in agent.capabilities

    def test_execute_task_creates_plan(self, scheduler, mock_llm, embed_engine,
                                        state_bus, memory_store):
        agent = PlannerAgent(
            scheduler, mock_llm, embed_engine, state_bus, memory_store,
        )
        result = agent.execute_task({
            "task_description": "Research solar energy technologies",
            "task_id": "test-1",
            "tags": ["research", "energy"],
        })

        assert "plan_id" in result
        assert "subtasks" in result
        assert len(result["subtasks"]) == 3
        assert result["task_id"] == "test-1"
        assert result["generated_by"] == "planner"
        assert result["elapsed_ms"] > 0

    def test_subtasks_have_required_fields(self, scheduler, mock_llm, embed_engine,
                                            state_bus, memory_store):
        agent = PlannerAgent(
            scheduler, mock_llm, embed_engine, state_bus, memory_store,
        )
        result = agent.execute_task({
            "task_description": "Test task",
            "task_id": "test-1",
        })

        for subtask in result["subtasks"]:
            assert "step" in subtask
            assert "description" in subtask
            assert "agent_role" in subtask
            assert "action" in subtask
            assert "params" in subtask
            assert "depends_on" in subtask
            assert subtask["agent_role"] in ["retriever", "executor", "summarizer"]

    def test_execute_task_stores_memory(self, scheduler, mock_llm, embed_engine,
                                         state_bus, memory_store):
        agent = PlannerAgent(
            scheduler, mock_llm, embed_engine, state_bus, memory_store,
        )
        result = agent.execute_task({
            "task_description": "Research solar energy",
            "task_id": "test-1",
        })
        assert "memory_id" in result
        assert result["memory_id"] is not None

        # Verify it's in the store
        mem = memory_store.get(result["memory_id"])
        assert mem is not None
        assert mem.memory_type == "strategy"
        assert "plan" in mem.tags

    def test_execute_task_with_llm_output(self, scheduler, mock_llm, embed_engine,
                                           state_bus, memory_store):
        agent = PlannerAgent(
            scheduler, mock_llm, embed_engine, state_bus, memory_store,
        )
        agent.execute_task({
            "task_description": "Test",
            "task_id": "test-1",
        })
        # Should have called structured LLM output
        assert mock_llm.structured_calls >= 1

    def test_generates_state_transfer(self, scheduler, mock_llm, embed_engine,
                                       state_bus, memory_store):
        agent = PlannerAgent(
            scheduler, mock_llm, embed_engine, state_bus, memory_store,
        )
        initial_transfers = state_bus.get_stats()["transfer_count"]
        agent.execute_task({
            "task_description": "Test task",
            "task_id": "test-1",
        })
        assert state_bus.get_stats()["transfer_count"] > initial_transfers

    def test_multiple_tasks(self, scheduler, mock_llm, embed_engine,
                             state_bus, memory_store):
        agent = PlannerAgent(
            scheduler, mock_llm, embed_engine, state_bus, memory_store,
        )
        result1 = agent.execute_task({
            "task_description": "Task A",
            "task_id": "task-a",
        })
        result2 = agent.execute_task({
            "task_description": "Task B",
            "task_id": "task-b",
        })
        assert result1["plan_id"] != result2["plan_id"]

    def test_fallback_plan_generation(self, scheduler, mock_llm, embed_engine,
                                       state_bus, memory_store):
        """Test fallback plan when LLM response can't be parsed."""
        agent = PlannerAgent(
            scheduler, mock_llm, embed_engine, state_bus, memory_store,
        )

        # Test research-type task
        plan = agent._generate_fallback_plan("research solar energy", "fb-1")
        assert len(plan["subtasks"]) == 3
        assert plan["subtasks"][0]["agent_role"] == "retriever"

        # Test code-type task
        plan = agent._generate_fallback_plan("fix a bug in the code", "fb-2")
        assert len(plan["subtasks"]) == 3

        # Test generic task
        plan = agent._generate_fallback_plan("do something random", "fb-3")
        assert len(plan["subtasks"]) == 3


class TestRetrieverAgent:
    """Test Retriever agent information retrieval."""

    def test_agent_initialization(self, scheduler, mock_llm, embed_engine,
                                   state_bus, memory_store):
        agent = RetrieverAgent(
            scheduler, mock_llm, embed_engine, state_bus, memory_store,
        )
        assert agent.agent_id == "retriever"
        assert "retrieve" in agent.capabilities
        assert "search" in agent.capabilities

    def test_knowledge_base_has_content(self):
        assert "renewable energy" in KNOWLEDGE_BASE
        assert "solar" in KNOWLEDGE_BASE["renewable energy"]
        assert "code analysis" in KNOWLEDGE_BASE
        assert "security_patterns" in KNOWLEDGE_BASE["code analysis"]

    def test_search_knowledge_base_solar(self, scheduler, mock_llm, embed_engine,
                                          state_bus, memory_store):
        agent = RetrieverAgent(
            scheduler, mock_llm, embed_engine, state_bus, memory_store,
        )
        hits = agent._search_knowledge_base("solar energy efficiency")
        assert len(hits) > 0
        assert any("solar" in str(h.get("tags", [])).lower()
                   or "solar" in h.get("topic", "").lower()
                   for h in hits)

    def test_search_knowledge_base_security(self, scheduler, mock_llm, embed_engine,
                                            state_bus, memory_store):
        agent = RetrieverAgent(
            scheduler, mock_llm, embed_engine, state_bus, memory_store,
        )
        hits = agent._search_knowledge_base("SQL injection vulnerability")
        assert len(hits) > 0

    def test_search_knowledge_base_database_tags(self, scheduler, mock_llm, embed_engine,
                                                  state_bus, memory_store):
        agent = RetrieverAgent(
            scheduler, mock_llm, embed_engine, state_bus, memory_store,
        )
        hits = agent._search_knowledge_base(
            "optimization", tags=["database", "sql"]
        )
        assert len(hits) > 0
        assert all(h.get("category") == "database systems" for h in hits)

    def test_search_knowledge_base_no_match(self, scheduler, mock_llm, embed_engine,
                                             state_bus, memory_store):
        agent = RetrieverAgent(
            scheduler, mock_llm, embed_engine, state_bus, memory_store,
        )
        hits = agent._search_knowledge_base("zzzz_nonexistent_topic_zzzz")
        assert len(hits) == 0

    def test_execute_task_returns_results(self, scheduler, mock_llm, embed_engine,
                                           state_bus, memory_store):
        agent = RetrieverAgent(
            scheduler, mock_llm, embed_engine, state_bus, memory_store,
        )
        result = agent.execute_task({
            "query": "solar energy photovoltaic technology",
            "task_id": "test-1",
            "tags": ["solar", "energy"],
        })

        assert "query" in result
        assert "combined_results" in result
        assert "hit_count" in result
        assert "memory_hit_count" in result
        assert result["elapsed_ms"] > 0

    def test_execute_task_with_memory_hits(self, scheduler, mock_llm, embed_engine,
                                            state_bus, memory_store):
        """When memory has relevant data, it should be found."""
        # Pre-populate memory
        from src.memory.models import MemoryUnit
        mem = MemoryUnit(
            source_agent="retriever",
            task_topic="Solar Research",
            summary="Solar efficiency data found",
            tags=["solar", "energy", "research"],
            embedding=embed_engine.encode("solar energy efficiency photovoltaic"),
        )
        memory_store.store(mem)

        agent = RetrieverAgent(
            scheduler, mock_llm, embed_engine, state_bus, memory_store,
        )
        result = agent.execute_task({
            "query": "solar energy efficiency",
            "task_id": "test-1",
            "tags": ["solar"],
        })

        # Should find the pre-populated memory
        assert result["memory_hit_count"] >= 1

    def test_execute_task_stores_memory(self, scheduler, mock_llm, embed_engine,
                                         state_bus, memory_store):
        agent = RetrieverAgent(
            scheduler, mock_llm, embed_engine, state_bus, memory_store,
        )
        result = agent.execute_task({
            "query": "solar energy",
            "task_id": "test-1",
        })
        if result.get("hit_count", 0) > 0:
            assert len(result["memory_refs"]) > 0


class TestExecutorAgent:
    """Test Executor agent processing and CodeAct execution."""

    def test_agent_initialization(self, scheduler, mock_llm, embed_engine,
                                   state_bus, memory_store):
        agent = ExecutorAgent(
            scheduler, mock_llm, embed_engine, state_bus, memory_store,
        )
        assert agent.agent_id == "executor"
        assert "execute" in agent.capabilities
        assert "process" in agent.capabilities

    def test_execute_task_processes_input(self, scheduler, mock_llm, embed_engine,
                                           state_bus, memory_store):
        agent = ExecutorAgent(
            scheduler, mock_llm, embed_engine, state_bus, memory_store,
            sandbox_enabled=False,
        )
        result = agent.execute_task({
            "input": "Analyze solar energy data and compute efficiency metrics",
            "task_id": "test-1",
        })

        assert "output" in result
        assert result["elapsed_ms"] > 0

    def test_execute_task_with_code(self, scheduler, mock_llm, embed_engine,
                                     state_bus, memory_store):
        agent = ExecutorAgent(
            scheduler, mock_llm, embed_engine, state_bus, memory_store,
            sandbox_enabled=True,
        )
        result = agent.execute_task({
            "input": "Calculate solar efficiency",
            "code": "print('Efficiency: 20%')",
            "task_id": "test-1",
        })

        assert result["code_executed"] is True

    def test_execute_task_with_retrieval_data(self, scheduler, mock_llm, embed_engine,
                                               state_bus, memory_store):
        agent = ExecutorAgent(
            scheduler, mock_llm, embed_engine, state_bus, memory_store,
            sandbox_enabled=False,
        )
        result = agent.execute_task({
            "input": "Process data",
            "retrieval_results": {
                "knowledge_base_hits": [
                    {"summary": "Solar efficiency 18-22%"},
                ],
            },
            "task_id": "test-1",
        })
        assert "input_from_retrieval" in result

    def test_execute_task_stores_memory(self, scheduler, mock_llm, embed_engine,
                                         state_bus, memory_store):
        agent = ExecutorAgent(
            scheduler, mock_llm, embed_engine, state_bus, memory_store,
            sandbox_enabled=False,
        )
        result = agent.execute_task({
            "input": "Test processing",
            "task_id": "test-1",
        })
        assert len(result["memory_refs"]) > 0

        mem = memory_store.get(result["memory_refs"][0])
        assert mem is not None
        assert mem.memory_type == "result"
        assert "execution" in mem.tags

    def test_code_act_generation(self, scheduler, mock_llm, embed_engine,
                                  state_bus, memory_store):
        """Test that CodeAct mode calls LLM for code generation."""
        agent = ExecutorAgent(
            scheduler, mock_llm, embed_engine, state_bus, memory_store,
            sandbox_enabled=True,
        )
        agent.execute_task({
            "input": "Calculate the sum of solar efficiency values",
            "task_id": "test-1",
        })
        # Code generation should trigger an LLM call
        assert mock_llm.chat_calls >= 1


class TestSummarizerAgent:
    """Test Summarizer agent synthesis and reporting."""

    def test_agent_initialization(self, scheduler, mock_llm, embed_engine,
                                   state_bus, memory_store):
        agent = SummarizerAgent(
            scheduler, mock_llm, embed_engine, state_bus, memory_store,
        )
        assert agent.agent_id == "summarizer"
        assert "summarize" in agent.capabilities
        assert "synthesize" in agent.capabilities

    def test_execute_task_synthesizes(self, scheduler, mock_llm, embed_engine,
                                       state_bus, memory_store):
        agent = SummarizerAgent(
            scheduler, mock_llm, embed_engine, state_bus, memory_store,
        )
        result = agent.execute_task({
            "task_id": "test-1",
            "plan": {
                "task_description": "Research solar energy technologies",
                "expected_outcome": "Comprehensive report on solar energy",
            },
            "retrieval_results": {
                1: {
                    "combined_results": "Solar efficiency: 15-22%, Costs down 90%",
                },
            },
            "execution_results": {
                2: {
                    "output": "Processed solar energy data successfully",
                },
            },
        })

        assert "summary" in result
        assert len(result["summary"]) > 0
        assert result["elapsed_ms"] > 0

    def test_execute_task_with_all_steps(self, scheduler, mock_llm, embed_engine,
                                          state_bus, memory_store):
        agent = SummarizerAgent(
            scheduler, mock_llm, embed_engine, state_bus, memory_store,
        )
        result = agent.execute_task({
            "task_id": "test-1",
            "plan": {
                "task_description": "Test",
                "expected_outcome": "Done",
                "memory_refs": ["ref-1"],
            },
            "retrieval_results": {
                1: {"combined_results": "Data found", "memory_refs": ["ref-2"]},
            },
            "execution_results": {
                2: {"output": "Processed", "memory_refs": ["ref-3"]},
            },
        })
        assert len(result["evidence_refs"]) > 0

    def test_execute_task_stores_memory(self, scheduler, mock_llm, embed_engine,
                                         state_bus, memory_store):
        agent = SummarizerAgent(
            scheduler, mock_llm, embed_engine, state_bus, memory_store,
        )
        result = agent.execute_task({
            "task_id": "test-1",
            "plan": {"task_description": "Test research task"},
            "retrieval_results": {1: {"combined_results": "Some data"}},
            "execution_results": {2: {"output": "Processed"}},
        })

        memory_ids = result.get("memory_refs", [])
        assert len(memory_ids) > 0
        mem = memory_store.get(memory_ids[0])
        assert mem is not None
        assert mem.memory_type == "result"
        assert "summary" in mem.tags

    def test_execute_task_empty_input(self, scheduler, mock_llm, embed_engine,
                                       state_bus, memory_store):
        agent = SummarizerAgent(
            scheduler, mock_llm, embed_engine, state_bus, memory_store,
        )
        result = agent.execute_task({"task_id": "test-1"})
        assert "summary" in result
