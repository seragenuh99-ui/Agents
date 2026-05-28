"""Unit tests for structured communication protocol."""

import json
import msgpack
import pytest
import time

from src.protocol import (
    Message, MessageType, ActionType, ProtocolParser, HandshakeMessage,
)


# ============================================================
# Message creation and serialization
# ============================================================

class TestMessageCreation:
    """Test Message dataclass creation and defaults."""

    def test_default_message(self):
        msg = Message()
        assert msg.msg_id
        assert len(msg.msg_id) == 36  # UUID format
        assert msg.timestamp > 0
        assert msg.from_agent == ""
        assert msg.to_agent == ""
        assert msg.msg_type == MessageType.REQUEST
        assert msg.action == ActionType.NOOP
        assert msg.params == {}
        assert msg.result is None
        assert msg.state_embedding is None
        assert msg.memory_refs == []
        assert msg.error is None

    def test_full_message(self, sample_message):
        assert sample_message.from_agent == "planner"
        assert sample_message.to_agent == "retriever"
        assert sample_message.action == ActionType.RETRIEVE
        assert sample_message.params["query"] == "solar energy technologies"
        assert len(sample_message.state_embedding) == 5
        assert "mem-001" in sample_message.memory_refs

    def test_message_with_error(self):
        msg = Message(
            from_agent="retriever",
            to_agent="planner",
            msg_type=MessageType.ERROR,
            error="Query failed: timeout",
        )
        assert msg.msg_type == MessageType.ERROR
        assert "timeout" in msg.error


class TestMessageSerialization:
    """Test MessagePack serialization round-trip."""

    def test_round_trip_basic(self):
        msg = Message(
            from_agent="agent1",
            to_agent="agent2",
            msg_type=MessageType.REQUEST,
            action=ActionType.RETRIEVE,
            params={"key": "value", "nested": {"a": 1}},
        )
        serialized = msg.serialize()
        assert isinstance(serialized, bytes)
        assert len(serialized) > 0

        restored = Message.deserialize(serialized)
        assert restored.from_agent == "agent1"
        assert restored.to_agent == "agent2"
        assert restored.action == ActionType.RETRIEVE
        assert restored.params["key"] == "value"
        assert restored.params["nested"]["a"] == 1

    def test_round_trip_with_embedding(self):
        msg = Message(
            from_agent="planner",
            to_agent="executor",
            state_embedding=[0.1, -0.2, 0.3] * 128,  # 384 floats
            memory_refs=["mem-001", "mem-002"],
        )
        serialized = msg.serialize()
        restored = Message.deserialize(serialized)
        assert len(restored.state_embedding) == 384
        assert pytest.approx(restored.state_embedding[0]) == 0.1
        assert restored.memory_refs == ["mem-001", "mem-002"]

    def test_round_trip_with_result(self):
        msg = Message(
            from_agent="retriever",
            to_agent="planner",
            result={"hits": 5, "items": [{"id": 1, "title": "Solar"}]},
        )
        serialized = msg.serialize()
        restored = Message.deserialize(serialized)
        assert restored.result["hits"] == 5
        assert restored.result["items"][0]["title"] == "Solar"

    def test_round_trip_empty(self):
        msg = Message()
        serialized = msg.serialize()
        restored = Message.deserialize(serialized)
        assert restored.msg_id == msg.msg_id
        assert restored.params == {}

    def test_msgpack_vs_json_size(self):
        """Verify MessagePack is more compact than JSON."""
        msg = Message(
            from_agent="planner",
            to_agent="retriever",
            action=ActionType.RETRIEVE,
            params={"query": "test", "limit": 10, "tags": ["a", "b", "c"]},
            result={"hits": 3, "data": [{"id": i, "val": f"item_{i}"} for i in range(5)]},
        )
        msgpack_size = len(msg.serialize())
        json_size = len(json.dumps(msg.to_dict()).encode("utf-8"))
        assert msgpack_size < json_size, f"msgpack={msgpack_size}, json={json_size}"


class TestTokenEstimation:
    """Test token count estimation for structured vs text modes."""

    def test_estimated_token_count(self, sample_message):
        tokens = sample_message.estimated_token_count()
        assert tokens > 0
        assert isinstance(tokens, int)

    def test_text_token_count(self, sample_message):
        tokens = sample_message.text_token_count()
        assert tokens > 0
        assert isinstance(tokens, int)

    def test_text_equivalent_contains_key_info(self, sample_message):
        text = sample_message.to_text_equivalent()
        assert "planner" in text
        assert "retriever" in text
        assert "retrieve" in text.lower()

    def test_token_comparison_structured_vs_text(self):
        """Structured message should have fewer or equal tokens."""
        msg = Message(
            from_agent="planner",
            to_agent="retriever",
            action=ActionType.RETRIEVE,
            params={"query": "solar energy", "limit": 10},
            result="Found 3 results: monocrystalline 18-22%, polycrystalline 15-18%",
        )
        structured_tokens = msg.estimated_token_count()
        text_tokens = msg.text_token_count()
        # Both should be positive
        assert structured_tokens > 0
        assert text_tokens > 0

    def test_text_mode_verbose_output(self):
        """Text mode message should generate readable natural language."""
        msg = Message(
            from_agent="executor",
            to_agent="summarizer",
            action=ActionType.EXECUTE,
            params={"input": "Process data"},
            result={"output": "Calculation complete", "error": None},
            state_embedding=[0.1] * 10,
        )
        text = msg.to_text_equivalent()
        assert "executor" in text
        assert "summarizer" in text
        assert "State embedding" in text


class TestMessageTypes:
    """Test all message type and action type enums."""

    def test_all_message_types(self):
        types = list(MessageType)
        assert MessageType.REQUEST in types
        assert MessageType.RESPONSE in types
        assert MessageType.HANDSHAKE in types
        assert MessageType.CAPABILITY_QUERY in types
        assert MessageType.STATE_TRANSFER in types
        assert MessageType.MEMORY_STORE in types
        assert MessageType.MEMORY_QUERY in types
        assert MessageType.ERROR in types

    def test_all_action_types(self):
        actions = list(ActionType)
        assert ActionType.PLAN in actions
        assert ActionType.RETRIEVE in actions
        assert ActionType.EXECUTE in actions
        assert ActionType.SUMMARIZE in actions
        assert ActionType.STORE_MEMORY in actions
        assert ActionType.QUERY_MEMORY in actions


class TestProtocolParser:
    """Test ProtocolParser message creation and validation."""

    def test_validate_valid_message(self):
        msg = Message(from_agent="test", to_agent="other")
        assert ProtocolParser.validate(msg) is True

    def test_validate_invalid_message(self):
        msg = Message(from_agent="", to_agent="")
        assert ProtocolParser.validate(msg) is False

    def test_create_handshake(self):
        msg = ProtocolParser.create_handshake(
            "agent1", "planner", ["plan", "decompose"]
        )
        assert msg.msg_type == MessageType.HANDSHAKE
        assert msg.from_agent == "agent1"
        assert "plan" in msg.capabilities
        assert "decompose" in msg.capabilities
        assert msg.context["role"] == "planner"

    def test_create_request(self):
        msg = ProtocolParser.create_request(
            from_agent="planner",
            to_agent="retriever",
            action=ActionType.RETRIEVE,
            params={"query": "test"},
            embedding=[0.1, 0.2],
            memory_refs=["mem-1"],
        )
        assert msg.msg_type == MessageType.REQUEST
        assert msg.from_agent == "planner"
        assert msg.to_agent == "retriever"
        assert msg.action == ActionType.RETRIEVE
        assert msg.state_embedding == [0.1, 0.2]
        assert msg.memory_refs == ["mem-1"]

    def test_create_response(self):
        request = ProtocolParser.create_request(
            "planner", "retriever", ActionType.RETRIEVE, {"q": "test"}
        )
        response = ProtocolParser.create_response(
            request, result={"data": "result"}, embedding=[0.5, 0.6]
        )
        assert response.msg_type == MessageType.RESPONSE
        assert response.from_agent == "retriever"
        assert response.to_agent == "planner"
        assert response.result == {"data": "result"}
        assert response.params["in_response_to"] == request.msg_id

    def test_create_error(self):
        request = ProtocolParser.create_request(
            "planner", "retriever", ActionType.RETRIEVE, {}
        )
        error_msg = ProtocolParser.create_error(request, "Not found")
        assert error_msg.msg_type == MessageType.ERROR
        assert error_msg.error == "Not found"

    def test_create_request_without_embedding(self):
        msg = ProtocolParser.create_request(
            "a", "b", ActionType.EXECUTE, {"input": "test"}
        )
        assert msg.state_embedding is None
        assert msg.memory_refs == []


class TestHandshakeMessage:
    """Test HandshakeMessage helper."""

    def test_handshake_conversion(self):
        hs = HandshakeMessage(
            agent_id="agent-1",
            agent_role="planner",
            capabilities=["plan", "route"],
        )
        msg = hs.to_message()
        assert msg.msg_type == MessageType.HANDSHAKE
        assert msg.from_agent == "agent-1"
        assert msg.to_agent == "broadcast"
        assert "plan" in msg.capabilities


# ============================================================
# Scheduler, Registry, MessageBus tests
# ============================================================

class TestAgentRegistry:
    """Test AgentRegistry for capability discovery and routing."""

    def test_register_agent(self, registry):
        registry.register("agent-1", "planner", ["plan", "decompose"])
        info = registry.get_info("agent-1")
        assert info is not None
        assert info.role == "planner"
        assert info.status == "idle"

    def test_find_by_capability(self, registry):
        registry.register("agent-1", "planner", ["plan"])
        registry.register("agent-2", "retriever", ["retrieve", "search"])
        registry.register("agent-3", "executor", ["execute", "process"])

        planners = registry.find_by_capability("plan")
        assert "agent-1" in planners
        assert len(planners) == 1

        searchers = registry.find_by_capability("search")
        assert "agent-2" in searchers

    def test_find_by_role(self, registry):
        registry.register("agent-1", "planner", ["plan"])
        registry.register("agent-2", "retriever", ["retrieve"])

        assert len(registry.find_by_role("planner")) == 1
        assert len(registry.find_by_role("retriever")) == 1
        assert len(registry.find_by_role("nonexistent")) == 0

    def test_list_agents(self, registry):
        registry.register("a1", "role1", ["cap1"])
        registry.register("a2", "role2", ["cap2"])
        agents = registry.list_agents()
        assert len(agents) == 2

    def test_update_status(self, registry):
        registry.register("agent-1", "planner", ["plan"])
        registry.update_status("agent-1", "busy")
        assert registry.get_info("agent-1").status == "busy"

    def test_record_message(self, registry):
        registry.register("agent-1", "planner", ["plan"])
        registry.record_message("agent-1", "sent")
        registry.record_message("agent-1", "sent")
        registry.record_message("agent-1", "received")
        info = registry.get_info("agent-1")
        assert info.messages_sent == 2
        assert info.messages_received == 1


class TestMessageBus:
    """Test MessageBus for message routing and history."""

    def test_send_and_receive(self, message_bus):
        msg = Message(from_agent="a", to_agent="b")
        message_bus.send(msg)
        received = message_bus.receive("b")
        assert len(received) == 1
        assert received[0].from_agent == "a"

    def test_receive_clears_queue(self, message_bus):
        msg = Message(from_agent="a", to_agent="b")
        message_bus.send(msg)
        message_bus.receive("b")
        assert message_bus.receive("b") == []

    def test_message_count(self, message_bus):
        for i in range(5):
            message_bus.send(Message(from_agent=f"a{i}", to_agent=f"b{i}"))
        assert message_bus.message_count == 5

    def test_broadcast(self, message_bus):
        # Pre-populate queues for known agents
        message_bus.send(Message(from_agent="init", to_agent="b"))
        message_bus.send(Message(from_agent="init", to_agent="c"))
        message_bus.receive("b")
        message_bus.receive("c")

        msg = Message(from_agent="a", to_agent="broadcast")
        message_bus.broadcast(msg)
        assert len(message_bus.receive("b")) == 1
        assert len(message_bus.receive("c")) == 1
        # Sender should not receive own broadcast
        assert len(message_bus.receive("a")) == 0

    def test_token_counting(self, message_bus):
        for i in range(3):
            msg = Message(
                from_agent="planner",
                to_agent="retriever",
                action=ActionType.RETRIEVE,
                params={"query": f"test_{i}", "limit": 10},
                result=f"Result data for query {i}: Found relevant information.",
            )
            message_bus.send(msg)

        assert message_bus.structured_token_count > 0
        assert message_bus.text_token_count > 0
        # Message count should be accurate
        assert message_bus.message_count == 3


class TestScheduler:
    """Test Scheduler for task routing and orchestration."""

    def test_handshake(self, scheduler, registry):
        msg = scheduler.handshake("agent-1", "planner", ["plan", "decompose"])
        assert msg.msg_type == MessageType.HANDSHAKE
        info = registry.get_info("agent-1")
        assert info.role == "planner"

    def test_discover_capable_agents(self, scheduler):
        scheduler.handshake("agent-1", "planner", ["plan"])
        scheduler.handshake("agent-2", "retriever", ["retrieve", "search"])

        planners = scheduler.discover_capable_agents("plan")
        assert "agent-1" in planners
        assert len(planners) == 1

    def test_route_task(self, scheduler):
        scheduler.handshake("retriever-agent", "retriever", ["retrieve"])

        msg = scheduler.route_task(
            action=ActionType.RETRIEVE,
            params={"query": "test"},
            from_agent="planner-agent",
            to_agent="retriever-agent",
        )
        assert msg is not None
        assert msg.action == ActionType.RETRIEVE
        assert msg.from_agent == "planner-agent"

    def test_route_task_with_embedding(self, scheduler):
        scheduler.handshake("exec", "executor", ["execute"])

        msg = scheduler.route_task(
            action=ActionType.EXECUTE,
            params={"input": "test"},
            from_agent="planner",
            to_agent="exec",
            embedding=[0.1, 0.2, 0.3],
            memory_refs=["mem-1", "mem-2"],
        )
        assert msg.state_embedding == [0.1, 0.2, 0.3]
        assert msg.memory_refs == ["mem-1", "mem-2"]

    def test_send_response(self, scheduler):
        scheduler.handshake("retriever", "retriever", ["retrieve"])
        request = scheduler.route_task(
            action=ActionType.RETRIEVE,
            params={"query": "test"},
            from_agent="planner",
            to_agent="retriever",
        )
        response = scheduler.send_response(request, result={"hits": 3})
        assert response.msg_type == MessageType.RESPONSE
        assert response.from_agent == "retriever"
        assert response.to_agent == "planner"
        assert response.result == {"hits": 3}

    def test_get_statistics(self, scheduler):
        scheduler.handshake("agent-1", "planner", ["plan"])
        scheduler.handshake("agent-2", "retriever", ["retrieve"])
        stats = scheduler.get_statistics()
        assert stats["agents"] == 2
        assert "total_messages" in stats
        assert "structured_tokens" in stats
        assert "text_equivalent_tokens" in stats
        assert "token_savings_pct" in stats

    def test_record_metric(self, scheduler):
        scheduler.record_metric("custom_metric", 42)
        scheduler.record_metric("custom_metric", 43)
        metrics = scheduler.get_metrics()
        assert metrics["custom_metric"] == [42, 43]
