"""Agent message scheduler and router.

Manages agent registry, capability discovery, message routing, and
execution orchestration for the multi-agent system.
"""

from __future__ import annotations

import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from ..protocol import (
    Message,
    MessageType,
    ActionType,
    ProtocolParser,
)


@dataclass
class AgentInfo:
    agent_id: str
    role: str
    capabilities: List[str]
    status: str = "idle"
    registered_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    messages_sent: int = 0
    messages_received: int = 0


class AgentRegistry:
    """Registry for agent capability discovery and routing."""

    def __init__(self):
        self._agents: Dict[str, AgentInfo] = {}
        self._capability_index: Dict[str, Set[str]] = defaultdict(set)
        self._handlers: Dict[str, Callable] = {}

    def register(self, agent_id: str, role: str, capabilities: List[str]) -> None:
        info = AgentInfo(
            agent_id=agent_id,
            role=role,
            capabilities=capabilities,
        )
        self._agents[agent_id] = info
        for cap in capabilities:
            self._capability_index[cap].add(agent_id)

    def find_by_capability(self, capability: str) -> List[str]:
        return list(self._capability_index.get(capability, set()))

    def find_by_role(self, role: str) -> List[str]:
        return [aid for aid, info in self._agents.items() if info.role == role]

    def get_info(self, agent_id: str) -> Optional[AgentInfo]:
        return self._agents.get(agent_id)

    def list_agents(self) -> Dict[str, AgentInfo]:
        return dict(self._agents)

    def update_status(self, agent_id: str, status: str) -> None:
        if agent_id in self._agents:
            self._agents[agent_id].status = status
            self._agents[agent_id].last_active = time.time()

    def record_message(self, agent_id: str, direction: str) -> None:
        if agent_id in self._agents:
            if direction == "sent":
                self._agents[agent_id].messages_sent += 1
            else:
                self._agents[agent_id].messages_received += 1


class MessageBus:
    """Asynchronous message bus for inter-agent communication."""

    def __init__(self):
        self._queues: Dict[str, List[Message]] = defaultdict(list)
        self._history: List[Message] = []

    def send(self, message: Message) -> None:
        self._queues[message.to_agent].append(message)
        self._history.append(message)

    def receive(self, agent_id: str) -> List[Message]:
        messages = self._queues.get(agent_id, [])
        self._queues[agent_id] = []
        return messages

    def broadcast(self, message: Message, exclude: Optional[List[str]] = None) -> None:
        exclude = exclude or []
        for agent_id in list(self._queues.keys()):
            if agent_id not in exclude and agent_id != message.from_agent:
                self.send(Message(
                    from_agent=message.from_agent,
                    to_agent=agent_id,
                    msg_type=message.msg_type,
                    action=message.action,
                    params=message.params,
                    result=message.result,
                    state_embedding=message.state_embedding,
                    memory_refs=message.memory_refs,
                    context=message.context,
                ))

    @property
    def message_count(self) -> int:
        return len(self._history)

    @property
    def structured_token_count(self) -> int:
        return sum(m.estimated_token_count() for m in self._history)

    @property
    def text_token_count(self) -> int:
        return sum(m.text_token_count() for m in self._history)


class Scheduler:
    """Task scheduler and orchestrator for multi-agent execution."""

    def __init__(self, registry: AgentRegistry, bus: MessageBus):
        self.registry = registry
        self.bus = bus
        self._task_queue: List[Dict[str, Any]] = []
        self._completed_tasks: List[Dict[str, Any]] = []
        self._task_history: List[Dict[str, Any]] = []
        self._metrics: Dict[str, Any] = defaultdict(list)

    def handshake(self, agent_id: str, role: str, capabilities: List[str]) -> Message:
        """Perform handshake and register agent capabilities."""
        self.registry.register(agent_id, role, capabilities)
        msg = ProtocolParser.create_handshake(agent_id, role, capabilities)
        self.bus.broadcast(msg, exclude=[agent_id])
        return msg

    def discover_capable_agents(self, capability: str) -> List[str]:
        return self.registry.find_by_capability(capability)

    def route_task(
        self,
        action: ActionType,
        params: Dict[str, Any],
        from_agent: str,
        to_agent: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        memory_refs: Optional[List[str]] = None,
    ) -> Message:
        """Route a task to a capable agent."""
        if to_agent is None:
            candidates = self.registry.find_by_capability(action.value)
            if not candidates:
                raise ValueError(f"No agent found with capability: {action.value}")
            to_agent = candidates[0]

        msg = ProtocolParser.create_request(
            from_agent=from_agent,
            to_agent=to_agent,
            action=action,
            params=params,
            embedding=embedding,
            memory_refs=memory_refs,
        )
        self.bus.send(msg)
        self.registry.record_message(from_agent, "sent")
        return msg

    def send_response(
        self,
        request: Message,
        result: Any,
        embedding: Optional[List[float]] = None,
        memory_refs: Optional[List[str]] = None,
    ) -> Message:
        msg = ProtocolParser.create_response(request, result, embedding, memory_refs)
        self.bus.send(msg)
        self.registry.record_message(msg.from_agent, "sent")
        return msg

    def record_metric(self, name: str, value: Any) -> None:
        self._metrics[name].append(value)

    def get_metrics(self) -> Dict[str, Any]:
        return dict(self._metrics)

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "total_messages": self.bus.message_count,
            "structured_tokens": self.bus.structured_token_count,
            "text_equivalent_tokens": self.bus.text_token_count,
            "token_savings_pct": (
                (1 - self.bus.structured_token_count / max(self.bus.text_token_count, 1)) * 100
            ),
            "agents": len(self.registry.list_agents()),
            "tasks_completed": len(self._completed_tasks),
        }
