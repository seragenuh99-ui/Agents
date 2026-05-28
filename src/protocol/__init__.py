"""Structured communication protocol for multi-agent collaboration.

This module implements a compact, structured message format that replaces
verbose natural language communication between agents, reducing token overhead
and enabling efficient state transfer.
"""

from __future__ import annotations

import msgpack
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class MessageType(str, Enum):
    REQUEST = "request"
    RESPONSE = "response"
    HANDSHAKE = "handshake"
    CAPABILITY_QUERY = "capability_query"
    CAPABILITY_ADVERTISE = "capability_advertise"
    STATE_TRANSFER = "state_transfer"
    MEMORY_STORE = "memory_store"
    MEMORY_QUERY = "memory_query"
    ERROR = "error"


class ActionType(str, Enum):
    PLAN = "plan"
    RETRIEVE = "retrieve"
    EXECUTE = "execute"
    SUMMARIZE = "summarize"
    STORE_MEMORY = "store_memory"
    QUERY_MEMORY = "query_memory"
    REPORT = "report"
    NOOP = "noop"


@dataclass
class Message:
    """Structured message for inter-agent communication."""
    msg_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    from_agent: str = ""
    to_agent: str = ""
    msg_type: MessageType = MessageType.REQUEST
    action: ActionType = ActionType.NOOP
    params: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Any] = None
    state_embedding: Optional[List[float]] = None
    memory_refs: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "h": {  # header
                "id": self.msg_id,
                "ts": self.timestamp,
                "fr": self.from_agent,
                "to": self.to_agent,
                "mt": self.msg_type.value,
            },
            "a": {  # action
                "tp": self.action.value,
                "pm": self.params,
                "cp": self.capabilities,
            },
            "s": {  # state
                "em": self.state_embedding,
                "mr": self.memory_refs,
            },
            "p": self.result,  # payload
            "c": self.context,  # context
            "e": self.error,  # error
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        h = data.get("h", {})
        a = data.get("a", {})
        s = data.get("s", {})
        return cls(
            msg_id=h.get("id", str(uuid.uuid4())),
            timestamp=h.get("ts", time.time()),
            from_agent=h.get("fr", ""),
            to_agent=h.get("to", ""),
            msg_type=MessageType(h.get("mt", "request")),
            action=ActionType(a.get("tp", "noop")),
            params=a.get("pm", {}),
            capabilities=a.get("cp", []),
            state_embedding=s.get("em"),
            memory_refs=s.get("mr", []),
            result=data.get("p"),
            context=data.get("c", {}),
            error=data.get("e"),
        )

    def serialize(self) -> bytes:
        return msgpack.packb(self.to_dict(), use_bin_type=True)

    @classmethod
    def deserialize(cls, data: bytes) -> "Message":
        return cls.from_dict(msgpack.unpackb(data, raw=False))

    def estimated_token_count(self) -> int:
        """Estimate token count for this structured message (1 token ≈ 3 chars)."""
        text = str(self.to_dict())
        return len(text) // 3

    def to_text_equivalent(self) -> str:
        """Generate equivalent natural language representation for comparison."""
        parts = [
            f"Message from {self.from_agent} to {self.to_agent}",
            f"Type: {self.msg_type.value}",
            f"Action: {self.action.value}",
        ]
        if self.params:
            parts.append(f"Parameters: {self.params}")
        if self.result:
            parts.append(f"Result: {self.result}")
        if self.state_embedding:
            parts.append(f"State embedding: [{len(self.state_embedding)} dimensions]")
        if self.memory_refs:
            parts.append(f"Memory references: {self.memory_refs}")
        if self.error:
            parts.append(f"Error: {self.error}")
        return "\n".join(parts)

    def text_token_count(self) -> int:
        """Token count of the natural language equivalent."""
        return len(self.to_text_equivalent()) // 3


@dataclass
class HandshakeMessage:
    """Initial handshake between agents for capability discovery."""
    agent_id: str
    agent_role: str
    capabilities: List[str]
    protocol_version: str = "1.0"

    def to_message(self) -> Message:
        return Message(
            from_agent=self.agent_id,
            to_agent="broadcast",
            msg_type=MessageType.HANDSHAKE,
            action=ActionType.NOOP,
            capabilities=self.capabilities,
            context={"role": self.agent_role, "proto_ver": self.protocol_version},
        )


class ProtocolParser:
    """Parses and validates structured protocol messages."""

    PROTOCOL_VERSION = "1.0"
    MIN_CAPABILITIES = 1

    @classmethod
    def validate(cls, message: Message) -> bool:
        """Validate a message against protocol requirements."""
        if not message.from_agent:
            return False
        if not message.msg_id:
            return False
        return True

    @classmethod
    def create_handshake(cls, agent_id: str, role: str, capabilities: List[str]) -> Message:
        return HandshakeMessage(
            agent_id=agent_id,
            agent_role=role,
            capabilities=capabilities,
        ).to_message()

    @classmethod
    def create_request(
        cls,
        from_agent: str,
        to_agent: str,
        action: ActionType,
        params: Dict[str, Any],
        embedding: Optional[List[float]] = None,
        memory_refs: Optional[List[str]] = None,
    ) -> Message:
        return Message(
            from_agent=from_agent,
            to_agent=to_agent,
            msg_type=MessageType.REQUEST,
            action=action,
            params=params,
            state_embedding=embedding,
            memory_refs=memory_refs or [],
        )

    @classmethod
    def create_response(
        cls,
        request: Message,
        result: Any,
        embedding: Optional[List[float]] = None,
        memory_refs: Optional[List[str]] = None,
    ) -> Message:
        return Message(
            from_agent=request.to_agent,
            to_agent=request.from_agent,
            msg_type=MessageType.RESPONSE,
            action=request.action,
            params={"in_response_to": request.msg_id},
            result=result,
            state_embedding=embedding,
            memory_refs=memory_refs or [],
        )

    @classmethod
    def create_error(cls, request: Message, error: str) -> Message:
        return Message(
            from_agent=request.to_agent,
            to_agent=request.from_agent,
            msg_type=MessageType.ERROR,
            action=request.action,
            params={"in_response_to": request.msg_id},
            error=error,
        )
