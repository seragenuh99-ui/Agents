"""结构化通信协议 — 多 Agent 间的紧凑二进制消息格式。

本模块实现了替代自然语言 Agent 通信的结构化协议：
- MessagePack 二进制序列化，字段名缩写以减小体积
- 消息类型（请求/响应/握手/错误）与动作类型（规划/检索/执行/总结）
- 每条消息携带动作、参数、可选的状态向量与记忆引用

核心设计：
  to_dict() → 缩写键名的字典（"h"=header, "a"=action, "s"=state, "p"=payload）
  serialize() → MessagePack 二进制（实际通信使用 to_dict/from_dict 路径）
  estimated_token_count() → 结构化消息的 token 估算，用于与纯文本对比
"""

from __future__ import annotations

import msgpack
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ============================================================
# 枚举定义 — 消息类型与动作类型
# ============================================================

class MessageType(str, Enum):
    """Agent 间消息的类型分类。"""
    REQUEST = "request"               # 任务请求（主动发起）
    RESPONSE = "response"             # 任务响应（携带结果）
    HANDSHAKE = "handshake"           # 握手注册（能力宣告）
    CAPABILITY_QUERY = "capability_query"       # 能力查询
    CAPABILITY_ADVERTISE = "capability_advertise"  # 能力广播
    STATE_TRANSFER = "state_transfer" # 非文本状态传递
    MEMORY_STORE = "memory_store"     # 记忆存储请求
    MEMORY_QUERY = "memory_query"     # 记忆查询请求
    ERROR = "error"                   # 错误响应


class ActionType(str, Enum):
    """Agent 可执行的具体动作。"""
    PLAN = "plan"              # 任务分解与规划
    RETRIEVE = "retrieve"      # 信息检索
    EXECUTE = "execute"        # 代码执行与数据处理
    SUMMARIZE = "summarize"    # 结果综合与总结
    STORE_MEMORY = "store_memory"  # 存储记忆
    QUERY_MEMORY = "query_memory"  # 查询记忆
    REPORT = "report"          # 生成报告
    NOOP = "noop"              # 空操作


# ============================================================
# Message — 核心消息数据结构
# ============================================================

@dataclass
class Message:
    """Agent 间通信的结构化消息。

    使用缩写字段名序列化为紧凑字典：
      h (header):  id=消息ID, ts=时间戳, fr=发送者, to=接收者, mt=消息类型
      a (action):  tp=动作类型, pm=参数, cp=能力列表
      s (state):   em=状态嵌入向量, mr=记忆引用列表
      p (payload): 结果数据
      c (context): 上下文元信息
      e (error):   错误信息
    """

    # 基础标识
    msg_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    from_agent: str = ""       # 发送者 ID
    to_agent: str = ""         # 接收者 ID

    # 消息类型与动作
    msg_type: MessageType = MessageType.REQUEST
    action: ActionType = ActionType.NOOP

    # 载荷
    params: Dict[str, Any] = field(default_factory=dict)     # 动作参数
    result: Optional[Any] = None                              # 执行结果

    # 优化机制
    state_embedding: Optional[List[float]] = None  # 384 维语义向量（供非文本传递）
    memory_refs: List[str] = field(default_factory=list)      # 相关记忆 ID 列表
    capabilities: List[str] = field(default_factory=list)     # Agent 能力声明
    context: Dict[str, Any] = field(default_factory=dict)     # 上下文元信息
    error: Optional[str] = None                                # 错误描述

    # ---- 序列化 ----

    def to_dict(self) -> Dict[str, Any]:
        """转为紧凑字典（缩写键名以减小体积）。"""
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
            "p": self.result,   # payload
            "c": self.context,  # context
            "e": self.error,    # error
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """从紧凑字典恢复 Message 对象。"""
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
        """MessagePack 二进制序列化。"""
        return msgpack.packb(self.to_dict(), use_bin_type=True)

    @classmethod
    def deserialize(cls, data: bytes) -> "Message":
        """从 MessagePack 二进制反序列化。"""
        return cls.from_dict(msgpack.unpackb(data, raw=False))

    # ---- Token 估算（用于与纯文本模式对比） ----

    def estimated_token_count(self) -> int:
        """估算结构化消息的 token 数（1 token ≈ 3 字符）。"""
        return len(str(self.to_dict())) // 3

    def to_text_equivalent(self) -> str:
        """生成等效的自然语言表示（用于对比实验）。"""
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
        """等效自然语言的 token 数。"""
        return len(self.to_text_equivalent()) // 3


# ============================================================
# HandshakeMessage — Agent 注册握手
# ============================================================

@dataclass
class HandshakeMessage:
    """Agent 初次连接时的能力宣告消息。"""
    agent_id: str
    agent_role: str
    capabilities: List[str]
    protocol_version: str = "1.0"

    def to_message(self) -> Message:
        """转为标准 Message 格式。"""
        return Message(
            from_agent=self.agent_id,
            to_agent="broadcast",
            msg_type=MessageType.HANDSHAKE,
            action=ActionType.NOOP,
            capabilities=self.capabilities,
            context={"role": self.agent_role, "proto_ver": self.protocol_version},
        )


# ============================================================
# ProtocolParser — 消息构造与验证的工厂类
# ============================================================

class ProtocolParser:
    """协议消息的构造、验证与错误处理。

    所有消息创建都通过此类方法，确保格式一致性。
    """

    PROTOCOL_VERSION = "1.0"

    # ---- 验证 ----

    @classmethod
    def validate(cls, message: Message) -> bool:
        """验证消息是否满足协议最小要求。"""
        if not message.from_agent:
            return False
        if not message.msg_id:
            return False
        return True

    # ---- 消息工厂方法 ----

    @classmethod
    def create_handshake(
        cls, agent_id: str, role: str, capabilities: List[str]
    ) -> Message:
        """创建 Agent 握手注册消息。"""
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
        """创建任务请求消息。

        Args:
            from_agent: 发送者 ID
            to_agent: 接收者 ID
            action: 要执行的动作类型
            params: 动作参数
            embedding: 可选的语义向量（用于非文本状态传递）
            memory_refs: 可选的记忆引用 ID 列表
        """
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
        """创建对请求消息的响应。

        自动将 from/to 对调，并携带 in_response_to 引用原始消息 ID。
        """
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
        """创建错误响应消息。"""
        return Message(
            from_agent=request.to_agent,
            to_agent=request.from_agent,
            msg_type=MessageType.ERROR,
            action=request.action,
            params={"in_response_to": request.msg_id},
            error=error,
        )
