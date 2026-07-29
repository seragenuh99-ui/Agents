"""Agent 调度器与消息路由 — 多 Agent 系统的通信基础设施。

本模块实现了 Agent 间通信所需的三大组件：
- AgentRegistry: Agent 注册与能力发现，维护能力→Agent 的反向索引
- MessageBus: 异步消息总线，点对点发送 + 广播 + 历史记录
- Scheduler: 任务调度器，握手注册、任务路由、响应回传、统计收集

核心流程：
  handshake() → 注册能力 → 广播握手消息
  route_task() → 能力匹配 → 发送请求 → 更新状态机
  send_response() → 回传结果 → 标记完成 → 记录指标
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


# ============================================================
# AgentInfo — Agent 元信息
# ============================================================

@dataclass
class AgentInfo:
    """Agent 的注册信息，包含身份、能力、运行时状态。"""
    agent_id: str                              # Agent 唯一标识
    role: str                                  # 角色（planner/retriever/executor/summarizer）
    capabilities: List[str]                    # 能力列表（对应 ActionType 值）
    status: str = "idle"                       # 当前状态：idle / busy
    registered_at: float = field(default_factory=time.time)   # 注册时间戳
    last_active: float = field(default_factory=time.time)     # 最后活跃时间
    messages_sent: int = 0                     # 已发送消息计数
    messages_received: int = 0                 # 已接收消息计数


# ============================================================
# AgentRegistry — Agent 注册中心
# ============================================================

class AgentRegistry:
    """Agent 注册中心：管理 Agent 的注册、能力发现与状态追踪。

    维护两个核心索引：
    - _agents: agent_id → AgentInfo 的主表
    - _capability_index: capability → {agent_id, ...} 的反向索引，用于快速能力匹配
    """

    def __init__(self):
        self._agents: Dict[str, AgentInfo] = {}                 # agent_id → AgentInfo
        self._capability_index: Dict[str, Set[str]] = defaultdict(set)  # 能力 → 拥有该能力的 Agent 集合
        self._handlers: Dict[str, Callable] = {}                # 预留：消息处理器注册

    # ---- 注册 ----

    def register(self, agent_id: str, role: str, capabilities: List[str]) -> None:
        """注册 Agent 及其能力声明，同时更新能力反向索引。"""
        info = AgentInfo(
            agent_id=agent_id,
            role=role,
            capabilities=capabilities,
        )
        self._agents[agent_id] = info
        for cap in capabilities:
            self._capability_index[cap].add(agent_id)

    # ---- 查询 ----

    def find_by_capability(self, capability: str) -> List[str]:
        """按能力查找拥有该能力的 Agent ID 列表。"""
        return list(self._capability_index.get(capability, set()))

    def find_by_role(self, role: str) -> List[str]:
        """按角色查找 Agent ID 列表。"""
        return [aid for aid, info in self._agents.items() if info.role == role]

    def get_info(self, agent_id: str) -> Optional[AgentInfo]:
        """获取单个 Agent 的注册信息。"""
        return self._agents.get(agent_id)

    def list_agents(self) -> Dict[str, AgentInfo]:
        """返回所有已注册 Agent 的信息（副本）。"""
        return dict(self._agents)

    # ---- 状态更新 ----

    def update_status(self, agent_id: str, status: str) -> None:
        """更新 Agent 的运行状态（idle/busy）并刷新最后活跃时间。"""
        if agent_id in self._agents:
            self._agents[agent_id].status = status
            self._agents[agent_id].last_active = time.time()

    def record_message(self, agent_id: str, direction: str) -> None:
        """记录 Agent 的消息收发计数。direction: "sent" | "received"。"""
        if agent_id in self._agents:
            if direction == "sent":
                self._agents[agent_id].messages_sent += 1
            else:
                self._agents[agent_id].messages_received += 1


# ============================================================
# MessageBus — 异步消息总线
# ============================================================

class MessageBus:
    """Agent 间异步消息总线。

    支持三种通信模式：
    - send(): 点对点发送，消息进入目标 Agent 的队列
    - receive(): 目标 Agent 拉取并清空自己的队列
    - broadcast(): 向所有已注册 Agent 广播（排除发送者及指定列表）

    同时维护 _history 全量消息记录，用于统计 token 节省量。
    """

    def __init__(self):
        self._queues: Dict[str, List[Message]] = defaultdict(list)  # agent_id → 消息队列
        self._history: List[Message] = []                            # 全量消息历史

    # ---- 发送与接收 ----

    def send(self, message: Message) -> None:
        """点对点发送：消息入队到目标 Agent 的队列，同时记入历史。"""
        self._queues[message.to_agent].append(message)
        self._history.append(message)

    def receive(self, agent_id: str) -> List[Message]:
        """拉取指定 Agent 的所有待处理消息，并清空其队列。"""
        messages = self._queues.get(agent_id, [])
        self._queues[agent_id] = []
        return messages

    def broadcast(self, message: Message, exclude: Optional[List[str]] = None) -> None:
        """向所有已有队列的 Agent 广播消息。

        Args:
            message: 要广播的消息模板
            exclude: 排除的 Agent ID 列表（不含发送者自身）
        """
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

    # ---- 统计属性 ----

    @property
    def message_count(self) -> int:
        """历史消息总数。"""
        return len(self._history)

    @property
    def structured_token_count(self) -> int:
        """结构化消息的估算 token 总量。"""
        return sum(m.estimated_token_count() for m in self._history)

    @property
    def text_token_count(self) -> int:
        """等效自然语言消息的估算 token 总量。"""
        return sum(m.text_token_count() for m in self._history)


# ============================================================
# Scheduler — 任务调度器
# ============================================================

class Scheduler:
    """多 Agent 系统的任务调度器。

    职责：
    1. 握手注册：handshake() 注册 Agent 能力并广播
    2. 任务路由：route_task() 按能力匹配目标 Agent，发送请求
    3. 响应回传：send_response() 将结果发送回请求方
    4. 状态追踪：维护 _task_queue、_completed_tasks、_task_history 三个状态机
    5. 指标收集：record_metric() 记录运行指标，get_statistics() 输出汇总
    """

    def __init__(self, registry: AgentRegistry, bus: MessageBus):
        self.registry = registry
        self.bus = bus
        self._task_queue: List[Dict[str, Any]] = []       # 待完成的任务队列
        self._completed_tasks: List[Dict[str, Any]] = []  # 已完成的任务列表
        self._task_history: List[Dict[str, Any]] = []     # 全量任务历史（含状态）
        self._metrics: Dict[str, Any] = defaultdict(list) # 指标收集器

    # ---- 握手 ----

    def handshake(self, agent_id: str, role: str, capabilities: List[str]) -> Message:
        """执行 Agent 握手：注册能力并广播上线消息。"""
        self.registry.register(agent_id, role, capabilities)
        msg = ProtocolParser.create_handshake(agent_id, role, capabilities)
        self.bus.broadcast(msg, exclude=[agent_id])
        return msg

    # ---- 能力发现 ----

    def discover_capable_agents(self, capability: str) -> List[str]:
        """查找拥有指定能力的 Agent 列表。"""
        return self.registry.find_by_capability(capability)

    # ---- 任务路由 ----

    def route_task(
        self,
        action: ActionType,
        params: Dict[str, Any],
        from_agent: str,
        to_agent: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        memory_refs: Optional[List[str]] = None,
    ) -> Message:
        """将任务路由到具备相应能力的 Agent。

        若未指定目标 Agent，则自动按 action 匹配能力最匹配的 Agent。
        路由后更新：消息总线、Agent 消息计数、Agent 状态、调度器状态机。

        Returns:
            发送的请求消息
        """
        # 未指定目标时，按能力自动匹配
        if to_agent is None:
            candidates = self.registry.find_by_capability(action.value)
            if not candidates:
                raise ValueError(f"No agent found with capability: {action.value}")
            to_agent = candidates[0]

        # 创建并发送请求消息
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
        self.registry.record_message(to_agent, "received")
        self.registry.update_status(to_agent, "busy")

        # 更新调度器状态机
        task_entry = {
            "msg_id": msg.msg_id,
            "action": action.value,
            "from": from_agent,
            "to": to_agent,
            "routed_at": time.time(),
        }
        self._task_queue.append(task_entry)
        self._task_history.append({**task_entry, "status": "routed"})
        self.record_metric("tasks_routed", 1)

        return msg

    # ---- 响应回传 ----

    def send_response(
        self,
        request: Message,
        result: Any,
        embedding: Optional[List[float]] = None,
        memory_refs: Optional[List[str]] = None,
    ) -> Message:
        """向请求方发送任务执行结果。

        自动创建响应消息（from/to 对调），更新 Agent 状态为 idle，
        并将对应任务从 _task_queue 移至 _completed_tasks。
        """
        msg = ProtocolParser.create_response(request, result, embedding, memory_refs)
        self.bus.send(msg)
        self.registry.record_message(msg.from_agent, "sent")
        self.registry.update_status(msg.from_agent, "idle")
        self.record_metric("tasks_completed", 1)

        # 将对应任务从队列中完成
        completed = None
        for i, entry in enumerate(self._task_queue):
            if entry["msg_id"] == request.msg_id:
                completed = self._task_queue.pop(i)
                break
        if completed:
            completed["completed_at"] = time.time()
            completed["status"] = "completed"
            self._completed_tasks.append(completed)
            # 同步更新历史记录中的状态
            for entry in self._task_history:
                if entry["msg_id"] == request.msg_id:
                    entry["status"] = "completed"
                    entry["completed_at"] = time.time()
                    break

        return msg

    # ---- 指标与统计 ----

    def record_metric(self, name: str, value: Any) -> None:
        """记录一个运行指标（追加到同名列表）。"""
        self._metrics[name].append(value)

    def get_metrics(self) -> Dict[str, Any]:
        """返回所有已记录指标的副本。"""
        return dict(self._metrics)

    def get_statistics(self) -> Dict[str, Any]:
        """汇总调度器运行统计。

        Returns:
            包含消息量、token 节省率、Agent 数、任务队列状态等统计信息。
        """
        return {
            "total_messages": self.bus.message_count,
            "structured_tokens": self.bus.structured_token_count,
            "text_equivalent_tokens": self.bus.text_token_count,
            "token_savings_pct": (
                (1 - self.bus.structured_token_count / max(self.bus.text_token_count, 1)) * 100
            ),
            "agents": len(self.registry.list_agents()),
            "tasks_queued": len(self._task_queue),
            "tasks_completed": len(self._completed_tasks),
            "tasks_total": len(self._task_history),
        }
