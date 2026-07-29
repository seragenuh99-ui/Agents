"""Agent 基类 — LLM 后端集成、消息处理、记忆操作与状态传递。

本模块定义了所有 Agent 的公共基础设施：
- LLMBackend: 统一的 LLM API 调用接口（支持 OpenAI/DeepSeek/自定义兼容 API）
- BaseAgent: 所有 Agent 的抽象基类，提供消息收发、记忆查询/存储、
  状态传递、模板提升（Promoter）等公共能力

LLM 调用链：
  chat() → 原始文本响应，自动记录 token 用量与 prompt 缓存命中
  chat_structured() → JSON 结构化响应，多策略提取有效 JSON
  _call_llm() / _call_llm_structured() → BaseAgent 便捷封装

记忆操作链：
  query_memory() → 三路搜索（关键词 + 标签 + 语义相似度）→ 合并去重
  store_memory() → 编码嵌入 → 创建 MemoryUnit → MemoryStore.store()
  _maybe_promote_to_template() → N>=2 个具体记忆 → LLM 合成领域模板
"""

from __future__ import annotations

import json
import os
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import requests

from ..protocol import Message, MessageType, ActionType, ProtocolParser
from ..protocol.scheduler import Scheduler, AgentRegistry, MessageBus
from ..state.embeddings import EmbeddingEngine
from ..state.exchange import StateExchangeBus, StatePacket
from ..memory.store import MemoryStore
from ..memory.models import MemoryUnit


# ============================================================
# LLMError — LLM 调用异常
# ============================================================

class LLMError(Exception):
    """LLM API 调用失败时抛出的异常。"""
    pass


# ============================================================
# Provider 预设 — 不同 LLM 提供商的默认配置
# ============================================================

_PROVIDERS = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "env_key": "OPENAI_API_KEY",
        "default_model": "gpt-4o",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "env_key": "DEEPSEEK_API_KEY",
        "default_model": "deepseek-chat",
    },
}


# ============================================================
# LLMBackend — 统一 LLM 后端
# ============================================================

class LLMBackend:
    """统一的 LLM 后端，支持 OpenAI 兼容 API。

    支持的 provider：openai、deepseek、custom。
    DeepSeek 使用 https://api.deepseek.com/v1，需配置 DEEPSEEK_API_KEY。

    自动追踪真实 token 用量和 prompt 缓存命中率（DeepSeek 的 cached_tokens）。
    支持 on_status 回调，用于向控制台输出实时状态。
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        provider: str = "custom",
    ):
        # 解析 provider 预设
        preset = _PROVIDERS.get(provider, {})
        self.provider = provider

        # Base URL 优先级：显式参数 > OPENAI_BASE_URL 环境变量 > provider 默认值
        self.base_url = (
            base_url
            or os.environ.get("OPENAI_BASE_URL")
            or preset.get("base_url", "https://api.openai.com/v1")
        )

        # API Key 优先级：显式参数 > provider 环境变量 > OPENAI_API_KEY
        env_key_name = preset.get("env_key", "OPENAI_API_KEY")
        self.api_key = (
            api_key
            or os.environ.get(env_key_name)
            or os.environ.get("OPENAI_API_KEY")
            or ""
        )

        # Model：显式参数 > provider 默认值
        self.model = model or preset.get("default_model", "gpt-4o")

        # Token 用量追踪
        self.call_count = 0                 # API 调用次数
        self.total_prompt_tokens = 0        # 累计 prompt tokens
        self.total_completion_tokens = 0    # 累计 completion tokens
        self.total_cached_tokens = 0        # 累计缓存命中 tokens
        self.last_usage: Dict[str, Any] = {}  # 最近一次调用的用量详情
        self.on_status: Optional[Callable[[str], None]] = None  # 状态回调

    def _headers(self) -> Dict[str, str]:
        """构建 API 请求头。"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """发送 Chat Completion 请求，返回文本响应。

        自动受益于 DeepSeek 的 prompt 缓存：重复前缀消息以 64 token 粒度命中 KV-cache。
        失败时抛出 LLMError。

        Kwargs:
            temperature: 温度参数（默认 0.3）
            max_tokens: 最大输出 token 数（默认 4096）
            timeout: 超时时间（默认 (15, 90) 秒）
            status_hint: 状态提示文本
        """
        self.last_elapsed_ms = 0.0
        if not self.api_key:
            raise LLMError(
                f"No API key configured. Set {self.provider.upper()}_API_KEY "
                f"or OPENAI_API_KEY environment variable."
            )

        url = f"{self.base_url}/chat/completions"
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.3),
            "max_tokens": kwargs.get("max_tokens", 4096),
        }
        response_format = kwargs.get("response_format")
        if response_format:
            body["response_format"] = response_format

        status_hint = kwargs.pop("status_hint", None)
        timeout = kwargs.get("timeout", (15, 90))
        if isinstance(timeout, (int, float)):
            timeout = (15, int(timeout))

        if self.on_status:
            if status_hint:
                self.on_status(status_hint)
            self.on_status(
                "  …正在等待 DeepSeek API 响应（单次最多约 90 秒，并非卡死）…"
            )

        t_call = time.time()
        try:
            resp = requests.post(
                url,
                headers=self._headers(),
                json=body,
                timeout=timeout,
            )
            self.last_elapsed_ms = (time.time() - t_call) * 1000
            if resp.status_code == 200:
                data = resp.json()
                self._record_usage(data)
                if self.on_status:
                    u = data.get("usage", {})
                    tot = u.get("total_tokens", 0)
                    self.on_status(f"  …DeepSeek 已返回（约 {tot} tokens）")
                return data["choices"][0]["message"]["content"]
            raise LLMError(
                f"API returned {resp.status_code}: {resp.text[:500]}"
            )
        except requests.exceptions.Timeout as e:
            raise LLMError(
                f"API 超时（连接 15s / 读取 90s）：{self.base_url} — "
                f"请检查网络或 DEEPSEEK_API_KEY。详情: {e}"
            )
        except requests.RequestException as e:
            raise LLMError(f"Request failed: {e}")

    def chat_structured(
        self,
        messages: List[Dict[str, str]],
        output_format: Dict[str, Any],
        **kwargs,
    ) -> Dict[str, Any]:
        """发送 Chat Completion 请求，期望返回结构化 JSON。

        通过多策略提取有效 JSON（```json``` 代码块 → ``` ``` 通用代码块 →
        花括号/方括号边界匹配 → 原始文本）。失败时抛出 LLMError。
        """
        json_mode = kwargs.pop("json_mode", False)
        response_format = {"type": "json_object"} if json_mode else None
        raw = self.chat(messages, response_format=response_format, **kwargs)
        return _extract_json(raw)

    def _record_usage(self, data: Dict[str, Any]) -> None:
        """从 API 响应中提取并累积 token 用量。"""
        usage = data.get("usage", {})
        if not usage:
            return

        self.call_count += 1
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.last_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": usage.get("total_tokens", 0),
        }

        # DeepSeek prompt 缓存：prompt_tokens_details.cached_tokens
        details = usage.get("prompt_tokens_details", {})
        cached = details.get("cached_tokens", 0)
        if cached:
            self.total_cached_tokens += cached
            self.last_usage["cached_tokens"] = cached
            self.last_usage["cache_hit_pct"] = (
                cached / prompt_tokens * 100 if prompt_tokens > 0 else 0
            )

    def get_usage_stats(self) -> Dict[str, Any]:
        """获取累积 token 用量和缓存统计。"""
        return {
            "call_count": self.call_count,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "total_cached_tokens": self.total_cached_tokens,
            "cache_hit_rate": (
                self.total_cached_tokens / self.total_prompt_tokens * 100
                if self.total_prompt_tokens > 0
                else 0
            ),
            "last_usage": self.last_usage,
        }

    def reset_stats(self) -> None:
        """重置用量统计。"""
        self.call_count = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cached_tokens = 0
        self.last_usage = {}


# ============================================================
# _extract_json — LLM 响应 JSON 提取
# ============================================================

def _extract_json(raw: str) -> Dict[str, Any]:
    """从 LLM 原始响应中提取 JSON，多策略回退。

    策略顺序：
    1. ```json ... ``` 代码块提取
    2. ``` ... ``` 通用代码块提取（跳过语言标签）
    3. 花括号/方括号边界匹配提取
    4. 原始文本整体解析
    """
    candidates = []

    # Strategy 1: ```json ... ``` 代码块
    start = 0
    while True:
        idx = raw.find("```json", start)
        if idx == -1:
            break
        block_start = idx + 7
        block_end = raw.find("```", block_start)
        if block_end != -1:
            candidates.append(raw[block_start:block_end].strip())
            start = block_end + 3
        else:
            break

    # Strategy 2: ``` ... ``` 通用代码块（无语言标签）
    start = 0
    while True:
        idx = raw.find("```", start)
        if idx == -1:
            break
        block_start = idx + 3
        first_nl = raw.find("\n", block_start)
        if first_nl != -1 and first_nl < block_start + 20:
            block_start = first_nl + 1
        block_end = raw.find("```", block_start)
        if block_end != -1:
            candidates.append(raw[block_start:block_end].strip())
            start = block_end + 3
        else:
            break

    # Strategy 3: 花括号 { } 或方括号 [ ] 边界匹配
    for delim_start, delim_end in [("{", "}"), ("[", "]")]:
        si = raw.find(delim_start)
        ei = raw.rfind(delim_end)
        if si != -1 and ei != -1 and ei > si:
            candidates.append(raw[si : ei + 1])

    # 逐个尝试
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue

    # Strategy 4: 原始文本整体解析
    try:
        return json.loads(raw.strip())
    except (json.JSONDecodeError, ValueError):
        pass

    raise LLMError(
        f"Could not parse JSON from LLM response "
        f"(first 300 chars): {raw[:300]}"
    )


# ============================================================
# BaseAgent — Agent 抽象基类
# ============================================================

class BaseAgent(ABC):
    """所有 Agent 的抽象基类，集成协议通信、LLM 调用、记忆操作与状态传递。

    子类必须实现：
    - handle_message(message): 处理收到的结构化消息
    - execute_task(task_input): 执行具体任务并返回结果

    公共能力：
    - 消息收发：send_message(), receive_messages(), process_pending_messages()
    - 记忆操作：query_memory(), retrieve_memories(), store_memory()
    - 状态传递：transfer_state(), receive_state_packets()
    - 模板提升：_maybe_promote_to_template() — N>=2 具体记忆 → 领域模板
    - LLM 调用：_call_llm(), _call_llm_structured()
    """

    def __init__(
        self,
        agent_id: str,
        role: str,
        capabilities: List[str],
        scheduler: Scheduler,
        llm: LLMBackend,
        embedding_engine: EmbeddingEngine,
        state_bus: StateExchangeBus,
        memory_store: MemoryStore,
        use_structured_protocol: bool = True,
    ):
        self.agent_id = agent_id
        self.role = role
        self.capabilities = capabilities
        self.scheduler = scheduler
        self.llm = llm
        self.embedding_engine = embedding_engine
        self.state_bus = state_bus
        self.memory_store = memory_store
        self.use_structured_protocol = use_structured_protocol

        self._task_history: List[Dict[str, Any]] = []  # 任务执行历史
        self._context: Dict[str, Any] = {}              # 当前任务上下文

        # 向调度器注册自身
        self.scheduler.handshake(agent_id, role, capabilities)

    # ---- 消息收发 ----

    def receive_messages(self) -> List[Message]:
        """从消息总线拉取并清空自己的消息队列。"""
        return self.scheduler.bus.receive(self.agent_id)

    def process_pending_messages(self) -> List[Message]:
        """轮询消息总线，处理所有待处理消息。

        这是消息驱动分发路径：消息通过总线传递，每个 Agent 的
        handle_message() 路由到对应处理器（_handle_plan_request 等），
        处理器调用 execute_task() 并回传响应。

        Returns:
            处理器发送的响应消息列表
        """
        messages = self.receive_messages()
        responses: List[Message] = []
        for msg in messages:
            response = self.handle_message(msg)
            if response is not None:
                responses.append(response)
        return responses

    def send_message(
        self,
        to_agent: str,
        action: ActionType,
        params: Dict[str, Any],
        embedding: Optional[List[float]] = None,
        memory_refs: Optional[List[str]] = None,
    ) -> Message:
        """通过调度器向目标 Agent 发送结构化消息。"""
        msg = self.scheduler.route_task(
            action=action,
            params=params,
            from_agent=self.agent_id,
            to_agent=to_agent,
            embedding=embedding,
            memory_refs=memory_refs,
        )
        return msg

    def send_text_message(
        self,
        to_agent: str,
        action: ActionType,
        text_content: str,
    ) -> Message:
        """以文本模式发送消息（用于对比实验）。

        文本模式下，消息内容以自然语言传递，模拟传统 Agent 间文本通信。
        """
        msg = Message(
            from_agent=self.agent_id,
            to_agent=to_agent,
            msg_type=MessageType.REQUEST,
            action=action,
            params={"text_content": text_content},
            context={"mode": "text"},
        )
        self.scheduler.bus.send(msg)
        self.scheduler.registry.record_message(self.agent_id, "sent")
        return msg

    # ---- 记忆操作 ----

    def query_memory(
        self,
        query: str,
        tags: Optional[List[str]] = None,
        use_embedding: bool = True,
        limit: int = 5,
    ) -> List[MemoryUnit]:
        """三路搜索共享记忆：关键词 + 标签 + 语义相似度，合并去重。

        搜索路径：
        1. search_by_keyword(): SQL LIKE 全文搜索
        2. search_by_tags(): 标签 OR 匹配
        3. search_by_similarity(): FAISS 语义相似度（cos > 0.3 过滤）

        所有命中的记忆都会记录访问日志。
        """
        results: List[MemoryUnit] = []
        seen_ids: set = set()

        # 路径 1：关键词搜索
        kw_results = self.memory_store.search_by_keyword(query, limit=limit)
        for mem in kw_results:
            if mem.memory_id not in seen_ids:
                results.append(mem)
                seen_ids.add(mem.memory_id)

        # 路径 2：标签搜索
        if tags:
            tag_results = self.memory_store.search_by_tags(tags, limit=limit)
            for mem in tag_results:
                if mem.memory_id not in seen_ids:
                    results.append(mem)
                    seen_ids.add(mem.memory_id)

        # 路径 3：语义相似度搜索
        if use_embedding:
            query_emb = self.embedding_engine.encode(query)
            sim_results = self.memory_store.search_by_similarity(query_emb, limit=limit)
            for mem, score in sim_results:
                if mem.memory_id not in seen_ids and score > 0.3:
                    results.append(mem)
                    seen_ids.add(mem.memory_id)

        # 记录访问日志
        for mem in results:
            self.memory_store.record_access(
                mem.memory_id, self.agent_id, self._context.get("task_id", "unknown")
            )

        return results[:limit]

    def retrieve_memories(
        self, memory_ids: List[str]
    ) -> List[MemoryUnit]:
        """按 ID 列表批量检索记忆，跳过不存在的 ID。"""
        results = []
        for mid in memory_ids:
            mem = self.memory_store.get(mid)
            if mem is not None:
                self.memory_store.record_access(
                    mid, self.agent_id, self._context.get("task_id", "unknown")
                )
                results.append(mem)
        return results

    def store_memory(
        self,
        topic: str,
        summary: str,
        content: str,
        tags: Optional[List[str]] = None,
        memory_type: str = "result",
        evidence_chain: Optional[List[str]] = None,
        embedding_text: Optional[str] = None,
    ) -> str:
        """创建并存储一条记忆到共享记忆库。

        Args:
            topic: 记忆主题
            summary: 记忆摘要
            content: 完整内容
            tags: 标签列表
            memory_type: 记忆类型（result/evidence/strategy/fact/error）
            evidence_chain: 证据链（支持该记忆的其他记忆 ID）
            embedding_text: 用于生成嵌入向量的文本（默认用 topic+summary+tags 拼接）

        Returns:
            新创建的记忆 ID
        """
        text_for_embedding = embedding_text or f"{topic} {summary} {' '.join(tags or [])}"
        embedding = self.embedding_engine.encode(text_for_embedding)

        memory = MemoryUnit(
            source_agent=self.agent_id,
            task_topic=topic,
            task_id=self._context.get("task_id", "unknown"),
            summary=summary,
            content=content,
            tags=tags or [],
            evidence_chain=evidence_chain or [],
            embedding=embedding,
            memory_type=memory_type,
        )
        return self.memory_store.store(memory)

    # ---- 模板提升（Promoter） ----

    def _maybe_promote_to_template(
        self, tags: List[str], memory_type: str = "strategy", threshold: int = 2,
    ) -> Optional[str]:
        """当同类型具体记忆积累到阈值时，自动合成为领域模板。

        G-Memory / EVOLVE-MEM 启发：N 个同类型、标签重叠的具体记忆
        → LLM 提取公共结构 → 存储为 abstraction_level=1 的领域模板。
        Planner 优先搜索模板进行 template-fill，避免完整 LLM 生成。

        Args:
            tags: 标签列表
            memory_type: 记忆类型
            threshold: 触发阈值（默认 2 个具体记忆即可合成模板）

        Returns:
            新模板的 memory_id，或 None（未达阈值/模板已存在/合成失败）
        """
        if not tags:
            return None

        # 统计同类型、同标签的具体记忆数量
        count = self.memory_store.count_by_tags_and_type(
            tags, memory_type, abstraction_level=0
        )
        if count < threshold:
            return None

        # 检查是否已有该领域的模板
        existing_templates = self.memory_store.get_templates(tags, memory_type, limit=1)
        if existing_templates:
            return existing_templates[0].memory_id

        # 获取具体记忆用于合成
        concrete = self.memory_store.get_concrete_memories(tags, memory_type, limit=5)
        if len(concrete) < threshold:
            return None

        # 构建 LLM prompt 提取公共结构
        parts = []
        for i, mem in enumerate(concrete):
            parts.append(f"[{i+1}] {mem.task_topic}\n  Content: {mem.content[:300]}")
        concrete_text = "\n".join(parts)
        common_tags = list(set.intersection(*[set(m.tags) for m in concrete])) if len(concrete) > 1 else tags

        system_prompt = """Domain template→JSON. Identify the common task structure from examples.
Output: {template_for:"domain name", common_tags:[], subtask_pattern:[{step,role,action,params_template:{}}], notes:""}"""
        user_prompt = f"Examples:\n{concrete_text}\n\nExtract the common template:"

        try:
            template = self._call_llm_structured(system_prompt, user_prompt, {}, max_tokens=256)
        except Exception:
            return None

        if not isinstance(template, dict) or "subtask_pattern" not in template:
            return None

        template["_derived_from"] = [m.memory_id for m in concrete]
        template["_derived_count"] = len(concrete)

        template_text_for_embedding = " ".join(common_tags)
        template_embedding = self.embedding_engine.encode(template_text_for_embedding)

        template_mem = MemoryUnit(
            source_agent=self.agent_id,
            task_topic=f"Template: {template.get('template_for', 'unknown')}",
            task_id=self._context.get("task_id", "unknown"),
            summary=template.get("notes", f"Domain template from {len(concrete)} examples")[:200],
            content=json.dumps(template, ensure_ascii=False),
            tags=common_tags or tags,
            evidence_chain=[m.memory_id for m in concrete],
            embedding=template_embedding,
            memory_type=memory_type,
            abstraction_level=1,
        )
        tid = self.memory_store.store(template_mem)
        return tid

    # ---- 状态传递 ----

    def transfer_state(
        self, target_agent: str, state_data: Any, context: str = ""
    ) -> StatePacket:
        """将状态编码为嵌入向量并发送给目标 Agent（非文本传递）。"""
        return self.state_bus.transfer(
            state_data, self.agent_id, target_agent, context=context
        )

    def receive_state_packets(
        self, since: float = 0, limit: int = 0
    ) -> List[StatePacket]:
        """获取发送给本 Agent 的状态数据包。

        下游 Agent 调用此方法获取上游发送的嵌入向量，
        可直接用于语义记忆搜索，无需重新编码文本。
        """
        return self.state_bus.get_packets_for(
            self.agent_id, since=since, limit=limit
        )

    # ---- LLM 辅助 ----

    def _llm_judge_similar(self, task_a: str, task_b: str) -> bool:
        """用 LLM 判断两个任务是否属于同一类工作。

        P0 优化：FAISS 嵌入提供快速粗粒度召回，LLM 判断提供高精度验证。
        成本：每次判断约 50 prompt tokens。
        """
        if not task_a or not task_b:
            return False
        try:
            system = "Task similarity judge. Answer YES if two tasks are the same kind of work (share domain, goal, or methodology). Answer NO if they are fundamentally different."
            user = f"Task A: {task_a[:200]}\nTask B: {task_b[:200]}\nSame kind of task? YES/NO:"
            response = self.llm.chat(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_tokens=5,
            )
            return "YES" in response.upper()
        except Exception:
            return False

    def _emit_status(self, msg: str) -> None:
        """向状态回调发送消息（用于控制台实时输出）。"""
        if getattr(self.llm, "on_status", None):
            self.llm.on_status(msg)

    def _call_llm(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """便捷方法：以 system + user prompt 调用 LLM 并返回文本。"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.llm.chat(messages, **kwargs)

    def _call_llm_structured(
        self, system_prompt: str, user_prompt: str, output_format: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        """便捷方法：以 system + user prompt 调用 LLM 并返回结构化 JSON。"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.llm.chat_structured(messages, output_format, json_mode=True, **kwargs)

    def _mem_task_desc(self, mem: MemoryUnit) -> str:
        """从记忆的 task_topic 中提取干净的任务描述。"""
        topic = mem.task_topic or ""
        for prefix in ["Summary: ", "Plan: ", "Execution: ", "Retrieval: "]:
            if topic.startswith(prefix):
                return topic[len(prefix):]
        return topic

    def _context_from_memories(self, memories: List[MemoryUnit]) -> str:
        """将检索到的记忆格式化为紧凑的 LLM prompt 上下文。

        仅取最相关的 2 条记忆的摘要（各 120 字符），避免大段内容撑爆 prompt。
        """
        if not memories:
            return "No relevant memories found."

        parts = []
        for i, mem in enumerate(memories[:2]):
            parts.append(f"[{mem.memory_id[:8]}] {mem.summary[:120]}")
        return "\n".join(parts)

    # ---- 抽象方法（子类必须实现） ----

    @abstractmethod
    def handle_message(self, message: Message) -> Optional[Message]:
        """处理收到的结构化消息，可选地返回响应消息。"""
        ...

    @abstractmethod
    def execute_task(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """执行具体任务并返回结果。"""
        ...

    # ---- 统计 ----

    def get_stats(self) -> Dict[str, Any]:
        """返回 Agent 运行统计。"""
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "tasks_executed": len(self._task_history),
            "capabilities": self.capabilities,
        }
