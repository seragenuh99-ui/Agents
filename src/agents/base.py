"""Base agent class with LLM backend integration.

Each agent has a role, capabilities, and can operate in both
text mode and structured protocol mode for comparison.
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


class LLMError(Exception):
    """Raised when an LLM API call fails."""

    pass


# Provider presets
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


class LLMBackend:
    """Unified LLM backend supporting OpenAI-compatible APIs.

    Supports providers: openai, deepseek, custom.
    DeepSeek uses https://api.deepseek.com/v1 with your DEEPSEEK_API_KEY.

    Tracks real token usage and prompt cache hits from API responses.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        provider: str = "custom",
    ):
        # Resolve provider preset
        preset = _PROVIDERS.get(provider, {})
        self.provider = provider

        # Base URL: explicit arg > env OPENAI_BASE_URL > provider default
        self.base_url = (
            base_url
            or os.environ.get("OPENAI_BASE_URL")
            or preset.get("base_url", "https://api.openai.com/v1")
        )

        # API key: explicit arg > provider env var > OPENAI_API_KEY
        env_key_name = preset.get("env_key", "OPENAI_API_KEY")
        self.api_key = (
            api_key
            or os.environ.get(env_key_name)
            or os.environ.get("OPENAI_API_KEY")
            or ""
        )

        # Model: explicit arg > provider default
        self.model = model or preset.get("default_model", "gpt-4o")

        # Usage tracking
        self.call_count = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cached_tokens = 0
        self.last_usage: Dict[str, Any] = {}
        self.on_status: Optional[Callable[[str], None]] = None

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Send a chat completion request. Raises LLMError on failure.

        Automatically benefits from prompt caching: DeepSeek caches KV-cache
        for repeated prefix messages at 64-token granularity.
        """
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

        try:
            resp = requests.post(
                url,
                headers=self._headers(),
                json=body,
                timeout=timeout,
            )
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
        """Send a chat completion expecting structured JSON output.

        Tries multiple strategies to extract valid JSON from the response.
        Raises LLMError if no valid JSON can be parsed.
        """
        json_mode = kwargs.pop("json_mode", False)
        response_format = {"type": "json_object"} if json_mode else None
        raw = self.chat(messages, response_format=response_format, **kwargs)
        return _extract_json(raw)

    def _record_usage(self, data: Dict[str, Any]) -> None:
        """Extract and accumulate token usage from API response."""
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

        # DeepSeek prompt caching: cached_tokens in prompt_tokens_details
        details = usage.get("prompt_tokens_details", {})
        cached = details.get("cached_tokens", 0)
        if cached:
            self.total_cached_tokens += cached
            self.last_usage["cached_tokens"] = cached
            self.last_usage["cache_hit_pct"] = (
                cached / prompt_tokens * 100 if prompt_tokens > 0 else 0
            )

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get cumulative token usage and cache statistics."""
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
        """Reset usage statistics."""
        self.call_count = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cached_tokens = 0
        self.last_usage = {}


def _extract_json(raw: str) -> Dict[str, Any]:
    """Extract JSON from an LLM response, trying multiple strategies."""
    candidates = []

    # Strategy 1: ```json ... ``` block
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

    # Strategy 2: ``` ... ``` block (no language tag)
    start = 0
    while True:
        idx = raw.find("```", start)
        if idx == -1:
            break
        block_start = idx + 3
        # Skip past a language tag on the opening fence
        first_nl = raw.find("\n", block_start)
        if first_nl != -1 and first_nl < block_start + 20:
            block_start = first_nl + 1
        block_end = raw.find("```", block_start)
        if block_end != -1:
            candidates.append(raw[block_start:block_end].strip())
            start = block_end + 3
        else:
            break

    # Strategy 3: raw text between outermost { } or [ ]
    for delim_start, delim_end in [("{", "}"), ("[", "]")]:
        si = raw.find(delim_start)
        ei = raw.rfind(delim_end)
        if si != -1 and ei != -1 and ei > si:
            candidates.append(raw[si : ei + 1])

    # Try each candidate
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue

    # Strategy 4: the full raw text
    try:
        return json.loads(raw.strip())
    except (json.JSONDecodeError, ValueError):
        pass

    raise LLMError(
        f"Could not parse JSON from LLM response "
        f"(first 300 chars): {raw[:300]}"
    )


class BaseAgent(ABC):
    """Base agent with protocol, LLM, and memory capabilities."""

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

        self._task_history: List[Dict[str, Any]] = []
        self._context: Dict[str, Any] = {}

        # Register with scheduler
        self.scheduler.handshake(agent_id, role, capabilities)

    def receive_messages(self) -> List[Message]:
        return self.scheduler.bus.receive(self.agent_id)

    def send_message(
        self,
        to_agent: str,
        action: ActionType,
        params: Dict[str, Any],
        embedding: Optional[List[float]] = None,
        memory_refs: Optional[List[str]] = None,
    ) -> Message:
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
        """Send a message in text mode (for comparison experiments).

        In text mode, the entire message content is passed as natural language,
        simulating traditional Agent-to-Agent text communication.
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

    def query_memory(
        self,
        query: str,
        tags: Optional[List[str]] = None,
        use_embedding: bool = True,
        limit: int = 5,
    ) -> List[MemoryUnit]:
        """Search shared memory for relevant past results."""
        results: List[MemoryUnit] = []
        seen_ids: set = set()

        # Keyword search
        kw_results = self.memory_store.search_by_keyword(query, limit=limit)
        for mem in kw_results:
            if mem.memory_id not in seen_ids:
                results.append(mem)
                seen_ids.add(mem.memory_id)

        # Tag search
        if tags:
            tag_results = self.memory_store.search_by_tags(tags, limit=limit)
            for mem in tag_results:
                if mem.memory_id not in seen_ids:
                    results.append(mem)
                    seen_ids.add(mem.memory_id)

        # Semantic similarity search
        if use_embedding:
            query_emb = self.embedding_engine.encode(query)
            sim_results = self.memory_store.search_by_similarity(query_emb, limit=limit)
            for mem, score in sim_results:
                if mem.memory_id not in seen_ids and score > 0.3:
                    results.append(mem)
                    seen_ids.add(mem.memory_id)

        # Log accesses
        for mem in results:
            self.memory_store.record_access(
                mem.memory_id, self.agent_id, self._context.get("task_id", "unknown")
            )

        return results[:limit]

    def retrieve_memories(
        self, memory_ids: List[str]
    ) -> List[MemoryUnit]:
        """Retrieve multiple memories by ID. Skips missing IDs."""
        results = []
        for mid in memory_ids:
            mem = self.memory_store.get(mid)
            if mem is not None:
                self.memory_store.record_access(
                    mid, self.agent_id, self._context.get("task_id", "unknown")
                )
                mem.access_count += 1
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
        """Store a memory unit in shared memory.

        If embedding_text is provided, it is used for the semantic embedding
        instead of the default topic+summary+tags concatenation. This allows
        storing with an embedding that matches the search query distribution
        (e.g. storing a plan with the raw task description as embedding).
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

    def _maybe_promote_to_template(
        self, tags: List[str], memory_type: str = "strategy", threshold: int = 2,
    ) -> Optional[str]:
        """Promote concrete memories to a domain template when enough accumulate.

        G-Memory / EVOLVE-MEM inspired: after N concrete memories of the same type
        and overlapping tags exist, synthesize a template capturing the common
        structure. The template is stored at abstraction_level=1 and searched
        first by the planner, enabling template-fill instead of full generation.
        """
        if not tags:
            return None

        # Count existing concrete memories (level 0) of this type with these tags
        count = self.memory_store.count_by_tags_and_type(
            tags, memory_type, abstraction_level=0
        )
        if count < threshold:
            return None

        # Check if a template already exists for this domain
        existing_templates = self.memory_store.get_templates(tags, memory_type, limit=1)
        if existing_templates:
            return existing_templates[0].memory_id

        # Grab the concrete memories to synthesize from
        concrete = self.memory_store.get_concrete_memories(tags, memory_type, limit=5)
        if len(concrete) < threshold:
            return None

        # Build LLM prompt to create template
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

    def transfer_state(
        self, target_agent: str, state_data: Any, context: str = ""
    ) -> StatePacket:
        """Send non-text state to another agent via embedding."""
        return self.state_bus.transfer(
            state_data, self.agent_id, target_agent, context=context
        )

    def _llm_judge_similar(self, task_a: str, task_b: str) -> bool:
        """Use LLM to judge if two task descriptions are the same kind of work.

        P0 optimization: FAISS embedding provides fast coarse recall,
        this method provides high-precision verification before caching.
        Cost: ~50 prompt tokens per judgment.
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
        if getattr(self.llm, "on_status", None):
            self.llm.on_status(msg)

    def _call_llm(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """Call the LLM with system and user prompts."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.llm.chat(messages, **kwargs)

    def _call_llm_structured(
        self, system_prompt: str, user_prompt: str, output_format: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.llm.chat_structured(messages, output_format, json_mode=True, **kwargs)

    def _mem_task_desc(self, mem: MemoryUnit) -> str:
        """Extract a clean task description from a memory's task_topic."""
        topic = mem.task_topic or ""
        for prefix in ["Summary: ", "Plan: ", "Execution: ", "Retrieval: "]:
            if topic.startswith(prefix):
                return topic[len(prefix):]
        return topic

    def _context_from_memories(self, memories: List[MemoryUnit]) -> str:
        """Format retrieved memories as compact context for LLM prompts.

        Only includes the most relevant 2 memories with summary only,
        avoiding the 500-char content dump that bloated prompts.
        """
        if not memories:
            return "No relevant memories found."

        parts = []
        for i, mem in enumerate(memories[:2]):
            parts.append(f"[{mem.memory_id[:8]}] {mem.summary[:120]}")
        return "\n".join(parts)

    @abstractmethod
    def handle_message(self, message: Message) -> Optional[Message]:
        """Process an incoming message and optionally respond."""
        ...

    @abstractmethod
    def execute_task(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task based on input parameters."""
        ...

    def get_stats(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "tasks_executed": len(self._task_history),
            "capabilities": self.capabilities,
        }
