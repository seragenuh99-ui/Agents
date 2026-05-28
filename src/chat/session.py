"""Interactive chat session on top of the full multi-agent pipeline (v0.9)."""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from ..agents.base import LLMBackend
from ..evaluation.quality_validator import extract_summary_payload
from ..orchestrator import Orchestrator
from ..run_options import DEFAULT_OPTIONS, RunOptions


def _load_dotenv() -> None:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(root, ".env")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())


def infer_tags(text: str) -> List[str]:
    """Lightweight domain tags from user message."""
    t = text.lower()
    tags: List[str] = []
    if any(w in t for w in ("solar", "wind", "energy", "renewable", "photovoltaic", "hydro")):
        tags.extend(["energy", "research"])
        if "solar" in t:
            tags.append("solar")
        if "wind" in t:
            tags.append("wind")
    if any(w in t for w in ("security", "vuln", "sql injection", "csrf", "jwt", "xss", "python")):
        tags.extend(["security", "research"])
        if "web" in t or "csrf" in t or "oauth" in t:
            tags.append("web")
        if "python" in t:
            tags.append("python")
    if any(w in t for w in ("database", "sql", "nosql", "redis", "postgres", "mysql", "index")):
        tags.extend(["database", "research"])
    if not tags:
        tags = ["research", "general"]
    # dedupe preserve order
    seen = set()
    out = []
    for x in tags:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def format_answer(result: Dict[str, Any]) -> str:
    """Turn orchestrator result into user-readable markdown."""
    if result.get("_e2e_cached"):
        score = result.get("_e2e_score", "?")
        lines = [f"*（记忆复用 E2E，相似度 {score}，未调用 LLM）*\n"]
    else:
        lines = []

    summary_step = result.get("steps", {}).get("summary", {})
    obj, text = extract_summary_payload(summary_step)

    if isinstance(obj, dict) and obj.get("key_findings"):
        lines.append("**要点**")
        for i, f in enumerate(obj.get("key_findings", [])[:5], 1):
            lines.append(f"{i}. {f}")
        facts = obj.get("facts") or []
        if facts:
            lines.append("\n**补充事实**")
            for f in facts[:3]:
                lines.append(f"- {f}")
        if obj.get("conclusion"):
            lines.append(f"\n**结论**：{obj['conclusion']}")
    elif text:
        lines.append(text)
    else:
        lines.append("（未生成有效摘要，请换种问法或 /clear 后重试）")

    plan = result.get("steps", {}).get("plan", {})
    meta = []
    if plan.get("_template_filled"):
        meta.append("模板填空")
    if plan.get("_executor_dropped"):
        meta.append("已跳过 Executor")
    if plan.get("llm_skipped"):
        meta.append("计划缓存")
    if meta:
        lines.append(f"\n*策略：{', '.join(meta)}*")

    return "\n".join(lines)


class ChatSession:
    """One user session: persistent memory DB + multi-turn task execution."""

    def __init__(
        self,
        *,
        db_path: str = "chat_session.db",
        provider: str = "deepseek",
        model: Optional[str] = None,
        mode: str = "structured",
        run_options: Optional[RunOptions] = None,
    ):
        _load_dotenv()
        self.db_path = db_path
        self.mode = mode
        self.run_options = run_options or DEFAULT_OPTIONS
        self.turn = 0
        self.session_id = uuid.uuid4().hex[:8]
        self.default_tags: List[str] = ["research", "general"]

        api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key and provider == "deepseek":
            raise RuntimeError("请设置 DEEPSEEK_API_KEY（.env 或环境变量）")

        self.llm = LLMBackend(
            provider=provider,
            api_key=api_key,
            model=model or os.environ.get("CHAT_MODEL", "deepseek-chat"),
        )
        self._reset_orchestrator()

    def _reset_orchestrator(self) -> None:
        self.orch = Orchestrator(
            llm=self.llm,
            mode=self.mode,
            use_real_embeddings=True,
            sandbox_enabled=True,
            memory_db_path=self.db_path,
            run_options=self.run_options,
        )

    def clear_memory(self) -> None:
        for suffix in ("", "-shm", "-wal"):
            p = self.db_path + suffix
            try:
                os.unlink(p)
            except OSError:
                pass
        self._reset_orchestrator()
        self.turn = 0

    def ask(self, user_text: str) -> Tuple[str, Dict[str, Any]]:
        """Run one user message through the full agent pipeline."""
        self.turn += 1
        self.llm.reset_stats()
        task_id = f"chat_{self.session_id}_{self.turn:04d}"
        tags = self.default_tags or infer_tags(user_text)

        if self.mode == "text":
            result = self.orch.execute_task_text_mode(
                task_id=task_id,
                task_description=user_text,
                tags=tags,
            )
        else:
            result = self.orch.execute_task(
                task_id=task_id,
                task_description=user_text,
                tags=tags,
            )

        stats = self.llm.get_usage_stats()
        result["_chat_stats"] = {
            "tokens": stats["total_prompt_tokens"] + stats["total_completion_tokens"],
            "api_calls": stats["call_count"],
            "elapsed_ms": result.get("elapsed_ms", 0),
        }
        return format_answer(result), result

    def usage_footer(self, result: Dict[str, Any]) -> str:
        s = result.get("_chat_stats", {})
        mem = self.orch.memory_store.get_stats()
        return (
            f"本轮：{s.get('tokens', 0)} tokens · {s.get('api_calls', 0)} 次 API · "
            f"{s.get('elapsed_ms', 0):.0f} ms · 记忆库 {mem.get('total_memories', 0)} 条"
        )
