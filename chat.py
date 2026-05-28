#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

"""Interactive chat with 702solver (latest multi-agent pipeline + shared memory).

Usage:
  python3 chat.py
  python3 chat.py --db my_chat.db
  python3 chat.py --mode text          # compare with text-only mode

In-session commands:
  /help              show help
  /quit, /exit       leave
  /clear             wipe session memory (new topic)
  /stats             memory & session info
  /mode structured   switch mode (structured | text)
  /tags solar energy set default tags for following turns
"""

from __future__ import annotations

import argparse

from src.chat.session import ChatSession
from src.paths import DEFAULT_CHAT_SESSION_DB


def print_banner() -> None:
    print(
        """
╔══════════════════════════════════════════════════════════════╗
║  702solver 对话模式 · v0.9 多 Agent + 共享记忆               ║
║  输入问题直接回车；相似问题会复用记忆（E2E/模板）省 Token     ║
║  输入 /help 查看命令                                         ║
╚══════════════════════════════════════════════════════════════╝
""".strip()
    )


def print_help() -> None:
    print(
        """
命令：
  /help                 帮助
  /quit, /exit          退出
  /clear                清空本会话记忆（新话题建议执行）
  /stats                记忆库与配置
  /mode structured|text 切换通信模式
  /tags a b c           设置默认标签（影响缓存域），如 /tags security python

直接输入文字即可提问，将走 Planner → Retriever → Executor → Summarizer 全流程。
""".strip()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="702solver interactive chat")
    parser.add_argument(
        "--db",
        default=DEFAULT_CHAT_SESSION_DB,
        help="Session memory DB path (default: output/databases/chat_session.db)",
    )
    parser.add_argument("--mode", default="structured", choices=["structured", "text"])
    parser.add_argument("--provider", default="deepseek", choices=["deepseek", "openai", "custom"])
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    try:
        session = ChatSession(
            db_path=args.db,
            provider=args.provider,
            model=args.model,
            mode=args.mode,
        )
    except RuntimeError as e:
        print(f"启动失败: {e}")
        return 1

    print_banner()
    print(f"会话 ID: {session.session_id} · 记忆库: {args.db} · 模式: {args.mode}\n")

    while True:
        try:
            user_input = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break

        if not user_input:
            continue

        low = user_input.lower()
        if low in ("/quit", "/exit", "/q"):
            print("再见。")
            break
        if low == "/help":
            print_help()
            continue
        if low == "/clear":
            session.clear_memory()
            print("已清空会话记忆，开始新话题。")
            continue
        if low == "/stats":
            print(session.orch.print_system_status())
            print(f"默认标签: {session.default_tags}")
            continue
        if low.startswith("/mode"):
            parts = user_input.split()
            if len(parts) < 2 or parts[1] not in ("structured", "text"):
                print("用法: /mode structured  或  /mode text")
                continue
            session.mode = parts[1]
            session._reset_orchestrator()
            print(f"已切换为 {parts[1]} 模式。")
            continue
        if low.startswith("/tags"):
            parts = user_input.split()[1:]
            if not parts:
                print(f"当前标签: {session.default_tags}")
            else:
                session.default_tags = parts
                print(f"已设置标签: {session.default_tags}")
            continue

        print("\n处理中…（多 Agent 流水线，首次较慢）\n")
        try:
            answer, result = session.ask(user_input)
        except Exception as e:
            print(f"错误: {e}\n")
            continue

        print("助手>")
        print(answer)
        print()
        print(session.usage_footer(result))
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
