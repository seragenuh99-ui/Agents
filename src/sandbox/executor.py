"""沙盒代码执行 — AST 安全验证 + 子进程隔离。

安全层次（双重防御）：
1. AST 级别：CodeValidator 拦截危险导入和函数调用
2. OS 级别：subprocess 子进程隔离（受限环境变量、/tmp 工作目录、10s 超时）

使用方式：
- validate_code(code) → (is_safe, violations)     # 仅 AST 验证
- ExecutorAgent._run_sandbox() 调用 validate_code + subprocess  # 完整沙盒
- SandboxExecutor / SandboxSession → 备选执行模型（测试用）
"""

from __future__ import annotations

import ast
import sys
import time
import traceback
from io import StringIO
from typing import Any, Dict, Set


# ═══════════════════════════════════════════════════════════════════════════════
# CodeValidator — AST 级别的代码安全验证
# ═══════════════════════════════════════════════════════════════════════════════

class CodeValidator(ast.NodeVisitor):
    """基于 AST 的代码验证器，拦截危险操作。

    遍历 Python AST 树，检测：
    - 危险模块导入（os, subprocess, socket 等）
    - 危险函数调用（eval, exec, open 等）
    """

    # 危险函数：可执行任意代码或访问文件系统
    DANGEROUS_FUNCTIONS: Set[str] = {
        "eval", "exec", "compile", "__import__", "open",
        "os.system", "subprocess.call", "subprocess.run", "subprocess.Popen",
    }

    # 危险模块：可绕过沙盒或访问网络/文件系统
    DANGEROUS_MODULES: Set[str] = {
        "os", "subprocess", "shutil", "sys", "ctypes", "socket",
        "requests", "urllib", "http", "ftp", "ftplib",
    }

    def __init__(self):
        self.violations: list = []  # 违规记录列表

    def visit_Import(self, node: ast.Import) -> None:
        """拦截 import xxx 语句中的危险模块。"""
        for alias in node.names:
            if alias.name.split(".")[0] in self.DANGEROUS_MODULES:
                self.violations.append(f"Blocked import: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """拦截 from xxx import yyy 语句中的危险模块。"""
        if node.module and node.module.split(".")[0] in self.DANGEROUS_MODULES:
            self.violations.append(f"Blocked import from: {node.module}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """拦截危险函数调用（eval/exec/open 等）。"""
        if isinstance(node.func, ast.Name):
            if node.func.id in self.DANGEROUS_FUNCTIONS:
                self.violations.append(f"Blocked function call: {node.func.id}")
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr in self.DANGEROUS_FUNCTIONS:
                self.violations.append(f"Blocked function call: {node.func.attr}")
        self.generic_visit(node)


def validate_code(code: str) -> tuple:
    """验证 Python 代码安全性。

    先解析 AST（捕获语法错误），再遍历节点检测危险操作。

    Returns:
        (is_safe: bool, violations: list[str]) — is_safe 为 True 表示代码安全
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, [f"Syntax error: {e}"]

    validator = CodeValidator()
    validator.visit(tree)
    return len(validator.violations) == 0, validator.violations


# ═══════════════════════════════════════════════════════════════════════════════
# SandboxExecutor — 进程内受限执行（备选模型）
# ═══════════════════════════════════════════════════════════════════════════════

class SandboxExecutor:
    """轻量级进程内沙盒执行器。

    通过受限 builtins + stdout/stderr 捕获实现隔离。
    注意：生产代码使用 ExecutorAgent._run_sandbox() 的 subprocess 方案
    （AST 验证 + OS 级隔离），本类为备选执行模型，主要用于测试。

    特性：
    - AST 安全验证
    - 受限内置函数（无 open/eval/exec）
    - 输出截断（防止内存溢出）
    """

    def __init__(self, timeout_sec: float = 10.0, max_output_chars: int = 100000):
        self.timeout_sec = timeout_sec
        self.max_output_chars = max_output_chars
        self.execution_count = 0

    def execute(self, code: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """在受限环境中执行代码并返回结果。

        流程：AST 验证 → 设置受限 globals → exec() 执行 → 收集输出和变量
        """
        self.execution_count += 1
        t0 = time.time()

        # 第1层：AST 安全验证
        is_safe, violations = validate_code(code)
        if not is_safe:
            return {
                "success": False,
                "output": "",
                "error": f"Code validation failed: {'; '.join(violations)}",
                "execution_time_ms": (time.time() - t0) * 1000,
            }

        # 第2层：受限内置函数白名单
        safe_builtins = {
            "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
            "enumerate": enumerate, "filter": filter, "float": float,
            "int": int, "len": len, "list": list, "map": map,
            "max": max, "min": min, "print": print, "range": range,
            "reversed": reversed, "round": round, "set": set,
            "sorted": sorted, "str": str, "sum": sum, "tuple": tuple,
            "type": type, "zip": zip, "isinstance": isinstance,
            "True": True, "False": False, "None": None,
            "__import__": __import__,
        }

        globals_dict = {"__builtins__": safe_builtins}
        locals_dict = {}
        if context:
            locals_dict.update(context)

        # 捕获 stdout/stderr
        stdout = StringIO()
        stderr = StringIO()

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = stdout
        sys.stderr = stderr

        try:
            exec(compile(code, "<sandbox>", "exec"), globals_dict, locals_dict)
            output = stdout.getvalue()
            error_output = stderr.getvalue()

            if len(output) > self.max_output_chars:
                output = output[:self.max_output_chars] + "\n... [truncated]"

            return {
                "success": True,
                "output": output,
                "error": error_output or None,
                "execution_time_ms": (time.time() - t0) * 1000,
                "result_vars": {
                    k: str(v)[:500]
                    for k, v in locals_dict.items()
                    if not k.startswith("_") and k not in (context or {})
                },
            }
        except Exception as e:
            return {
                "success": False,
                "output": stdout.getvalue(),
                "error": f"{type(e).__name__}: {str(e)}\n{stderr.getvalue()}",
                "execution_time_ms": (time.time() - t0) * 1000,
            }
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


# ═══════════════════════════════════════════════════════════════════════════════
# SandboxSession — 持久化沙盒会话（备选模型）
# ═══════════════════════════════════════════════════════════════════════════════

class SandboxSession:
    """持久化子进程沙盒，变量在多次调用间保持。

    通过 stdin/stdout 管道与长期运行的 Python 子进程通信，
    避免每次执行 ~30ms 的冷启动开销。变量跨调用保持：
    x=1 在一次调用中设置，后续 y=x+1 可直接使用。

    注意：生产代码使用 ExecutorAgent._run_sandbox() 的临时子进程方案
    （每次新建进程，无状态残留风险），本类为备选执行模型，主要用于测试。
    """

    def __init__(self, timeout_sec: float = 10.0):
        self.timeout_sec = timeout_sec
        self._proc = None
        self._start()

    def _start(self) -> None:
        """启动持久化 Python 子进程，运行 REPL 风格的主循环。

        子进程通过 JSON 行协议通信：
        - 输入：{"code": "..."}  或  {"_cmd": "exit"}
        - 输出：{"ok": true/false, "error": "...", "vars": {...}}
        """
        code = r"""
import sys, json, traceback
_namespace = {}
while True:
    try:
        line = sys.stdin.readline()
        if not line:
            break
        cmd = json.loads(line)
        if cmd.get('_cmd') == 'exit':
            break
        src = cmd['code']
        try:
            exec(src, _namespace)
            result = {'ok': True}
        except Exception as e:
            result = {'ok': False, 'error': str(e), 'traceback': traceback.format_exc()}
        result['vars'] = {k: repr(v) for k, v in _namespace.items()
                          if not k.startswith('_')}
        sys.stdout.write(json.dumps(result) + '\n')
        sys.stdout.flush()
    except Exception:
        pass
"""
        import subprocess
        self._proc = subprocess.Popen(
            [sys.executable, "-c", code],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def execute(self, code: str) -> Dict[str, Any]:
        """在持久化会话中执行代码，变量跨调用保持。"""
        is_safe, violations = validate_code(code)
        if not is_safe:
            return {
                "success": False,
                "error": f"Code validation failed: {'; '.join(violations)}",
            }

        if self._proc is None:
            self._start()

        try:
            import json
            payload = json.dumps({"code": code}) + "\n"
            self._proc.stdin.write(payload)
            self._proc.stdin.flush()

            response = self._proc.stdout.readline()
            if not response:
                self._start()  # 进程已死，重启
                return {"success": False, "error": "Sandbox process terminated"}

            result = json.loads(response)
            return {
                "success": result.get("ok", False),
                "error": result.get("error"),
                "result_vars": result.get("vars", {}),
            }
        except Exception as e:
            return {"success": False, "error": f"Session error: {e}"}

    def close(self) -> None:
        """终止沙盒会话，清理子进程。"""
        if self._proc:
            try:
                import json
                self._proc.stdin.write(json.dumps({"_cmd": "exit"}) + "\n")
                self._proc.stdin.flush()
                self._proc.wait(timeout=2)
            except Exception:
                self._proc.kill()
            self._proc = None
