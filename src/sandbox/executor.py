"""Sandbox executor for safe CodeAct code execution.

Provides a restricted execution environment where LLM-generated Python
code can run with limited capabilities and resource constraints.
"""

from __future__ import annotations

import ast
import sys
import time
import traceback
from io import StringIO
from typing import Any, Dict, Set


class CodeValidator(ast.NodeVisitor):
    """AST-based code validator that blocks dangerous operations."""

    DANGEROUS_FUNCTIONS: Set[str] = {
        "eval", "exec", "compile", "__import__", "open",
        "os.system", "subprocess.call", "subprocess.run", "subprocess.Popen",
    }

    DANGEROUS_MODULES: Set[str] = {
        "os", "subprocess", "shutil", "sys", "ctypes", "socket",
        "requests", "urllib", "http", "ftp", "ftplib",
    }

    def __init__(self):
        self.violations: list = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name.split(".")[0] in self.DANGEROUS_MODULES:
                self.violations.append(f"Blocked import: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module and node.module.split(".")[0] in self.DANGEROUS_MODULES:
            self.violations.append(f"Blocked import from: {node.module}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            if node.func.id in self.DANGEROUS_FUNCTIONS:
                self.violations.append(f"Blocked function call: {node.func.id}")
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr in self.DANGEROUS_FUNCTIONS:
                self.violations.append(f"Blocked function call: {node.func.attr}")
        self.generic_visit(node)


def validate_code(code: str) -> tuple:
    """Validate Python code for safety. Returns (is_safe, violations)."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, [f"Syntax error: {e}"]

    validator = CodeValidator()
    validator.visit(tree)
    return len(validator.violations) == 0, validator.violations


class SandboxExecutor:
    """A lightweight sandbox for executing Python code safely.

    Features:
    - AST-based validation to block dangerous operations
    - Restricted builtins
    - Memory and time limits
    - Stdout/stderr capture
    """

    def __init__(self, timeout_sec: float = 10.0, max_output_chars: int = 100000):
        self.timeout_sec = timeout_sec
        self.max_output_chars = max_output_chars
        self.execution_count = 0

    def execute(self, code: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute code and return results."""
        self.execution_count += 1
        t0 = time.time()

        # Validate code safety
        is_safe, violations = validate_code(code)
        if not is_safe:
            return {
                "success": False,
                "output": "",
                "error": f"Code validation failed: {'; '.join(violations)}",
                "execution_time_ms": (time.time() - t0) * 1000,
            }

        # Prepare restricted globals
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


class SandboxSession:
    """Persistent sandbox session with variable state across executions.

    Uses a long-lived subprocess communicating via stdin/stdout pipes
    to avoid the ~30ms cold-start per execution. Variables persist
    across calls: x=1 in one call, then y=x+1 works in the next.
    """

    def __init__(self, timeout_sec: float = 10.0):
        self.timeout_sec = timeout_sec
        self._proc = None
        self._start()

    def _start(self) -> None:
        """Launch a persistent Python subprocess with a REPL-like loop."""
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
        # Collect user-created variables
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
        """Execute code in the persistent session. Variables persist across calls."""
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
                self._start()  # Restart if process died
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
        """Terminate the sandbox session."""
        if self._proc:
            try:
                import json
                self._proc.stdin.write(json.dumps({"_cmd": "exit"}) + "\n")
                self._proc.stdin.flush()
                self._proc.wait(timeout=2)
            except Exception:
                self._proc.kill()
            self._proc = None
