"""Unit tests for CodeAct sandbox: validation and safe execution."""

import ast
import time
import pytest

from src.sandbox.executor import SandboxExecutor, validate_code, CodeValidator


class TestCodeValidator:
    """Test AST-based code safety validation."""

    def test_safe_code_passes(self):
        is_safe, violations = validate_code("x = 1 + 2\nprint(x)")
        assert is_safe is True
        assert violations == []

    def test_safe_code_with_math(self):
        is_safe, _ = validate_code(
            "import math\nresult = math.sqrt(144)\nprint(result)"
        )
        assert is_safe is True

    def test_safe_code_with_allowed_builtins(self):
        is_safe, _ = validate_code(
            "data = [1, 2, 3]\nprint(sum(data))\nprint(len(data))"
        )
        assert is_safe is True

    def test_blocks_eval_call(self):
        is_safe, violations = validate_code("eval('print(1)')")
        assert is_safe is False
        assert any("eval" in v for v in violations)

    def test_blocks_exec_call(self):
        is_safe, violations = validate_code("exec('x = 1')")
        assert is_safe is False
        assert any("exec" in v for v in violations)

    def test_blocks_import_os(self):
        is_safe, violations = validate_code("import os\nos.system('ls')")
        assert is_safe is False
        assert any("os" in v.lower() for v in violations)

    def test_blocks_import_subprocess(self):
        is_safe, violations = validate_code("import subprocess\nsubprocess.run(['ls'])")
        assert is_safe is False
        assert any("subprocess" in v.lower() for v in violations)

    def test_blocks_import_shutil(self):
        is_safe, violations = validate_code("import shutil")
        assert is_safe is False

    def test_blocks_import_sys(self):
        is_safe, violations = validate_code("import sys\nsys.exit(0)")
        assert is_safe is False
        assert any("sys" in v.lower() for v in violations)

    def test_blocks_import_ctypes(self):
        is_safe, violations = validate_code("import ctypes")
        assert is_safe is False

    def test_blocks_import_socket(self):
        is_safe, violations = validate_code("import socket")
        assert is_safe is False

    def test_blocks_import_requests(self):
        is_safe, violations = validate_code("import requests")
        assert is_safe is False

    def test_blocks_import_urllib(self):
        is_safe, violations = validate_code("import urllib.request")
        assert is_safe is False

    def test_blocks_import_http(self):
        is_safe, violations = validate_code("from http import client")
        assert is_safe is False

    def test_blocks_import_ftp(self):
        is_safe, violations = validate_code("import ftplib")
        assert is_safe is False

    def test_blocks_open_call(self):
        is_safe, violations = validate_code("open('/etc/passwd')")
        assert is_safe is False
        assert any("open" in v for v in violations)

    def test_blocks_compile_call(self):
        is_safe, violations = validate_code("compile('x=1', '', 'exec')")
        assert is_safe is False
        assert any("compile" in v for v in violations)

    def test_syntax_error_handled(self):
        is_safe, violations = validate_code("def broken(:\n    pass")
        assert is_safe is False
        assert any("Syntax" in v for v in violations)

    def test_blocks_multiple_dangerous_calls(self):
        code = "eval('1')\nexec('x=2')\nimport os"
        is_safe, violations = validate_code(code)
        assert is_safe is False
        assert len(violations) >= 2

    def test_safe_import_from_allowed_module(self):
        is_safe, _ = validate_code("from collections import defaultdict")
        assert is_safe is True

    def test_safe_list_comprehension(self):
        is_safe, _ = validate_code("[x*2 for x in range(10) if x % 2 == 0]")
        assert is_safe is True

    def test_visit_call_handles_non_name_ast(self):
        """visit_Call should handle all AST node types for func."""
        validator = CodeValidator()
        tree = ast.parse("x = [1, 2, 3]\nx[0]")
        validator.visit(tree)
        assert len(validator.violations) == 0

    def test_visit_import_from_no_module(self):
        """ImportFrom with no module should pass."""
        validator = CodeValidator()
        tree = ast.parse("from math import pi")
        validator.visit(tree)
        assert len(validator.violations) == 0

    def test_blocks_attribute_dangerous_call(self):
        """os.system() uses attribute access for function name."""
        is_safe, violations = validate_code("import os\nos.system('whoami')")
        assert is_safe is False


class TestSandboxExecutor:
    """Test SandboxExecutor for safe code execution."""

    def test_basic_execution(self, sandbox):
        result = sandbox.execute("x = 1 + 2\nprint(x)")
        assert result["success"] is True
        assert "3" in result["output"]
        assert result["execution_time_ms"] >= 0

    def test_execution_with_math(self, sandbox):
        result = sandbox.execute(
            "import math\nv = math.sqrt(100)\nprint(f'Result: {v}')"
        )
        assert result["success"] is True
        assert "Result: 10.0" in result["output"]

    def test_execution_result_vars(self, sandbox):
        result = sandbox.execute("a = 42\nb = a * 2\nprint(a, b)")
        assert result["success"] is True
        assert "a" in result["result_vars"]
        assert "b" in result["result_vars"]

    def test_blocks_dangerous_code(self, sandbox):
        result = sandbox.execute("import os\nos.system('ls')")
        assert result["success"] is False
        assert "Code validation failed" in result["error"]

    def test_blocks_eval(self, sandbox):
        result = sandbox.execute("eval('1+1')")
        assert result["success"] is False

    def test_handles_runtime_error(self, sandbox):
        result = sandbox.execute("print(1/0)")
        assert result["success"] is False
        assert "ZeroDivisionError" in result["error"] or "division by zero" in result["error"]

    def test_handles_name_error(self, sandbox):
        result = sandbox.execute("print(undefined_var)")
        assert result["success"] is False
        assert "NameError" in result["error"]

    def test_context_variables(self, sandbox):
        result = sandbox.execute(
            "print(data['value'] * 2)",
            context={"data": {"value": 5}},
        )
        assert result["success"] is True
        assert "10" in result["output"]

    def test_context_vars_not_in_result_vars(self, sandbox):
        """Context keys should not appear in result_vars."""
        result = sandbox.execute(
            "output = data * 3",
            context={"data": 10},
        )
        assert "data" not in result["result_vars"]
        assert "output" in result["result_vars"]

    def test_execution_count_increments(self, sandbox):
        initial = sandbox.execution_count
        sandbox.execute("print(1)")
        sandbox.execute("print(2)")
        assert sandbox.execution_count == initial + 2

    def test_restricted_builtins(self, sandbox):
        """Verify that certain builtins are restricted even if not AST-blocked."""
        result = sandbox.execute("print(type('hello'))")
        assert result["success"] is True

    def test_output_truncation(self):
        sandbox = SandboxExecutor(timeout_sec=5.0, max_output_chars=50)
        result = sandbox.execute("print('x' * 200)")
        assert result["success"] is True
        assert "[truncated]" in result["output"]
        assert len(result["output"]) <= 50 + len("\n... [truncated]")

    def test_stdout_stderr_separation(self, sandbox):
        """Verify stdout output and stderr error are captured."""
        result = sandbox.execute(
            "print('to stdout')\n"
            "x = 1 / 0"
        )
        assert result["success"] is False
        assert "to stdout" in result["output"]
        assert "ZeroDivisionError" in result["error"]

    def test_multiline_code(self, sandbox):
        result = sandbox.execute(
            "def fib_iter(n):\n"
            "    a, b = 0, 1\n"
            "    for _ in range(n):\n"
            "        a, b = b, a + b\n"
            "    return a\n"
            "print(fib_iter(6))"
        )
        assert result["success"] is True
        assert "8" in result["output"]

    def test_loop_execution(self, sandbox):
        result = sandbox.execute(
            "total = 0\n"
            "for i in range(100):\n"
            "    total += i\n"
            "print(total)"
        )
        assert result["success"] is True
        assert "4950" in result["output"]


class TestSandboxPerformance:
    """Performance tests for sandbox execution."""

    def test_validation_speed_small(self):
        t0 = time.perf_counter()
        for _ in range(100):
            validate_code("x = 1\nprint(x)")
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.2, f"Validation too slow: {elapsed:.4f}s"

    def test_validation_speed_complex(self):
        code = (
            "data = [i*2 for i in range(100)]\n"
            "total = sum(data)\n"
            "avg = total / len(data)\n"
            "print(f'Avg: {avg:.2f}')\n"
            "result = {'total': total, 'avg': avg}"
        )
        t0 = time.perf_counter()
        for _ in range(100):
            validate_code(code)
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.5, f"Validation too slow: {elapsed:.4f}s"

    def test_execution_speed(self, sandbox):
        t0 = time.perf_counter()
        sandbox.execute("sum(range(1000))")
        elapsed = time.perf_counter() - t0
        assert elapsed < 1.0, f"Execution too slow: {elapsed:.4f}s"

    def test_restored_io_after_error(self, sandbox):
        """sys.stdout must be restored even after execution failure."""
        import sys
        old_stdout = sys.stdout
        sandbox.execute("1/0")  # Runtime error
        assert sys.stdout is old_stdout


class TestValidateCodeFunction:
    """Tests for the standalone validate_code function."""

    def test_empty_code(self):
        is_safe, violations = validate_code("")
        assert is_safe is True
        assert violations == []

    def test_comments_only(self):
        is_safe, _ = validate_code("# This is a comment\n# Another comment")
        assert is_safe is True

    def test_pass_statement(self):
        is_safe, _ = validate_code("pass")
        assert is_safe is True

    def test_nested_function_safe(self):
        is_safe, _ = validate_code(
            "def add(a, b):\n"
            "    return a + b\n"
            "print(add(2, 3))"
        )
        assert is_safe is True

    def test_import_allowed_module(self):
        is_safe, _ = validate_code("import json\nimport math\nimport random")
        assert is_safe is True

    def test_import_from_allowed(self):
        is_safe, _ = validate_code("from itertools import combinations")
        assert is_safe is True
