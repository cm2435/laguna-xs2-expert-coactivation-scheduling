from __future__ import annotations

import ast
import contextlib
import io
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class CodeScore:
    extracted_code: str
    parse_ok: bool
    tests_ok: bool
    test_stdout: str
    test_stderr: str


CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_python_code(text: str) -> str:
    match = CODE_BLOCK_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def parse_ok(code: str) -> bool:
    try:
        ast.parse(code)
    except SyntaxError:
        return False
    return True


def score_code_generation(text: str, tests: list[str]) -> CodeScore:
    code = extract_python_code(text)
    parsed = parse_ok(code)
    if not parsed:
        return CodeScore(code, False, False, "", "syntax error")

    namespace: dict[str, object] = {}
    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exec(code, namespace)
            for test in tests:
                exec(test, namespace)
    except Exception as exc:
        return CodeScore(code, True, False, stdout.getvalue(), f"{type(exc).__name__}: {exc}")

    return CodeScore(code, True, True, stdout.getvalue(), stderr.getvalue())
