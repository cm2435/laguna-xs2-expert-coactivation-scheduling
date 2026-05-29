from __future__ import annotations

import subprocess
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ToolResult:
    tool_name: str
    ok: bool
    output: str
    exit_code: int | None = None


class ToolExecutor:
    BLOCKED_COMMAND_PREFIXES = (
        ("pip", "install"),
        ("pip3", "install"),
        ("python", "-m", "pip", "install"),
        ("python3", "-m", "pip", "install"),
        ("uv", "pip", "install"),
        ("apt", "install"),
        ("apt-get", "install"),
        ("apt-get", "update"),
        ("conda", "install"),
        ("mamba", "install"),
    )

    def __init__(
        self,
        repo: str | Path,
        *,
        command_timeout_s: int = 120,
        output_limit_bytes: int = 20000,
    ) -> None:
        self.repo = Path(repo).resolve()
        self.command_timeout_s = command_timeout_s
        self.output_limit_bytes = output_limit_bytes

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        if tool_name == "shell":
            return self.shell(str(arguments.get("command", "")))
        if tool_name == "read_file":
            return self.read_file(
                str(arguments.get("path", "")),
                start_line=arguments.get("start_line"),
                max_lines=arguments.get("max_lines"),
            )
        if tool_name == "write_file":
            return self.write_file(str(arguments.get("path", "")), str(arguments.get("content", "")))
        if tool_name == "apply_patch":
            return self.apply_patch(str(arguments.get("patch", "")))
        if tool_name == "exit":
            return ToolResult("exit", True, str(arguments.get("summary", "")))
        return ToolResult(tool_name, False, f"unknown tool: {tool_name}")

    def shell(self, command: str) -> ToolResult:
        if not command.strip():
            return ToolResult("shell", False, "missing command")
        if self.is_blocked_shell_command(command):
            return ToolResult(
                "shell",
                False,
                "blocked command: package manager and system install commands are disabled "
                "for rollouts. Inspect and edit the repo using existing tools.",
            )
        try:
            result = subprocess.run(
                command,
                cwd=self.repo,
                shell=True,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.command_timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            output = (exc.stdout or "") + (exc.stderr or "")
            return ToolResult("shell", False, self.limit_output(output + "\n[TIMEOUT]"), None)
        output = self.limit_output((result.stdout or "") + (result.stderr or ""))
        return ToolResult("shell", result.returncode == 0, output, result.returncode)

    def read_file(
        self,
        path: str,
        *,
        start_line: Any = None,
        max_lines: Any = None,
    ) -> ToolResult:
        target = self.resolve_repo_path(path)
        if target is None or not target.is_file():
            return ToolResult("read_file", False, f"file not found or outside repo: {path}")
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(int(start_line or 1), 1)
        limit = int(max_lines or 200)
        selected = lines[start - 1 : start - 1 + limit]
        rendered = "\n".join(f"{idx}: {line}" for idx, line in enumerate(selected, start=start))
        return ToolResult("read_file", True, self.limit_output(rendered))

    def write_file(self, path: str, content: str) -> ToolResult:
        target = self.resolve_repo_path(path)
        if target is None:
            return ToolResult("write_file", False, f"path outside repo: {path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return ToolResult("write_file", True, f"wrote {target.relative_to(self.repo)}")

    def apply_patch(self, patch: str) -> ToolResult:
        try:
            changed = self.apply_simple_patch(patch)
        except ValueError as exc:
            return ToolResult("apply_patch", False, str(exc))
        return ToolResult("apply_patch", True, "updated " + ", ".join(changed))

    def apply_simple_patch(self, patch: str) -> list[str]:
        lines = patch.splitlines()
        if not lines or lines[0].strip() != "*** Begin Patch":
            raise ValueError("patch must start with *** Begin Patch")
        changed: list[str] = []
        i = 1
        while i < len(lines):
            line = lines[i]
            if line.strip() == "*** End Patch":
                return changed
            if not line.startswith("*** Update File: "):
                raise ValueError(f"unsupported patch line: {line}")
            rel_path = line.removeprefix("*** Update File: ").strip()
            target = self.resolve_repo_path(rel_path)
            if target is None or not target.exists():
                raise ValueError(f"cannot update missing/outside file: {rel_path}")
            i += 1
            if i >= len(lines) or not lines[i].startswith("@@"):
                raise ValueError(f"missing hunk for {rel_path}")
            i += 1
            old_block: list[str] = []
            new_block: list[str] = []
            while i < len(lines) and not lines[i].startswith("*** "):
                hunk_line = lines[i]
                if not hunk_line:
                    old_block.append("")
                    new_block.append("")
                elif hunk_line[0] == " ":
                    old_block.append(hunk_line[1:])
                    new_block.append(hunk_line[1:])
                elif hunk_line[0] == "-":
                    old_block.append(hunk_line[1:])
                elif hunk_line[0] == "+":
                    new_block.append(hunk_line[1:])
                else:
                    raise ValueError(f"bad hunk line: {hunk_line}")
                i += 1
            self.replace_block(target, old_block, new_block)
            changed.append(rel_path)
        raise ValueError("patch must end with *** End Patch")

    def replace_block(self, target: Path, old_block: list[str], new_block: list[str]) -> None:
        text = target.read_text(encoding="utf-8")
        old = "\n".join(old_block)
        new = "\n".join(new_block)
        if text.endswith("\n"):
            old += "\n"
            new += "\n"
        if old not in text:
            raise ValueError(f"hunk did not match {target.relative_to(self.repo)}")
        target.write_text(text.replace(old, new, 1), encoding="utf-8")

    def resolve_repo_path(self, path: str) -> Path | None:
        raw = Path(path)
        candidate = raw.resolve() if raw.is_absolute() else (self.repo / raw).resolve()
        try:
            candidate.relative_to(self.repo)
        except ValueError:
            return None
        return candidate

    def is_blocked_shell_command(self, command: str) -> bool:
        try:
            parts = shlex.split(command)
        except ValueError:
            return False
        if not parts:
            return False
        for prefix in self.BLOCKED_COMMAND_PREFIXES:
            if tuple(parts[: len(prefix)]) == prefix:
                return True
        return False

    def limit_output(self, output: str) -> str:
        data = output.encode("utf-8", errors="replace")
        if len(data) <= self.output_limit_bytes:
            return output
        truncated = data[: self.output_limit_bytes].decode("utf-8", errors="replace")
        return truncated + "\n[TRUNCATED]"
