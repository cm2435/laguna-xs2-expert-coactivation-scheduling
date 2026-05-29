from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TaskEnvironment:
    template_path: Path
    setup_command: str
    smoke_command: str


@dataclass(frozen=True)
class TaskGrader:
    hidden_command: str
    public_command: str = ""


@dataclass(frozen=True)
class TaskLimits:
    timeout_s: int
    max_turns: int
    network: str = "disabled"


@dataclass(frozen=True)
class TaskManifest:
    task_id: str
    suite: str
    repo: str
    repo_id: str
    base_commit: str
    problem_statement: str
    environment: TaskEnvironment
    grader: TaskGrader
    limits: TaskLimits
    path: Path


def load_task_manifest(path: str | Path) -> TaskManifest:
    manifest_path = Path(path)
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected task manifest mapping: {path}")
    env = raw.get("environment", {})
    grader = raw.get("grader", {})
    limits = raw.get("limits", {})
    return TaskManifest(
        task_id=str(raw["task_id"]),
        suite=str(raw["suite"]),
        repo=str(raw["repo"]),
        repo_id=str(raw["repo_id"]),
        base_commit=str(raw["base_commit"]),
        problem_statement=str(raw.get("problem_statement", "")),
        environment=TaskEnvironment(
            template_path=Path(env["template_path"]),
            setup_command=str(env.get("setup_command", "")),
            smoke_command=str(env.get("smoke_command", "")),
        ),
        grader=TaskGrader(
            hidden_command=str(grader.get("hidden_command", "")),
            public_command=str(grader.get("public_command", "")),
        ),
        limits=TaskLimits(
            timeout_s=int(limits.get("timeout_s", 1800)),
            max_turns=int(limits.get("max_turns", 30)),
            network=str(limits.get("network", "disabled")),
        ),
        path=manifest_path,
    )


def iter_registry(registry_path: str | Path) -> list[Path]:
    paths: list[Path] = []
    with Path(registry_path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row: dict[str, Any] = json.loads(line)
            paths.append(Path(row["manifest"]))
    return paths
