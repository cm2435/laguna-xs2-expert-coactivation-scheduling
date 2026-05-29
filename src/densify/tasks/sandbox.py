from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from densify.run_artifacts import write_json
from densify.tasks.manifest import TaskManifest


@dataclass(frozen=True)
class Sandbox:
    run_id: str
    root: Path
    repo: Path
    task_path: Path


def write_poolside_settings(repo: Path) -> Path:
    repo_path = repo.resolve()
    settings_dir = repo / ".poolside"
    settings_dir.mkdir(parents=True, exist_ok=True)
    settings_path = settings_dir / "settings.local.yaml"
    settings_path.write_text(
        f"""tools:
  shell:
    disabled: false
    allow:
      - "*"
paths:
  allow:
    - path: {repo_path}/**
      write: true
""",
        encoding="utf-8",
    )
    return settings_path


def prepare_sandbox(task: TaskManifest, run_id: str, output_root: str | Path) -> Sandbox:
    root = Path(output_root) / run_id
    repo_dst = root / "repo"
    if root.exists():
        raise FileExistsError(f"Sandbox already exists: {root}")
    root.mkdir(parents=True)
    shutil.copytree(task.environment.template_path, repo_dst, symlinks=True)
    write_poolside_settings(repo_dst)
    task_dst = root / "task.yaml"
    shutil.copy2(task.path, task_dst)
    write_json(
        root / "metadata.json",
        {
            "run_id": run_id,
            "task_id": task.task_id,
            "repo": task.repo,
            "repo_id": task.repo_id,
            "base_commit": task.base_commit,
        },
    )
    return Sandbox(run_id=run_id, root=root, repo=repo_dst, task_path=task_dst)
