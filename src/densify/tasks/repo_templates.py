from __future__ import annotations

import subprocess
from pathlib import Path

from densify.run_artifacts import write_json
from densify.tasks.manifest import TaskManifest, iter_registry, load_task_manifest


def prepare_repo_template(task: TaskManifest) -> Path:
    repo_path = task.environment.template_path
    if (repo_path / ".git").exists():
        return repo_path

    repo_path.parent.mkdir(parents=True, exist_ok=True)
    clone_url = f"https://github.com/{task.repo}.git"
    subprocess.run(["git", "clone", clone_url, str(repo_path)], check=True)
    subprocess.run(["git", "-C", str(repo_path), "checkout", task.base_commit], check=True)
    write_json(
        repo_path.parent / "metadata.json",
        {
            "task_id": task.task_id,
            "repo": task.repo,
            "repo_id": task.repo_id,
            "base_commit": task.base_commit,
            "clone_url": clone_url,
        },
    )
    return repo_path


def prepare_repo_templates_from_registry(
    registry_path: str | Path,
    *,
    offset: int = 0,
    limit: int | None = None,
) -> list[Path]:
    prepared: list[Path] = []
    seen: set[Path] = set()
    manifest_paths = iter_registry(registry_path)
    if offset:
        manifest_paths = manifest_paths[offset:]
    if limit is not None:
        manifest_paths = manifest_paths[:limit]

    for manifest_path in manifest_paths:
        task = load_task_manifest(manifest_path)
        if task.environment.template_path in seen:
            continue
        prepared.append(prepare_repo_template(task))
        seen.add(task.environment.template_path)
    return prepared
