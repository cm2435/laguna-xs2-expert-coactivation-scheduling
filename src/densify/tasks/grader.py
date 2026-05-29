from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from densify.run_artifacts import write_json
from densify.tasks.manifest import TaskManifest


@dataclass(frozen=True)
class GradeResult:
    task_id: str
    status: str
    repo_dirty: bool
    patch_path: Path
    public_exit_code: int | None
    hidden_exit_code: int | None


def export_patch(repo_path: str | Path, patch_path: str | Path) -> None:
    result = subprocess.run(
        ["git", "-C", str(repo_path), "diff", "--binary"],
        check=True,
        capture_output=True,
        text=True,
    )
    Path(patch_path).write_text(result.stdout, encoding="utf-8")


def repo_is_dirty(repo_path: str | Path) -> bool:
    result = subprocess.run(
        ["git", "-C", str(repo_path), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def grade_sandbox(task: TaskManifest, sandbox_root: str | Path) -> GradeResult:
    root = Path(sandbox_root)
    repo = root / "repo"
    patch_path = root / "patch.diff"
    export_patch(repo, patch_path)
    public_exit_code: int | None = None
    if task.grader.public_command:
        public = subprocess.run(task.grader.public_command, cwd=repo, shell=True, check=False)
        public_exit_code = public.returncode

    result = GradeResult(
        task_id=task.task_id,
        status="not_graded_yet",
        repo_dirty=repo_is_dirty(repo),
        patch_path=patch_path,
        public_exit_code=public_exit_code,
        hidden_exit_code=None,
    )
    write_json(root / "grade_result.json", result)
    return result
