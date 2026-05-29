from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from densify.run_artifacts import write_json
from densify.tasks.grader import grade_sandbox
from densify.tasks.manifest import TaskManifest, iter_registry, load_task_manifest
from densify.tasks.sandbox import prepare_sandbox


@dataclass(frozen=True)
class PoolRolloutResult:
    run_id: str
    task_id: str
    sandbox_root: Path
    exit_code: int
    rollout_dir: Path


def make_run_id(task_id: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{task_id}"


def pool_prompt(task: TaskManifest) -> str:
    return (
        "We need solve this repository task. Inspect the codebase, make the minimal fix, "
        "and leave the final patch in the working tree.\n\n"
        f"Issue:\n{task.problem_statement}\n"
    )


def run_pool_rollout(
    *,
    task: TaskManifest,
    api_url: str,
    output_dir: str | Path,
    sandbox_root: str | Path = "sandboxes/pool_runs",
    run_id: str | None = None,
    pool_binary: str = "pool",
) -> PoolRolloutResult:
    rid = run_id or make_run_id(task.task_id)
    rollout_dir = Path(output_dir) / rid
    rollout_dir.mkdir(parents=True, exist_ok=False)
    sandbox = prepare_sandbox(task, rid, sandbox_root)

    command = [
        pool_binary,
        "exec",
        "--sandbox",
        "disabled",
        "--api-url",
        api_url,
        "-a",
        "default",
        "-p",
        pool_prompt(task),
        "-o",
        "json",
    ]
    (rollout_dir / "pool_command.sh").write_text(" ".join(command) + "\n", encoding="utf-8")

    if shutil.which(pool_binary) is None:
        write_json(
            rollout_dir / "rollout_summary.json",
            {
                "run_id": rid,
                "task_id": task.task_id,
                "status": "pool_not_found",
                "sandbox_root": str(sandbox.root),
            },
        )
        grade_sandbox(task, sandbox.root)
        return PoolRolloutResult(rid, task.task_id, sandbox.root, 127, rollout_dir)

    result = subprocess.run(
        command,
        cwd=sandbox.repo,
        check=False,
        capture_output=True,
        text=True,
        env={"POOLSIDE_API_KEY": "dummy", **dict(__import__("os").environ)},
        timeout=task.limits.timeout_s,
    )
    (rollout_dir / "pool_stdout.json").write_text(result.stdout, encoding="utf-8")
    (rollout_dir / "pool_stderr.txt").write_text(result.stderr, encoding="utf-8")
    grade = grade_sandbox(task, sandbox.root)
    write_json(
        rollout_dir / "rollout_summary.json",
        {
            "run_id": rid,
            "task_id": task.task_id,
            "repo": task.repo,
            "sandbox_root": str(sandbox.root),
            "exit_code": result.returncode,
            "grade_result": str(sandbox.root / "grade_result.json"),
            "patch_path": str(grade.patch_path),
        },
    )
    return PoolRolloutResult(rid, task.task_id, sandbox.root, result.returncode, rollout_dir)


def run_rollout_batch(
    *,
    registry_path: str | Path,
    api_url: str,
    output_dir: str | Path,
    limit: int | None = None,
) -> list[PoolRolloutResult]:
    results: list[PoolRolloutResult] = []
    manifests = iter_registry(registry_path)
    if limit is not None:
        manifests = manifests[:limit]
    for manifest_path in manifests:
        task = load_task_manifest(manifest_path)
        results.append(run_pool_rollout(task=task, api_url=api_url, output_dir=output_dir))
    return results
