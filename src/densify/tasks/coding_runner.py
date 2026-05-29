from __future__ import annotations

from densify.coding_harness.runner import HarnessConfig, run_coding_harness
from densify.run_artifacts import write_json
from densify.tasks.grader import grade_sandbox
from densify.tasks.manifest import load_task_manifest
from densify.tasks.pool_runner import make_run_id
from densify.tasks.sandbox import prepare_sandbox


def run_task_rollout(
    *,
    task_path: str,
    api_url: str,
    model: str = "laguna",
    output_dir: str = "runs/coding_harness_rollouts",
    sandbox_root: str = "sandboxes/coding_harness",
    run_id: str | None = None,
    max_turns: int = 40,
    temperature: float = 0.0,
) -> str:
    task = load_task_manifest(task_path)
    rid = run_id or make_run_id(task.task_id)
    sandbox = prepare_sandbox(task, rid, sandbox_root)
    rollout_dir = f"{output_dir}/{rid}"

    result = run_coding_harness(
        HarnessConfig(
            repo=sandbox.repo,
            output_dir=rollout_dir,
            api_url=api_url,
            model=model,
            task=task.problem_statement,
            max_turns=max_turns,
            temperature=temperature,
        )
    )
    grade = grade_sandbox(task, sandbox.root)
    write_json(
        f"{rollout_dir}/rollout_summary.json",
        {
            "run_id": rid,
            "task_id": task.task_id,
            "repo": task.repo,
            "sandbox_root": sandbox.root,
            "harness_success": result.success,
            "turns": result.turns,
            "grade_result": sandbox.root / "grade_result.json",
            "patch_path": grade.patch_path,
        },
    )
    return rollout_dir
