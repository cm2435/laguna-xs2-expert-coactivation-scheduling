from __future__ import annotations

import subprocess

from densify.tasks.manifest import load_task_manifest
from densify.tasks.pool_runner import pool_prompt, run_pool_rollout


def test_pool_prompt_includes_problem_statement(tmp_path):
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        """
task_id: tiny
suite: synthetic
repo: owner/repo
repo_id: owner__repo
base_commit: local
problem_statement: Fix the frobnicator.
environment:
  template_path: /tmp/repo
grader:
  hidden_command: echo hidden
limits:
  timeout_s: 10
  max_turns: 2
""",
        encoding="utf-8",
    )
    assert "Fix the frobnicator" in pool_prompt(load_task_manifest(task_path))


def test_run_pool_rollout_records_missing_pool(tmp_path):
    template = tmp_path / "template" / "repo"
    template.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=template, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=template, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=template, check=True)
    (template / "hello.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "hello.py"], cwd=template, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=template, check=True, capture_output=True)

    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        f"""
task_id: tiny
suite: synthetic
repo: owner/repo
repo_id: owner__repo
base_commit: local
problem_statement: Fix it.
environment:
  template_path: {template}
grader:
  hidden_command: echo hidden
limits:
  timeout_s: 10
  max_turns: 2
""",
        encoding="utf-8",
    )

    result = run_pool_rollout(
        task=load_task_manifest(task_path),
        api_url="http://127.0.0.1:1/v1",
        output_dir=tmp_path / "runs",
        sandbox_root=tmp_path / "sandboxes",
        run_id="run1",
        pool_binary="definitely-missing-pool-binary",
    )

    assert result.exit_code == 127
    assert (result.rollout_dir / "rollout_summary.json").exists()
