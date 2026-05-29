from __future__ import annotations

import subprocess

from densify.tasks.grader import grade_sandbox
from densify.tasks.manifest import load_task_manifest
from densify.tasks.sandbox import prepare_sandbox


def test_prepare_sandbox_and_grade_exports_patch(tmp_path):
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
problem_statement: fix it
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
    task = load_task_manifest(task_path)

    sandbox = prepare_sandbox(task, "run1", tmp_path / "sandboxes")
    (sandbox.repo / "hello.py").write_text("VALUE = 2\n", encoding="utf-8")
    result = grade_sandbox(task, sandbox.root)

    assert sandbox.repo.exists()
    assert result.repo_dirty is True
    assert (sandbox.root / "patch.diff").read_text(encoding="utf-8")
