from __future__ import annotations

from densify.tasks.manifest import load_task_manifest


def test_load_task_manifest(tmp_path):
    path = tmp_path / "task.yaml"
    path.write_text(
        """
task_id: tiny
suite: synthetic
repo: owner/repo
repo_id: owner__repo
base_commit: abc
problem_statement: fix it
environment:
  template_path: envs/repo_templates/owner__repo/abc/repo
  setup_command: python -m pip install -e .
grader:
  hidden_command: echo hidden
limits:
  timeout_s: 10
  max_turns: 2
""",
        encoding="utf-8",
    )

    task = load_task_manifest(path)

    assert task.task_id == "tiny"
    assert task.environment.template_path.as_posix().endswith("owner__repo/abc/repo")
    assert task.limits.max_turns == 2
