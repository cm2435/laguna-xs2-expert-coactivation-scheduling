from __future__ import annotations

import json

import yaml

from densify.swebench.task_selection import manifest_from_swebench_row, select_tasks_by_repo


def test_select_tasks_by_repo_respects_repo_counts() -> None:
    rows = [
        {"instance_id": "a1", "repo": "a/a"},
        {"instance_id": "a2", "repo": "a/a"},
        {"instance_id": "b1", "repo": "b/b"},
    ]

    selected = select_tasks_by_repo(rows, {"a/a": 1, "b/b": 1})

    assert [row["instance_id"] for row in selected] == ["a1", "b1"]


def test_manifest_from_swebench_row_hides_gold_patch() -> None:
    manifest = manifest_from_swebench_row(
        {
            "instance_id": "astropy__astropy-1",
            "repo": "astropy/astropy",
            "base_commit": "abc123",
            "problem_statement": "fix bug",
            "patch": "SECRET_GOLD_PATCH",
            "test_patch": "SECRET_TEST_PATCH",
        }
    )

    dumped = yaml.safe_dump(manifest)
    assert manifest["environment"]["template_path"] == (
        "envs/repo_templates/astropy__astropy/abc123/repo"
    )
    assert "SECRET_GOLD_PATCH" not in dumped
    assert "SECRET_TEST_PATCH" not in dumped
    assert "test_patch" not in dumped
    assert json.dumps(manifest["visible_to_model"])
