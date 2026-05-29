from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from densify.config import load_yaml


def repo_id(repo: str) -> str:
    return repo.replace("/", "__")


def select_tasks_by_repo(
    rows: list[dict[str, Any]],
    repo_counts: dict[str, int],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    counts = {repo: 0 for repo in repo_counts}
    for row in rows:
        repo = str(row.get("repo", ""))
        if repo not in repo_counts:
            continue
        if counts[repo] >= repo_counts[repo]:
            continue
        selected.append(row)
        counts[repo] += 1
        if all(counts[name] >= target for name, target in repo_counts.items()):
            break
    return selected


def manifest_from_swebench_row(row: dict[str, Any]) -> dict[str, Any]:
    instance_id = str(row["instance_id"])
    repo = str(row["repo"])
    rid = repo_id(repo)
    base_commit = str(row["base_commit"])
    return {
        "task_id": instance_id,
        "suite": "swebench_verified",
        "repo": repo,
        "repo_id": rid,
        "base_commit": base_commit,
        "problem_statement": str(row.get("problem_statement", "")),
        "visible_to_model": {
            "issue_statement": True,
            "public_tests": False,
            "hidden_tests": False,
        },
        "environment": {
            "template_path": f"envs/repo_templates/{rid}/{base_commit}/repo",
            "setup_command": "python -m pip install -e .",
            "smoke_command": "",
        },
        "grader": {
            "hidden_command": (
                "uv run python scripts/grade_task_sandbox.py "
                f"--task tasks/swebench_verified/{instance_id}.yaml "
                "--sandbox {sandbox}"
            )
        },
        "limits": {
            "timeout_s": 1800,
            "max_turns": 30,
            "network": "disabled",
        },
        "metadata": {
            "dataset": "princeton-nlp/SWE-bench_Verified",
            "instance_id": instance_id,
        },
    }


def write_task_manifests(rows: list[dict[str, Any]], output_dir: Path, registry_path: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with registry_path.open("w", encoding="utf-8") as registry:
        for row in rows:
            manifest = manifest_from_swebench_row(row)
            task_id = manifest["task_id"]
            manifest_path = output_dir / f"{task_id}.yaml"
            manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

            grader_dir = Path("tasks/graders/swebench_verified") / task_id
            grader_dir.mkdir(parents=True, exist_ok=True)
            grader_metadata = {
                "task_id": task_id,
                "repo": row.get("repo"),
                "base_commit": row.get("base_commit"),
                "test_patch": row.get("test_patch", ""),
                "patch": row.get("patch", ""),
                "FAIL_TO_PASS": row.get("FAIL_TO_PASS", []),
                "PASS_TO_PASS": row.get("PASS_TO_PASS", []),
            }
            (grader_dir / "grader_metadata.json").write_text(
                json.dumps(grader_metadata, indent=2) + "\n",
                encoding="utf-8",
            )
            registry.write(
                json.dumps(
                    {
                        "task_id": task_id,
                        "suite": "swebench_verified",
                        "manifest": str(manifest_path),
                        "repo_id": manifest["repo_id"],
                    }
                )
                + "\n"
            )


def build_manifests_from_config(config_path: str | Path) -> list[dict[str, Any]]:
    from datasets import load_dataset

    cfg = load_yaml(config_path)
    dataset = load_dataset(str(cfg["dataset"]), split=str(cfg.get("split", "test")))
    rows = [dict(row) for row in dataset]
    repo_counts = {str(repo): int(count) for repo, count in dict(cfg["repos"]).items()}
    selected = select_tasks_by_repo(rows, repo_counts)
    target_total = int(cfg.get("target_total", sum(repo_counts.values())))
    selected = selected[:target_total]
    write_task_manifests(selected, Path(cfg["output_dir"]), Path(cfg["registry_path"]))
    return selected
