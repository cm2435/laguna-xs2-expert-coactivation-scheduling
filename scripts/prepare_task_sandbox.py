from __future__ import annotations

import argparse
from datetime import UTC, datetime

from densify.tasks.manifest import load_task_manifest
from densify.tasks.sandbox import prepare_sandbox


def default_run_id(task_id: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{task_id}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--output-root", default="sandboxes/pool_runs")
    args = parser.parse_args()

    task = load_task_manifest(args.task)
    sandbox = prepare_sandbox(task, args.run_id or default_run_id(task.task_id), args.output_root)
    print(sandbox.root)


if __name__ == "__main__":
    main()
