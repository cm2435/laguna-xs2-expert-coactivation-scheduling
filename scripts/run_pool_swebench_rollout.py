from __future__ import annotations

import argparse

from densify.tasks.manifest import load_task_manifest
from densify.tasks.pool_runner import run_pool_rollout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--output-dir", default="runs/swebench_pool_rollouts")
    parser.add_argument("--sandbox-root", default="sandboxes/pool_runs")
    parser.add_argument("--run-id")
    args = parser.parse_args()

    result = run_pool_rollout(
        task=load_task_manifest(args.task),
        api_url=args.api_url,
        output_dir=args.output_dir,
        sandbox_root=args.sandbox_root,
        run_id=args.run_id,
    )
    print(result.rollout_dir)


if __name__ == "__main__":
    main()
