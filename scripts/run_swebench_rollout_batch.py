from __future__ import annotations

import argparse

from densify.tasks.pool_runner import run_rollout_batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default="tasks/registry.jsonl")
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--output-dir", default="runs/swebench_pool_rollouts")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()

    if args.concurrency != 1:
        raise ValueError("Only --concurrency 1 is supported in the first implementation")
    results = run_rollout_batch(
        registry_path=args.registry,
        api_url=args.api_url,
        output_dir=args.output_dir,
        limit=args.limit,
    )
    print(f"Attempted {len(results)} rollouts")


if __name__ == "__main__":
    main()
