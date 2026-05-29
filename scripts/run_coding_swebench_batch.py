from __future__ import annotations

import argparse

from densify.run_artifacts import append_jsonl
from densify.tasks.coding_runner import run_task_rollout
from densify.tasks.manifest import iter_registry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default="tasks/registry.jsonl")
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--model", default="laguna")
    parser.add_argument("--output-dir", default="runs/coding_harness_rollouts")
    parser.add_argument("--sandbox-root", default="sandboxes/coding_harness")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    manifests = iter_registry(args.registry)
    if args.limit is not None:
        manifests = manifests[: args.limit]

    for manifest in manifests:
        rollout_dir = run_task_rollout(
            task_path=str(manifest),
            api_url=args.api_url,
            model=args.model,
            output_dir=args.output_dir,
            sandbox_root=args.sandbox_root,
            max_turns=args.max_turns,
            temperature=args.temperature,
        )
        append_jsonl(
            f"{args.output_dir}/batch_attempts.jsonl",
            {"task_manifest": str(manifest), "rollout_dir": rollout_dir},
        )


if __name__ == "__main__":
    main()
