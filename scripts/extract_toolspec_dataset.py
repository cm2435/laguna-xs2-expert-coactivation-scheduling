from __future__ import annotations

import argparse

from densify.tooltrace.speculation_dataset import extract_toolspec_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-root", default="runs/swebench_pool_rollouts")
    parser.add_argument("--output")
    args = parser.parse_args()

    out = extract_toolspec_dataset(args.rollout_root, args.output)
    print(out)


if __name__ == "__main__":
    main()
