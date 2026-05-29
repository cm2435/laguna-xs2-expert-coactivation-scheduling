from __future__ import annotations

import argparse

from densify.swebench.task_selection import build_manifests_from_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/swebench_verified_20.yaml")
    args = parser.parse_args()

    selected = build_manifests_from_config(args.config)
    print(f"Wrote {len(selected)} SWE-bench task manifests")


if __name__ == "__main__":
    main()
