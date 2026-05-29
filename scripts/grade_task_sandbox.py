from __future__ import annotations

import argparse

from densify.tasks.grader import grade_sandbox
from densify.tasks.manifest import load_task_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--sandbox", required=True)
    args = parser.parse_args()

    result = grade_sandbox(load_task_manifest(args.task), args.sandbox)
    print(result.status)


if __name__ == "__main__":
    main()
