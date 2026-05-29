from __future__ import annotations

import argparse

from densify.tasks.repo_templates import prepare_repo_templates_from_registry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default="tasks/registry.jsonl")
    args = parser.parse_args()

    prepared = prepare_repo_templates_from_registry(args.registry)
    for path in prepared:
        print(path)
    print(f"Prepared {len(prepared)} repo templates")


if __name__ == "__main__":
    main()
