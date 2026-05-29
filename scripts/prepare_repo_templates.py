from __future__ import annotations

import argparse

from densify.tasks.repo_templates import prepare_repo_templates_from_registry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default="tasks/registry.jsonl")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    prepared = prepare_repo_templates_from_registry(
        args.registry,
        offset=args.offset,
        limit=args.limit,
    )
    for path in prepared:
        print(path)
    print(f"Prepared {len(prepared)} repo templates")


if __name__ == "__main__":
    main()
