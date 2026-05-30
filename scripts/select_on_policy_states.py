from __future__ import annotations

import argparse
from pathlib import Path

from densify.on_policy.select_states import select_states_from_runs, write_states_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-states-per-task", type=int, default=5)
    args = parser.parse_args()

    states = select_states_from_runs(args.runs_dir, max_states_per_task=args.max_states_per_task)
    write_states_jsonl(states, args.output)
    print(f"wrote_states={len(states)}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
