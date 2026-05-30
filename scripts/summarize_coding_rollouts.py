from __future__ import annotations

import argparse
import json
from pathlib import Path

from densify.rollout_sft.summary import summarize_rollout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", required=True, type=Path)
    parser.add_argument("--sandboxes-dir", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows = [
        summarize_rollout(run_dir, args.sandboxes_dir).to_json()
        for run_dir in sorted(path for path in args.runs_dir.iterdir() if path.is_dir())
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"count": len(rows), "rollouts": rows}, indent=2) + "\n")
    args.output.with_suffix(".jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
