from __future__ import annotations

import argparse
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Shuffle a JSONL file with a fixed seed.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    rows = [line for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(row + "\n" for row in rows), encoding="utf-8")
    print(f"input_rows={len(rows)}")
    print(f"output={args.output}")
    print(f"seed={args.seed}")


if __name__ == "__main__":
    main()
