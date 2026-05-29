"""Stream an excerpt of allenai/c4 (en) to a local jsonl for calibration /
expert-activation analysis.

Usage: python3 scripts/build_c4_excerpt.py --n 400 --out c4_excerpt.jsonl
"""
from __future__ import annotations

import argparse
import json

from datasets import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400, help="number of docs to keep")
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--out", default="c4_excerpt.jsonl")
    args = ap.parse_args()

    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    n = 0
    with open(args.out, "w") as f:
        for ex in ds:
            txt = (ex.get("text") or "").strip()
            if len(txt) < args.min_chars:
                continue
            f.write(json.dumps({"text": txt}) + "\n")
            n += 1
            if n >= args.n:
                break
    print(f"wrote {n} C4 docs -> {args.out}")


if __name__ == "__main__":
    main()
