from __future__ import annotations

import argparse
from pathlib import Path

from densify.recovery_data.to_sft import (
    build_sft_rows_from_examples,
    read_parsed_examples,
    write_sft_rows,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert parsed recovery examples to SFT JSONL.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    examples = read_parsed_examples(args.input)
    rows = build_sft_rows_from_examples(examples)
    write_sft_rows(rows, args.output)
    print(f"examples={len(examples)}")
    print(f"rows={len(rows)}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
