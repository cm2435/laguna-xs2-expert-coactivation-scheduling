from __future__ import annotations

import argparse
import json
from pathlib import Path

from densify.recovery_data.to_sft import read_parsed_examples
from densify.recovery_data.validate import validate_recovery_dataset, validation_to_json
from densify.run_artifacts import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate parsed recovery examples and SFT rows.")
    parser.add_argument("--examples", required=True, type=Path)
    parser.add_argument("--sft", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fail-hard", action="store_true")
    args = parser.parse_args()

    examples = read_parsed_examples(args.examples)
    rows = [
        json.loads(line)
        for line in args.sft.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = validate_recovery_dataset(examples, rows)
    payload = validation_to_json(result)
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2))
    if args.fail_hard and result.hard_failures:
        raise SystemExit("hard validation failures: " + ", ".join(result.hard_failures))


if __name__ == "__main__":
    main()
