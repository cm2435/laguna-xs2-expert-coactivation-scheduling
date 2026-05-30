from __future__ import annotations

import argparse
import json
from pathlib import Path

from densify.recovery_data.parse import example_to_json, parse_recovery_text
from densify.run_artifacts import append_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse metacognitive recovery generation JSONL.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rejects", type=Path)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        args.output.unlink()
    rejects_path = args.rejects or args.output.with_suffix(".rejects.jsonl")
    if rejects_path.exists():
        rejects_path.unlink()

    accepted = 0
    rejected = 0
    for line_no, line in enumerate(args.input.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        text = str(row.get("response_text") or "")
        try:
            example = parse_recovery_text(text, fallback_id=str(row.get("id") or line_no))
        except Exception as exc:
            rejected += 1
            append_jsonl(rejects_path, {"line_no": line_no, "error": repr(exc), "row": row})
            continue
        append_jsonl(args.output, example_to_json(example))
        accepted += 1

    write_json(
        args.output.with_suffix(".summary.json"),
        {"accepted": accepted, "rejected": rejected},
    )
    print(f"accepted={accepted} rejected={rejected}")
    print(f"output={args.output}")
    print(f"rejects={rejects_path}")


if __name__ == "__main__":
    main()
