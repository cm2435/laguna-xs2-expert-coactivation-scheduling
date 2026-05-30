from __future__ import annotations

import argparse
import json
from pathlib import Path

from densify.on_policy.teacher_corrections import (
    build_teacher_payload,
    correction_row_from_response,
    query_teacher,
    read_jsonl,
    write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--teacher-api-url", default="http://127.0.0.1:8791/v1")
    parser.add_argument("--teacher-model", default="laguna")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = []
    for state in read_jsonl(args.states):
        payload = build_teacher_payload(state, teacher_model=args.teacher_model)
        if args.dry_run:
            response = {"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": []}}]}
        else:
            response = query_teacher(args.teacher_api_url, payload)
        row = correction_row_from_response(state, response, payload=payload)
        if args.dry_run:
            row["teacher_payload"] = payload
        rows.append(row)

    write_jsonl(rows, args.output)
    print(f"wrote_corrections={len(rows)}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
