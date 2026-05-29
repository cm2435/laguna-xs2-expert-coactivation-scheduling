from __future__ import annotations

import json
from pathlib import Path

from densify.run_artifacts import append_jsonl
from densify.tooltrace.pool_parse import parse_pool_tool_calls, tool_call_to_json


def extract_toolspec_dataset(
    rollout_root: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    root = Path(rollout_root)
    out = Path(output_path) if output_path is not None else root / "speculation_dataset.jsonl"
    if out.exists():
        out.unlink()
    for served_text_path in sorted(root.glob("**/model_calls/*/served_text.txt")):
        call_dir = served_text_path.parent
        text = served_text_path.read_text(encoding="utf-8")
        for call in parse_pool_tool_calls(text):
            row = tool_call_to_json(
                call,
                call_id=call_dir.name,
                served_text_path=str(served_text_path),
                request_path=str(call_dir / "request.json"),
            )
            append_jsonl(out, row)
    return out


def load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
