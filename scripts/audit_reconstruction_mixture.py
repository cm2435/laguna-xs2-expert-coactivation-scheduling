from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from densify.reconstruction_data import format_sft_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preflight audit for reconstruction/KL mixture rendering."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-examples", type=int, default=10)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if structured tool calls are lost in rendered text.",
    )
    return parser.parse_args()


def source_for(row: dict[str, Any]) -> str:
    return str(row.get("_reconstruction_source") or row.get("source") or row.get("quality") or "?")


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if line.strip():
                yield line_no, json.loads(line)


def assistant_tool_calls(row: dict[str, Any]) -> list[dict[str, Any]]:
    calls = []
    for message in row.get("messages") or []:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            if isinstance(call, dict):
                calls.append(call)
    return calls


def call_name(call: dict[str, Any]) -> str:
    function = call.get("function") if isinstance(call, dict) else None
    if not isinstance(function, dict):
        return ""
    return str(function.get("name") or "")


def call_arguments(call: dict[str, Any]) -> dict[str, Any]:
    function = call.get("function") if isinstance(call, dict) else None
    if not isinstance(function, dict):
        return {}
    raw = function.get("arguments")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {"_raw": raw}
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    if raw is None:
        return {}
    return {"value": raw}


def short(value: Any, limit: int = 140) -> str:
    text = str(value).replace("\n", "\\n")
    return text[:limit]


def normalized_contains(haystack: str, needle: Any, *, min_chars: int = 50) -> bool:
    text = str(needle)
    if not text:
        return True
    candidates = [text, text.replace("\\n", "\n"), text.replace("\n", "\\n")]
    for candidate in candidates:
        if candidate and candidate[:min_chars] in haystack:
            return True
    compact_haystack = " ".join(haystack.split())
    compact_needle = " ".join(text.split())
    if compact_needle and compact_needle[:min_chars] in compact_haystack:
        return True
    return False


def audit(path: Path, max_examples: int) -> dict[str, Any]:
    by_source: dict[str, Counter] = defaultdict(Counter)
    tool_counts: dict[str, Counter] = defaultdict(Counter)
    examples: list[dict[str, Any]] = []
    total = Counter()

    for line_no, row in iter_jsonl(path):
        src = source_for(row)
        total["rows"] += 1
        by_source[src]["rows"] += 1
        try:
            rendered = format_sft_row(row)
        except Exception as exc:
            total["render_errors"] += 1
            by_source[src]["render_errors"] += 1
            if len(examples) < max_examples:
                examples.append({"line": line_no, "source": src, "error": repr(exc)})
            continue

        calls = assistant_tool_calls(row)
        raw_tagged = json.dumps(row).count("<tool_call>")
        rendered_tagged = rendered.count("<tool_call>")
        total["raw_tagged_tool_calls"] += raw_tagged
        total["rendered_tagged_tool_calls"] += rendered_tagged
        by_source[src]["raw_tagged_tool_calls"] += raw_tagged
        by_source[src]["rendered_tagged_tool_calls"] += rendered_tagged
        if "<think>" in rendered:
            total["rendered_think_rows"] += 1
            by_source[src]["rendered_think_rows"] += 1
        if calls:
            total["rows_with_structured_tool_calls"] += 1
            by_source[src]["rows_with_structured_tool_calls"] += 1

        for call in calls:
            name = call_name(call)
            args = call_arguments(call)
            tool_counts[src][name or "<missing>"] += 1
            total["structured_tool_calls"] += 1
            by_source[src]["structured_tool_calls"] += 1
            missing_bits = []
            if name and name not in rendered:
                missing_bits.append(f"name:{name}")
            for key, value in args.items():
                if str(key) not in rendered:
                    missing_bits.append(f"arg_key:{key}")
                if value not in (None, "") and not normalized_contains(rendered, value):
                    missing_bits.append(f"arg_value:{key}")
            if missing_bits:
                total["structured_tool_calls_lost"] += 1
                by_source[src]["structured_tool_calls_lost"] += 1
                if len(examples) < max_examples:
                    examples.append(
                        {
                            "line": line_no,
                            "source": src,
                            "missing": missing_bits,
                            "tool": name,
                            "args": {key: short(value) for key, value in args.items()},
                            "rendered_preview": rendered[:800],
                        }
                    )

    return {
        "input": str(path),
        "total": dict(total),
        "by_source": {source: dict(counter) for source, counter in sorted(by_source.items())},
        "tool_counts": {source: dict(counter) for source, counter in sorted(tool_counts.items())},
        "examples": examples,
    }


def main() -> None:
    args = parse_args()
    report = audit(args.input, args.max_examples)
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    lost = int(report["total"].get("structured_tool_calls_lost", 0))
    errors = int(report["total"].get("render_errors", 0))
    if args.strict and (lost or errors):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
