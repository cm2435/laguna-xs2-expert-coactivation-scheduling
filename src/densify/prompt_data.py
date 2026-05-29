from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CodingPrompt:
    id: str
    prompt: str
    entrypoint: str | None
    tests: list[str]


def load_jsonl_prompts(path: str | Path) -> list[CodingPrompt]:
    rows: list[CodingPrompt] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            raw = json.loads(line)
            rows.append(
                CodingPrompt(
                    id=str(raw["id"]),
                    prompt=str(raw["prompt"]),
                    entrypoint=raw.get("entrypoint"),
                    tests=list(raw.get("tests", [])),
                )
            )
    if not rows:
        raise ValueError(f"No prompts loaded from {path}")
    return rows
