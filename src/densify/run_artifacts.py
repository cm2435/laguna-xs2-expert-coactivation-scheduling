from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


def new_run_dir(base_dir: str | Path, prefix: str = "run") -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = Path(base_dir) / f"{prefix}_{stamp}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    return value


def write_json(path: str | Path, payload: Any) -> None:
    Path(path).write_text(json.dumps(to_jsonable(payload), indent=2) + "\n", encoding="utf-8")


def write_yaml(path: str | Path, payload: Any) -> None:
    Path(path).write_text(yaml.safe_dump(to_jsonable(payload), sort_keys=False), encoding="utf-8")


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(to_jsonable(row)) + "\n")
