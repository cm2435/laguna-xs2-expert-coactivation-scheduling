from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_metric_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"no metric rows found in {path}")
    return rows


def summarize_metric_rows(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    first = rows[0]
    last = rows[-1]
    first_loss = float(first["loss"])
    last_loss = float(last["loss"])
    delta = last_loss - first_loss
    elapsed = float(last.get("elapsed_sec") or 0.0) - float(first.get("elapsed_sec") or 0.0)
    step_delta = int(last["step"]) - int(first["step"])
    return {
        "steps": len(rows),
        "first_step": int(first["step"]),
        "last_step": int(last["step"]),
        "first_loss": first_loss,
        "last_loss": last_loss,
        "best_loss": min(float(row["loss"]) for row in rows),
        "loss_delta": delta,
        "loss_delta_pct": 100.0 * delta / first_loss if first_loss else 0.0,
        "elapsed_sec": float(last.get("elapsed_sec") or 0.0),
        "steps_per_sec": step_delta / elapsed if elapsed > 0 else 0.0,
    }
