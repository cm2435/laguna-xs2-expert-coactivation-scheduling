from __future__ import annotations

import json

from densify.plotting.pretraining_curve import load_metric_rows, summarize_metric_rows


def test_load_metric_rows_ignores_blank_lines(tmp_path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        json.dumps({"step": 1, "loss": 3.0, "per_layer": {"0": {"mse": 2.0}}})
        + "\n\n"
        + json.dumps({"step": 2, "loss": 1.0, "per_layer": {"0": {"mse": 0.5}}})
        + "\n",
        encoding="utf-8",
    )

    rows = load_metric_rows(path)

    assert [row["step"] for row in rows] == [1, 2]


def test_summarize_metric_rows_reports_loss_improvement() -> None:
    rows = [
        {"step": 1, "loss": 4.0, "elapsed_sec": 1.0},
        {"step": 11, "loss": 2.0, "elapsed_sec": 6.0},
    ]

    summary = summarize_metric_rows(rows)

    assert summary["steps"] == 2
    assert summary["first_step"] == 1
    assert summary["last_step"] == 11
    assert summary["first_loss"] == 4.0
    assert summary["last_loss"] == 2.0
    assert summary["loss_delta"] == -2.0
    assert summary["loss_delta_pct"] == -50.0
    assert summary["steps_per_sec"] == 2.0
