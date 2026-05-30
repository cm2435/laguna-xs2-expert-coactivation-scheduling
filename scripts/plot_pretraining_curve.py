from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from densify.plotting.pretraining_curve import load_metric_rows, summarize_metric_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot dense reconstruction/pretraining metrics.jsonl.")
    parser.add_argument("metrics_jsonl", type=Path, help="Path to metrics.jsonl from train_dense_reconstruction.py")
    parser.add_argument("--output", type=Path, help="Output PNG path")
    parser.add_argument("--rolling-window", type=int, default=5)
    parser.add_argument("--show", action="store_true", help="Open an interactive matplotlib window")
    parser.add_argument("--title", default="Dense Reconstruction Pretraining")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_metric_rows(args.metrics_jsonl)
    summary = summarize_metric_rows(rows)
    output = args.output or args.metrics_jsonl.with_suffix(".png")
    output.parent.mkdir(parents=True, exist_ok=True)

    fig = plot_rows(rows, title=args.title, rolling_window=args.rolling_window)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    print(json.dumps({"plot": str(output), "summary": summary}, indent=2))
    if args.show:
        plt.show()
    plt.close(fig)


def plot_rows(rows: list[dict[str, Any]], *, title: str, rolling_window: int):
    steps = [int(row["step"]) for row in rows]
    losses = [float(row["loss"]) for row in rows]
    per_layer = _collect_per_layer(rows)
    has_per_layer = bool(per_layer)

    fig, axes = plt.subplots(2 if has_per_layer else 1, 1, figsize=(10, 7 if has_per_layer else 4.5), sharex=True)
    if not isinstance(axes, list) and not hasattr(axes, "__len__"):
        axes = [axes]

    ax = axes[0]
    ax.plot(steps, losses, label="loss", color="#1f77b4", linewidth=1.8)
    if rolling_window > 1 and len(losses) >= rolling_window:
        ax.plot(
            steps,
            _rolling_mean(losses, rolling_window),
            label=f"loss rolling mean ({rolling_window})",
            color="#ff7f0e",
            linewidth=1.6,
        )
    ax.set_title(title)
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    if has_per_layer:
        layer_ax = axes[1]
        for layer_id, values in sorted(per_layer.items(), key=lambda item: int(item[0]) if item[0].isdigit() else item[0]):
            layer_ax.plot(steps, values, label=f"layer {layer_id}", linewidth=1.0, alpha=0.85)
        layer_ax.set_ylabel("per-layer mse/loss")
        layer_ax.set_xlabel("step")
        layer_ax.grid(True, alpha=0.25)
        if len(per_layer) <= 12:
            layer_ax.legend(loc="best", fontsize=8)
        else:
            layer_ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7, ncol=1)
    else:
        ax.set_xlabel("step")

    return fig


def _collect_per_layer(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    layer_ids: set[str] = set()
    for row in rows:
        per_layer = row.get("per_layer")
        if isinstance(per_layer, dict):
            layer_ids.update(str(key) for key in per_layer)
    series = {layer_id: [] for layer_id in layer_ids}
    for row in rows:
        per_layer = row.get("per_layer") if isinstance(row.get("per_layer"), dict) else {}
        for layer_id in layer_ids:
            layer_row = per_layer.get(layer_id) or per_layer.get(int(layer_id)) or {}
            series[layer_id].append(_layer_scalar(layer_row))
    return {layer_id: values for layer_id, values in series.items() if any(value == value for value in values)}


def _layer_scalar(layer_row: Any) -> float:
    if isinstance(layer_row, dict):
        for key in ("mse", "loss", "reconstruction_loss"):
            if key in layer_row:
                return float(layer_row[key])
    if isinstance(layer_row, (int, float)):
        return float(layer_row)
    return float("nan")


def _rolling_mean(values: list[float], window: int) -> list[float]:
    result: list[float] = []
    for index in range(len(values)):
        start = max(0, index + 1 - window)
        chunk = values[start : index + 1]
        result.append(sum(chunk) / len(chunk))
    return result


if __name__ == "__main__":
    main()
