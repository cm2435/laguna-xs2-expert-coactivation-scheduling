from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class UsageCostSummary:
    response_files: int
    prompt_tokens: int
    completion_tokens: int
    estimated_cost_usd: float


MODEL_PRICES: dict[str, tuple[float, float]] = {
    "openai/gpt-5.4-mini": (0.75, 4.5),
    "openai/gpt-5-mini": (0.25, 2.0),
    "openai/gpt-5.5-mini": (0.25, 2.0),
    "openai/gpt-5.5": (5.0, 30.0),
    "openai/gpt-5.5-pro": (30.0, 180.0),
    "anthropic/claude-sonnet-4.5": (3.0, 15.0),
    "anthropic/claude-sonnet-4.6": (3.0, 15.0),
}


def summarize_usage_cost(
    run_dir: str | Path,
    *,
    input_price_per_million: float,
    output_price_per_million: float,
) -> UsageCostSummary:
    prompt_tokens = 0
    completion_tokens = 0
    response_files = 0
    response_paths = sorted(
        set(Path(run_dir).glob("responses/turn_*.json"))
        | set(Path(run_dir).glob("*/responses/turn_*.json"))
    )
    for response_path in response_paths:
        response_files += 1
        payload = json.loads(response_path.read_text(encoding="utf-8"))
        usage = payload.get("usage") or {}
        prompt_tokens += int(usage.get("prompt_tokens") or 0)
        completion_tokens += int(usage.get("completion_tokens") or 0)
    cost = (prompt_tokens / 1_000_000 * input_price_per_million) + (
        completion_tokens / 1_000_000 * output_price_per_million
    )
    return UsageCostSummary(
        response_files=response_files,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        estimated_cost_usd=round(cost, 6),
    )
