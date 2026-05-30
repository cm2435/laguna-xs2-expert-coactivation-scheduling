from __future__ import annotations

import json

from densify.tasks.openrouter_cost import MODEL_PRICES, summarize_usage_cost


def test_summarize_usage_cost_reads_response_usage(tmp_path):
    responses = tmp_path / "run" / "task" / "responses"
    responses.mkdir(parents=True)
    (responses / "turn_0001.json").write_text(
        json.dumps(
            {
                "usage": {
                    "prompt_tokens": 1000,
                    "completion_tokens": 100,
                }
            }
        ),
        encoding="utf-8",
    )
    (responses / "turn_0002.json").write_text(
        json.dumps(
            {
                "usage": {
                    "prompt_tokens": 2000,
                    "completion_tokens": 200,
                }
            }
        ),
        encoding="utf-8",
    )

    summary = summarize_usage_cost(
        tmp_path / "run",
        input_price_per_million=5.0,
        output_price_per_million=30.0,
    )

    assert summary.response_files == 2
    assert summary.prompt_tokens == 3000
    assert summary.completion_tokens == 300
    assert summary.estimated_cost_usd == 0.024


def test_model_prices_include_cheap_gpt_mini_aliases():
    assert MODEL_PRICES["openai/gpt-5.4-mini"] == (0.75, 4.5)
    assert MODEL_PRICES["openai/gpt-5-mini"] == (0.25, 2.0)
    assert MODEL_PRICES["openai/gpt-5.5-mini"] == (0.25, 2.0)
    assert MODEL_PRICES["anthropic/claude-sonnet-4.6"] == (3.0, 15.0)


def test_summarize_usage_cost_reads_direct_rollout_responses(tmp_path):
    response_dir = tmp_path / "responses"
    response_dir.mkdir()
    (response_dir / "turn_0001.json").write_text(
        json.dumps({"usage": {"prompt_tokens": 100, "completion_tokens": 50}}),
        encoding="utf-8",
    )

    summary = summarize_usage_cost(
        tmp_path,
        input_price_per_million=0.25,
        output_price_per_million=2.0,
    )

    assert summary.response_files == 1
    assert summary.prompt_tokens == 100
    assert summary.completion_tokens == 50
    assert summary.estimated_cost_usd == 0.000125
