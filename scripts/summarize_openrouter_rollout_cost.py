from __future__ import annotations

import argparse

from densify.tasks.openrouter_cost import MODEL_PRICES, summarize_usage_cost


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize OpenRouter rollout token usage/cost.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model", default="openai/gpt-5.5")
    parser.add_argument("--input-price-per-million", type=float)
    parser.add_argument("--output-price-per-million", type=float)
    args = parser.parse_args()

    default_prices = MODEL_PRICES.get(args.model)
    input_price = args.input_price_per_million
    output_price = args.output_price_per_million
    if input_price is None or output_price is None:
        if default_prices is None:
            raise SystemExit("Unknown model price; pass --input-price-per-million and --output-price-per-million.")
        input_price, output_price = default_prices

    summary = summarize_usage_cost(
        args.run_dir,
        input_price_per_million=input_price,
        output_price_per_million=output_price,
    )
    print(f"response_files={summary.response_files}")
    print(f"prompt_tokens={summary.prompt_tokens}")
    print(f"completion_tokens={summary.completion_tokens}")
    print(f"estimated_cost_usd={summary.estimated_cost_usd:.6f}")


if __name__ == "__main__":
    main()
