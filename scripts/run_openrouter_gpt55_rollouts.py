from __future__ import annotations

import argparse
import asyncio
import os

from densify.tasks.async_coding_batch import run_manifests_async
from densify.tasks.manifest import iter_registry


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run coding-harness SWE-bench rollouts through OpenRouter GPT-5.5."
    )
    parser.add_argument("--registry", default="tasks/registry.jsonl")
    parser.add_argument("--api-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--model", default="openai/gpt-5.5")
    parser.add_argument("--output-dir", default="runs/openrouter_gpt55_rollouts")
    parser.add_argument("--sandbox-root", default="sandboxes/openrouter_gpt55")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--max-turns", type=int, default=25)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--http-referer", default="https://github.com/cm2435/laguna-xs2-experiments")
    parser.add_argument("--x-title", default="Laguna XS.2 Hackathon Rollouts")
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key. Export {args.api_key_env}=...")

    manifests = iter_registry(args.registry)
    if args.offset:
        manifests = manifests[args.offset :]
    if args.limit is not None:
        manifests = manifests[: args.limit]

    rows = asyncio.run(
        run_manifests_async(
            manifests,
            api_url=args.api_url,
            model=args.model,
            output_dir=args.output_dir,
            sandbox_root=args.sandbox_root,
            max_turns=args.max_turns,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            concurrency=args.concurrency,
            api_key=api_key,
            extra_headers={"HTTP-Referer": args.http_referer, "X-Title": args.x_title},
        )
    )
    ok = sum(1 for row in rows if row.get("ok"))
    print(f"attempted={len(rows)} ok={ok} failed={len(rows) - ok}")
    print(f"output_dir={args.output_dir}")
    print(f"sandbox_root={args.sandbox_root}")


if __name__ == "__main__":
    main()
