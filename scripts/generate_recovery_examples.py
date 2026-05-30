from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from densify.recovery_data.prompts import (
    build_teacher_messages,
    expand_prompt_schedule,
    task_seed_from_manifest,
)
from densify.recovery_data.schema import PROMPT_FAMILIES
from densify.run_artifacts import append_jsonl
from densify.tasks.manifest import iter_registry

Caller = Callable[..., dict[str, Any]]


def normalized_model_id(model: str) -> str:
    return model.lower().removeprefix("openrouter/")


def is_openai_gpt5_family(model: str) -> bool:
    normalized = normalized_model_id(model)
    return normalized.startswith("openai/gpt-5") or normalized.startswith("gpt-5")


def build_chat_completion_payload(
    *,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    if is_openai_gpt5_family(model):
        payload["max_completion_tokens"] = max_tokens
    else:
        payload["temperature"] = temperature
        payload["max_tokens"] = max_tokens
    return payload


def build_prompt_schedule(
    *,
    phase: str,
    samples_per_family: int | None,
    limit: int | None,
    families: str | None,
    schedule_offset: int = 0,
) -> list[str]:
    if families:
        family_ids = [family.strip() for family in families.split(",") if family.strip()]
        unknown = [family for family in family_ids if family not in PROMPT_FAMILIES]
        if unknown:
            raise ValueError(f"Unknown prompt families: {', '.join(unknown)}")
        repeats = samples_per_family or 1
        schedule = [family for _ in range(repeats) for family in family_ids]
    else:
        schedule = expand_prompt_schedule(phase, samples_per_family=samples_per_family)
    if schedule_offset:
        schedule = schedule[schedule_offset:]
    if limit is not None:
        schedule = schedule[:limit]
    return schedule


def build_request_rows(
    *,
    manifests: Sequence[Path],
    schedule: Sequence[str],
    model: str,
    temperature: float,
    max_tokens: int,
    row_index_offset: int = 0,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, family_id in enumerate(schedule):
        global_index = row_index_offset + index
        manifest = manifests[global_index % len(manifests)]
        task = task_seed_from_manifest(str(manifest))
        messages = build_teacher_messages(task, prompt_family_id=family_id)
        request_payload = build_chat_completion_payload(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        rows.append(
            {
                "id": f"{task.task_id}:{family_id}:{global_index:04d}",
                "task_id": task.task_id,
                "manifest": str(manifest),
                "prompt_family_id": family_id,
                "prompt_family_name": PROMPT_FAMILIES[family_id]["name"],
                "model": model,
                "request": request_payload,
            }
        )
    return rows


async def run_requests_async(
    request_rows: Sequence[dict[str, Any]],
    *,
    output: Path,
    api_url: str,
    api_key: str,
    extra_headers: dict[str, str],
    concurrency: int,
    caller: Caller | None = None,
) -> list[dict[str, Any]]:
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    semaphore = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    active_caller = caller or call_chat_completions

    async def run_one(row: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            started = time.perf_counter()
            output_row = dict(row)
            try:
                response = await asyncio.to_thread(
                    active_caller,
                    api_url,
                    row["request"],
                    api_key=api_key,
                    extra_headers=extra_headers,
                )
                output_row["ok"] = True
                output_row["latency_s"] = round(time.perf_counter() - started, 3)
                output_row["response"] = response
                choices = response.get("choices") or []
                message = (choices[0].get("message") if choices else {}) or {}
                output_row["response_text"] = str(message.get("content") or "")
                output_row["usage"] = response.get("usage") or {}
            except Exception as exc:
                output_row["ok"] = False
                output_row["latency_s"] = round(time.perf_counter() - started, 3)
                output_row["response_text"] = ""
                output_row["usage"] = {}
                output_row["error"] = repr(exc)
                print(
                    "[recovery-generation-failed] "
                    f"id={row.get('id')} "
                    f"family={row.get('prompt_family_id')} "
                    f"task={row.get('task_id')} "
                    f"model={row.get('model')} "
                    f"error={exc!r}",
                    flush=True,
                )
            async with write_lock:
                append_jsonl(output, output_row)
            return output_row

    return list(await asyncio.gather(*(run_one(row) for row in request_rows)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate metacognitive recovery rollouts.")
    parser.add_argument("--tasks", required=True, help="Registry JSONL with manifest paths.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--phase",
        choices=["smoke", "pilot", "repair", "scale", "stretch"],
        default="smoke",
    )
    parser.add_argument("--samples-per-family", type=int)
    parser.add_argument("--families", help="Comma-separated prompt family ids, e.g. P1,P3,P5,P7.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--schedule-offset", type=int, default=0)
    parser.add_argument("--provider", choices=["openrouter", "none"], default="none")
    parser.add_argument("--api-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--model", default="anthropic/claude-sonnet-4.6")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=12000)
    parser.add_argument("--http-referer", default="https://github.com/cm2435/laguna-xs2-experiments")
    parser.add_argument("--x-title", default="Laguna XS.2 Recovery Data")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifests = iter_registry(args.tasks)
    if args.offset:
        manifests = manifests[args.offset :]
    schedule = build_prompt_schedule(
        phase=args.phase,
        samples_per_family=args.samples_per_family,
        limit=args.limit,
        families=args.families,
        schedule_offset=args.schedule_offset,
    )
    if not manifests:
        raise SystemExit("No task manifests found")
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be >= 1")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        args.output.unlink()

    api_key = os.environ.get(args.api_key_env)
    if args.provider == "openrouter" and not args.dry_run and not api_key:
        raise SystemExit(f"Missing API key. Export {args.api_key_env}=...")

    request_rows = build_request_rows(
        manifests=manifests,
        schedule=schedule,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        row_index_offset=args.schedule_offset,
    )

    if args.dry_run or args.provider == "none":
        for row in request_rows:
            row["response_text"] = ""
            row["dry_run"] = True
            append_jsonl(args.output, row)
    else:
        asyncio.run(
            run_requests_async(
                request_rows,
                output=args.output,
                api_url=args.api_url,
                api_key=str(api_key),
                extra_headers={"HTTP-Referer": args.http_referer, "X-Title": args.x_title},
                concurrency=args.concurrency,
            )
        )
    print(f"wrote={args.output}")
    print(f"rollouts={len(schedule)}")


def call_chat_completions(
    api_url: str,
    payload: dict[str, Any],
    *,
    api_key: str,
    extra_headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    url = api_url.rstrip("/") + "/chat/completions"
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    headers.update(extra_headers or {})
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter request failed: {exc.code} {detail}") from exc


if __name__ == "__main__":
    main()
