from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from densify.run_artifacts import append_jsonl
from densify.tasks.coding_runner import run_task_rollout


Runner = Callable[..., str]


async def run_manifests_async(
    manifests: Sequence[Path],
    *,
    api_url: str,
    model: str,
    output_dir: str | Path,
    sandbox_root: str | Path,
    max_turns: int,
    temperature: float,
    max_tokens: int | None,
    concurrency: int,
    api_key: str | None = None,
    extra_headers: dict[str, str] | None = None,
    runner: Runner = run_task_rollout,
) -> list[dict[str, Any]]:
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(sandbox_root).mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    attempts_path = Path(output_dir) / "batch_attempts.jsonl"

    async def run_one(manifest: Path) -> dict[str, Any]:
        async with semaphore:
            try:
                rollout_dir = await asyncio.to_thread(
                    runner,
                    task_path=str(manifest),
                    api_url=api_url,
                    model=model,
                    output_dir=str(output_dir),
                    sandbox_root=str(sandbox_root),
                    max_turns=max_turns,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    api_key=api_key,
                    extra_headers=extra_headers,
                )
                row: dict[str, Any] = {
                    "task_manifest": str(manifest),
                    "rollout_dir": rollout_dir,
                    "model": model,
                    "ok": True,
                }
            except Exception as exc:  # keep long batches moving
                row = {
                    "task_manifest": str(manifest),
                    "rollout_dir": "",
                    "model": model,
                    "ok": False,
                    "error": repr(exc),
                }
            async with write_lock:
                append_jsonl(attempts_path, row)
            return row

    return list(await asyncio.gather(*(run_one(manifest) for manifest in manifests)))
