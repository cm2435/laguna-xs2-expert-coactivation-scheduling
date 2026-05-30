from __future__ import annotations

import asyncio
import time
from pathlib import Path

from densify.tasks.async_coding_batch import run_manifests_async


def test_run_manifests_async_honors_concurrency_limit(tmp_path):
    active = 0
    max_active = 0
    started = 0

    def fake_runner(**kwargs):
        nonlocal active, max_active, started
        started += 1
        active += 1
        max_active = max(max_active, active)
        time.sleep(0.02)
        active -= 1
        return f"run-{Path(kwargs['task_path']).name}"

    manifests = [tmp_path / f"task_{index}.yaml" for index in range(6)]

    rows = asyncio.run(
        run_manifests_async(
            manifests,
            api_url="https://openrouter.ai/api/v1",
            model="openai/gpt-5.5",
            output_dir=tmp_path / "runs",
            sandbox_root=tmp_path / "sandboxes",
            max_turns=5,
            temperature=0.0,
            max_tokens=8192,
            concurrency=2,
            runner=fake_runner,
        )
    )

    assert started == 6
    assert max_active == 2
    assert [row["model"] for row in rows] == ["openai/gpt-5.5"] * 6
    assert (tmp_path / "runs" / "batch_attempts.jsonl").exists()
