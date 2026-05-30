from __future__ import annotations

import asyncio
import importlib.util
import time
from pathlib import Path
from typing import Any


def load_generate_recovery_examples() -> Any:
    script_path = Path(__file__).parents[1] / "scripts" / "generate_recovery_examples.py"
    spec = importlib.util.spec_from_file_location("generate_recovery_examples", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_recovery_generator_uses_strict_payload_for_openai_gpt5_family() -> None:
    module = load_generate_recovery_examples()

    payload = module.build_chat_completion_payload(
        model="openai/gpt-5.5-mini",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.2,
        max_tokens=123,
    )

    assert payload["max_completion_tokens"] == 123
    assert "max_tokens" not in payload
    assert "temperature" not in payload


def test_recovery_generator_keeps_temperature_for_non_openai_models() -> None:
    module = load_generate_recovery_examples()

    payload = module.build_chat_completion_payload(
        model="anthropic/claude-sonnet-4.6",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.2,
        max_tokens=123,
    )

    assert payload["temperature"] == 0.2
    assert payload["max_tokens"] == 123
    assert "max_completion_tokens" not in payload


def test_recovery_generator_can_select_specific_prompt_families() -> None:
    module = load_generate_recovery_examples()

    schedule = module.build_prompt_schedule(
        phase="smoke",
        samples_per_family=1,
        limit=None,
        families="P1,P3,P5,P7",
    )

    assert schedule == ["P1", "P3", "P5", "P7"]


def test_recovery_generator_can_resume_prompt_schedule() -> None:
    module = load_generate_recovery_examples()

    schedule = module.build_prompt_schedule(
        phase="smoke",
        samples_per_family=2,
        limit=3,
        families=None,
        schedule_offset=5,
    )

    assert schedule == ["P3", "P4", "P4"]


def test_recovery_generator_uses_schedule_offset_for_row_ids_and_tasks(tmp_path: Path) -> None:
    module = load_generate_recovery_examples()
    manifest_a = tmp_path / "a.yaml"
    manifest_b = tmp_path / "b.yaml"
    manifest_a.write_text(
        "task_id: task_a\n"
        "suite: test\n"
        "repo: org/a\n"
        "repo_id: org__a\n"
        "base_commit: abc\n"
        "problem_statement: A\n"
        "environment:\n"
        "  template_path: template-a\n"
        "grader:\n"
        "  hidden_command: pytest\n",
        encoding="utf-8",
    )
    manifest_b.write_text(
        "task_id: task_b\n"
        "suite: test\n"
        "repo: org/b\n"
        "repo_id: org__b\n"
        "base_commit: def\n"
        "problem_statement: B\n"
        "environment:\n"
        "  template_path: template-b\n"
        "grader:\n"
        "  hidden_command: pytest\n",
        encoding="utf-8",
    )

    rows = module.build_request_rows(
        manifests=[manifest_a, manifest_b],
        schedule=["P4", "P4"],
        model="test-model",
        temperature=0.2,
        max_tokens=128,
        row_index_offset=77,
    )

    assert rows[0]["id"] == "task_b:P4:0077"
    assert rows[0]["task_id"] == "task_b"
    assert rows[1]["id"] == "task_a:P4:0078"


def test_recovery_generator_runs_requests_concurrently(tmp_path: Path) -> None:
    module = load_generate_recovery_examples()
    calls: list[str] = []

    def fake_call(_api_url: str, payload: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
        time.sleep(0.05)
        calls.append(payload["model"])
        return {"choices": [{"message": {"content": "<example />"}}], "usage": {}}

    requests = [
        {
            "id": f"row-{index}",
            "task_id": "task",
            "manifest": "manifest",
            "prompt_family_id": "P1",
            "prompt_family_name": "family",
            "model": "test-model",
            "request": {"model": "test-model", "messages": []},
        }
        for index in range(4)
    ]
    output = tmp_path / "rows.jsonl"

    started = time.perf_counter()
    rows = asyncio.run(
        module.run_requests_async(
            requests,
            output=output,
            api_url="http://example.test",
            api_key="key",
            extra_headers={},
            concurrency=4,
            caller=fake_call,
        )
    )
    elapsed = time.perf_counter() - started

    assert len(rows) == 4
    assert len(calls) == 4
    assert len(output.read_text(encoding="utf-8").splitlines()) == 4
    assert elapsed < 0.16


def test_recovery_generator_logs_failed_requests_and_keeps_going(
    tmp_path: Path,
    capsys: Any,
) -> None:
    module = load_generate_recovery_examples()

    def fake_call(_api_url: str, payload: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
        if payload["model"] == "bad-model":
            raise RuntimeError("rate limited")
        return {"choices": [{"message": {"content": "<example />"}}], "usage": {}}

    requests = [
        {
            "id": "bad-row",
            "task_id": "task",
            "manifest": "manifest",
            "prompt_family_id": "P5",
            "prompt_family_name": "family",
            "model": "bad-model",
            "request": {"model": "bad-model", "messages": []},
        },
        {
            "id": "good-row",
            "task_id": "task",
            "manifest": "manifest",
            "prompt_family_id": "P1",
            "prompt_family_name": "family",
            "model": "good-model",
            "request": {"model": "good-model", "messages": []},
        },
    ]
    output = tmp_path / "rows.jsonl"

    rows = asyncio.run(
        module.run_requests_async(
            requests,
            output=output,
            api_url="http://example.test",
            api_key="key",
            extra_headers={},
            concurrency=2,
            caller=fake_call,
        )
    )

    captured = capsys.readouterr()
    assert "[recovery-generation-failed]" in captured.out
    assert "bad-row" in captured.out
    assert "P5" in captured.out
    assert "rate limited" in captured.out
    assert [row["ok"] for row in rows] == [False, True]
