from __future__ import annotations

from densify.recovery_data.prompts import build_teacher_messages, expand_prompt_schedule
from densify.recovery_data.schema import TaskSeed


def test_expand_prompt_schedule_smoke_has_one_per_family() -> None:
    schedule = expand_prompt_schedule("smoke")

    assert len(schedule) == 12
    assert schedule[0] == "P1"
    assert schedule[-1] == "P12"


def test_build_teacher_messages_contains_contract_and_family_constraints() -> None:
    messages = build_teacher_messages(
        TaskSeed(
            task_id="django__django-10999",
            repo="django/django",
            problem_statement="Fix parse_duration() for negative durations.",
        ),
        prompt_family_id="P3",
    )

    assert [message["role"] for message in messages] == ["system", "user"]
    assert "<scheme>" in messages[0]["content"]
    assert "<trajectory_json>" in messages[0]["content"]
    assert "Never mention the prompt" in messages[0]["content"]
    assert (
        'Role strings must be exactly "system", "user", "assistant", or "tool"'
        in messages[0]["content"]
    )
    assert (
        "Completeness and valid JSON are more important than trajectory length"
        in messages[0]["content"]
    )
    assert "Escape every newline inside JSON string values as \\n" in messages[0]["content"]
    assert "- shell(command)" in messages[0]["content"]
    assert (
        "Prefer read_file after shell has discovered a concrete file path"
        in messages[0]["content"]
    )
    assert "Use 3-7 assistant turns" in messages[0]["content"]
    assert (
        "Do not continue exploratory debugging once the lesson has been demonstrated"
        in messages[0]["content"]
    )
    assert "PR URL / issue URL trap" in messages[1]["content"]
    assert "External URLs and issue IDs are not repository paths" in messages[1]["content"]
