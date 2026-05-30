from __future__ import annotations

from recovery_fixtures import make_recovery_raw

from densify.recovery_data.parse import parse_recovery_text
from densify.recovery_data.to_sft import build_sft_rows_from_example
from densify.recovery_data.validate import validate_recovery_dataset


def test_validate_recovery_dataset_counts_balance_and_think_leaks() -> None:
    example = parse_recovery_text(make_recovery_raw())
    rows = build_sft_rows_from_example(example)

    result = validate_recovery_dataset([example], rows)

    assert result.num_rollouts == 1
    assert result.num_rows == 1
    assert result.first_action_tool_counts == {"shell": 1}
    assert result.child_facing_think_rows == 1
    assert result.parseable_tool_call_rows == 1
    assert result.passes_hard_balance is True
    assert result.hard_failures == []


def test_validate_recovery_dataset_flags_child_think_scheme_leak() -> None:
    example = parse_recovery_text(
        make_recovery_raw(think="The data generation prompt asked for this.")
    )
    rows = build_sft_rows_from_example(example)

    result = validate_recovery_dataset([example], rows)

    assert result.think_leak_count == 1
    assert "think blocks mention scheme/data generation/synthetic prompt" in result.hard_failures
