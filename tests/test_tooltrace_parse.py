from __future__ import annotations

from densify.tooltrace.pool_parse import parse_pool_tool_calls
from densify.tooltrace.speculation_dataset import extract_toolspec_dataset, load_jsonl


def test_parse_pool_tool_calls_xmlish_format() -> None:
    calls = parse_pool_tool_calls(
        """
before
<tool_call>shell
<arg_key>cmd</arg_key><arg_value>pytest -q</arg_value>
</tool_call>
after
"""
    )

    assert len(calls) == 1
    assert calls[0].tool_name == "shell"
    assert calls[0].arguments == {"cmd": "pytest -q"}


def test_extract_toolspec_dataset_from_rollout_tree(tmp_path) -> None:
    call_dir = tmp_path / "run1" / "model_calls" / "call_000001"
    call_dir.mkdir(parents=True)
    (call_dir / "request.json").write_text("{}", encoding="utf-8")
    (call_dir / "served_text.txt").write_text(
        "<tool_call>shell\n<arg_key>cmd</arg_key><arg_value>pytest -q</arg_value></tool_call>",
        encoding="utf-8",
    )

    out = extract_toolspec_dataset(tmp_path)
    rows = load_jsonl(out)

    assert rows[0]["call_id"] == "call_000001"
    assert rows[0]["tool_name"] == "shell"
    assert rows[0]["arguments"]["cmd"] == "pytest -q"
