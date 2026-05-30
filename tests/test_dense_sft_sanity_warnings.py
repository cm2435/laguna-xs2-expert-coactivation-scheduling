from __future__ import annotations

import importlib.util
from pathlib import Path


def load_sanity_module():
    path = Path("scripts/run_dense_sft_sanity.py")
    spec = importlib.util.spec_from_file_location("run_dense_sft_sanity", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summarize_reports_loop_warning_counts() -> None:
    sanity = load_sanity_module()
    rows = [
        {
            "kind": "rollout_prefix",
            "raw_generation": "<tool_call>read_file</tool_call>",
            "generated_tokens": 1,
            "latency_s": 1.0,
            "heuristics": {
                "repeated_tool_call": True,
                "contains_error_marker": False,
            },
        },
        {
            "kind": "rollout_prefix",
            "raw_generation": "file not found",
            "generated_tokens": 1,
            "latency_s": 1.0,
            "heuristics": {
                "repeated_tool_call": False,
                "contains_error_marker": True,
            },
        },
    ]

    summary = sanity.summarize(rows)

    assert summary["warning_counts"] == {
        "loop_warning_rows": 1,
        "error_marker_rows": 1,
    }


def test_loop_warning_heuristics_detects_repeated_tagged_calls() -> None:
    sanity = load_sanity_module()
    text = "\n".join(
        [
            "<tool_call>read_file\n<arg_key>path</arg_key>\n<arg_value>x.py</arg_value>\n</tool_call>",
            "<tool_call>read_file\n<arg_key>path</arg_key>\n<arg_value>x.py</arg_value>\n</tool_call>",
            "<tool_call>read_file\n<arg_key>path</arg_key>\n<arg_value>x.py</arg_value>\n</tool_call>",
        ]
    )

    heuristics = sanity.loop_warning_heuristics(text)

    assert heuristics["repeated_tool_call"]
    assert heuristics["max_consecutive_tool_call_repeat"] == 3
