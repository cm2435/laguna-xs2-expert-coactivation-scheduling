import importlib.util
import json
from pathlib import Path

from densify.reconstruction_data import format_sft_row


def load_canary_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_reconstruction_turn_canary.py"
    spec = importlib.util.spec_from_file_location("build_reconstruction_turn_canary", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


canary = load_canary_module()


def test_iter_turn_rows_extracts_tool_call_turns_and_wraps_thinking():
    rows = [
        {
            "messages": [
                {"role": "user", "content": "Find symbol."},
                {
                    "role": "assistant",
                    "content": "Search before reading.",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "shell",
                                "arguments": {"command": "grep -rn Symbol ."},
                            }
                        }
                    ],
                },
                {"role": "tool", "content": "matches"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "function": {
                                "name": "read_file",
                                "arguments": {"path": "pkg/file.py"},
                            }
                        }
                    ],
                },
            ]
        }
    ]

    turn_rows = canary.iter_turn_rows(rows)

    assert len(turn_rows) == 2
    first = format_sft_row(turn_rows[0])
    assert "<think>\nSearch before reading.\n</think>" in first
    assert "<tool_call>\nshell" in first
    second = format_sft_row(turn_rows[1])
    assert "</think>\n<tool_call>\nread_file" in second
    assert "matches" in second


def test_build_turn_canary_deduplicates_filters_and_repeats(tmp_path, monkeypatch):
    class TinyTokenizer:
        def __call__(self, text, add_special_tokens=True):
            del add_special_tokens
            return {"input_ids": text.split()}

    monkeypatch.setattr(
        canary.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: TinyTokenizer(),
    )
    source = tmp_path / "source.jsonl"
    row = {
        "messages": [
            {"role": "user", "content": "u"},
            {
                "role": "assistant",
                "content": "think",
                "tool_calls": [{"function": {"name": "shell", "arguments": {"command": "ls"}}}],
            },
        ]
    }
    source.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    output = tmp_path / "canary.jsonl"

    summary = canary.build_turn_canary(
        input_path=source,
        output_path=output,
        tokenizer_name="tiny",
        max_unique_rows=12,
        repeat=3,
        seq_len=100,
        seed=1,
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert summary["unique_rows"] == 1
    assert summary["rows_written"] == 3
    assert len(rows) == 3
