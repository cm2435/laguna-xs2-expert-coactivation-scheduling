from __future__ import annotations

import json
import importlib.util
from pathlib import Path


def _load_validator():
    path = Path(__file__).resolve().parents[1] / "scripts" / "validate_sft_dataset.py"
    spec = importlib.util.spec_from_file_location("validate_sft_dataset", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validator = _load_validator()
summarize_sft_rows = validator.summarize_sft_rows
compare_template_rendering = validator.compare_template_rendering


class FakeTokenizer:
    chat_template = "fake"

    def __call__(self, text, add_special_tokens=True, truncation=False, max_length=None):
        ids = list(range(len(text.split())))
        if add_special_tokens:
            ids = [0] + ids
        return {"input_ids": ids}

    def decode(self, token_ids):
        return " ".join(f"tok{token_id}" for token_id in token_ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        rendered = "".join(f"<|{message['role']}|>{message.get('content', '')}" for message in messages)
        if add_generation_prompt:
            rendered += "<|assistant|>"
        if tokenize:
            return self(rendered, add_special_tokens=False)["input_ids"]
        return rendered


def test_summarize_sft_rows_counts_targets_and_rejects_tool_rows() -> None:
    rows = [
        {
            "id": "ok",
            "quality": "silver",
            "messages": [
                {"role": "user", "content": "fix bug"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"function": {"name": "shell", "arguments": "{}"}}],
                },
            ],
        },
        {
            "id": "bad",
            "quality": "bronze",
            "messages": [
                {"role": "assistant", "content": "use tool"},
                {"role": "tool", "content": "observation"},
            ],
        },
    ]

    summary = summarize_sft_rows(rows, FakeTokenizer(), seq_len=128, max_examples=1)

    assert summary["row_count"] == 2
    assert summary["valid_row_count"] == 1
    assert summary["invalid_row_count"] == 1
    assert summary["assistant_target_count"] == 1
    assert summary["tool_call_target_count"] == 1
    assert summary["quality_counts"] == {"silver": 1, "bronze": 1}
    assert summary["invalid_rows"][0]["id"] == "bad"
    assert summary["examples"][0]["id"] == "ok"


def test_compare_template_rendering_reports_token_delta() -> None:
    row = {
        "id": "ok",
        "messages": [
            {"role": "user", "content": "fix bug"},
            {"role": "assistant", "content": "done"},
        ],
    }

    comparison = compare_template_rendering(row, FakeTokenizer())

    assert comparison["has_chat_template"] is True
    assert comparison["manual_token_count"] > 0
    assert comparison["chat_template_token_count"] > 0
    assert "manual_preview" in comparison
    assert "chat_template_preview" in comparison
