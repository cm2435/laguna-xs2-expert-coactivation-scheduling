from __future__ import annotations

from densify.rollout_sft.tokenize import render_message, split_sft_texts, tokenize_sft_row


class FakeTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def __init__(self):
        self.add_special_tokens_calls = []

    def __call__(self, text, add_special_tokens=True, truncation=False, max_length=None):
        self.add_special_tokens_calls.append(add_special_tokens)
        tokens = text.split()
        ids = list(range(1, len(tokens) + 1))
        if add_special_tokens:
            ids = [0] + ids
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids}


class FakeChatTokenizer(FakeTokenizer):
    chat_template = "fake"

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        return_tensors=None,
        enable_thinking=False,
    ):
        rendered = " ".join(
            f"<|{message['role']}|> {message.get('content', '')}" for message in messages
        )
        if add_generation_prompt:
            rendered += " <|assistant|>"
        if enable_thinking:
            rendered += " <think>"
        if tokenize:
            return self(rendered, add_special_tokens=False)["input_ids"]
        return rendered


def test_render_message_preserves_tool_calls() -> None:
    rendered = render_message(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "shell", "arguments": "{}"}}],
        }
    )

    assert "<assistant>" in rendered
    assert "tool_calls" in rendered
    assert "shell" in rendered


def test_tokenize_sft_row_masks_prefix_and_trains_final_assistant() -> None:
    row = {
        "messages": [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "fix bug"},
            {"role": "assistant", "content": "use tool"},
        ]
    }

    tokenized = tokenize_sft_row(row, FakeTokenizer(), seq_len=128)

    assert len(tokenized["input_ids"]) == len(tokenized["labels"])
    assert any(label == -100 for label in tokenized["labels"])
    assert any(label != -100 for label in tokenized["labels"])
    first_target = next(index for index, label in enumerate(tokenized["labels"]) if label != -100)
    assert all(label == -100 for label in tokenized["labels"][:first_target])


def test_split_sft_texts_rejects_tool_observation_target() -> None:
    row = {
        "messages": [
            {"role": "user", "content": "fix bug"},
            {"role": "assistant", "content": "use tool"},
            {"role": "tool", "content": "observation"},
        ]
    }

    try:
        split_sft_texts(row)
    except ValueError as exc:
        assert "final message must be assistant" in str(exc)
    else:
        raise AssertionError("expected tool-target row to be rejected")


def test_split_sft_texts_returns_trainable_target_text() -> None:
    row = {
        "messages": [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "fix bug"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "shell", "arguments": "{}"}}],
            },
        ]
    }

    split = split_sft_texts(row)

    assert "system prompt" in split.prefix_text
    assert "tool_calls" in split.target_text
    assert "shell" in split.target_text


def test_tokenize_sft_row_uses_chat_template_when_available() -> None:
    row = {
        "messages": [
            {"role": "user", "content": "fix bug"},
            {"role": "assistant", "content": "done"},
        ]
    }

    tokenizer = FakeChatTokenizer()
    tokenized = tokenize_sft_row(row, tokenizer, seq_len=128)

    assert len(tokenized["input_ids"]) == len(tokenized["labels"])
    first_target = next(index for index, label in enumerate(tokenized["labels"]) if label != -100)
    assert first_target > 0
    assert all(label == -100 for label in tokenized["labels"][:first_target])
    assert tokenizer.add_special_tokens_calls == [False, False]


def test_chat_template_masks_prior_assistant_and_tool_turns() -> None:
    row = {
        "messages": [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "fix bug"},
            {"role": "assistant", "content": "bad read"},
            {"role": "tool", "content": "file not found"},
            {"role": "assistant", "content": "recover with shell"},
        ]
    }

    tokenizer = FakeChatTokenizer()
    tokenized = tokenize_sft_row(row, tokenizer, seq_len=128)
    labels = tokenized["labels"]
    first_target = next(index for index, label in enumerate(labels) if label != -100)

    assert all(label == -100 for label in labels[:first_target])
    assert all(label != -100 for label in labels[first_target:])
    assert first_target > 0


def test_tokenize_sft_row_can_disable_thinking_for_prompt_template() -> None:
    row = {
        "messages": [
            {"role": "user", "content": "fix bug"},
            {"role": "assistant", "content": "done"},
        ]
    }

    tokenized = tokenize_sft_row(row, FakeChatTokenizer(), seq_len=128, enable_thinking=False)

    assert any(label != -100 for label in tokenized["labels"])
