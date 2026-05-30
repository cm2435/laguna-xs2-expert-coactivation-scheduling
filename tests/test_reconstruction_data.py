import importlib.util
from pathlib import Path

from densify.reconstruction_data import format_sft_row


def load_reconstruction_train_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "train_dense_reconstruction.py"
    spec = importlib.util.spec_from_file_location("train_dense_reconstruction", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


train_reconstruction = load_reconstruction_train_module()


def test_format_sft_row_handles_instruction_output_rows():
    row = {
        "instruction": "Write add.",
        "input": "Use Python.",
        "output": "def add(a,b): return a+b",
    }

    text = format_sft_row(row)

    assert "<user>" in text
    assert "Write add." in text
    assert "Use Python." in text
    assert "<assistant>" in text
    assert "def add" in text


def test_format_sft_row_handles_messages_rows():
    row = {
        "messages": [
            {"role": "user", "content": "Fix bug"},
            {"role": "assistant", "content": "Patch follows"},
        ]
    }

    text = format_sft_row(row)

    assert "<user>\nFix bug\n</user>" in text
    assert "<assistant>\nPatch follows\n</assistant>" in text


def test_format_sft_row_renders_structured_assistant_tool_calls():
    row = {
        "messages": [
            {"role": "user", "content": "Find the validator."},
            {
                "role": "assistant",
                "content": "I should search before reading.",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "shell",
                            "arguments": {"command": "grep -rn URLValidator django/"},
                        },
                    }
                ],
            },
        ]
    }

    text = format_sft_row(row)

    assert "I should search before reading." in text
    assert "<tool_call>\nshell" in text
    assert "<arg_key>command</arg_key>" in text
    assert "<arg_value>grep -rn URLValidator django/</arg_value>" in text


def test_format_sft_row_preserves_tool_call_when_assistant_content_is_empty():
    row = {
        "messages": [
            {"role": "user", "content": "Read the file."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path": "django/core/validators.py"}',
                        },
                    }
                ],
            },
        ]
    }

    text = format_sft_row(row)

    assert "<assistant>" in text
    assert "<tool_call>\nread_file" in text
    assert "<arg_key>path</arg_key>" in text
    assert "<arg_value>django/core/validators.py</arg_value>" in text


def test_format_sft_row_handles_prompt_completion_rows():
    row = {"prompt": "Q", "completion": "A"}

    assert format_sft_row(row) == "<user>\nQ\n</user>\n<assistant>\nA\n</assistant>"


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 99
    vocab = {
        "a": 10,
        "b": 11,
        "c": 12,
        "d": 13,
        "e": 14,
        "f": 15,
    }

    def __call__(self, text, add_special_tokens=True, truncation=False, max_length=None):
        del add_special_tokens
        ids = [self.vocab[token] for token in text.split()]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids}


def test_iter_token_batches_packs_short_rows_with_eos_boundaries():
    rows = [
        {"text": "a b"},
        {"text": "c d"},
        {"text": "e f"},
    ]

    batches = list(
        train_reconstruction.iter_token_batches(
            rows,
            TinyTokenizer(),
            seq_len=8,
            batch_size=1,
            device="cpu",
            pack_sequences=True,
        )
    )

    assert int(batches[0]["attention_mask"].sum().item()) == 8
    assert int(batches[1]["attention_mask"].sum().item()) == 1
    assert batches[0]["input_ids"][0].tolist() == [10, 11, 99, 12, 13, 99, 14, 15]
    assert batches[1]["input_ids"][0].tolist() == [99, 0, 0, 0, 0, 0, 0, 0]
    assert int(batches[0]["packing_doc_start_count"].sum().item()) == 3
    assert int(batches[0]["packing_boundary_count"].sum().item()) == 2
    assert int(batches[1]["packing_doc_start_count"].sum().item()) == 0
    assert int(batches[1]["packing_boundary_count"].sum().item()) == 1


def test_iter_token_batches_can_mask_kl_to_final_assistant_turn():
    class CharTokenizer:
        pad_token_id = 0
        eos_token_id = 99

        def __call__(self, text, add_special_tokens=True, truncation=False, max_length=None):
            del add_special_tokens
            ids = [ord(char) % 89 + 10 for char in text]
            if truncation and max_length is not None:
                ids = ids[:max_length]
            return {"input_ids": ids}

    rows = [
        {
            "messages": [
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"},
            ]
        }
    ]
    tokenizer = CharTokenizer()

    batches = list(
        train_reconstruction.iter_token_batches(
            rows,
            tokenizer,
            seq_len=128,
            batch_size=1,
            device="cpu",
            pack_sequences=False,
            kl_target_mode="assistant_last",
        )
    )

    mask = batches[0]["kl_attention_mask"][0].tolist()
    assert any(mask)
    assert mask[0] == 0
    assert mask[-1] == 0


def test_iter_token_batches_can_preserve_one_row_per_sequence_with_padding():
    rows = [
        {"text": "a b"},
        {"text": "c d"},
    ]

    batches = list(
        train_reconstruction.iter_token_batches(
            rows,
            TinyTokenizer(),
            seq_len=4,
            batch_size=1,
            device="cpu",
            pack_sequences=False,
        )
    )

    assert [int(batch["attention_mask"].sum().item()) for batch in batches] == [2, 2]
    assert [int(batch["packing_boundary_count"].sum().item()) for batch in batches] == [0, 0]
