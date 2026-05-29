from densify.reconstruction_data import format_sft_row


def test_format_sft_row_handles_instruction_output_rows():
    row = {"instruction": "Write add.", "input": "Use Python.", "output": "def add(a,b): return a+b"}

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


def test_format_sft_row_handles_prompt_completion_rows():
    row = {"prompt": "Q", "completion": "A"}

    assert format_sft_row(row) == "<user>\nQ\n</user>\n<assistant>\nA\n</assistant>"
