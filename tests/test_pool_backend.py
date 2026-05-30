from __future__ import annotations

from densify.pool_backend import (
    content_to_text,
    normalize_messages,
    parse_generated_tool_calls,
    strip_pool_unfriendly_markup,
)


def test_content_to_text_handles_openai_content_parts() -> None:
    assert (
        content_to_text(
            [
                {"type": "text", "text": "hello"},
                {"type": "image_url", "image_url": {"url": "ignored"}},
                {"type": "text", "text": "world"},
            ]
        )
        == "hello\nworld"
    )


def test_normalize_messages_flattens_pool_payload_messages() -> None:
    messages = normalize_messages(
        [
            {"role": "system", "content": [{"type": "text", "text": "sys"}]},
            {"role": "user", "content": "task"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{\"path\": \"x.py\"}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "name": "read_file", "content": "obs"},
        ]
    )

    assert messages == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": {"path": "x.py"}},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "name": "read_file", "content": "obs"},
    ]


def test_strip_pool_unfriendly_markup_removes_assistant_close_tags() -> None:
    assert strip_pool_unfriendly_markup("\nREADY\n</assistant>\n</assistant>") == "READY"


def test_parse_generated_tool_calls_extracts_structured_calls() -> None:
    text = (
        "None\n"
        '{"tool_calls": [{"id": "call_1", "type": "function", '
        '"function": {"name": "read_file", "arguments": "{\\"path\\": \\"src/app.py\\"}"}}]}'
        "\n</assistant>"
    )

    content, tool_calls = parse_generated_tool_calls(text)

    assert content == "None"
    assert tool_calls == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "read_file", "arguments": '{"path": "src/app.py"}'},
        }
    ]


def test_parse_generated_tool_calls_extracts_laguna_tagged_calls() -> None:
    text = """
<tool_call>read_file
<arg_key>path</arg_key>
<arg_value>sympy/functions/elementary/hyperbolic.py</arg_value>
<arg_key>start_line</arg_key>
<arg_value>580</arg_value>
<arg_key>max_lines</arg_key>
<arg_value>20</arg_value>
</tool_call>
</assistant>
<tool_call>apply_patch
<arg_key>patch</arg_key>
<arg_value>ignored run-on text</arg_value>
</tool_call>
"""

    content, tool_calls = parse_generated_tool_calls(text)

    assert content == ""
    assert tool_calls == [
        {
            "id": "generated_tool_1",
            "type": "function",
            "function": {
                "name": "read_file",
                "arguments": '{"path": "sympy/functions/elementary/hyperbolic.py", "start_line": 580, "max_lines": 20}',
            },
        }
    ]


def test_parse_generated_tool_calls_accepts_partial_laguna_tagged_calls() -> None:
    text = """
<tool_call>read_file
<arg_key>path</arg_key>
<arg_value>sympy/functions/elementary/hyperbolic.py</arg_value>
<arg_key>content</arg_key>
<arg_value>run-on text without a close tag
"""

    _, tool_calls = parse_generated_tool_calls(text)

    assert tool_calls == [
        {
            "id": "generated_tool_1",
            "type": "function",
            "function": {
                "name": "read_file",
                "arguments": '{"path": "sympy/functions/elementary/hyperbolic.py"}',
            },
        }
    ]
