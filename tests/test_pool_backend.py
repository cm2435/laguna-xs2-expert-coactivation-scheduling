from __future__ import annotations

from densify.pool_backend import content_to_text, normalize_messages, strip_pool_unfriendly_markup


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
        ]
    )

    assert messages == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
    ]


def test_strip_pool_unfriendly_markup_removes_assistant_close_tags() -> None:
    assert strip_pool_unfriendly_markup("\nREADY\n</assistant>\n</assistant>") == "READY"
