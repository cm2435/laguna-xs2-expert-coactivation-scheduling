from __future__ import annotations

from densify.rollout_sft.logprobs import (
    build_kd_row,
    normalize_chat_top_logprobs,
    sampled_token_ids_from_chat_logprobs,
    split_last_assistant_target,
)


class FakeTokenizer:
    def __call__(self, text: str, add_special_tokens: bool = False):
        ids = [ord(char) for char in text]
        if add_special_tokens:
            ids = [0] + ids
        return {"input_ids": ids}


def test_split_last_assistant_target_ignores_following_tool_observation() -> None:
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "call tool"},
        {"role": "tool", "content": "observation"},
    ]

    context, target = split_last_assistant_target(messages)

    assert [message["role"] for message in context] == ["system", "user"]
    assert target == {"role": "assistant", "content": "call tool"}


def test_normalize_chat_top_logprobs_preserves_one_entry_per_target_token() -> None:
    response = {
        "choices": [
            {
                "logprobs": {
                    "content": [
                        {
                            "token": "A",
                            "logprob": -0.1,
                            "top_logprobs": [
                                {"token": "A", "logprob": -0.1},
                                {"token": "B", "logprob": -2.3},
                            ],
                        },
                        {
                            "token": "B",
                            "logprob": -0.2,
                            "top_logprobs": [
                                {"token": "B", "logprob": -0.2},
                                {"token": "A", "logprob": -1.9},
                            ],
                        },
                    ]
                }
            }
        ]
    }

    top_logprobs = normalize_chat_top_logprobs(response, FakeTokenizer(), top_k=2)

    assert top_logprobs == [
        [{"token_id": 65, "logprob": -0.1}, {"token_id": 66, "logprob": -2.3}],
        [{"token_id": 66, "logprob": -0.2}, {"token_id": 65, "logprob": -1.9}],
    ]


def test_sampled_token_ids_from_chat_logprobs_reads_generated_tokens() -> None:
    response = {
        "choices": [
            {
                "logprobs": {
                    "content": [
                        {"token": "A", "logprob": -0.1, "top_logprobs": []},
                        {"token": "B", "logprob": -0.2, "top_logprobs": []},
                    ]
                }
            }
        ]
    }

    assert sampled_token_ids_from_chat_logprobs(response, FakeTokenizer()) == [65, 66]


def test_build_kd_row_targets_last_assistant_span() -> None:
    sft_row = {
        "id": "demo",
        "task_id": "task",
        "source_rollout": "runs/demo",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "OK"},
            {"role": "tool", "content": "observation"},
        ],
    }
    top_logprobs = [
        [{"token_id": 79, "logprob": -0.1}],
        [{"token_id": 75, "logprob": -0.2}],
    ]

    row = build_kd_row(sft_row, FakeTokenizer(), top_logprobs, source="test", top_k=1)

    assert row["id"] == "demo"
    assert row["target_text"] == "OK"
    assert row["target_token_ids"] == [79, 75]
    assert row["teacher_top_logprobs"] == top_logprobs
    assert [message["role"] for message in row["context_messages"]] == ["system", "user"]
