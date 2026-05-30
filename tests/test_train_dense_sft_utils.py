from __future__ import annotations

import math
import importlib.util
from pathlib import Path

import torch
from torch import nn

from densify.rollout_sft.train import collate_kd_tokenized, set_sft_trainable_parameters, tokenize_kd_row


def _load_train_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "train_dense_sft.py"
    spec = importlib.util.spec_from_file_location("train_dense_sft", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


train_script = _load_train_script()


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.routed_dense = nn.Linear(2, 2)
        self.norm = nn.LayerNorm(2)
        self.lm_head = nn.Linear(2, 2)
        self.attn = nn.Linear(2, 2)


def test_set_sft_trainable_parameters_trains_only_routed_dense_by_default() -> None:
    model = FakeModel()

    trainable = set_sft_trainable_parameters(model)

    assert trainable == sum(param.numel() for param in model.routed_dense.parameters())
    assert all(param.requires_grad for param in model.routed_dense.parameters())
    assert not any(param.requires_grad for param in model.norm.parameters())
    assert not any(param.requires_grad for param in model.lm_head.parameters())
    assert not any(param.requires_grad for param in model.attn.parameters())


def test_set_sft_trainable_parameters_can_train_norms_and_lm_head() -> None:
    model = FakeModel()

    set_sft_trainable_parameters(model, train_norms=True, train_lm_head=True)

    assert all(param.requires_grad for param in model.routed_dense.parameters())
    assert all(param.requires_grad for param in model.norm.parameters())
    assert all(param.requires_grad for param in model.lm_head.parameters())
    assert not any(param.requires_grad for param in model.attn.parameters())


class FakeTokenizer:
    def __init__(self):
        self.vocab = {"<system>\ns\n</system>": [10, 11], " target": [20, 21]}

    def __call__(self, text: str, add_special_tokens: bool = True, truncation=False, max_length=None):
        if text == "<system>\ns\n</system>":
            ids = [10, 11]
        elif text == "<system>\ns\n</system> target":
            ids = [10, 11, 20, 21]
        else:
            ids = [ord(char) % 127 for char in text]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids}


def test_tokenize_kd_row_aligns_teacher_topk_to_target_positions() -> None:
    row = {
        "context_messages": [{"role": "system", "content": "s"}],
        "target_text": "target",
        "teacher_top_logprobs": [
            [{"token_id": 20, "logprob": -0.1}, {"token_id": 99, "logprob": -2.0}],
            [{"token_id": 21, "logprob": -0.2}, {"token_id": 98, "logprob": -3.0}],
        ],
    }

    tokenized = tokenize_kd_row(row, FakeTokenizer(), seq_len=8)

    assert tokenized["input_ids"] == [10, 11, 20, 21]
    assert tokenized["labels"] == [-100, -100, 20, 21]
    assert tokenized["target_mask"] == [False, False, True, True]
    assert tokenized["teacher_token_ids"] == [[0, 0], [0, 0], [20, 99], [21, 98]]
    assert tokenized["teacher_logprobs"][2] == [-0.1, -2.0]


def test_collate_kd_tokenized_pads_teacher_arrays() -> None:
    rows = [
        {
            "input_ids": [1, 2],
            "labels": [-100, 2],
            "target_mask": [False, True],
            "teacher_token_ids": [[0], [2]],
            "teacher_logprobs": [[0.0], [-0.1]],
        },
        {
            "input_ids": [3],
            "labels": [-100],
            "target_mask": [False],
            "teacher_token_ids": [[0]],
            "teacher_logprobs": [[0.0]],
        },
    ]

    input_ids, labels, attention_mask, target_mask, token_ids, logprobs = collate_kd_tokenized(rows, pad_id=0)

    assert input_ids.tolist() == [[1, 2], [3, 0]]
    assert labels.tolist() == [[-100, 2], [-100, -100]]
    assert attention_mask.tolist() == [[1, 1], [1, 0]]
    assert target_mask.tolist() == [[False, True], [False, False]]
    assert token_ids.tolist() == [[[0], [2]], [[0], [0]]]
    assert torch.isinf(logprobs[1, 1, 0])
    assert logprobs[1, 1, 0].item() < 0


def test_split_train_validation_rows_groups_by_task_id() -> None:
    rows = [
        {"id": "a1", "task_id": "task-a"},
        {"id": "a2", "task_id": "task-a"},
        {"id": "b1", "task_id": "task-b"},
        {"id": "b2", "task_id": "task-b"},
        {"id": "c1", "task_id": "task-c"},
    ]

    train_rows, val_rows = train_script.split_train_validation_rows(rows, 0.5)

    split_by_task = {}
    for row in train_rows:
        split_by_task.setdefault(row["task_id"], "train")
        assert split_by_task[row["task_id"]] == "train"
    for row in val_rows:
        split_by_task.setdefault(row["task_id"], "val")
        assert split_by_task[row["task_id"]] == "val"
    assert train_rows
    assert val_rows


class LengthTokenizer:
    def __call__(self, text, add_special_tokens=True, truncation=False, max_length=None):
        ids = list(range(len(text.split())))
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids}


def test_sort_rows_by_token_length_orders_and_rejects_zero_target() -> None:
    rows = [
        {"id": "long", "messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a b c"}]},
        {"id": "short", "messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]},
    ]

    sorted_rows = train_script.sort_rows_by_token_length(
        rows,
        LengthTokenizer(),
        seq_len=128,
        enable_thinking=False,
    )

    assert [row["id"] for row in sorted_rows] == ["short", "long"]


def test_rotate_rows_wraps_by_consumed_count() -> None:
    rows = [{"id": "a"}, {"id": "b"}, {"id": "c"}]

    assert train_script.rotate_rows(rows, 0) == rows
    assert train_script.rotate_rows(rows, 1) == [{"id": "b"}, {"id": "c"}, {"id": "a"}]
    assert train_script.rotate_rows(rows, 4) == [{"id": "b"}, {"id": "c"}, {"id": "a"}]


def test_weighted_ce_loss_applies_row_weights() -> None:
    logits = torch.tensor(
        [
            [[5.0, 0.0], [5.0, 0.0], [5.0, 0.0]],
            [[5.0, 0.0], [0.0, 5.0], [0.0, 5.0]],
        ]
    )
    labels = torch.tensor(
        [
            [-100, 0, 0],
            [-100, 0, 0],
        ]
    )

    unweighted = train_script.weighted_ce_loss(logits, labels, torch.tensor([1.0, 1.0]))
    weighted = train_script.weighted_ce_loss(logits, labels, torch.tensor([1.0, 10.0]))

    assert weighted > unweighted


def test_rejects_copied_shell_placeholder_without_override(tmp_path) -> None:
    model_dir = tmp_path / "laguna-xs2-dense-k8-copied-shell"
    model_dir.mkdir()
    (model_dir / "copied_shell_report.json").write_text(
        '{"random_routed_dense_keys": 117, "copied_shared_expert_keys": 117}\n',
        encoding="utf-8",
    )

    try:
        train_script.assert_not_placeholder_base(str(model_dir), allow_placeholder_base=False)
    except SystemExit as exc:
        assert "random routed dense" in str(exc).lower()
    else:
        raise AssertionError("expected placeholder base rejection")


def test_allows_reconstructed_base_without_placeholder_report(tmp_path) -> None:
    model_dir = tmp_path / "laguna-xs2-dense-k8-recon-2k"
    model_dir.mkdir()

    train_script.assert_not_placeholder_base(str(model_dir), allow_placeholder_base=False)


def test_allows_placeholder_base_with_explicit_override(tmp_path) -> None:
    model_dir = tmp_path / "laguna-xs2-dense-k8-copied-shell"
    model_dir.mkdir()
    (model_dir / "copied_shell_report.json").write_text(
        '{"random_routed_dense_keys": 117}\n',
        encoding="utf-8",
    )

    train_script.assert_not_placeholder_base(str(model_dir), allow_placeholder_base=True)
