from __future__ import annotations

import torch

from densify.rollout_sft.tokenize import render_messages


def set_sft_trainable_parameters(
    model: torch.nn.Module,
    *,
    train_norms: bool = False,
    train_lm_head: bool = False,
) -> int:
    trainable = 0
    for name, param in model.named_parameters():
        should_train = ".routed_dense." in name or "routed_dense" in name
        if train_norms and "norm" in name:
            should_train = True
        if train_lm_head and "lm_head" in name:
            should_train = True
        param.requires_grad_(should_train)
        if should_train:
            trainable += param.numel()
    return trainable


def collate_tokenized(
    rows: list[dict[str, list[int]]],
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len = max(len(row["input_ids"]) for row in rows)
    input_ids = torch.full((len(rows), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(rows), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(rows), max_len), dtype=torch.long)
    for idx, row in enumerate(rows):
        length = len(row["input_ids"])
        input_ids[idx, :length] = torch.tensor(row["input_ids"], dtype=torch.long)
        labels[idx, :length] = torch.tensor(row["labels"], dtype=torch.long)
        attention_mask[idx, :length] = 1
    return input_ids, labels, attention_mask


def tokenize_kd_row(row, tokenizer, seq_len: int) -> dict[str, list]:
    prefix_text = render_messages(list(row["context_messages"]))
    target_text = " " + str(row["target_text"])
    full_text = prefix_text + target_text
    prefix_ids = tokenizer(prefix_text, add_special_tokens=True)["input_ids"] if prefix_text else []
    full_ids = tokenizer(full_text, add_special_tokens=True, truncation=True, max_length=seq_len)[
        "input_ids"
    ]
    target_start = min(len(prefix_ids), len(full_ids))
    labels = [-100] * len(full_ids)
    target_mask = [False] * len(full_ids)
    teacher_top_logprobs = list(row["teacher_top_logprobs"])
    top_k = max((len(position) for position in teacher_top_logprobs), default=0)
    teacher_token_ids = [[0] * top_k for _ in full_ids]
    teacher_logprobs = [[0.0] * top_k for _ in full_ids]

    for offset, idx in enumerate(range(target_start, len(full_ids))):
        if offset >= len(teacher_top_logprobs):
            break
        labels[idx] = full_ids[idx]
        target_mask[idx] = True
        entries = teacher_top_logprobs[offset]
        for entry_idx, entry in enumerate(entries[:top_k]):
            teacher_token_ids[idx][entry_idx] = int(entry["token_id"])
            teacher_logprobs[idx][entry_idx] = float(entry["logprob"])

    return {
        "input_ids": full_ids,
        "labels": labels,
        "target_mask": target_mask,
        "teacher_token_ids": teacher_token_ids,
        "teacher_logprobs": teacher_logprobs,
    }


def collate_kd_tokenized(
    rows: list[dict[str, list]],
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len = max(len(row["input_ids"]) for row in rows)
    top_k = max((len(position) for row in rows for position in row["teacher_token_ids"]), default=0)
    input_ids = torch.full((len(rows), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(rows), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(rows), max_len), dtype=torch.long)
    target_mask = torch.zeros((len(rows), max_len), dtype=torch.bool)
    teacher_token_ids = torch.zeros((len(rows), max_len, top_k), dtype=torch.long)
    teacher_logprobs = torch.full((len(rows), max_len, top_k), -float("inf"), dtype=torch.float32)
    for idx, row in enumerate(rows):
        length = len(row["input_ids"])
        input_ids[idx, :length] = torch.tensor(row["input_ids"], dtype=torch.long)
        labels[idx, :length] = torch.tensor(row["labels"], dtype=torch.long)
        attention_mask[idx, :length] = 1
        target_mask[idx, :length] = torch.tensor(row["target_mask"], dtype=torch.bool)
        row_token_ids = torch.tensor(row["teacher_token_ids"], dtype=torch.long)
        row_logprobs = torch.tensor(row["teacher_logprobs"], dtype=torch.float32)
        teacher_token_ids[idx, :length, : row_token_ids.shape[-1]] = row_token_ids
        teacher_logprobs[idx, :length, : row_logprobs.shape[-1]] = row_logprobs
    return input_ids, labels, attention_mask, target_mask, teacher_token_ids, teacher_logprobs
