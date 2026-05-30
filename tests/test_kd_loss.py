from __future__ import annotations

import torch

from densify.rollout_sft.kd import topk_kl_loss


def test_topk_kl_loss_gathers_student_logits_at_teacher_token_ids() -> None:
    student_logits = torch.tensor(
        [
            [
                [0.0, 4.0, 1.0, -1.0],
                [3.0, 0.0, 1.0, -2.0],
            ]
        ]
    )
    teacher_token_ids = torch.tensor([[[1, 2], [0, 2]]])
    teacher_logprobs = torch.log(torch.tensor([[[0.9, 0.1], [0.8, 0.2]]]))
    target_mask = torch.tensor([[True, True]])

    loss = topk_kl_loss(
        student_logits=student_logits,
        teacher_token_ids=teacher_token_ids,
        teacher_logprobs=teacher_logprobs,
        target_mask=target_mask,
    )

    assert loss.item() < 0.2


def test_topk_kl_loss_ignores_masked_positions() -> None:
    student_logits = torch.tensor(
        [
            [
                [0.0, 5.0, -1.0],
                [-10.0, -10.0, 10.0],
            ]
        ]
    )
    teacher_token_ids = torch.tensor([[[1, 0], [0, 1]]])
    teacher_logprobs = torch.log(torch.tensor([[[0.95, 0.05], [0.95, 0.05]]]))
    target_mask = torch.tensor([[True, False]])

    loss = topk_kl_loss(
        student_logits=student_logits,
        teacher_token_ids=teacher_token_ids,
        teacher_logprobs=teacher_logprobs,
        target_mask=target_mask,
    )

    assert loss.item() < 0.1
