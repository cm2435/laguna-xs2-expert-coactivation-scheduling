from __future__ import annotations

import torch
import torch.nn.functional as F


def topk_kl_loss(
    *,
    student_logits: torch.Tensor,
    teacher_token_ids: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    target_mask: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    if student_logits.ndim != 3:
        raise ValueError("student_logits must have shape [batch, seq, vocab]")
    if teacher_token_ids.shape != teacher_logprobs.shape:
        raise ValueError("teacher_token_ids and teacher_logprobs must have the same shape")
    if teacher_token_ids.shape[:2] != student_logits.shape[:2]:
        raise ValueError("teacher rows must align with student batch and sequence dimensions")
    if target_mask.shape != student_logits.shape[:2]:
        raise ValueError("target_mask must have shape [batch, seq]")

    active = target_mask.bool()
    if not torch.any(active):
        return student_logits.new_tensor(0.0)

    token_ids = teacher_token_ids[active]
    logprobs = teacher_logprobs[active]
    logits = student_logits[active]
    student_logprobs = F.log_softmax(logits / temperature, dim=-1)
    student_topk_logprobs = student_logprobs.gather(dim=-1, index=token_ids)
    teacher_probs = torch.softmax(logprobs / temperature, dim=-1)
    return F.kl_div(student_topk_logprobs, teacher_probs, reduction="batchmean", log_target=False)
