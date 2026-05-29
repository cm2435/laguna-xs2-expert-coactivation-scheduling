from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "auto": "auto",
}


def load_tokenizer(model_id: str, trust_remote_code: bool = True):
    tokenizer: Any = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_teacher_model(
    model_id: str,
    torch_dtype: str = "bfloat16",
    trust_remote_code: bool = True,
    device_map: str = "auto",
):
    dtype = DTYPES.get(torch_dtype)
    if dtype is None:
        expected = ", ".join(sorted(DTYPES))
        raise ValueError(f"Unsupported torch_dtype={torch_dtype!r}; expected one of {expected}")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model
