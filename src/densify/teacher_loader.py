from __future__ import annotations

from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.utils.quantization_config import CompressedTensorsConfig

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
    compressed_tensors_run_compressed: bool = False,
):
    dtype = DTYPES.get(torch_dtype)
    if dtype is None:
        expected = ", ".join(sorted(DTYPES))
        raise ValueError(f"Unsupported torch_dtype={torch_dtype!r}; expected one of {expected}")

    model_config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    quantization_config = getattr(model_config, "quantization_config", None)
    if isinstance(quantization_config, dict) and quantization_config.get(
        "quant_method"
    ) == "compressed-tensors":
        quantization_config = CompressedTensorsConfig.from_dict(
            quantization_config,
            run_compressed=compressed_tensors_run_compressed,
        )

    load_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": trust_remote_code,
        "device_map": device_map,
        "low_cpu_mem_usage": True,
    }
    if quantization_config is not None:
        load_kwargs["quantization_config"] = quantization_config

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        **load_kwargs,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model
