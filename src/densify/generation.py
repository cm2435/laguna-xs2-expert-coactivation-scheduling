from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from densify.config import GenerationConfig
from densify.prompt_data import CodingPrompt


@dataclass(frozen=True)
class GenerationResult:
    text: str
    latency_s: float
    generated_tokens: int


def first_model_device(model) -> torch.device:
    return next(model.parameters()).device


def format_coding_messages(prompt: CodingPrompt) -> list[dict[str, str]]:
    user_prompt = (
        "Return a single Python code block with the requested function.\n\n"
        f"Task:\n{prompt.prompt}"
    )
    return [{"role": "user", "content": user_prompt}]


@torch.inference_mode()
def generate_one(model, tokenizer, prompt: CodingPrompt, cfg: GenerationConfig) -> GenerationResult:
    device = first_model_device(model)
    input_ids = tokenizer.apply_chat_template(
        format_coding_messages(prompt),
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=cfg.enable_thinking,
    ).to(device)
    input_len = int(input_ids.shape[-1])

    generation_kwargs = {
        "inputs": input_ids,
        "max_new_tokens": cfg.max_new_tokens,
        "do_sample": cfg.do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if cfg.do_sample:
        generation_kwargs["temperature"] = cfg.temperature
        if cfg.top_k is not None:
            generation_kwargs["top_k"] = cfg.top_k
        generation_kwargs["top_p"] = cfg.top_p

    start = time.perf_counter()
    output_ids = model.generate(**generation_kwargs)
    latency_s = time.perf_counter() - start
    generated = output_ids[0, input_len:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return GenerationResult(
        text=text,
        latency_s=latency_s,
        generated_tokens=int(generated.numel()),
    )
