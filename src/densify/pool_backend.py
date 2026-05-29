from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from densify.run_artifacts import write_json
from densify.teacher_loader import load_teacher_model, load_tokenizer


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
        return "\n".join(part for part in parts if part)
    return str(content or "")


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalized = []
    for message in messages:
        role = str(message.get("role", "user"))
        normalized.append({"role": role, "content": content_to_text(message.get("content"))})
    return normalized


@dataclass(frozen=True)
class BackendGeneration:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    call_dir: Path | None


class LocalHFBackend:
    def __init__(
        self,
        *,
        model_id: str,
        torch_dtype: str,
        device_map: str,
        trust_remote_code: bool,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_k: int | None,
        top_p: float,
        enable_thinking: bool,
        output_dir: Path,
    ) -> None:
        self.model_id = model_id
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self.trust_remote_code = trust_remote_code
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.enable_thinking = enable_thinking
        self.output_dir = output_dir
        self.tokenizer = None
        self.model = None
        self.call_index = 0

    def load(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return
        self.tokenizer = load_tokenizer(self.model_id, trust_remote_code=self.trust_remote_code)
        self.model = load_teacher_model(
            self.model_id,
            torch_dtype=self.torch_dtype,
            trust_remote_code=self.trust_remote_code,
            device_map=self.device_map,
        )

    @torch.inference_mode()
    def generate(self, payload: dict[str, Any]) -> BackendGeneration:
        self.load()
        assert self.model is not None
        assert self.tokenizer is not None

        messages = normalize_messages(payload.get("messages", []))
        encoded = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=self.enable_thinking,
        )
        if isinstance(encoded, torch.Tensor):
            input_ids = encoded.to(next(self.model.parameters()).device)
            inputs = {"input_ids": input_ids}
        else:
            device = next(self.model.parameters()).device
            inputs = {key: value.to(device) for key, value in encoded.items()}
            input_ids = inputs["input_ids"]

        generation_kwargs: dict[str, Any] = {
            **inputs,
            "max_new_tokens": int(payload.get("max_completion_tokens") or self.max_new_tokens),
            "do_sample": self.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.do_sample:
            generation_kwargs["temperature"] = self.temperature
            if self.top_k is not None:
                generation_kwargs["top_k"] = self.top_k
            generation_kwargs["top_p"] = self.top_p

        start = time.perf_counter()
        output_ids = self.model.generate(**generation_kwargs)
        latency_s = time.perf_counter() - start

        input_len = int(input_ids.shape[-1])
        generated_tokens = output_ids[0, input_len:]
        text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        call_dir = self.next_call_dir()
        write_json(call_dir / "request.json", payload)
        write_json(
            call_dir / "metadata.json",
            {
                "model_id": self.model_id,
                "input_token_count": int(input_ids.numel()),
                "generated_token_count": int(generated_tokens.numel()),
                "latency_s": latency_s,
                "tokens_per_second": int(generated_tokens.numel()) / max(latency_s, 1e-6),
            },
        )
        (call_dir / "generated_text.txt").write_text(text, encoding="utf-8")
        torch.save(input_ids.detach().cpu(), call_dir / "input_tokens.pt")
        torch.save(generated_tokens.detach().cpu(), call_dir / "generated_tokens.pt")

        return BackendGeneration(
            text=text,
            input_tokens=int(input_ids.numel()),
            output_tokens=int(generated_tokens.numel()),
            latency_s=latency_s,
            call_dir=call_dir,
        )

    def next_call_dir(self) -> Path:
        self.call_index += 1
        call_dir = self.output_dir / "model_calls" / f"call_{self.call_index:06d}"
        call_dir.mkdir(parents=True, exist_ok=False)
        return call_dir
