from __future__ import annotations

import json
import re
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


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for message in messages:
        role = str(message.get("role", "user"))
        clean: dict[str, Any] = {"role": role, "content": content_to_text(message.get("content"))}
        if message.get("content") is None:
            clean["content"] = None
        if message.get("tool_calls"):
            clean["tool_calls"] = normalize_tool_calls_for_template(message["tool_calls"])
        if message.get("tool_call_id"):
            clean["tool_call_id"] = message["tool_call_id"]
        if message.get("name"):
            clean["name"] = message["name"]
        normalized.append(clean)
    return normalized


def normalize_tool_calls_for_template(tool_calls: Any) -> list[dict[str, Any]]:
    normalized = []
    if not isinstance(tool_calls, list):
        return normalized
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        clean = dict(call)
        function = dict(clean.get("function") or {})
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                parsed = {"_raw": arguments}
            function["arguments"] = parsed if isinstance(parsed, dict) else {"value": parsed}
        elif arguments is None:
            function["arguments"] = {}
        clean["function"] = function
        normalized.append(clean)
    return normalized


def strip_pool_unfriendly_markup(text: str) -> str:
    return text.replace("</assistant>", "").strip()


def parse_generated_tool_calls(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Extract OpenAI-style tool calls from Laguna text generations.

    The SFT data renders tool calls as assistant text followed by a JSON object:
    {"tool_calls": [...]}. The coding harness needs those calls in the structured
    Chat Completions field, otherwise it will never execute tools.
    """
    cleaned = strip_pool_unfriendly_markup(text)
    payload = first_json_object_with_key(cleaned, "tool_calls")
    if payload is None:
        tagged_call = first_tagged_tool_call(cleaned)
        if tagged_call is None:
            return cleaned, []
        content, call = tagged_call
        return content, [call]
    calls = payload.get("tool_calls")
    if not isinstance(calls, list):
        return cleaned, []
    tool_calls = [normalize_tool_call(call, idx) for idx, call in enumerate(calls, start=1)]
    tool_calls = [call for call in tool_calls if call]
    content = cleaned.replace(json.dumps(payload, sort_keys=True), "").strip()
    if content == cleaned:
        content = cleaned[: cleaned.find("{")].strip() if "{" in cleaned else ""
    return content, tool_calls


def first_tagged_tool_call(text: str) -> tuple[str, dict[str, Any]] | None:
    match = re.search(r"<tool_call>(.*?)</tool_call>", text, flags=re.DOTALL)
    if match:
        body = match.group(1)
        content = text[: match.start()].strip()
    else:
        start = text.find("<tool_call>")
        if start < 0:
            return None
        body = text[start + len("<tool_call>") :]
        content = text[:start].strip()
    if not body.strip():
        return None
    lines = body.splitlines()
    tool_name = ""
    while lines and not tool_name:
        tool_name = lines.pop(0).strip()
    if not tool_name:
        return None
    arguments: dict[str, Any] = {}
    for arg_key, arg_value in re.findall(
        r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>",
        body,
        flags=re.DOTALL,
    ):
        arguments[arg_key.strip()] = coerce_tagged_arg_value(arg_value.strip())
    return content, {
        "id": "generated_tool_1",
        "type": "function",
        "function": {"name": tool_name, "arguments": json.dumps(arguments)},
    }


def coerce_tagged_arg_value(value: str) -> Any:
    if value in {"true", "false"}:
        return value == "true"
    try:
        return int(value)
    except ValueError:
        return value


def first_json_object_with_key(text: str, key: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and key in parsed:
            return parsed
    return None


def normalize_tool_call(call: Any, idx: int) -> dict[str, Any]:
    if not isinstance(call, dict):
        return {}
    function = call.get("function")
    if not isinstance(function, dict):
        return {}
    name = str(function.get("name") or "")
    arguments = function.get("arguments")
    if isinstance(arguments, dict):
        arguments = json.dumps(arguments)
    elif arguments is None:
        arguments = "{}"
    elif not isinstance(arguments, str):
        arguments = json.dumps(arguments)
    if not name:
        return {}
    return {
        "id": str(call.get("id") or f"tool_{idx}"),
        "type": str(call.get("type") or "function"),
        "function": {"name": name, "arguments": arguments},
    }


@dataclass(frozen=True)
class BackendGeneration:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    call_dir: Path | None
    tool_calls: list[dict[str, Any]] | None = None


class LocalHFBackend:
    def __init__(
        self,
        *,
        model_id: str,
        tokenizer_id: str | None = None,
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
        self.tokenizer_id = tokenizer_id or model_id
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
        self.tokenizer = load_tokenizer(self.tokenizer_id, trust_remote_code=self.trust_remote_code)
        self.model = load_teacher_model(
            self.model_id,
            torch_dtype=self.torch_dtype,
            trust_remote_code=self.trust_remote_code,
            device_map=self.device_map,
        )
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = True

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
        raw_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        served_text, tool_calls = parse_generated_tool_calls(raw_text)

        call_dir = self.next_call_dir()
        write_json(call_dir / "request.json", payload)
        write_json(
            call_dir / "metadata.json",
            {
                "model_id": self.model_id,
                "tokenizer_id": self.tokenizer_id,
                "input_token_count": int(input_ids.numel()),
                "generated_token_count": int(generated_tokens.numel()),
                "latency_s": latency_s,
                "tokens_per_second": int(generated_tokens.numel()) / max(latency_s, 1e-6),
            },
        )
        (call_dir / "generated_text.txt").write_text(raw_text, encoding="utf-8")
        (call_dir / "served_text.txt").write_text(served_text, encoding="utf-8")
        write_json(call_dir / "tool_calls.json", tool_calls)
        torch.save(input_ids.detach().cpu(), call_dir / "input_tokens.pt")
        torch.save(generated_tokens.detach().cpu(), call_dir / "generated_tokens.pt")

        return BackendGeneration(
            text=served_text,
            input_tokens=int(input_ids.numel()),
            output_tokens=int(generated_tokens.numel()),
            latency_s=latency_s,
            call_dir=call_dir,
            tool_calls=tool_calls,
        )

    def next_call_dir(self) -> Path:
        self.call_index += 1
        call_dir = self.output_dir / "model_calls" / f"call_{self.call_index:06d}"
        call_dir.mkdir(parents=True, exist_ok=False)
        return call_dir
