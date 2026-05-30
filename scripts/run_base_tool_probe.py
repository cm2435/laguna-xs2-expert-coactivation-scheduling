from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from densify.pool_backend import parse_generated_tool_calls

REQUIRED_ARGS = {
    "shell": "command",
    "read_file": "path",
    "apply_patch": "patch",
    "exit": None,
}


def summarize_generation(text: str) -> dict:
    _, calls = parse_generated_tool_calls(text)
    if not calls:
        return {"parseable_tool_call": False, "tool_name": None, "has_required_arg": False}
    call = calls[0]
    name = call["function"]["name"]
    args = json.loads(call["function"].get("arguments") or "{}")
    required = REQUIRED_ARGS.get(name)
    has_required = required is None if name in REQUIRED_ARGS else False
    if required:
        has_required = bool(args.get(required))
    return {
        "parseable_tool_call": name in REQUIRED_ARGS,
        "tool_name": name,
        "has_required_arg": has_required,
        "arguments": args,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe a dense base for one-shot Laguna tool-call ability."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokenizer", default="poolside/Laguna-XS.2")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--disable-cudnn-sdpa", action="store_true")
    args = parser.parse_args()

    if args.disable_cudnn_sdpa and torch.cuda.is_available():
        torch.backends.cuda.enable_cudnn_sdp(False)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    messages = [
        {
            "role": "system",
            "content": (
                "You are a coding agent. Emit exactly one Laguna tagged tool call. "
                "Valid tools: shell, read_file, apply_patch, exit."
            ),
        },
        {
            "role": "user",
            "content": (
                "Repository: django/django. Issue: URLValidator accepts invalid characters in "
                "username/password. Inspect the repository to find the likely validator file. "
                "Emit one tool call."
            ),
        },
    ]
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=True,
    )
    if isinstance(encoded, torch.Tensor):
        inputs = {"input_ids": encoded.to(model.device)}
    else:
        inputs = {key: value.to(model.device) for key, value in encoded.items()}
    input_len = int(inputs["input_ids"].shape[-1])
    start = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0, input_len:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    summary = summarize_generation(text)
    summary.update(
        {
            "model_path": args.model_path,
            "generated_tokens": int(generated.numel()),
            "latency_s": time.perf_counter() - start,
        }
    )
    (args.output_dir / "generation.txt").write_text(text, encoding="utf-8")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
