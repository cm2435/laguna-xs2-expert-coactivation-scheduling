from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from densify.code_scoring import score_code_generation
from densify.prompt_data import load_jsonl_prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tiny coding smoke eval for a dense Laguna checkpoint.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokenizer", default="poolside/Laguna-XS.2")
    parser.add_argument("--prompt-path", default="data/prompts/python_smoke.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
        dtype=dtype_from_name(args.dtype),
        device_map=args.device_map,
        low_cpu_mem_usage=True,
    )
    model.eval()

    summary = {
        "num_prompts": 0,
        "non_empty": 0,
        "python_like": 0,
        "parse_ok": 0,
        "tests_ok": 0,
        "total_generated_tokens": 0,
        "total_latency_s": 0.0,
    }
    rows: list[dict[str, object]] = []

    for prompt in load_jsonl_prompts(args.prompt_path)[: args.limit]:
        messages = [
            {
                "role": "user",
                "content": (
                    "Return a single Python code block with the requested function.\n\n"
                    f"Task:\n{prompt.prompt}"
                ),
            }
        ]
        encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=args.enable_thinking,
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
        latency_s = time.perf_counter() - start
        generated = output_ids[0, input_len:]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        score = score_code_generation(text, prompt.tests)

        row = {
            "id": prompt.id,
            "prompt": prompt.prompt,
            "raw_generation": text,
            "extracted_code": score.extracted_code,
            "parse_ok": score.parse_ok,
            "tests_ok": score.tests_ok,
            "test_stdout": score.test_stdout,
            "test_stderr": score.test_stderr,
            "latency_s": latency_s,
            "generated_tokens": int(generated.numel()),
        }
        rows.append(row)

        summary["num_prompts"] += 1
        summary["non_empty"] += int(bool(text.strip()))
        summary["python_like"] += int("def " in score.extracted_code or "```python" in text.lower())
        summary["parse_ok"] += int(score.parse_ok)
        summary["tests_ok"] += int(score.tests_ok)
        summary["total_generated_tokens"] += int(generated.numel())
        summary["total_latency_s"] += latency_s

        print(
            json.dumps(
                {
                    "id": prompt.id,
                    "generated_tokens": int(generated.numel()),
                    "parse_ok": score.parse_ok,
                    "tests_ok": score.tests_ok,
                    "preview": text[:160].replace("\n", "\\n"),
                }
            ),
            flush=True,
        )

    n = max(int(summary["num_prompts"]), 1)
    summary.update(
        {
            "non_empty_rate": summary["non_empty"] / n,
            "python_like_rate": summary["python_like"] / n,
            "parse_ok_rate": summary["parse_ok"] / n,
            "tests_ok_rate": summary["tests_ok"] / n,
            "tokens_per_second": summary["total_generated_tokens"]
            / max(float(summary["total_latency_s"]), 1e-6),
        }
    )

    (args.output_dir / "generations.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("SUMMARY", json.dumps(summary, indent=2))
    print(f"wrote={args.output_dir}")


if __name__ == "__main__":
    main()
