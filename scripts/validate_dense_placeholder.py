from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a Laguna dense placeholder checkpoint.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path) if Path(args.model_path).exists() else args.model_path
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=args.trust_remote_code,
    ).to(args.device)
    model.eval()

    input_ids = torch.tensor([[config.bos_token_id or 1, 7, 8]], device=args.device)
    with torch.no_grad():
        out = model(input_ids=input_ids)
        generated = model.generate(input_ids=input_ids, max_new_tokens=2, do_sample=False)

    dense_replacements = sum(
        1
        for _, module in model.named_modules()
        if module.__class__.__name__ == "LagunaDenseMoEReplacement" or hasattr(module, "routed_dense")
    )

    print("tokenizer_loaded=skipped")
    print("model_loaded=true")
    print(f"model_type={config.model_type}")
    print(f"num_dense_replacements={dense_replacements}")
    print(f"forward_logits_shape={list(out.logits.shape)}")
    print(f"generate_shape={list(generated.shape)}")
    print("generate_ok=true")


if __name__ == "__main__":
    main()
