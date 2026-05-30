"""SFT the dense Laguna student for CUDA kernel generation.

Data: SakanaAI/AI-CUDA-Engineer-Archive (PyTorch_Code_Module -> CUDA_Code), correct kernels only,
chat-formatted with Laguna's template. Standard causal-LM cross-entropy on the assistant span
(prompt masked). Recovers chat behaviour + teaches PyTorch->CUDA. Inspired by
dhaya98/gpt-oss-20b-cuda-sft (TRL SFT on the same dataset).

Trainable: routed_dense + lm_head + norms (attention frozen). No teacher needed.
"""
from __future__ import annotations

import argparse
import json
import time
from itertools import islice
from pathlib import Path

import torch
from datasets import load_dataset
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

SYS = ("You are an expert GPU kernel engineer. Convert PyTorch modules into correct, "
       "optimized CUDA kernels.")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-model", default="EvanOLeary/laguna-xs2-dense-k8-kernelmix")
    ap.add_argument("--dataset", default="SakanaAI/AI-CUDA-Engineer-Archive")
    ap.add_argument("--splits", default="level_1,level_2")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--grad-accum-steps", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=1e-5)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


def build_example(tok, row, seq_len):
    py = str(row.get("PyTorch_Code_Module") or "").strip()
    cuda = str(row.get("CUDA_Code") or "").strip()
    if not py or not cuda or not row.get("Correct", False):
        return None
    user = f"Convert this PyTorch module into an optimized CUDA kernel:\n\n```python\n{py}\n```"
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": user}]
    # render to string (apply_chat_template(tokenize=True) is buggy for this tokenizer), then tokenize
    prompt_str = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                         enable_thinking=False)
    full_str = tok.apply_chat_template(
        msgs + [{"role": "assistant", "content": f"```cpp\n{cuda}\n```"}],
        add_generation_prompt=False, tokenize=False, enable_thinking=False)
    prompt_ids = tok(prompt_str, add_special_tokens=False).input_ids
    full_ids = tok(full_str, add_special_tokens=False).input_ids[:seq_len]
    if len(full_ids) < len(prompt_ids) + 8:
        return None
    labels = list(full_ids)
    for i in range(min(len(prompt_ids), len(labels))):
        labels[i] = -100  # mask the prompt
    return full_ids, labels


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = (args.output_dir / "metrics.jsonl").open("a")

    tok = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    print("[load] student", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.student_model, trust_remote_code=True, dtype=torch.bfloat16, device_map={"": args.device})
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # trainable: routed_dense + lm_head + norms
    trainable = 0
    for name, p in model.named_parameters():
        p.requires_grad_(("routed_dense" in name) or ("lm_head" in name) or ("norm" in name.lower()))
        if p.requires_grad:
            trainable += p.numel()
    print(f"[freeze] trainable params: {trainable/1e9:.2f} B", flush=True)
    model.train()
    opt = AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)

    # stream + interleave splits, filter correct kernels
    def rows():
        its = [iter(load_dataset(args.dataset, split=s, streaming=True)) for s in args.splits.split(",")]
        i = 0
        while its:
            try:
                yield next(its[i % len(its)])
            except StopIteration:
                its.pop(i % len(its)); continue
            i += 1

    dev = args.device
    accum = args.grad_accum_steps
    json.dump({"student": args.student_model, "dataset": args.dataset, "splits": args.splits,
               "lr": args.learning_rate, "seq_len": args.seq_len,
               "trainable_b": trainable / 1e9}, (args.output_dir / "config.json").open("w"), indent=2)

    start = time.time()
    step = 0
    micro = 0
    opt.zero_grad(set_to_none=True)
    kept = 0
    for row in rows():
        ex = build_example(tok, row, args.seq_len)
        if ex is None:
            continue
        ids, labels = ex
        kept += 1
        input_ids = torch.tensor([ids], device=dev)
        lab = torch.tensor([labels], device=dev)
        out = model(input_ids=input_ids, labels=lab)
        (out.loss / accum).backward()
        micro += 1
        if micro % accum != 0:
            continue
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        step += 1
        if step > args.max_steps:
            break
        if step == 1 or step % args.log_every == 0:
            row_m = {"step": step, "loss": float(out.loss.detach().cpu()),
                     "elapsed_sec": time.time() - start, "examples_used": kept}
            metrics.write(json.dumps(row_m) + "\n"); metrics.flush()
            print(json.dumps(row_m), flush=True)
        if args.save_every and step % args.save_every == 0:
            model.save_pretrained(args.output_dir / f"checkpoint-step-{step}", safe_serialization=True)
    model.save_pretrained(args.output_dir / "checkpoint-final", safe_serialization=True)
    print(f"[done] {step} steps, {kept} examples", flush=True)


if __name__ == "__main__":
    main()
