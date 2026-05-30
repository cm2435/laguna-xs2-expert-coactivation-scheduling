from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import time
from itertools import islice
from pathlib import Path
from typing import Any, Iterable

import torch
from datasets import load_dataset
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SFT recovery for the dense Laguna student (chat-template formatted).")
    p.add_argument("--model", default="./recon_model", help="student model: local dir or HF id")
    p.add_argument("--dataset", default="nvidia/OpenCodeInstruct")
    p.add_argument("--dataset-config")
    p.add_argument("--split", default="train")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--warmup-steps", type=int, default=-1, help="LR warmup steps; -1 = 3%% of max-steps")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--max-examples", type=int)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=list(DTYPES))
    p.add_argument("--trust-remote-code", action="store_true", default=True)
    p.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def device_map_for(device: str):
    return device if device in {"cpu", "auto"} else {"": device}


def row_to_messages(row: dict[str, Any]) -> list[dict[str, str]] | None:
    """Turn a dataset row into chat messages for apply_chat_template."""
    if row.get("python_code") and row.get("triton_code"):
        prompt = "Convert this PyTorch module into an optimized Triton kernel:" + chr(10) + chr(10) + str(row["python_code"]).strip()
        return [{"role": "user", "content": prompt}, {"role": "assistant", "content": str(row["triton_code"]).strip()}]
    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        out = []
        for m in messages:
            if isinstance(m, dict) and m.get("content"):
                role = str(m.get("role", "user")).lower().strip()
                out.append({"role": "assistant" if role == "assistant" else "user", "content": str(m["content"]).strip()})
        return out or None
    instruction = str(row.get("instruction") or row.get("prompt") or row.get("question") or "").strip()
    extra = str(row.get("input") or "").strip()
    output = str(row.get("output") or row.get("completion") or row.get("response") or row.get("answer") or "").strip()
    if extra:
        instruction = f"{instruction}\n\n{extra}".strip() if instruction else extra
    if instruction and output:
        return [{"role": "user", "content": instruction}, {"role": "assistant", "content": output}]
    return None


def _collate(seqs: list[torch.Tensor], pad_id: int, device: str) -> dict[str, torch.Tensor]:
    maxlen = max(int(s.numel()) for s in seqs)
    input_ids = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), maxlen), dtype=torch.long)
    for i, s in enumerate(seqs):
        input_ids[i, : s.numel()] = s
        attn[i, : s.numel()] = 1
    return {"input_ids": input_ids.to(device), "attention_mask": attn.to(device)}


def iter_token_batches(rows, tokenizer, seq_len, batch_size, device):
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    pending: list[torch.Tensor] = []
    for row in rows:
        messages = row_to_messages(row)
        if not messages:
            continue
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        except Exception:
            continue
        ids = tokenizer(text, add_special_tokens=False, truncation=True, max_length=seq_len)["input_ids"]
        if len(ids) < 2:
            continue
        pending.append(torch.tensor(ids, dtype=torch.long))
        if len(pending) == batch_size:
            yield _collate(pending, pad_id, device)
            pending = []
    if pending:
        yield _collate(pending, pad_id, device)


def sft_loss(model, batch):
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = out.logits[:, :-1, :]
    labels = batch["input_ids"][:, 1:].clone()
    labels[batch["attention_mask"][:, 1:] == 0] = -100
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(), labels.reshape(-1), ignore_index=-100
    )


def save_checkpoint(model, source_model, checkpoint_dir):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir, safe_serialization=True)
    if os.path.isdir(source_model):
        for py in glob.glob(os.path.join(source_model, "*.py")):
            shutil.copy2(py, checkpoint_dir)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.jsonl"
    dtype = DTYPES[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code, dtype=dtype,
        device_map=device_map_for(args.device),
    )
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model.train()

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    warmup = args.warmup_steps if args.warmup_steps >= 0 else max(10, int(0.03 * args.max_steps))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, args.max_steps)
    dataset = load_dataset(args.dataset, args.dataset_config, split=args.split, streaming=args.streaming)
    rows = dataset if not args.max_examples else islice(dataset, args.max_examples)
    batches = iter_token_batches(rows, tokenizer, args.seq_len, args.batch_size, args.device)

    (args.output_dir / "config.json").write_text(json.dumps({
        "model": args.model, "dataset": args.dataset, "split": args.split,
        "seq_len": args.seq_len, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate, "max_steps": args.max_steps,
        "warmup_steps": warmup, "lr_schedule": "cosine",
        "format": "apply_chat_template",
    }, indent=2) + "\n")

    start = time.time()
    for step, batch in enumerate(batches, start=1):
        if step > args.max_steps:
            break
        optimizer.zero_grad(set_to_none=True)
        loss = sft_loss(model, batch)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step == 1 or step % args.log_every == 0:
            row = {"step": step, "loss": float(loss.detach().cpu()), "lr": scheduler.get_last_lr()[0], "elapsed_sec": time.time() - start}
            with metrics_path.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        if args.save_every and step % args.save_every == 0:
            save_checkpoint(model, args.model, args.output_dir / f"checkpoint-step-{step}")
    save_checkpoint(model, args.model, args.output_dir / "checkpoint-final")


if __name__ == "__main__":
    main()
