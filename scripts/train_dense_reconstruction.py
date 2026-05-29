from __future__ import annotations

import argparse
import json
import time
from itertools import islice
from pathlib import Path
from typing import Any, Iterable

import torch
from datasets import load_dataset
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from densify.reconstruction import (
    compute_parallel_reconstruction_loss,
    find_reconstruction_layer_ids,
    freeze_for_dense_reconstruction,
)
from densify.reconstruction_data import format_sft_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Laguna dense routed layers with teacher-forced layer reconstruction."
    )
    parser.add_argument("--teacher-model", default="poolside/Laguna-XS.2")
    parser.add_argument("--student-model", default="cm2435-new/laguna-xs2-dense-k8-copied-shell")
    parser.add_argument("--dataset", default="nvidia/OpenCodeInstruct")
    parser.add_argument("--dataset-config")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--cosine-weight", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--optimizer", default="adamw", choices=["adamw", "adafactor"],
                        help="adafactor uses ~0 extra state (fits all-layer training on 80GB).")
    parser.add_argument("--grad-accum-steps", type=int, default=1,
                        help="Accumulate this many batches per optimizer step.")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--layer-ids", help="Comma-separated layer ids. Default: all student routed_dense layers.")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def device_map_for(device: str) -> str | dict[str, str]:
    if device in {"cpu", "auto"}:
        return device
    return {"": device}


def iter_token_batches(
    rows: Iterable[dict[str, Any]],
    tokenizer: Any,
    seq_len: int,
    batch_size: int,
    device: str,
) -> Iterable[dict[str, torch.Tensor]]:
    pending: list[torch.Tensor] = []
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    for row in rows:
        try:
            text = format_sft_row(row)
        except ValueError:
            continue
        ids = tokenizer(text, add_special_tokens=True, truncation=True, max_length=seq_len)["input_ids"]
        if len(ids) < 2:
            continue
        tensor = torch.tensor(ids[:seq_len], dtype=torch.long)
        if tensor.numel() < seq_len:
            padded = torch.full((seq_len,), int(pad_id), dtype=torch.long)
            padded[: tensor.numel()] = tensor
            mask = torch.zeros((seq_len,), dtype=torch.long)
            mask[: tensor.numel()] = 1
        else:
            padded = tensor
            mask = torch.ones((seq_len,), dtype=torch.long)
        pending.append(torch.stack([padded, mask]))
        if len(pending) == batch_size:
            stacked = torch.stack(pending)
            pending.clear()
            yield {
                "input_ids": stacked[:, 0].to(device),
                "attention_mask": stacked[:, 1].to(device),
            }


def parse_layer_ids(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.jsonl"
    dtype = dtype_from_name(args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        trust_remote_code=args.trust_remote_code,
        dtype=dtype,
        device_map=device_map_for(args.device),
    )
    student = AutoModelForCausalLM.from_pretrained(
        args.student_model,
        trust_remote_code=args.trust_remote_code,
        dtype=dtype,
        device_map=device_map_for(args.device),
    )

    teacher.eval()
    teacher.requires_grad_(False)
    if hasattr(teacher.config, "use_cache"):
        teacher.config.use_cache = False
    if hasattr(student.config, "use_cache"):
        student.config.use_cache = False
    trainable_tensors = freeze_for_dense_reconstruction(student)
    student.train()
    layer_ids = parse_layer_ids(args.layer_ids) or find_reconstruction_layer_ids(teacher, student)
    if not layer_ids:
        raise RuntimeError("No dense reconstruction layers found")

    trainable_params = [param for param in student.parameters() if param.requires_grad]
    if args.optimizer == "adafactor":
        from transformers.optimization import Adafactor
        optimizer = Adafactor(trainable_params, lr=args.learning_rate,
                              scale_parameter=False, relative_step=False, warmup_init=False)
    else:
        optimizer = AdamW(trainable_params, lr=args.learning_rate)
    dataset = load_dataset(
        args.dataset,
        args.dataset_config,
        split=args.split,
        streaming=args.streaming,
    )
    rows: Iterable[dict[str, Any]] = dataset
    if args.max_examples:
        rows = islice(rows, args.max_examples)
    batches = iter_token_batches(rows, tokenizer, args.seq_len, args.batch_size, args.device)

    config = {
        "teacher_model": args.teacher_model,
        "student_model": args.student_model,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "cosine_weight": args.cosine_weight,
        "layer_ids": layer_ids,
        "trainable_tensors": trainable_tensors,
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    start = time.time()
    accum = max(1, args.grad_accum_steps)
    step = 0
    micro = 0
    optimizer.zero_grad(set_to_none=True)
    for batch in batches:
        result = compute_parallel_reconstruction_loss(
            teacher,
            student,
            batch,
            layer_ids=layer_ids,
            cosine_weight=args.cosine_weight,
        )
        (result.loss / accum).backward()
        micro += 1
        if micro % accum != 0:
            continue
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        if step > args.max_steps:
            break

        if step == 1 or step % args.log_every == 0:
            row = {
                "step": step,
                "loss": float(result.loss.detach().cpu()),
                "elapsed_sec": time.time() - start,
                "per_layer": result.per_layer,
            }
            with metrics_path.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)

        if args.save_every and step % args.save_every == 0:
            student.save_pretrained(args.output_dir / f"checkpoint-step-{step}", safe_serialization=True)

    student.save_pretrained(args.output_dir / "checkpoint-final", safe_serialization=True)


if __name__ == "__main__":
    main()
