from __future__ import annotations

import argparse
import hashlib
import json
import time
from itertools import cycle
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from densify.rollout_sft.kd import topk_kl_loss
from densify.rollout_sft.tokenize import tokenize_sft_row
from densify.rollout_sft.train import (
    collate_kd_tokenized,
    collate_tokenized,
    set_sft_trainable_parameters,
    tokenize_kd_row,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SFT a dense Laguna checkpoint on rollout JSONL.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer-model")
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--validation-dataset", type=Path)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--kd-dataset", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sort-by-length", action="store_true")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--sft-weight", type=float, default=1.0)
    parser.add_argument("--kd-weight", type=float, default=0.0)
    parser.add_argument("--kd-temperature", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--attn-implementation", choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--disable-cudnn-sdpa", action="store_true")
    parser.add_argument("--train-norms", action="store_true")
    parser.add_argument("--train-lm-head", action="store_true")
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument(
        "--allow-placeholder-base",
        action="store_true",
        help="Allow SFT directly from a copied-shell placeholder with random routed dense weights.",
    )
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.disable_cudnn_sdpa and torch.cuda.is_available():
        torch.backends.cuda.enable_cudnn_sdp(False)
    args.output_dir.mkdir(parents=True, exist_ok=bool(args.resume_from_checkpoint))
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    rows = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise SystemExit(f"empty dataset: {args.dataset}")
    if args.validation_dataset is not None:
        val_rows = [
            json.loads(line) for line in args.validation_dataset.read_text(encoding="utf-8").splitlines() if line
        ]
        train_rows = rows
    elif args.validation_fraction > 0:
        train_rows, val_rows = split_train_validation_rows(rows, args.validation_fraction)
    else:
        train_rows, val_rows = rows, []
    if not train_rows:
        raise SystemExit("empty training split after validation split")
    kd_rows = []
    if args.kd_dataset is not None:
        kd_rows = [json.loads(line) for line in args.kd_dataset.read_text(encoding="utf-8").splitlines() if line]
        if not kd_rows:
            raise SystemExit(f"empty KD dataset: {args.kd_dataset}")
        if args.kd_weight == 0.0:
            args.kd_weight = 0.3

    tokenizer_model = args.tokenizer_model or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.sort_by_length:
        train_rows = sort_rows_by_token_length(
            train_rows,
            tokenizer,
            seq_len=args.seq_len,
            enable_thinking=not args.disable_thinking,
        )
        val_rows = sort_rows_by_token_length(
            val_rows,
            tokenizer,
            seq_len=args.seq_len,
            enable_thinking=not args.disable_thinking,
        )
    model_path = args.resume_from_checkpoint or args.model
    if args.resume_from_checkpoint is None:
        assert_not_placeholder_base(args.model, allow_placeholder_base=args.allow_placeholder_base)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
    ).to(args.device)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    trainable = set_sft_trainable_parameters(
        model,
        train_norms=args.train_norms,
        train_lm_head=args.train_lm_head,
    )
    model.train()
    optimizer = AdamW((param for param in model.parameters() if param.requires_grad), lr=args.lr)
    start_step = 1
    if args.resume_from_checkpoint:
        state_path = args.resume_from_checkpoint / "trainer_state.pt"
        if not state_path.exists():
            raise FileNotFoundError(f"missing trainer state: {state_path}")
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"]) + 1
        consumed_rows = (start_step - 1) * args.batch_size
        train_rows = rotate_rows(train_rows, consumed_rows)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    metrics = args.output_dir / "metrics.jsonl"
    (args.output_dir / "config.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "model_load_path": str(model_path),
                "resume_from_checkpoint": str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None,
                "tokenizer_model": tokenizer_model,
                "dataset": str(args.dataset),
                "validation_dataset": str(args.validation_dataset) if args.validation_dataset else None,
                "validation_fraction": args.validation_fraction,
                "train_rows": len(train_rows),
                "validation_rows": len(val_rows),
                "eval_every": args.eval_every,
                "eval_batches": args.eval_batches,
                "no_save": args.no_save,
                "kd_dataset": str(args.kd_dataset) if args.kd_dataset else None,
                "seq_len": args.seq_len,
                "max_steps": args.max_steps,
                "start_step": start_step,
                "batch_size": args.batch_size,
                "sort_by_length": args.sort_by_length,
                "lr": args.lr,
                "sft_weight": args.sft_weight,
                "kd_weight": args.kd_weight,
                "kd_temperature": args.kd_temperature,
                "dtype": args.dtype,
                "attn_implementation": args.attn_implementation,
                "disable_cudnn_sdpa": args.disable_cudnn_sdpa,
                "trainable_parameters": trainable,
                "enable_thinking": not args.disable_thinking,
                "allow_placeholder_base": args.allow_placeholder_base,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    started = time.time()
    iterator = cycle(train_rows)
    val_iterator = cycle(val_rows) if val_rows else None
    kd_iterator = cycle(kd_rows) if kd_rows else None
    for step in range(start_step, args.max_steps + 1):
        batch_rows = [next(iterator) for _ in range(args.batch_size)]
        tokenized = [
            tokenize_sft_row(row, tokenizer, args.seq_len, enable_thinking=not args.disable_thinking)
            for row in batch_rows
        ]
        input_ids, labels, attention_mask = collate_tokenized(tokenized, pad_id)
        batch_input_tokens = int(attention_mask.sum().item())
        batch_trainable_tokens = int((labels != -100).sum().item())
        if batch_trainable_tokens <= 0:
            raise RuntimeError(
                "batch has zero trainable tokens; check dataset filtering, seq_len, and loss masking"
            )
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        attention_mask = attention_mask.to(args.device)
        optimizer.zero_grad(set_to_none=True)
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        row_weights = torch.tensor(
            [float(row.get("weight", 1.0)) for row in batch_rows],
            dtype=output.logits.dtype,
            device=output.logits.device,
        )
        ce_loss = weighted_ce_loss(output.logits, labels, row_weights)
        kd_loss = output.loss.new_tensor(0.0)
        if kd_iterator is not None:
            kd_batch_rows = [next(kd_iterator) for _ in range(args.batch_size)]
            kd_tokenized = [tokenize_kd_row(row, tokenizer, args.seq_len) for row in kd_batch_rows]
            (
                kd_input_ids,
                kd_labels,
                kd_attention_mask,
                kd_target_mask,
                teacher_token_ids,
                teacher_logprobs,
            ) = collate_kd_tokenized(kd_tokenized, pad_id)
            kd_input_ids = kd_input_ids.to(args.device)
            kd_labels = kd_labels.to(args.device)
            kd_attention_mask = kd_attention_mask.to(args.device)
            kd_target_mask = kd_target_mask.to(args.device)
            teacher_token_ids = teacher_token_ids.to(args.device)
            teacher_logprobs = teacher_logprobs.to(args.device)
            kd_output = model(input_ids=kd_input_ids, attention_mask=kd_attention_mask, labels=kd_labels)
            kd_loss = topk_kl_loss(
                student_logits=kd_output.logits,
                teacher_token_ids=teacher_token_ids,
                teacher_logprobs=teacher_logprobs,
                target_mask=kd_target_mask,
                temperature=args.kd_temperature,
            )
        loss = args.sft_weight * ce_loss + args.kd_weight * kd_loss
        loss.backward()
        optimizer.step()
        if step == 1 or step % args.log_every == 0:
            val_ce_loss = None
            if val_iterator is not None and args.eval_every > 0 and (step == 1 or step % args.eval_every == 0):
                val_ce_loss = evaluate_ce_loss(
                    model,
                    tokenizer,
                    val_iterator,
                    pad_id=pad_id,
                    seq_len=args.seq_len,
                    batch_size=args.batch_size,
                    eval_batches=args.eval_batches,
                    device=args.device,
                    enable_thinking=not args.disable_thinking,
                )
            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "ce_loss": float(ce_loss.detach().cpu()),
                "kd_loss": float(kd_loss.detach().cpu()),
                "val_ce_loss": val_ce_loss,
                "batch_input_tokens": batch_input_tokens,
                "batch_trainable_tokens": batch_trainable_tokens,
                "tokens_per_second_since_start": batch_input_tokens * step / max(time.time() - started, 1e-6),
                "peak_cuda_memory_gb": _peak_cuda_memory_gb(args.device),
                "elapsed_sec": time.time() - started,
            }
            with metrics.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        if args.save_every and step % args.save_every == 0:
            save_training_checkpoint(
                args.output_dir / f"checkpoint-step-{step}",
                model=model,
                optimizer=optimizer,
                step=step,
                args=args,
            )
    if not args.no_save:
        save_training_checkpoint(
            args.output_dir / "checkpoint-final",
            model=model,
            optimizer=optimizer,
            step=args.max_steps,
            args=args,
        )


def split_train_validation_rows(
    rows: list[dict],
    validation_fraction: float,
) -> tuple[list[dict], list[dict]]:
    if not 0 <= validation_fraction < 1:
        raise ValueError("--validation-fraction must be in [0, 1)")
    train_rows = []
    val_rows = []
    for row in rows:
        split_key = str(row.get("task_id") or row.get("source_rollout") or row.get("id") or "")
        bucket = _stable_bucket(split_key)
        if bucket < validation_fraction:
            val_rows.append(row)
        else:
            train_rows.append(row)
    if not val_rows and len(rows) > 1 and validation_fraction > 0:
        val_rows = rows[-1:]
        train_rows = rows[:-1]
    return train_rows, val_rows


def sort_rows_by_token_length(
    rows: list[dict],
    tokenizer,
    *,
    seq_len: int,
    enable_thinking: bool,
) -> list[dict]:
    rows_with_lengths = []
    for row in rows:
        tokenized = tokenize_sft_row(row, tokenizer, seq_len, enable_thinking=enable_thinking)
        trainable_tokens = sum(label != -100 for label in tokenized["labels"])
        if trainable_tokens <= 0:
            raise RuntimeError(f"row has zero trainable tokens after tokenization: {row.get('id')}")
        rows_with_lengths.append((len(tokenized["input_ids"]), row))
    return [row for _, row in sorted(rows_with_lengths, key=lambda item: item[0])]


def rotate_rows(rows: list[dict], consumed_rows: int) -> list[dict]:
    if not rows:
        return rows
    offset = consumed_rows % len(rows)
    return rows[offset:] + rows[:offset]


def assert_not_placeholder_base(model_path: str, *, allow_placeholder_base: bool) -> None:
    if allow_placeholder_base:
        return
    report_path = Path(model_path) / "copied_shell_report.json"
    if not report_path.exists():
        return
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Could not parse copied shell report at {report_path}: {exc}") from exc
    random_keys = int(report.get("random_routed_dense_keys") or 0)
    if random_keys > 0:
        raise SystemExit(
            f"Refusing to SFT from placeholder base {model_path}: "
            f"{random_keys} random routed dense tensors remain. "
            "Use a reconstructed checkpoint or pass --allow-placeholder-base intentionally."
        )


def weighted_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    row_weights: torch.Tensor,
) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    token_losses = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(shift_labels)
    target_mask = shift_labels != -100
    weights = row_weights.to(device=token_losses.device, dtype=token_losses.dtype).view(-1, 1)
    weighted_mask = target_mask.to(token_losses.dtype) * weights
    denominator = weighted_mask.sum().clamp_min(1.0)
    return (token_losses * weighted_mask).sum() / denominator


def save_training_checkpoint(
    path: Path,
    *,
    model,
    optimizer,
    step: int,
    args: argparse.Namespace,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
        },
        path / "trainer_state.pt",
    )


def evaluate_ce_loss(
    model,
    tokenizer,
    iterator,
    *,
    pad_id: int,
    seq_len: int,
    batch_size: int,
    eval_batches: int,
    device: str,
    enable_thinking: bool,
) -> float:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    batches = 0
    with torch.no_grad():
        for _ in range(eval_batches):
            batch_rows = [next(iterator) for _ in range(batch_size)]
            tokenized = [
                tokenize_sft_row(row, tokenizer, seq_len, enable_thinking=enable_thinking)
                for row in batch_rows
            ]
            input_ids, labels, attention_mask = collate_tokenized(tokenized, pad_id)
            output = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                labels=labels.to(device),
            )
            total_loss += float(output.loss.detach().cpu())
            batches += 1
    if was_training:
        model.train()
    return total_loss / max(batches, 1)


def _stable_bucket(value: str) -> float:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    integer = int.from_bytes(digest[:8], byteorder="big")
    return integer / float(2**64)


def _peak_cuda_memory_gb(device: str) -> float | None:
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / 1024**3


if __name__ == "__main__":
    main()
