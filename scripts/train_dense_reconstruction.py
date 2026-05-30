from __future__ import annotations

import argparse
import inspect
import json
import shutil
import time
from collections.abc import Iterable
from itertools import islice
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from densify.reconstruction import (
    compute_parallel_reconstruction_loss,
    find_reconstruction_layer_ids,
    freeze_for_dense_reconstruction,
)
from densify.reconstruction_data import _format_message, format_sft_row


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
    parser.add_argument(
        "--no-pack-sequences",
        action="store_true",
        help="Disable concatenating short examples into seq_len blocks.",
    )
    parser.add_argument(
        "--require-block-diagonal-attention",
        action="store_true",
        help=(
            "Fail if packed batches use a plain 1D padding mask instead of "
            "block-diagonal attention."
        ),
    )
    parser.add_argument(
        "--max-pad-fraction-warning",
        type=float,
        default=0.25,
        help="Print a loud warning when logged batches exceed this padding fraction.",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--structural-weight", type=float, default=1.0)
    parser.add_argument("--cosine-weight", type=float, default=0.05)
    parser.add_argument("--logit-kl-weight", type=float, default=0.0)
    parser.add_argument(
        "--kl-target-mode",
        choices=["all", "assistant_last"],
        default="all",
        help=(
            "Which logit positions receive KL loss. assistant_last masks KL to "
            "positions predicting the final assistant message in a messages row."
        ),
    )
    parser.add_argument(
        "--train-norms",
        action="store_true",
        help="Also train normalization parameters during reconstruction.",
    )
    parser.add_argument(
        "--train-lm-head",
        action="store_true",
        help="Also train lm_head parameters during reconstruction.",
    )
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument(
        "--save-train-state",
        action="store_true",
        help="Save optimizer/global-step state in each checkpoint for exact resume.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        help="Resume optimizer/global-step state from a checkpoint containing training_state.pt.",
    )
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--disable-cudnn-sdpa", action="store_true")
    parser.add_argument(
        "--layer-ids",
        help="Comma-separated layer ids. Default: all student routed_dense layers.",
    )
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
    pack_sequences: bool = True,
    kl_target_mode: str = "all",
) -> Iterable[dict[str, torch.Tensor]]:
    pending: list[dict[str, torch.Tensor]] = []
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    separator_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else pad_id
    packed_ids: list[int] = []
    packed_kl_mask: list[int] = []
    packed_boundaries: list[int] = []
    packed_doc_starts: list[int] = []

    def append_pending(
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        kl_attention_mask: torch.Tensor,
        boundary_markers: torch.Tensor,
        doc_start_markers: torch.Tensor,
    ) -> dict[str, torch.Tensor] | None:
        pending.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "kl_attention_mask": kl_attention_mask,
                "packing_boundary_markers": boundary_markers,
                "packing_doc_start_markers": doc_start_markers,
                "packing_boundary_count": boundary_markers.sum().reshape(()),
                "packing_doc_start_count": doc_start_markers.sum().reshape(()),
            }
        )
        if len(pending) != batch_size:
            return None
        stacked = {
            key: torch.stack([item[key] for item in pending]).to(device)
            for key in pending[0]
        }
        pending.clear()
        return stacked

    for row in rows:
        try:
            ids, kl_mask_ids = tokenize_reconstruction_row(
                row,
                tokenizer,
                seq_len=seq_len,
                kl_target_mode=kl_target_mode,
            )
        except ValueError:
            continue
        if len(ids) < 2:
            continue
        if pack_sequences:
            if packed_ids and packed_ids[-1] != separator_id:
                packed_ids.append(int(separator_id))
                packed_kl_mask.append(0)
                packed_boundaries.append(1)
                packed_doc_starts.append(0)
            doc_start = [0] * len(ids)
            doc_start[0] = 1
            packed_ids.extend(ids)
            packed_kl_mask.extend(kl_mask_ids)
            packed_boundaries.extend([0] * len(ids))
            packed_doc_starts.extend(doc_start)
            if packed_ids[-1] != separator_id:
                packed_ids.append(int(separator_id))
                packed_kl_mask.append(0)
                packed_boundaries.append(1)
                packed_doc_starts.append(0)
            while len(packed_ids) >= seq_len:
                tensor = torch.tensor(packed_ids[:seq_len], dtype=torch.long)
                kl_mask = torch.tensor(packed_kl_mask[:seq_len], dtype=torch.long)
                boundary_markers = torch.tensor(packed_boundaries[:seq_len], dtype=torch.long)
                doc_start_markers = torch.tensor(packed_doc_starts[:seq_len], dtype=torch.long)
                packed_ids = packed_ids[seq_len:]
                packed_kl_mask = packed_kl_mask[seq_len:]
                packed_boundaries = packed_boundaries[seq_len:]
                packed_doc_starts = packed_doc_starts[seq_len:]
                batch = append_pending(
                    tensor,
                    torch.ones((seq_len,), dtype=torch.long),
                    kl_mask,
                    boundary_markers,
                    doc_start_markers,
                )
                if batch is not None:
                    yield batch
        else:
            tensor = torch.tensor(ids[:seq_len], dtype=torch.long)
            kl_tensor = torch.tensor(kl_mask_ids[:seq_len], dtype=torch.long)
            if tensor.numel() < seq_len:
                padded = torch.full((seq_len,), int(pad_id), dtype=torch.long)
                kl_padded = torch.zeros((seq_len,), dtype=torch.long)
                padded[: tensor.numel()] = tensor
                kl_padded[: kl_tensor.numel()] = kl_tensor
                mask = torch.zeros((seq_len,), dtype=torch.long)
                mask[: tensor.numel()] = 1
            else:
                padded = tensor
                kl_padded = kl_tensor
                mask = torch.ones((seq_len,), dtype=torch.long)
            boundary_markers = torch.zeros((seq_len,), dtype=torch.long)
            doc_start_markers = torch.zeros((seq_len,), dtype=torch.long)
            if mask.sum().item() > 0:
                doc_start_markers[0] = 1
            batch = append_pending(padded, mask, kl_padded, boundary_markers, doc_start_markers)
            if batch is not None:
                yield batch

    if pack_sequences and packed_ids:
        tensor = torch.tensor(packed_ids, dtype=torch.long)
        padded = torch.full((seq_len,), int(pad_id), dtype=torch.long)
        kl_padded = torch.zeros((seq_len,), dtype=torch.long)
        mask = torch.zeros((seq_len,), dtype=torch.long)
        padded[: tensor.numel()] = tensor
        kl_padded[: len(packed_kl_mask)] = torch.tensor(packed_kl_mask, dtype=torch.long)
        mask[: tensor.numel()] = 1
        boundary_markers = torch.zeros((seq_len,), dtype=torch.long)
        doc_start_markers = torch.zeros((seq_len,), dtype=torch.long)
        boundary_markers[: len(packed_boundaries)] = torch.tensor(
            packed_boundaries,
            dtype=torch.long,
        )
        doc_start_markers[: len(packed_doc_starts)] = torch.tensor(
            packed_doc_starts,
            dtype=torch.long,
        )
        batch = append_pending(padded, mask, kl_padded, boundary_markers, doc_start_markers)
        if batch is not None:
            yield batch
    if pending:
        while len(pending) < batch_size:
            padded = torch.full((seq_len,), int(pad_id), dtype=torch.long)
            mask = torch.zeros((seq_len,), dtype=torch.long)
            kl_mask = torch.zeros((seq_len,), dtype=torch.long)
            boundary_markers = torch.zeros((seq_len,), dtype=torch.long)
            doc_start_markers = torch.zeros((seq_len,), dtype=torch.long)
            pending.append(
                {
                    "input_ids": padded,
                    "attention_mask": mask,
                    "kl_attention_mask": kl_mask,
                    "packing_boundary_markers": boundary_markers,
                    "packing_doc_start_markers": doc_start_markers,
                    "packing_boundary_count": boundary_markers.sum().reshape(()),
                    "packing_doc_start_count": doc_start_markers.sum().reshape(()),
                }
            )
        yield {
            key: torch.stack([item[key] for item in pending]).to(device)
            for key in pending[0]
        }


def tokenize_reconstruction_row(
    row: dict[str, Any],
    tokenizer: Any,
    *,
    seq_len: int,
    kl_target_mode: str,
) -> tuple[list[int], list[int]]:
    text = format_sft_row(row)
    ids = tokenizer(text, add_special_tokens=True, truncation=True, max_length=seq_len)[
        "input_ids"
    ]
    if kl_target_mode == "all":
        return ids, [1] * len(ids)
    if kl_target_mode != "assistant_last":
        raise ValueError(f"Unsupported KL target mode: {kl_target_mode}")

    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        return ids, [0] * len(ids)
    target = messages[-1]
    if not isinstance(target, dict) or str(target.get("role", "")).lower() != "assistant":
        return ids, [0] * len(ids)

    prefix_parts = []
    for message in messages[:-1]:
        if isinstance(message, dict):
            rendered = _format_message(message)
            if rendered:
                prefix_parts.append(rendered)
    prefix_text = "\n".join(prefix_parts)
    prefix_ids = (
        tokenizer(prefix_text, add_special_tokens=True)["input_ids"]
        if prefix_text
        else []
    )
    # KL at logit position i trains the distribution over token i+1.
    target_first_logit = max(len(prefix_ids) - 1, 0)
    target_last_logit = max(len(ids) - 2, -1)
    mask = [0] * len(ids)
    for index in range(target_first_logit, target_last_logit + 1):
        if 0 <= index < len(mask):
            mask[index] = 1
    return ids, mask


def parse_layer_ids(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def copy_remote_code_files(model: Any, checkpoint_dir: Path) -> None:
    """Keep local checkpoints reloadable when the model uses HF remote code."""
    source_files: set[Path] = set()
    for obj in (model.__class__, model.config.__class__):
        try:
            source_files.add(Path(inspect.getfile(obj)))
        except TypeError:
            continue

    for source in source_files:
        if source.suffix == ".py" and source.exists():
            shutil.copy2(source, checkpoint_dir / source.name)


def save_checkpoint(
    model: Any,
    checkpoint_dir: Path,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    step: int | None = None,
) -> None:
    model.save_pretrained(checkpoint_dir, safe_serialization=True)
    copy_remote_code_files(model, checkpoint_dir)
    if optimizer is not None:
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "step": step,
            },
            checkpoint_dir / "training_state.pt",
        )


def load_training_state(
    checkpoint_dir: Path,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> int:
    state_path = checkpoint_dir / "training_state.pt"
    if not state_path.exists():
        raise FileNotFoundError(
            f"{state_path} does not exist; this checkpoint is model-only and cannot "
            "be exact-resumed. Restart from it as --student-model to accept a fresh optimizer."
        )
    state = torch.load(state_path, map_location=device)
    optimizer.load_state_dict(state["optimizer"])
    return int(state.get("step") or 0)


def main() -> None:
    args = parse_args()
    if args.disable_cudnn_sdpa and torch.cuda.is_available():
        torch.backends.cuda.enable_cudnn_sdp(False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.jsonl"
    dtype = dtype_from_name(args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        args.teacher_model,
        trust_remote_code=args.trust_remote_code,
    )
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
    trainable_tensors = freeze_for_dense_reconstruction(
        student,
        train_norms=args.train_norms,
        train_lm_head=args.train_lm_head,
    )
    student.train()
    layer_ids = parse_layer_ids(args.layer_ids) or find_reconstruction_layer_ids(teacher, student)
    if not layer_ids:
        raise RuntimeError("No dense reconstruction layers found")

    optimizer = AdamW(
        (param for param in student.parameters() if param.requires_grad),
        lr=args.learning_rate,
    )
    resume_step = 0
    if args.resume_from_checkpoint:
        resume_step = load_training_state(args.resume_from_checkpoint, optimizer, args.device)
        print(
            json.dumps(
                {
                    "event": "resumed_training_state",
                    "checkpoint": str(args.resume_from_checkpoint),
                    "resume_step": resume_step,
                }
            ),
            flush=True,
        )
    dataset_kwargs: dict[str, Any] = {"split": args.split, "streaming": args.streaming}
    if args.dataset == "json" and args.dataset_config:
        dataset_kwargs["data_files"] = args.dataset_config
        dataset = load_dataset(args.dataset, **dataset_kwargs)
    else:
        dataset = load_dataset(args.dataset, args.dataset_config, **dataset_kwargs)
    rows: Iterable[dict[str, Any]] = dataset
    if args.max_examples:
        rows = islice(rows, args.max_examples)
    batches = iter_token_batches(
        rows,
        tokenizer,
        args.seq_len,
        args.batch_size,
        args.device,
        pack_sequences=not args.no_pack_sequences,
        kl_target_mode=args.kl_target_mode,
    )

    config = {
        "teacher_model": args.teacher_model,
        "student_model": args.student_model,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "pack_sequences": not args.no_pack_sequences,
        "pack_separator_token_id": tokenizer.eos_token_id,
        "require_block_diagonal_attention": args.require_block_diagonal_attention,
        "max_pad_fraction_warning": args.max_pad_fraction_warning,
        "learning_rate": args.learning_rate,
        "structural_weight": args.structural_weight,
        "cosine_weight": args.cosine_weight,
        "logit_kl_weight": args.logit_kl_weight,
        "kl_target_mode": args.kl_target_mode,
        "train_norms": args.train_norms,
        "train_lm_head": args.train_lm_head,
        "layer_ids": layer_ids,
        "trainable_tensors": trainable_tensors,
        "save_train_state": args.save_train_state,
        "resume_from_checkpoint": (
            str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None
        ),
        "resume_step": resume_step,
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    start = time.time()
    last_saved_step = resume_step
    for step, batch in enumerate(batches, start=1):
        if step <= resume_step:
            continue
        if step > args.max_steps:
            break
        model_batch = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "kl_attention_mask": batch["kl_attention_mask"],
        }
        if args.require_block_diagonal_attention and (
            model_batch["attention_mask"].dim() < 3
            or model_batch["attention_mask"].shape[-1]
            != model_batch["attention_mask"].shape[-2]
        ):
            raise RuntimeError(
                "PACKING CONTAMINATION RISK: --require-block-diagonal-attention was "
                f"set, but attention_mask has shape {tuple(model_batch['attention_mask'].shape)}. "
                "Packed documents can attend across EOS boundaries."
            )
        optimizer.zero_grad(set_to_none=True)
        result = compute_parallel_reconstruction_loss(
            teacher,
            student,
            model_batch,
            layer_ids=layer_ids,
            cosine_weight=args.cosine_weight,
            logit_kl_weight=args.logit_kl_weight,
            structural_weight=args.structural_weight,
        )
        result.loss.backward()
        optimizer.step()

        if step == 1 or step % args.log_every == 0:
            nominal_tokens = int(model_batch["input_ids"].numel())
            real_tokens = int(model_batch["attention_mask"].sum().item())
            kl_tokens = int(model_batch["kl_attention_mask"].sum().item())
            pad_fraction = 1 - (float(real_tokens) / float(nominal_tokens))
            boundary_count = int(batch["packing_boundary_count"].sum().item())
            doc_start_count = int(batch["packing_doc_start_count"].sum().item())
            boundary_warning = (
                not args.no_pack_sequences
                and doc_start_count > 1
                and boundary_count < doc_start_count - 1
            )
            pad_warning = pad_fraction > args.max_pad_fraction_warning
            row = {
                "step": step,
                "loss": float(result.loss.detach().cpu()),
                "nominal_tokens": nominal_tokens,
                "real_tokens": real_tokens,
                "kl_tokens": kl_tokens,
                "pad_fraction": pad_fraction,
                "packing_boundary_count": boundary_count,
                "packing_doc_start_count": doc_start_count,
                "packing_soft_boundary_warning": boundary_warning,
                "packing_pad_warning": pad_warning,
                "elapsed_sec": time.time() - start,
                "per_layer": result.per_layer,
            }
            if boundary_warning:
                print(
                    "PACKING BOUNDARY WARNING: packed batch has fewer inserted EOS "
                    f"boundaries ({boundary_count}) than doc joins implied by starts "
                    f"({doc_start_count}).",
                    flush=True,
                )
            if pad_warning:
                print(
                    "PACKING PAD WARNING: "
                    f"pad_fraction={pad_fraction:.3f} exceeds "
                    f"--max-pad-fraction-warning={args.max_pad_fraction_warning:.3f}.",
                    flush=True,
                )
            with metrics_path.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)

        if args.save_every and step % args.save_every == 0:
            save_checkpoint(
                student,
                args.output_dir / f"checkpoint-step-{step}",
                optimizer=optimizer if args.save_train_state else None,
                step=step,
            )
            last_saved_step = step

    final_step = min(step, args.max_steps) if "step" in locals() else resume_step
    if final_step != last_saved_step:
        save_checkpoint(
            student,
            args.output_dir / "checkpoint-final",
            optimizer=optimizer if args.save_train_state else None,
            step=final_step,
        )


if __name__ == "__main__":
    main()
