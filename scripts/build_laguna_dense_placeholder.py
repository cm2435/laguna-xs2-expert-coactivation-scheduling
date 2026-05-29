from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from densify.dense_checkpoint.build_placeholder import (
    build_tiny_placeholder,
    copied_shell_key_report,
    write_laguna_dense_remote_code,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Laguna XS.2 dense placeholder checkpoints.")
    parser.add_argument("--target-dir", required=True, type=Path)
    parser.add_argument("--k-routed", default=8, type=int)
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="Build a tiny local checkpoint for HF remote-code smoke testing.",
    )
    parser.add_argument("--source-model", default="poolside/Laguna-XS.2")
    parser.add_argument("--source-meta-dir", type=Path)
    parser.add_argument(
        "--emit-remote-code-only",
        action="store_true",
        help="Write patched Laguna dense remote-code/config without model weights.",
    )
    parser.add_argument("--init", default="random", choices=["random", "selected-concat"])
    parser.add_argument("--copy-non-moe", action="store_true")
    parser.add_argument("--copy-shared-expert", action="store_true")
    parser.add_argument("--push-to-hub")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--max-shard-size", default="5GB")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.tiny:
        path = build_tiny_placeholder(args.target_dir, k_routed=args.k_routed)
        print(f"wrote_tiny_placeholder={path}")
        return

    if args.emit_remote_code_only:
        source_meta_dir = args.source_meta_dir
        if source_meta_dir is None:
            source_meta_dir = Path(
                snapshot_download(
                    args.source_model,
                    allow_patterns=["*.py", "config.json", "tokenizer_config.json", "generation_config.json"],
                    local_dir="repos/hf_laguna_xs2_meta",
                )
            )
        path = write_laguna_dense_remote_code(
            source_meta_dir,
            args.target_dir,
            k_routed=args.k_routed,
        )
        print(f"wrote_laguna_dense_remote_code={path}")
        return

    if not args.copy_non_moe or not args.copy_shared_expert:
        raise SystemExit(
            "Full-size placeholders must pass --copy-non-moe and --copy-shared-expert. "
            "Use --tiny for local all-random smoke tests."
        )

    source_dir = args.source_meta_dir
    if source_dir is None:
        source_dir = Path(
            snapshot_download(
                args.source_model,
                allow_patterns=["*"],
            )
        )
    write_laguna_dense_remote_code(source_dir, args.target_dir, k_routed=args.k_routed)

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.torch_dtype]

    target_config = AutoConfig.from_pretrained(args.target_dir, trust_remote_code=True)
    target_model = AutoModelForCausalLM.from_config(
        target_config,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    source_model = AutoModelForCausalLM.from_pretrained(
        source_dir,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )

    source_state = source_model.state_dict()
    target_state = target_model.state_dict()
    report = copied_shell_key_report(source_state.keys(), target_state.keys())
    copied = 0
    for key, target_tensor in target_state.items():
        source_tensor = source_state.get(key)
        if source_tensor is None or source_tensor.shape != target_tensor.shape:
            continue
        target_tensor.copy_(source_tensor.to(device=target_tensor.device, dtype=target_tensor.dtype))
        copied += 1

    report["copied_tensors_with_matching_shape"] = copied
    report_path = args.target_dir / "copied_shell_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    del source_model
    del source_state

    target_model.save_pretrained(
        args.target_dir,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    print(f"wrote_full_copied_shell={args.target_dir}")
    print(f"copied_shell_report={report_path}")

    if args.push_to_hub:
        from huggingface_hub import upload_folder

        upload_folder(
            repo_id=args.push_to_hub,
            folder_path=args.target_dir,
            repo_type="model",
        )
        print(f"pushed_to_hub={args.push_to_hub}")


if __name__ == "__main__":
    main()
