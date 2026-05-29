from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors import safe_open

from densify.dense_checkpoint.moe_tensor_map import classify_laguna_tensor_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Laguna MoE tensor keys from safetensor shards.")
    parser.add_argument("--source-model", default="poolside/Laguna-XS.2")
    parser.add_argument("--local-dir", type=Path, default=Path("repos/hf_laguna_xs2_weights"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/dense_placeholder/laguna_moe_tensor_map.json"),
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Only download config/modeling metadata and inspect available local shards if present.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    patterns = ["*.py", "config.json", "*.safetensors"] if not args.metadata_only else ["*.py", "config.json"]
    snapshot_path = Path(
        snapshot_download(args.source_model, allow_patterns=patterns, local_dir=args.local_dir)
    )
    layer_map: dict[int, dict[str, object]] = defaultdict(
        lambda: {"router": [], "shared_experts": [], "routed_experts": []}
    )
    tensor_files = sorted(snapshot_path.glob("*.safetensors"))
    for tensor_file in tensor_files:
        with safe_open(tensor_file, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                classified = classify_laguna_tensor_key(key)
                if classified.layer_id is None or classified.kind == "other":
                    continue
                shape = list(handle.get_slice(key).get_shape())
                entry = {
                    "key": key,
                    "shape": shape,
                    "expert_id": classified.expert_id,
                    "proj": classified.proj,
                }
                layer = layer_map[classified.layer_id]
                if classified.kind == "router":
                    layer["router"].append(entry)
                elif classified.kind == "shared_expert":
                    layer["shared_experts"].append(entry)
                elif classified.kind == "routed_expert":
                    layer["routed_experts"].append(entry)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_model": args.source_model,
        "snapshot_path": str(snapshot_path),
        "tensor_file_count": len(tensor_files),
        "layers": [
            {"layer_id": layer_id, **values} for layer_id, values in sorted(layer_map.items())
        ],
    }
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"wrote={args.output}")
    print(f"tensor_file_count={len(tensor_files)}")
    print(f"layer_count={len(layer_map)}")


if __name__ == "__main__":
    main()
