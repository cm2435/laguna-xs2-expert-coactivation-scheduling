from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int
    temperature: float
    top_k: int | None
    top_p: float
    do_sample: bool
    enable_thinking: bool
    batch_size: int


@dataclass(frozen=True)
class TeacherSmokeConfig:
    model_id: str
    torch_dtype: str
    trust_remote_code: bool
    device_map: str
    compressed_tensors_run_compressed: bool
    prompt_path: Path
    output_dir: Path
    generation: GenerationConfig


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in config file: {path}")
    return data


def load_teacher_smoke_config(path: str | Path) -> TeacherSmokeConfig:
    raw = load_yaml(path)
    gen = raw["generation"]
    return TeacherSmokeConfig(
        model_id=str(raw["model_id"]),
        torch_dtype=str(raw.get("torch_dtype", "bfloat16")),
        trust_remote_code=bool(raw.get("trust_remote_code", True)),
        device_map=str(raw.get("device_map", "auto")),
        compressed_tensors_run_compressed=bool(
            raw.get("compressed_tensors_run_compressed", False)
        ),
        prompt_path=Path(raw["prompt_path"]),
        output_dir=Path(raw["output_dir"]),
        generation=GenerationConfig(
            max_new_tokens=int(gen.get("max_new_tokens", 256)),
            temperature=float(gen.get("temperature", 0.7)),
            top_k=int(gen["top_k"]) if gen.get("top_k") is not None else None,
            top_p=float(gen.get("top_p", 0.95)),
            do_sample=bool(gen.get("do_sample", True)),
            enable_thinking=bool(gen.get("enable_thinking", True)),
            batch_size=int(gen.get("batch_size", 1)),
        ),
    )
