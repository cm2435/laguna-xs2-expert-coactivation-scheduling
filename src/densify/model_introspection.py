from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

LAYER_CLASS_MARKERS = (
    "decoderlayer",
    "transformerlayer",
    "lagunalayer",
)


@dataclass(frozen=True)
class ModuleSummary:
    name: str
    class_name: str
    parameter_count: int


def count_direct_parameters(module: Any) -> int:
    return sum(param.numel() for param in module.parameters(recurse=False))


def find_candidate_moe_modules(model: Any) -> list[ModuleSummary]:
    candidates: list[ModuleSummary] = []
    for name, module in model.named_modules():
        cls = module.__class__.__name__.lower()
        child_names = {child_name.lower() for child_name, _ in module.named_children()}
        looks_like_moe = (
            "moe" in cls
            or "expert" in cls
            or "experts" in child_names
            or ("gate" in child_names and "shared_expert" in child_names)
        )
        if looks_like_moe:
            candidates.append(
                ModuleSummary(
                    name=name,
                    class_name=module.__class__.__name__,
                    parameter_count=count_direct_parameters(module),
                )
            )
    return candidates


def architecture_summary(model: Any, model_id: str, torch_dtype: str) -> dict[str, Any]:
    transformer_layers = [
        name
        for name, module in model.named_modules()
        if module.__class__.__name__.lower().endswith(LAYER_CLASS_MARKERS)
    ]
    candidates = find_candidate_moe_modules(model)
    return {
        "model_id": model_id,
        "torch_dtype": torch_dtype,
        "model_class": model.__class__.__name__,
        "num_parameters_seen": sum(p.numel() for p in model.parameters()),
        "num_transformer_layers": len(transformer_layers),
        "candidate_moe_modules": [asdict(item) for item in candidates],
    }
