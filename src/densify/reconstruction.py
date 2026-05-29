from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ReconstructionResult:
    loss: torch.Tensor
    per_layer: dict[int, dict[str, float | int]]


def _get_layers(model: nn.Module) -> nn.ModuleList | list[nn.Module]:
    layers = getattr(getattr(model, "model", model), "layers", None)
    if layers is None:
        raise ValueError("Expected model.model.layers or model.layers")
    return layers


def _first_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            if isinstance(item, torch.Tensor):
                return item
    raise TypeError(f"Expected tensor output, got {type(value)!r}")


def find_reconstruction_layer_ids(teacher: nn.Module, student: nn.Module) -> list[int]:
    teacher_layers = _get_layers(teacher)
    student_layers = _get_layers(student)
    layer_ids: list[int] = []
    for layer_id, (teacher_layer, student_layer) in enumerate(zip(teacher_layers, student_layers)):
        teacher_mlp = getattr(teacher_layer, "mlp", None)
        student_mlp = getattr(student_layer, "mlp", None)
        if teacher_mlp is not None and student_mlp is not None and hasattr(student_mlp, "routed_dense"):
            layer_ids.append(layer_id)
    return layer_ids


def freeze_for_dense_reconstruction(student: nn.Module) -> int:
    trainable = 0
    for name, param in student.named_parameters():
        should_train = ".routed_dense." in name
        param.requires_grad_(should_train)
        if should_train:
            trainable += 1
    return trainable


def capture_teacher_mlp_io(
    teacher: nn.Module,
    batch: dict[str, torch.Tensor],
    layer_ids: list[int],
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    layers = _get_layers(teacher)
    inputs: dict[int, torch.Tensor] = {}
    outputs: dict[int, torch.Tensor] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def make_hook(layer_id: int):
        def hook(module: nn.Module, module_inputs: tuple[Any, ...], module_output: Any) -> None:
            del module
            inputs[layer_id] = _first_tensor(module_inputs).detach()
            outputs[layer_id] = _first_tensor(module_output).detach()

        return hook

    try:
        for layer_id in layer_ids:
            mlp = getattr(layers[layer_id], "mlp")
            handles.append(mlp.register_forward_hook(make_hook(layer_id)))
        with torch.no_grad():
            teacher(**batch)
    finally:
        for handle in handles:
            handle.remove()

    missing = [layer_id for layer_id in layer_ids if layer_id not in inputs or layer_id not in outputs]
    if missing:
        raise RuntimeError(f"Teacher hooks did not capture layers: {missing}")
    return inputs, outputs


def _masked_mean(values: torch.Tensor, attention_mask: torch.Tensor | None) -> tuple[torch.Tensor, int]:
    if attention_mask is None:
        return values.mean(), values.numel()
    mask = attention_mask.to(device=values.device, dtype=torch.bool)
    selected = values[mask]
    if selected.numel() == 0:
        raise ValueError("attention_mask contains no valid tokens")
    return selected.mean(), int(selected.numel())


def compute_parallel_reconstruction_loss(
    teacher: nn.Module,
    student: nn.Module,
    batch: dict[str, torch.Tensor],
    layer_ids: list[int] | None = None,
    cosine_weight: float = 0.05,
) -> ReconstructionResult:
    if layer_ids is None:
        layer_ids = find_reconstruction_layer_ids(teacher, student)
    if not layer_ids:
        raise ValueError("No reconstruction layers found")

    teacher.eval()
    teacher_inputs, teacher_outputs = capture_teacher_mlp_io(teacher, batch, layer_ids)
    student_layers = _get_layers(student)
    attention_mask = batch.get("attention_mask")
    layer_losses: list[torch.Tensor] = []
    metrics: dict[int, dict[str, float | int]] = {}

    for layer_id in layer_ids:
        student_mlp = getattr(student_layers[layer_id], "mlp")
        x = teacher_inputs[layer_id]
        target = teacher_outputs[layer_id]
        pred = student_mlp(x)
        target = target.to(device=pred.device, dtype=pred.dtype)

        mse_per_token = (pred - target).pow(2).mean(dim=-1)
        mse, token_count = _masked_mean(mse_per_token, attention_mask)
        loss = mse
        cosine_value = torch.zeros((), device=pred.device, dtype=pred.dtype)
        if cosine_weight:
            cosine_per_token = 1 - F.cosine_similarity(pred.float(), target.float(), dim=-1).to(pred.dtype)
            cosine_value, _ = _masked_mean(cosine_per_token, attention_mask)
            loss = loss + cosine_weight * cosine_value
        layer_losses.append(loss)
        metrics[layer_id] = {
            "mse": float(mse.detach().cpu()),
            "cosine_loss": float(cosine_value.detach().cpu()),
            "token_count": token_count,
        }

    return ReconstructionResult(loss=torch.stack(layer_losses).mean(), per_layer=metrics)
