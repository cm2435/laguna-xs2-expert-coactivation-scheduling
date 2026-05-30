from __future__ import annotations

import re
from dataclasses import dataclass


_LAYER_RE = re.compile(r"(?:^|\.)layers\.(?P<layer>\d+)\.")
_ROUTED_RE = re.compile(
    r"(?:^|\.)experts\.(?P<expert>\d+)\.(?P<proj>gate_proj|up_proj|down_proj)\.weight$"
)
_PACKED_ROUTED_RE = re.compile(r"(?:^|\.)experts\.(?P<proj>gate_up_proj|down_proj)$")
_SHARED_RE = re.compile(
    r"(?:^|\.)(?:shared_expert|shared_experts)\.(?P<proj>gate_proj|up_proj|down_proj)\.weight$"
)
_ROUTER_RE = re.compile(r"(?:^|\.)(?:router|gate)\.weight$")


@dataclass(frozen=True)
class TensorKeyClassification:
    key: str
    kind: str
    layer_id: int | None = None
    expert_id: int | None = None
    proj: str | None = None


def classify_laguna_tensor_key(key: str) -> TensorKeyClassification:
    layer_match = _LAYER_RE.search(key)
    layer_id = int(layer_match.group("layer")) if layer_match else None

    if routed_match := _ROUTED_RE.search(key):
        return TensorKeyClassification(
            key=key,
            kind="routed_expert",
            layer_id=layer_id,
            expert_id=int(routed_match.group("expert")),
            proj=routed_match.group("proj"),
        )
    if packed_match := _PACKED_ROUTED_RE.search(key):
        return TensorKeyClassification(
            key=key,
            kind="packed_routed_experts",
            layer_id=layer_id,
            proj=packed_match.group("proj"),
        )
    if shared_match := _SHARED_RE.search(key):
        return TensorKeyClassification(
            key=key,
            kind="shared_expert",
            layer_id=layer_id,
            proj=shared_match.group("proj"),
        )
    if _ROUTER_RE.search(key):
        return TensorKeyClassification(key=key, kind="router", layer_id=layer_id)
    return TensorKeyClassification(key=key, kind="other", layer_id=layer_id)
