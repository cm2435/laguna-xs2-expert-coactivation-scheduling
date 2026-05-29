"""MoE -> dense FFN densification for a single Laguna sparse layer.

Implements the score -> select -> concatenate pipeline from
"Pruning and Distilling Mixture-of-Experts into Dense Language Models"
(arXiv:2605.28207), instantiated for Laguna-XS.2's sigmoid-router MoE block.

Scoring methods compared:
  * frequency  - selection count (baseline; arXiv:2605.28207 shows it picks
                 redundant experts)
  * acp        - activation-weighted conditional probability:
                 CP_e * sqrt(E_t||f_e(t)||^2)
  * do-acp     - D-optimal greedy selection maximizing log det of the
                 importance-weighted expert-output Gram kernel (the paper's
                 best method; jointly maximizes importance and diversity)

The selected experts are concatenated (pure pruning, no merge) into one dense
FFN with per-expert magnitude scaling folded into the down projection. We then
measure how well the dense FFN reconstructs the true MoE block on held-out
activations -- the pre-distillation signal that the paper shows is dominated by
the scoring choice.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RoutingStats:
    freq: torch.Tensor          # [E] selection counts
    cp: torch.Tensor            # [E] mean routing weight when selected
    alpha: torch.Tensor         # [E] marginal routing weight = E_t[rw * 1{selected}]
    n_tokens: int


@dataclass
class ExpertStats:
    out_norm_sq: torch.Tensor   # [E] E_t||f_e(t)||^2
    gram: torch.Tensor          # [E, E] E_t<f_i(t), f_j(t)>


def compute_routing_stats(mlp, x: torch.Tensor) -> RoutingStats:
    """Replicate LagunaTopKRouter on inputs x [N, H] and gather routing stats."""
    gate = mlp.gate
    E = gate.num_experts
    logits = torch.nn.functional.linear(x.float(), gate.weight.float())
    if gate.router_logit_softcapping and gate.router_logit_softcapping > 0.0:
        sc = gate.router_logit_softcapping
        logits = torch.tanh(logits / sc) * sc
    scores = torch.sigmoid(logits)
    sel_scores = scores + gate.e_score_correction_bias.float()
    _, sel = torch.topk(sel_scores, gate.top_k, dim=-1)              # [N, k]
    rw = scores.gather(-1, sel)
    rw = rw / rw.sum(dim=-1, keepdim=True)                            # normalized, [N, k]

    N = x.shape[0]
    freq = torch.zeros(E, device=x.device)
    alpha = torch.zeros(E, device=x.device)            # sum of rw over tokens
    cp_sum = torch.zeros(E, device=x.device)
    onehot = torch.zeros(N, E, device=x.device)
    onehot.scatter_(1, sel, 1.0)
    freq = onehot.sum(0)
    # scatter routing weights to expert slots
    rw_full = torch.zeros(N, E, device=x.device)
    rw_full.scatter_(1, sel, rw)
    alpha = rw_full.sum(0) / N                          # marginal weight
    cp_sum = rw_full.sum(0)
    cp = torch.where(freq > 0, cp_sum / freq.clamp(min=1), torch.zeros_like(freq))
    return RoutingStats(freq=freq, cp=cp, alpha=alpha, n_tokens=N)


@torch.no_grad()
def compute_expert_stats(mlp, x: torch.Tensor, chunk: int = 256) -> ExpertStats:
    """Run all E experts over all N tokens in chunks; accumulate Gram + norms.

    Memory stays bounded: only a [E, chunk, *] tensor is live at a time.
    Uses no_grad (not inference_mode) so the returned Gram/norms are normal
    tensors usable by the autograd-touching selection ops downstream.
    """
    experts = mlp.experts
    gate_up = experts.gate_up_proj           # [E, 2*I, H]
    down = experts.down_proj                 # [E, H, I]
    E, twoI, H = gate_up.shape
    I = twoI // 2
    act = mlp.shared_experts.act_fn
    dev = x.device
    N = x.shape[0]

    gram = torch.zeros(E, E, device=dev, dtype=torch.float64)
    out_norm_sq = torch.zeros(E, device=dev, dtype=torch.float64)

    for s in range(0, N, chunk):
        xc = x[s:s + chunk].to(gate_up.dtype)            # [n, H]
        # gate_up: [E, 2I, H] x [n, H] -> [E, n, 2I]
        gu = torch.einsum("nh,eth->ent", xc, gate_up)
        g, u = gu[..., :I], gu[..., I:]
        inter = act(g) * u                               # [E, n, I]
        f = torch.einsum("eni,ehi->enh", inter, down)    # [E, n, H]
        f = f.double()
        gram += torch.einsum("enh,fnh->ef", f, f)
        out_norm_sq += torch.einsum("enh,enh->e", f, f)
    gram /= N
    out_norm_sq /= N
    return ExpertStats(out_norm_sq=out_norm_sq, gram=gram)


def acp_scores(routing: RoutingStats, experts: ExpertStats) -> torch.Tensor:
    """ACP = conditional prob * sqrt(mean output norm^2)."""
    return routing.cp * torch.sqrt(experts.out_norm_sq.clamp(min=0).to(routing.cp.dtype))


def select_frequency(routing: RoutingStats, k: int) -> list[int]:
    return torch.topk(routing.freq, k).indices.tolist()


def select_acp(routing: RoutingStats, experts: ExpertStats, k: int) -> list[int]:
    return torch.topk(acp_scores(routing, experts), k).indices.tolist()


def select_do_acp(routing: RoutingStats, experts: ExpertStats, k: int) -> list[int]:
    """Greedy D-optimal selection on the importance-weighted Gram kernel.

    K_ij = sqrt(I_i I_j) * G_ij with I = ACP; pick experts that maximize
    log det(K_S + lambda I) one at a time.
    """
    I = acp_scores(routing, experts).double()
    G = experts.gram                                     # [E,E] float64
    sqrtI = torch.sqrt(I.clamp(min=1e-12))
    K = (sqrtI[:, None] * sqrtI[None, :]) * G            # importance-weighted kernel
    E = K.shape[0]
    lam = K.diagonal().mean() / k                        # regularization ~ mean diag / k
    eye = torch.eye(E, device=K.device, dtype=K.dtype)
    Kr = K + lam * eye

    selected: list[int] = []
    remaining = set(range(E))
    for _ in range(k):
        best_idx, best_gain = -1, -float("inf")
        if not selected:
            base = None
        for e in list(remaining):
            cand = selected + [e]
            sub = Kr[cand][:, cand]
            sign, logdet = torch.linalg.slogdet(sub)
            val = logdet.item() if sign.item() > 0 else -float("inf")
            if val > best_gain:
                best_gain, best_idx = val, e
        selected.append(best_idx)
        remaining.discard(best_idx)
    return selected


def effective_rank(experts: ExpertStats, idx: list[int]) -> float:
    """Entropy-based effective rank of the selected experts' output Gram."""
    sub = experts.gram[idx][:, idx]
    ev = torch.linalg.eigvalsh(sub).clamp(min=0)
    s = ev.sum()
    if s <= 0:
        return 0.0
    p = ev / s
    p = p[p > 0]
    return float(torch.exp(-(p * torch.log(p)).sum()))


@torch.inference_mode()
def build_dense_ffn(mlp, idx: list[int], routing: RoutingStats,
                    scaling: str = "marginal") -> dict:
    """Concatenate selected experts into dense (gate, up, down) weights.

    The routed-scaling factor (2.5) and per-expert magnitude alpha are folded
    into the down projection. `scaling`:
      * "marginal" - alpha_e = E_t[rw * 1{selected}] (expected routing weight)
      * "uniform"  - alpha_e = 1/k
    Returns dense weight tensors for an MLP: x -> down(act(gate(x)) * up(x)).
    """
    experts = mlp.experts
    gate_up = experts.gate_up_proj           # [E, 2I, H]
    down = experts.down_proj                 # [E, H, I]
    twoI = gate_up.shape[1]
    I = twoI // 2
    rs = mlp.routed_scaling_factor

    if scaling == "uniform":
        alpha = {e: 1.0 / len(idx) for e in idx}
    else:
        alpha = {e: float(routing.alpha[e]) for e in idx}

    gate_blocks, up_blocks, down_blocks = [], [], []
    for e in idx:
        gu = gate_up[e]                      # [2I, H]
        gate_blocks.append(gu[:I])           # [I, H]
        up_blocks.append(gu[I:])             # [I, H]
        scale = rs * alpha[e]
        down_blocks.append(down[e] * scale)  # [H, I] scaled
    gate_w = torch.cat(gate_blocks, dim=0)   # [k*I, H]
    up_w = torch.cat(up_blocks, dim=0)       # [k*I, H]
    down_w = torch.cat(down_blocks, dim=1)   # [H, k*I]
    return {"gate": gate_w, "up": up_w, "down": down_w, "intermediate": len(idx) * I}


@torch.inference_mode()
def dense_forward(mlp, dense: dict, x: torch.Tensor) -> torch.Tensor:
    """shared(x) + concat-experts(x). Mirrors LagunaSparseMoeBlock but static."""
    act = mlp.shared_experts.act_fn
    xc = x.to(dense["gate"].dtype)
    g = act(torch.nn.functional.linear(xc, dense["gate"]))
    u = torch.nn.functional.linear(xc, dense["up"])
    routed = torch.nn.functional.linear(g * u, dense["down"])
    shared = mlp.shared_experts(x.to(next(mlp.shared_experts.parameters()).dtype))
    return routed.to(shared.dtype) + shared


def reconstruction_error(y_true: torch.Tensor, y_pred: torch.Tensor) -> dict:
    yt, yp = y_true.float(), y_pred.float()
    rel_l2 = (yp - yt).norm() / yt.norm().clamp(min=1e-9)
    cos = torch.nn.functional.cosine_similarity(yp, yt, dim=-1).mean()
    return {"rel_l2": float(rel_l2), "cosine": float(cos)}
