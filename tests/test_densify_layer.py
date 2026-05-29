"""CPU unit tests for densify_layer using a small synthetic MoE block that
mirrors LagunaSparseMoeBlock's attributes. No model download required."""
from __future__ import annotations

from types import SimpleNamespace

import torch

from densify import densify_layer as dl

H, I, E, TOPK = 16, 4, 8, 2


def make_mock_mlp(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    act = torch.nn.SiLU()

    gate = SimpleNamespace(
        num_experts=E,
        top_k=TOPK,
        hidden_dim=H,
        weight=torch.randn(E, H, generator=g),
        e_score_correction_bias=torch.zeros(E),
        router_logit_softcapping=0.0,
    )
    experts = SimpleNamespace(
        gate_up_proj=torch.randn(E, 2 * I, H, generator=g) * 0.1,
        down_proj=torch.randn(E, H, I, generator=g) * 0.1,
    )
    shared = torch.nn.Sequential()  # placeholder so .parameters() works
    shared_mlp = SimpleNamespace(
        act_fn=act,
        # callable returning zeros so reconstruction isolates the routed path
        __call__=lambda x: torch.zeros_like(x),
    )

    # build a tiny callable shared-expert module with a parameter for dtype/device
    class Shared(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.act_fn = act
            self.lin = torch.nn.Linear(H, H, bias=False)

        def forward(self, x):
            return self.lin(x)

    return SimpleNamespace(
        gate=gate,
        experts=experts,
        shared_experts=Shared(),
        routed_scaling_factor=2.5,
    )


def test_routing_and_expert_stats_shapes():
    mlp = make_mock_mlp()
    x = torch.randn(32, H)
    routing = dl.compute_routing_stats(mlp, x)
    assert routing.freq.shape == (E,)
    assert routing.freq.sum().item() == 32 * TOPK          # top-k per token
    # normalized routing weights sum to 1 per token -> marginal alpha sums to 1
    assert torch.allclose(routing.alpha.sum(), torch.tensor(1.0), atol=1e-5)
    es = dl.compute_expert_stats(mlp, x, chunk=8)
    assert es.gram.shape == (E, E)
    assert es.out_norm_sq.shape == (E,)
    # Gram diagonal equals output norms
    assert torch.allclose(es.gram.diagonal(), es.out_norm_sq, atol=1e-6)


def test_selectors_return_k_unique_experts():
    mlp = make_mock_mlp()
    x = torch.randn(64, H)
    routing = dl.compute_routing_stats(mlp, x)
    es = dl.compute_expert_stats(mlp, x, chunk=16)
    for k in (2, 4):
        for sel in (
            dl.select_frequency(routing, k),
            dl.select_acp(routing, es, k),
            dl.select_do_acp(routing, es, k),
        ):
            assert len(sel) == k
            assert len(set(sel)) == k
            assert all(0 <= e < E for e in sel)


def test_do_acp_increases_diversity_under_redundancy():
    """With duplicated experts, frequency may pick redundant ones; D-optimal
    selection should achieve >= effective rank."""
    mlp = make_mock_mlp(seed=1)
    # make experts 0 and 1 near-duplicates in output space
    mlp.experts.gate_up_proj[1] = mlp.experts.gate_up_proj[0].clone()
    mlp.experts.down_proj[1] = mlp.experts.down_proj[0].clone()
    x = torch.randn(128, H)
    routing = dl.compute_routing_stats(mlp, x)
    es = dl.compute_expert_stats(mlp, x, chunk=32)
    k = 3
    do = dl.select_do_acp(routing, es, k)
    fr = dl.select_frequency(routing, k)
    assert dl.effective_rank(es, do) >= dl.effective_rank(es, fr) - 1e-6


def test_dense_ffn_build_and_forward_shapes():
    mlp = make_mock_mlp()
    x = torch.randn(20, H)
    routing = dl.compute_routing_stats(mlp, x)
    es = dl.compute_expert_stats(mlp, x, chunk=8)
    idx = dl.select_do_acp(routing, es, 4)
    dense = dl.build_dense_ffn(mlp, idx, routing, scaling="marginal")
    assert dense["gate"].shape == (4 * I, H)
    assert dense["up"].shape == (4 * I, H)
    assert dense["down"].shape == (H, 4 * I)
    y = dl.dense_forward(mlp, dense, x)
    assert y.shape == (20, H)
    rec = dl.reconstruction_error(x, y)
    assert "rel_l2" in rec and "cosine" in rec
