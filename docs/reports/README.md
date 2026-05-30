# Reports — Laguna-XS.2 expert activation & densification

Diagnostics behind the MoE→dense densification recipe. All numbers come from
forward-pass router introspection on the frozen `poolside/Laguna-XS.2` teacher
(top-8 of 256 experts, 39 sparse layers).

| Report | What it covers |
|---|---|
| [expert-activation-c4.md](expert-activation-c4.md) | Expert usage on a C4 (web-text) excerpt — coverage, effective experts, Gini, co-activation. |
| [expert-activation-c4-param-budget.md](expert-activation-c4-param-budget.md) | Same C4 run + full 33.4B→3.0B-active parameter budget and where the weights/compute live. |
| [expert-activation-by-dataset.md](expert-activation-by-dataset.md) | Dataset-by-dataset walkthrough across the densification mix (+SWE-bench) with 16×16 expert grids. |
| [distillation-pretraining.md](distillation-pretraining.md) | Swapping routers out: MoE→dense FFN, the reconstruction objective, training tricks, real V1/V2 curves + GIF — all code-referenced. |

**Headline:** the lower-level the input, the narrower the routing — CUDA/Triton
collapse onto ~100–108 effective experts/layer; NL instructions stay as broad as web
text (~158–183). Three near-disjoint expert neighborhoods (web / code-instruct /
kernel) → a kernel-anchored data mix is required so kernel-specialist experts are
reconstructed. See [../MODEL_CHANGES.md](../MODEL_CHANGES.md) for how this feeds the recipe.

## Layout
- `figures/` — square 16×16 expert-activation grids + summary/overlap diagrams.
- `data/` — per-dataset and C4 metrics JSON (`global` + `per_layer`), plus `samples.json`.
- Generators: `scripts/analyze_datasets_expert.py`, `scripts/analyze_expert_activation_c4.py`.
