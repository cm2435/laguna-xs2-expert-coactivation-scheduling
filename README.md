# Laguna XS.2 ToolSpec Rollout Study

Hackathon workspace for studying ToolSpec-style speculative decoding on Laguna XS.2 inside Pool/SWE-bench coding-agent rollouts.

Current active plan:

- [docs/ACTIVE_PLAN.md](docs/ACTIVE_PLAN.md)
- [docs/MASTER_PLAN.md](docs/MASTER_PLAN.md)
- [docs/rfcs/008-toolspec-swebench-rollout-speculation.md](docs/rfcs/008-toolspec-swebench-rollout-speculation.md)
- [docs/eng_plans/003-vllm-swebench-rollout-data-infra.md](docs/eng_plans/003-vllm-swebench-rollout-data-infra.md)

Older MoE densification documents are retained as archived background under [docs/rfcs](docs/rfcs).

## MoE→Dense densification reports

End-to-end study of densifying the Laguna-XS.2 MoE into a ~3B dense coding model — expert-activation
analysis, the router swap, and reconstruction pre-training. See [docs/reports/](docs/reports/):

- [Expert activation on C4](docs/reports/expert-activation-c4.md) + [parameter budget](docs/reports/expert-activation-c4-param-budget.md)
- [Expert activation across the training mix (by dataset)](docs/reports/expert-activation-by-dataset.md)
- [Swapping routers out — distillation pre-training](docs/reports/distillation-pretraining.md)
- [Model changes & rationale](docs/MODEL_CHANGES.md)

## Related repositories

- **[Tyronita/laguna-dense-cuda-kernels](https://github.com/Tyronita/laguna-dense-cuda-kernels)** —
  the ~3B dense CUDA/Triton **kernel generator** densified from Laguna-XS.2 (SFT + verifiable reward +
  isolated eval); the downstream task this densification feeds.
