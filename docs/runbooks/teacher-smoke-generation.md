# Teacher Smoke Generation Runbook

## Goal

Run the unmodified teacher model through HF/PyTorch on a small Python coding set and verify that generations are sane before activation capture or surrogate training.

## Commands

Quick spike:

```bash
uv sync --extra dev
uv run python scripts/build_python_smoke_prompts.py
uv run python scripts/smoke_load_teacher.py --config configs/teacher_smoke_h100.yaml
uv run python scripts/run_teacher_smoke_eval.py --config configs/teacher_smoke_h100.yaml --limit 5
```

The default H100/B300 config uses the `poolside/Laguna-XS.2` BF16 checkpoint. Use
`poolside/Laguna-XS.2-FP8` only for serving-runtime experiments; the HF/PyTorch
training scaffold needs ordinary BF16/FP16 matmuls for hooks, gradients, and layer
replacement.

Pass-gate run:

```bash
uv run python scripts/run_teacher_smoke_eval.py --config configs/teacher_smoke_h100.yaml --limit 20
```

## Pass Gate

Promote to activation capture only if:

- `architecture.json` lists candidate MoE modules.
- `generations.jsonl` has one row per prompt.
- `summary.json` reports non-empty generations.
- At least three examples in `examples.md` are readable Python attempts.

## Failure Handling

- If the model does not load, run the proxy config and inspect Laguna remote-code support.
- If generations are empty, check tokenizer EOS/PAD config and chat template.
- If snippets fail parsing, inspect `examples.md` before changing prompts.
- If H100 memory is insufficient, retry with batch size 1 and lower `max_new_tokens`.
