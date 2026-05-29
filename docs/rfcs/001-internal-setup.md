# RFC 001: Internal Setup

## Purpose

Build the local and cloud infrastructure needed to inspect Laguna XS.2, capture activations, train dense surrogate layers, and assemble a densified checkpoint.

This workstream owns the repository layout, environment, model-loading path, activation hooks, and data formats shared by the rest of the project.

## Scope

Internal setup covers:

- GPU environment.
- Python environment.
- Model loading.
- Layer discovery.
- Activation capture.
- Dataset/rollout storage.
- Checkpoint storage.
- Reproducibility.

It does not own surrogate training recipes or eval scoring logic.

## Hardware Assumptions

Preferred:

```text
1-8x H100 80GB or equivalent
fast local NVMe
>= 256GB host RAM if possible
```

Fallback:

```text
single 80GB GPU
smaller model/proxy for early code-path validation
```

If full Laguna is awkward, validate the code path on a smaller MoE model first, then transfer the hooks to Laguna.

## Environment

Use a single Python environment with:

```text
torch
transformers
accelerate
safetensors
datasets
numpy
einops
tqdm
scikit-learn
pytest
```

Optional:

```text
deepspeed
bitsandbytes
flash-attn
sglang
vllm
lm-eval
evaluate
```

## Repository Layout

Proposed project-owned files:

```text
src/densify/
  __init__.py
  config.py
  model_introspection.py
  activation_capture.py
  surrogate_modules.py
  checkpoint_io.py
  logging.py

scripts/
  inspect_laguna.py
  collect_rollouts.py
  capture_activations.py
  train_surrogate_layer.py
  assemble_densified_model.py
  run_smoke_generation.py

configs/
  local_smoke.yaml
  laguna_xs2_full.yaml

tests/
  test_model_introspection.py
  test_surrogate_shapes.py
  test_activation_capture.py
```

## Layer Discovery

We need a robust way to discover sparse MoE layers:

```text
for each module in model.named_modules():
  detect Laguna sparse MLP / MoE module
  record layer id
  record hidden size
  record expert count
  record expert intermediate size
  record shared expert presence
```

Output:

```json
{
  "num_sparse_layers": 39,
  "layers": [
    {
      "layer_id": 1,
      "module_name": "...",
      "hidden_size": 2048,
      "num_experts": 256,
      "num_experts_per_tok": 8,
      "expert_intermediate_size": 512,
      "has_shared_expert": true
    }
  ]
}
```

Note: Laguna's first MLP layer may be dense, based on the config pattern. We should confirm rather than assume every layer is sparse.

## Activation Capture

For each sparse layer, capture:

```text
x_layer: input to sparse MoE block
y_teacher: output of sparse MoE block
optional router logits
optional selected expert ids
optional router weights
```

Use PyTorch forward hooks:

```text
pre-hook on sparse block -> save input
forward hook on sparse block -> save output
```

Store in shard files:

```text
data/activations/{run_id}/layer_{layer_id:02d}/shard_{shard_id:05d}.pt
```

Each shard:

```python
{
  "input": FloatTensor[num_tokens, hidden_size],
  "target": FloatTensor[num_tokens, hidden_size],
  "attention_mask_info": optional,
  "metadata": {
    "prompt_ids": [...],
    "token_offsets": [...],
    "layer_id": int
  }
}
```

## Storage Management

Activation storage can explode. Start small.

Initial capture target:

```text
100k-1M tokens total
one layer first
then all sparse layers
bf16 or fp16 activations
```

Storage estimates:

```text
one tensor token = 2048 hidden * 2 bytes ~= 4KB
input + target ~= 8KB/token/layer
100k tokens ~= 800MB per layer
all layers is large
```

Therefore use:

- token subsampling,
- per-layer capture runs,
- streaming writes,
- optional fp16 storage,
- optional random projection diagnostics before storing all layers.

## Deliverables

- Script that prints Laguna layer inventory.
- Script that captures one layer's activations on a sorting prompt.
- Script that captures all sparse layers on a small prompt batch.
- Activation shard format documented and tested.
- Config files for local smoke and full run.

## Open Questions

- Can we load full Laguna locally, or do we need a smaller MoE proxy for code-path development?
- Which backend exposes the cleanest module objects: HF transformers, SGLang model code, or vLLM?
- How much activation data can we store on available disk?
- Do we need router logits for initialization, or are selected expert ids and weights enough?

