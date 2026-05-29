# RFC 001: Internal Setup

**Status:** Archived background. This was written for the earlier MoE densification track and is not the active hackathon implementation path.

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

Budget:

```text
$200 Prime Intellect / Prime Select credits
```

Operating assumption:

```text
Most development happens on one VM.
We opportunistically spin up multiple H100 VMs for parallel activation capture or layer training.
We avoid workflows that require always-on multi-node infrastructure.
```

Preferred:

```text
1x H100 80GB for the main VM
2-4x H100 80GB burst capacity when credits and availability allow
fast local NVMe
>= 256GB host RAM if possible
```

Fallback:

```text
single 80GB GPU
smaller model/proxy for early code-path validation
```

If full Laguna is awkward, validate the code path on a smaller MoE model first, then transfer the hooks to Laguna.

## Backend Decision

The primary backend for this workstream is **HF/PyTorch**.

Reason:

```text
MoE densification needs gradients, hooks, direct module replacement, and easy checkpoint surgery.
HF/PyTorch exposes those directly.
Serving engines optimize the opposite direction: they fuse, wrap, shard, and hide model internals.
```

Backend roles:

```text
HF/PyTorch:
  primary path for teacher loading, activation capture, surrogate training,
  module replacement, and checkpoint assembly

SGLang/vLLM:
  optional path for high-throughput rollout generation and final inference
  benchmarking after we have a densified checkpoint

pool:
  optional harness for producing realistic coding prompts/agent turns;
  not the model backend for activation capture
```

### Why Not Start With SGLang/vLLM?

SGLang and vLLM both have Laguna support, and they are useful later. But they are inference runtimes. For this project we need:

- `model.named_modules()` introspection.
- PyTorch forward pre-hooks and forward hooks.
- Access to intermediate tensors before and after each MoE block.
- Gradients through dense surrogate modules.
- Simple replacement of `layer.mlp`.
- Saving a modified checkpoint.

These are simpler in HF/PyTorch than in a serving runtime.

### When To Use SGLang/vLLM

Use SGLang or vLLM only when they solve a specific bottleneck:

1. **High-throughput rollout generation**
   - If HF generation is too slow, use SGLang/vLLM to generate text continuations or coding-agent traces.
   - Then replay those prompts/continuations through HF/PyTorch for activation capture.

2. **Final serving benchmark**
   - Once the densified model exists, test whether it can be served in SGLang/vLLM or a simpler PyTorch loop.
   - Measure tokens/sec, requests/sec, and memory.

3. **Compatibility check**
   - If HF remote-code loading is blocked, inspect SGLang/vLLM Laguna model code for module names and weight layout.

The first milestone should not depend on a serving backend.

## Rollout and Activation Capture Strategy

Separate "sample generation" from "activation capture":

```text
Path A: simple first loop

prompts
  -> HF/PyTorch teacher generate/forward with hooks
  -> activation shards
  -> train surrogate

Path B: high-throughput later loop

prompts
  -> SGLang/vLLM rollout generation
  -> saved prompts/continuations
  -> HF/PyTorch replay with hooks
  -> activation shards
  -> train surrogate
```

Path A is the default because it is simplest and keeps one source of truth. Path B is only needed if rollout generation becomes the bottleneck.

## Pool Setup

`pool` is the Poolside coding-agent harness. In this project it is optional and upstream of model training.

Use pool to produce realistic coding-agent prompts if time allows:

```text
pool exec
  -> realistic coding task trajectory
  -> capture or reconstruct prompts/turns
  -> store under data/prompts or data/rollouts
```

Do not make the first activation-capture path depend on pool. The first capture path should use a small local JSONL prompt file so we can debug hooks quickly.

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
  smoke_load_teacher.py
  collect_rollouts.py
  capture_activations.py
  train_surrogate_layer.py
  assemble_densified_model.py
  run_smoke_generation.py

configs/
  local_smoke.yaml
  laguna_xs2_full.yaml
  prime_h100.yaml

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

## Model Loading Plan

Start with the HF/PyTorch teacher path:

```text
AutoModelForCausalLM.from_pretrained(
  "poolside/Laguna-XS.2-FP8",
  trust_remote_code=True,
  torch_dtype=...,
  device_map=...
)
```

If FP8 loading through HF is awkward, try in this order:

1. Load BF16 if it fits for teacher-only inference.
2. Load FP8 with remote code if supported.
3. Load a smaller MoE proxy to validate hooks.
4. Use SGLang/vLLM Laguna code as implementation reference for module names and checkpoint layout.

Teacher parameters must be frozen:

```text
for p in teacher.parameters():
  p.requires_grad_(False)
teacher.eval()
```

Surrogate parameters are the only trainable parameters in the first training loop.

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

Default capture mode:

```text
torch.no_grad()
teacher.eval()
cache x_layer and y_teacher to CPU
write shard to disk
```

For surrogate training:

```text
load activation shard
surrogate(x_layer) -> y_student
loss(y_student, y_teacher)
backprop only through surrogate
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

## Prime Credit Management

Because we only have approximately $200 in credits, every expensive run should have:

```text
run_id
expected GPU hours
expected disk use
success criterion
kill criterion
```

Recommended spending order:

1. **Local/small smoke without full Laguna**
   - Validate code paths cheaply.

2. **Single H100 full-teacher load**
   - Confirm we can load Laguna and inspect layers.

3. **One-layer activation capture**
   - Confirm hooks and storage.

4. **One-layer surrogate training**
   - Confirm reconstruction improves.

5. **Parallel layer training burst**
   - Only after one layer works.

Avoid spending credits on large rollouts until the one-layer loop works.

## Deliverables

- Script that prints Laguna layer inventory.
- Script that loads the HF/PyTorch teacher and runs one smoke prompt.
- Script that captures one layer's activations on a sorting prompt.
- Script that captures all sparse layers on a small prompt batch.
- Activation shard format documented and tested.
- Config files for local smoke and full run.
- Decision log for whether SGLang/vLLM are needed for rollout generation.

## Open Questions

- Can we load full Laguna locally, or do we need a smaller MoE proxy for code-path development?
- Does HF/PyTorch remote-code loading work cleanly for `poolside/Laguna-XS.2-FP8`?
- If HF FP8 loading is blocked, should the teacher be BF16, SGLang-backed, or a smaller proxy?
- How much activation data can we store on available disk?
- Do we need router logits for initialization, or are selected expert ids and weights enough?
- How many H100-hours should we reserve for final eval/inference versus training?
