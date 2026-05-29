# RFC 004: Inventory

**Status:** Background. Keep the inventory discipline, but the active artifacts now center on Pool/SWE-bench rollouts and ToolSpec traces rather than densified checkpoints.

## Purpose

Maintain an explicit inventory of models, data, hardware, checkpoints, metrics, and open implementation risks.

This workstream prevents the project from becoming a pile of untracked experiments.

## Model Inventory

Track:

```text
teacher model
model revision / commit
precision
checkpoint source
license
local path
load command
```

Initial models:

```text
poolside/Laguna-XS.2-FP8
poolside/Laguna-XS.2-BF16 if available/needed
smaller MoE proxy model for local tests
```

For each densified checkpoint:

```text
checkpoint id
surrogate width
initialization method
training data id
training steps
loss recipe
eval status
artifact path
```

## Data Inventory

Track:

```text
prompt dataset
rollout dataset
activation dataset
eval dataset
```

For each:

```text
dataset id
source
num prompts
num tokens
task types
license/usage constraints
storage path
checksum
```

Key questions from the notes:

- How hard is this dataset?
- How much data do we need?
- How much eval generation can we afford?
- How much expert diversity does the data cover?

## Hardware Inventory

Track:

```text
GPU type
GPU count
VRAM
host RAM
disk
interconnect
cloud provider
hourly cost
availability window
```

For training:

```text
activation capture speed
surrogate layer training speed
max batch size
storage throughput
```

For inference:

```text
load memory
decode tokens/sec
prefill tokens/sec
requests/sec
p50/p95 latency
```

## Checkpoint Inventory

Suggested structure:

```text
checkpoints/
  surrogate_layers/
    run_001/
      layer_01.pt
      layer_02.pt
      metrics.json
  densified_model/
    run_001_width2048_shared_init/
      config.json
      model.safetensors
      manifest.json
```

Manifest fields:

```json
{
  "checkpoint_id": "run_001_width2048_shared_init",
  "teacher": "poolside/Laguna-XS.2-FP8",
  "surrogate_width": 2048,
  "init": "shared_expert_clone",
  "training_data": "rollouts_001",
  "activation_data": "activations_001",
  "layers_replaced": [1, 2, 3],
  "metrics": {
    "mean_relative_error": 0.0,
    "smoke_pass_rate": 0.0
  }
}
```

## Experiment Registry

Every experiment gets:

```text
run id
owner
start time
git commit
config path
input artifacts
output artifacts
summary result
decision
```

Decision values:

```text
keep
retry
discard
promote_to_eval
```

## Inference / Device Inventory

Track possible serving paths:

### Server GPU

```text
transformers generate
SGLang
vLLM
TensorRT-LLM
```

### Local / Consumer GPU

```text
llama.cpp feasibility
MLX feasibility
exllama-style feasibility
```

### Mobile / Edge

Questions:

- Can the densified architecture export to a simpler runtime?
- What quantization would be required?
- What is the expected model size after densification?
- Can attention/KV memory fit?
- Is the tokenizer/runtime supported?

Mobile is not a first milestone, but it is worth tracking because dense replacement removes MoE dispatch complexity.

## Reporting Inventory

Keep final hackathon material under:

```text
reports/
  figures/
  tables/
  final_summary.md
```

Required tables:

- Model size comparison.
- Reconstruction metrics by layer.
- Eval scores by checkpoint.
- Inference metrics by checkpoint.
- Hardware/device feasibility.

Required figures:

- Densification architecture diagram.
- Layer-wise reconstruction loss.
- Width vs memory and eval quality.
- Tokens/sec vs model size.

## Deliverables

- `inventory/models.md`
- `inventory/data.md`
- `inventory/hardware.md`
- `inventory/checkpoints.md`
- `inventory/experiments.md`
- `inventory/inference_targets.md`

## Open Questions

- Which exact Laguna checkpoint should be the teacher?
- Do we have enough GPU memory for full teacher activation capture?
- How much host disk can we spend on activations?
- What is the smallest dense width that remains useful?
- Which runtime can serve the densified checkpoint fastest?
