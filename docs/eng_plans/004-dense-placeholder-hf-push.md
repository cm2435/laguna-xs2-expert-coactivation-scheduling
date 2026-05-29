# Engineering Plan 004: Dense Placeholder Architecture And Hugging Face Push

**Status:** Active next build.

**Goal:** Create a Hugging Face checkpoint that has the final dense-serving architecture shape, even before training recovers quality. This unblocks serving/inference work while the training team works on dense-layer recovery.

The artifact is explicitly a placeholder:

```text
cm2435/laguna-xs2-dense-k8-copied-shell
```

It should load with Hugging Face, expose the same tokenizer/chat template as Laguna XS.2, preserve the non-MoE Laguna shell, and replace routed MoE blocks with dense SwiGLU blocks of the correct shape. The model card must clearly state that routed dense weights are random or structurally initialized and not useful for inference quality yet.

## Architecture Decision

Use the teammate proposal as the architecture target:

```text
Laguna XS.2 shell
  same tokenizer
  same embeddings
  same attention blocks
  same norms
  same lm_head
  every MoE FFN -> dense routed SwiGLU surrogate
  shared expert kept as an additive dense path where possible
```

Dense routed surrogate:

```text
K = 8 routed expert slots
expert_intermediate_size ~= original expert intermediate size
dense_routed_intermediate_size = K * expert_intermediate_size

gate_proj: hidden_size -> dense_routed_intermediate_size
up_proj:   hidden_size -> dense_routed_intermediate_size
down_proj: dense_routed_intermediate_size -> hidden_size
act:       SiLU / hidden_act from Laguna config
```

Forward shape:

```text
routed_dense = down_proj(silu(gate_proj(x)) * up_proj(x))
shared = original_shared_expert(x) if retained
output = shared + routed_dense
```

K=8 is the first placeholder because it matches the active routed width. K=16 is the likely quality fallback and can become a second checkpoint after the serving path works.

## Why Not A Generic Dense Width

Earlier we considered a free dense width such as 6144. The better serving/research story is K-based expert concatenation:

```text
K=8  -> directly corresponds to top-8 routed active experts
K=16 -> quality fallback while still removing routing/expert-union variance
```

This matches the MoE-to-dense paper recipe: score, select/group experts, concatenate into a dense FFN, then distill.

## Inputs

Required:

```text
source model: poolside/Laguna-XS.2
target repo: cm2435/laguna-xs2-dense-k8-copied-shell
HF token with write permission
GPU VM with enough RAM/storage to load model metadata and write shards
```

Useful source details to confirm programmatically:

```text
num layers
hidden_size
expert intermediate size
expert tensor names
shared expert tensor names
router/gate tensor names
routed scaling factor
hidden_act
```

## Technical Implementation Plan

The fastest reliable path is to prove the dense architecture on a tiny/random checkpoint first, then make the pushed full-size placeholder a copied-shell checkpoint. Phase E is mandatory: the artifact we publish should preserve Laguna's non-MoE backbone and shared expert weights, with only the new routed dense surrogate initialized or structurally filled.

Execution phases:

```text
Phase A: local tiny fake-model tests
Phase B: inspect source Laguna config and tensor names
Phase C: build dense config/model code
Phase D: instantiate dense checkpoint skeleton from config
Phase E: copy non-MoE/shared weights; initialize or concatenate K routed experts
Phase F: validate local load/generate
Phase G: push to Hugging Face Hub
Phase H: serving teammate load attempt
```

### Proposed Code Layout

Add:

```text
src/densify/dense_checkpoint/
  __init__.py
  config.py
  moe_tensor_map.py
  modeling_laguna_dense.py
  build_placeholder.py
  model_card.py

scripts/
  inspect_laguna_moe_tensors.py
  build_laguna_dense_placeholder.py
  validate_dense_placeholder.py

tests/
  test_dense_checkpoint_config.py
  test_dense_checkpoint_model.py
  test_moe_tensor_map.py
```

Responsibilities:

```text
config.py
  derive dense config from source AutoConfig
  set K, dense widths, architecture metadata

moe_tensor_map.py
  inspect module names and/or safetensor keys
  classify router/shared/routed expert tensors

modeling_laguna_dense.py
  custom HF remote-code model implementation
  dense SwiGLU replacement classes

build_placeholder.py
  orchestrate local checkpoint directory creation
  copy tokenizer and config
  initialize weights
  save_pretrained

model_card.py
  generate README.md with warnings and architecture metadata
```

### Fast Path Versus Quality Path

Build these in order:

```text
v0 tiny architecture-only random:
  random all model weights
  local-only, tiny/fast to validate custom code and serving interface

v1 copied shell:
  copy embeddings, attention, norms, lm_head, shared expert from Laguna
  random routed dense surrogate

v2 structural concat:
  copy non-MoE shell
  copy shared expert
  concatenate selected K routed experts into dense routed surrogate
```

For the hackathon split, v0 is only a local smoke test. The first pushed HF artifact should be v1 so the serving teammate works against the real Laguna shell. v2 is the first checkpoint worth recovering with SFT if structural initialization lands quickly.

### Test Strategy

Use fake configs and tiny models before touching Laguna:

```text
hidden_size = 16
num_layers = 2
expert_intermediate_size = 4
k_routed = 2
vocab_size = 64
```

Required tests:

```text
test_dense_config_records_conversion_metadata
test_dense_routed_mlp_forward_shape
test_dense_moe_replacement_adds_shared_path
test_tiny_dense_model_forward_logits_shape
test_tiny_dense_model_generate_two_tokens
test_tensor_map_classifies_fake_laguna_style_keys
```

Only after these pass should we run source-model inspection on the VM.

### Serving Contract

The placeholder repo must have a stable serving-facing contract:

```text
config.json
tokenizer files
modeling_laguna_dense.py
configuration_laguna_dense.py if needed
model.safetensors shards
README.md
```

The config must expose:

```json
{
  "model_type": "laguna_dense",
  "architectures": ["LagunaDenseForCausalLM"],
  "auto_map": {
    "AutoConfig": "configuration_laguna_dense.LagunaDenseConfig",
    "AutoModelForCausalLM": "modeling_laguna_dense.LagunaDenseForCausalLM"
  },
  "moe_dense_conversion": {
    "source_model": "poolside/Laguna-XS.2",
    "k_routed": 8,
    "expert_intermediate_size": 512,
    "dense_routed_intermediate_size": 4096,
    "shared_expert": "kept"
  }
}
```

If we can subclass the source config without a custom config class, do that. If not, ship `configuration_laguna_dense.py` with `trust_remote_code=True`.

### Weight Initialization Details

Random initialization:

```text
use source model initializer_range if present
initialize linear weights with normal_(0, initializer_range)
zero biases if any
tie lm_head to embeddings if source model ties them
```

Shared expert copy:

```text
copy shared_expert.gate_proj
copy shared_expert.up_proj
copy shared_expert.down_proj
leave shared path trainable metadata false/true configurable
```

Selected expert concat for K=8:

```text
gate_dense rows [i*expert_width:(i+1)*expert_width] = expert_i.gate_proj
up_dense rows   [i*expert_width:(i+1)*expert_width] = expert_i.up_proj
down_dense cols [:, i*expert_width:(i+1)*expert_width] = alpha_i * expert_i.down_proj
```

Initial expert selection:

```text
v2-fast: first K experts per layer
v2-better: top K by router frequency from captured traces
v2-best: DO-ACP / diversity-aware selection
```

Do not block v0/v1 on router trace availability.

### Memory And Runtime Notes

For v0 random:

```text
do not load source weights
download only config/tokenizer or use AutoConfig/AutoTokenizer
instantiate dense model directly from derived config
```

For v1/v2:

```text
load source model with low_cpu_mem_usage=True
prefer safetensors streaming if possible
copy one shard/layer at a time if full load is too memory-heavy
write target with safe_serialization=True
```

On the VM, use `uv run --no-sync` for any environment that relies on the currently working vLLM/Torch stack.

### Task 1: Inspect Laguna Module And Tensor Names

Create:

```text
scripts/inspect_laguna_moe_tensors.py
```

Output:

```text
runs/dense_placeholder/laguna_moe_tensor_map.json
```

The JSON should include, per MoE layer:

```json
{
  "layer_id": 0,
  "moe_module": "...",
  "router": "...",
  "shared_expert": {
    "gate_proj": "...",
    "up_proj": "...",
    "down_proj": "..."
  },
  "routed_experts": [
    {
      "expert_id": 0,
      "gate_proj": "...",
      "up_proj": "...",
      "down_proj": "..."
    }
  ]
}
```

Acceptance:

```text
all MoE layers discovered
256 routed experts discovered per MoE layer
shared expert discovered per MoE layer
gate/up/down shapes written
```

### Task 2: Define Dense Config

Create a derived config:

```text
runs/dense_placeholder/config_dense_k8.json
```

Add explicit fields:

```json
{
  "architectures": ["LagunaDenseForCausalLM"],
  "moe_dense_conversion": {
    "source_model": "poolside/Laguna-XS.2",
    "kind": "routed_moe_to_dense_swiglu",
    "k_routed": 8,
    "shared_expert": "kept",
    "routed_scaling_factor_folded": false,
    "placeholder_weights": "random"
  }
}
```

If custom code is required, the placeholder repo must include the modeling code and use `trust_remote_code=True`.

Acceptance:

```text
AutoConfig.from_pretrained(local_path, trust_remote_code=True) works
config records enough metadata for serving teammate to identify dense layers
```

### Task 3: Implement Dense Layer Class

Create:

```text
src/densify/dense_checkpoint/modeling_laguna_dense.py
```

Minimum classes:

```text
LagunaDenseRoutedMLP
LagunaDenseMoEReplacement
LagunaDenseForCausalLM or architecture-specific subclass/wrapper
```

The simplest approach is to copy or subclass the Laguna remote-code modeling file and replace the MoE module class. Do not rewrite attention.

Acceptance:

```text
model instantiates from config with random weights
forward(input_ids) returns logits with expected shape
generate(max_new_tokens=2) runs on a tiny prompt
```

### Task 4: Placeholder Weight Construction

The pushed placeholder must preserve the Laguna shell:

```text
copy all non-MoE weights from source Laguna
copy shared expert weights exactly
random-init routed dense K=8 projections
```

If full non-MoE copying is too slow, stop and fix the copier rather than pushing a full-size all-random architecture shell. A local tiny all-random checkpoint is acceptable for code tests only.

Potential structural init after placeholder:

```text
select first K experts per layer
concatenate their gate_proj/up_proj rows
concatenate down_proj columns
optionally divide/fold scaling into down_proj
```

Acceptance:

```text
state_dict loads without missing/unexpected keys except documented omissions
save_pretrained(local_path, safe_serialization=True) writes shards
reload local checkpoint succeeds
```

### Task 5: Push To Hub

Create a model card:

```text
README.md
```

Required warning:

```text
This is an architecture placeholder for Laguna XS.2 MoE-to-dense serving integration.
The non-MoE backbone and shared experts are copied from Laguna XS.2.
The routed dense FFN weights are random or structurally initialized.
It is not a quality checkpoint.
```

Push:

```bash
huggingface-cli login
python scripts/build_laguna_dense_placeholder.py \
  --source-model poolside/Laguna-XS.2 \
  --target-dir checkpoints/laguna-xs2-dense-k8-copied-shell \
  --k-routed 8 \
  --init random \
  --copy-non-moe \
  --copy-shared-expert \
  --push-to-hub cm2435/laguna-xs2-dense-k8-copied-shell
```

Acceptance:

```text
HF repo exists
AutoTokenizer loads
AutoModelForCausalLM loads with trust_remote_code=True
vLLM/SGLang serving teammate can attempt load
```

## Validation Script

Create:

```text
scripts/validate_dense_placeholder.py
```

It should:

```text
load tokenizer
load model
run one forward pass
run one 2-token generation
print parameter count
print dense replacement count
```

Validation commands:

```bash
python scripts/validate_dense_placeholder.py \
  --model-path checkpoints/laguna-xs2-dense-k8-copied-shell \
  --trust-remote-code

python scripts/validate_dense_placeholder.py \
  --model-path cm2435/laguna-xs2-dense-k8-copied-shell \
  --trust-remote-code
```

Expected output:

```text
tokenizer_loaded=true
model_loaded=true
num_dense_replacements=40
forward_logits_shape=[1, N, vocab_size]
generate_ok=true
```

## Suggested Implementation Order

1. Add tiny dense config/model classes and tests.
2. Add `build_laguna_dense_placeholder.py --tiny` to produce a tiny local checkpoint.
3. Validate tiny local checkpoint.
4. Add Laguna config derivation without loading source weights.
5. Build full-size copied-shell dense checkpoint with random routed dense layers.
6. Validate full-size copied-shell checkpoint on CPU/GPU with one forward pass if memory allows.
7. Push full-size copied-shell checkpoint to HF.
8. Add copied-shell mode.
9. Add selected-concat mode.

Do not start with selected-concat. It is tempting, but it couples architecture work to expert-selection logic. The serving teammate needs the real copied Laguna shell first; selected-concat can follow once the mandatory copied-shell path is stable.

## Concrete Commands

Tiny smoke:

```bash
uv run python scripts/build_laguna_dense_placeholder.py \
  --tiny \
  --target-dir checkpoints/tiny-laguna-dense-k2 \
  --k-routed 2 \
  --init random

uv run python scripts/validate_dense_placeholder.py \
  --model-path checkpoints/tiny-laguna-dense-k2 \
  --trust-remote-code
```

Full copied-shell placeholder:

```bash
uv run python scripts/build_laguna_dense_placeholder.py \
  --source-model poolside/Laguna-XS.2 \
  --target-dir checkpoints/laguna-xs2-dense-k8-copied-shell \
  --k-routed 8 \
  --init random \
  --copy-non-moe \
  --copy-shared-expert
```

Push:

```bash
uv run python scripts/build_laguna_dense_placeholder.py \
  --source-model poolside/Laguna-XS.2 \
  --target-dir checkpoints/laguna-xs2-dense-k8-copied-shell \
  --k-routed 8 \
  --init random \
  --copy-non-moe \
  --copy-shared-expert \
  --push-to-hub cm2435/laguna-xs2-dense-k8-copied-shell
```

Structural-concat follow-up:

```bash
uv run python scripts/build_laguna_dense_placeholder.py \
  --source-model poolside/Laguna-XS.2 \
  --target-dir checkpoints/laguna-xs2-dense-k8-concat \
  --k-routed 8 \
  --init selected-concat \
  --copy-non-moe \
  --copy-shared-expert
```

## Risks

1. Laguna remote code may not be easy to subclass.
   - Fallback: copy remote modeling file into the placeholder repo and make minimal edits.

2. vLLM/SGLang may not accept custom `trust_remote_code` model immediately.
   - Fallback: serving teammate starts with HF Transformers backend first.

3. Copying full non-MoE weights may be slow or storage-heavy.
   - Fallback: random architecture checkpoint first, copied-shell checkpoint second.

4. Keeping shared expert as a separate additive path may require custom serving kernels.
   - Fallback: fold shared expert into a wider dense FFN only for serving experiments, while preserving the research target separately.

## Done Definition

This plan is done when:

```text
cm2435/laguna-xs2-dense-k8-copied-shell exists
model/tokenizer load from HF
forward pass works
serving teammate confirms whether their stack can load it
training team has fixed tensor names and shapes to target
```
