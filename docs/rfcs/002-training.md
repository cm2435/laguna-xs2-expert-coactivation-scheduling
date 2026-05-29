# RFC 002: Training Dense Surrogate MoE Replacements

**Status:** Archived background. This MoE-to-dense training plan is retained for reference but is not the active hackathon implementation path.

## Purpose

Train dense surrogate MLPs that replace Laguna XS.2's sparse MoE MLP blocks on coding-task rollouts.

The training workstream owns initialization, layer-wise reconstruction, optional global fine-tuning, and checkpoint assembly.

## Core Training Objective

For each sparse MoE layer:

```text
y_teacher = MoE_layer(x)
y_student = DenseSurrogate_layer(x)
loss = reconstruction_loss(y_student, y_teacher)
```

Primary loss:

```text
MSE(y_student, y_teacher)
```

Recommended combined loss:

```text
loss = mse(y_student, y_teacher)
     + 0.1 * (1 - cosine_similarity(y_student, y_teacher))
```

Optional later:

```text
KL(student_logits, teacher_logits)
task CE loss on teacher continuations
layer weighting by downstream sensitivity
```

## Surrogate Architecture

First architecture:

```text
DenseSurrogateMLP:
  up_proj: hidden_size -> intermediate_size
  gate_proj: hidden_size -> intermediate_size
  activation: SiLU
  down_proj: intermediate_size -> hidden_size
  output = down_proj(SiLU(gate_proj(x)) * up_proj(x))
```

This mirrors common gated MLP blocks and is likely closer to Laguna expert structure than a plain two-layer MLP.

Width sweep:

```text
intermediate_size = 1024
intermediate_size = 2048
intermediate_size = 4096
intermediate_size = 4608
```

The 4608 option roughly matches active expert intermediate width:

```text
8 routed experts * 512 + 1 shared expert * 512 = 4608
```

## Initialization Experiments

We need to compare initialization methods, not just training recipes.

### Init A: Random Baseline

Use standard Kaiming/Xavier initialization.

Purpose:

```text
Measures how hard reconstruction is with no architecture knowledge.
```

### Init B: Shared Expert Clone

Initialize the surrogate from the shared expert where shape-compatible.

Purpose:

```text
The shared expert is always active, so it may be a strong base function.
```

If dimensions differ, copy compatible submatrices and randomly initialize remaining rows/columns.

### Init C: Router-Frequency Expert Merge

Collect router usage on rollouts:

```text
expert_frequency[layer, expert]
mean_router_weight[layer, expert]
```

Build weighted average expert weights:

```text
W_merged = sum_e alpha_e * W_expert_e
```

where:

```text
alpha_e = normalized frequency or normalized mean router contribution
```

Purpose:

```text
Approximates the expected expert mixture on the coding distribution.
```

### Init D: Least-Squares Output Head

For a chosen random/copied input transform:

```text
h = activation(gate(x)) * up(x)
solve down_proj so h @ W_down ~= y_teacher
```

Purpose:

```text
Fast data-aware initialization before gradient training.
```

### Init E: SVD Expert Merge

Construct a larger merged expert operator and compress it to the chosen dense intermediate width using SVD or low-rank projection.

Purpose:

```text
Potential high-quality init, but likely too slow for first pass.
```

## Training Modes

### Mode 1: Independent Layer Training

Train each dense surrogate independently on teacher activations.

Pros:

- Parallelizable.
- Stable.
- Easy to debug.
- Good layer-wise metrics.

Cons:

- Does not account for compounding error after full model swap.

This is the first mode.

### Mode 2: Sequential Replacement Training

Train layer 1, swap it, collect new activations for layer 2, train layer 2, and continue.

Pros:

- Accounts for distribution shift.

Cons:

- Slower.
- Harder to parallelize.

Use only if independent full swap fails.

### Mode 3: End-to-End Fine-Tuning

Assemble all surrogates, freeze attention/embeddings, and fine-tune dense MLPs on teacher-generated coding data.

Pros:

- Can repair compounding errors.

Cons:

- More expensive.
- Needs careful memory planning.

Stretch goal.

## Data Recipe

Start with cheap prompts:

```text
sorting function
binary search
parse unified diff
small bug fix
small test repair
```

Then expand:

```text
HumanEval/MBPP prompts
CruxEval prompts
pool/coding-agent turns
SWE-bench issue prompts
```

For each prompt:

```text
run teacher
record activations at sparse layers
optionally record teacher tokens/logits
```

## Training Metrics

Per layer:

```text
train MSE
validation MSE
cosine similarity
relative error = ||y_student - y_teacher|| / ||y_teacher||
variance explained
```

Across layers:

```text
mean reconstruction error
worst-layer reconstruction error
correlation with downstream eval drop
```

Checkpoint acceptance:

```text
relative error < chosen threshold
cosine similarity > 0.95 for first smoke
downstream smoke generation still coherent after swap
```

## Deliverables

- Train one layer with multiple initializations.
- Train all sparse layers independently.
- Save surrogate checkpoints per layer.
- Assemble full densified model.
- Record reconstruction metrics by layer and initialization.
- Produce plots showing width vs reconstruction quality.

## Open Questions

- How much data is needed per layer?
- Which layers are hardest to reconstruct?
- Does shared-expert initialization dominate random?
- Is 2048 intermediate width enough?
- Does layer-wise reconstruction predict code eval performance?
