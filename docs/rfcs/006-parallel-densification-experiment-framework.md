# RFC 006: Parallel Densification Experiment Framework

**Status:** Archived background. This experiment framework belongs to the MoE densification track and is not the active implementation path.

## Purpose

Design the codebase so we can test multiple MoE-to-dense block replacement strategies in parallel and compare them on shared metrics.

The goal is to buy down experiment risk quickly:

```text
same teacher activations
same layer(s)
same eval prompts
many dense replacement strategies
many distillation losses
parallel jobs
comparable curves
```

This RFC extends [RFC 005](005-moe-to-dense-method-selection.md), which selected the method family. This document defines how the code should be structured so those methods can be implemented, launched, and compared without each experiment becoming a bespoke script.

## Design Goals

1. **Strategy isolation**
   - Random init, shared-expert init, router-weighted merge, selected-expert concat, MergeMoE-style init, and future methods should live behind the same interface.

2. **Shared artifacts**
   - Expensive teacher passes should be reused across experiments.
   - Activation shards and router traces are immutable inputs.

3. **Parallel execution**
   - We should be able to run `N` layer/strategy jobs at once across one or more GPUs/VMs.
   - Jobs should be restartable and independent.

4. **Comparable metrics**
   - Every job should emit the same metric schema so we can plot curves directly.

5. **Fast partial answers**
   - We should not wait for full-model densification to learn something.
   - One-layer reconstruction curves and partial-swap evals are first-class outputs.

## Core Experiment Unit

The atomic experiment is:

```text
teacher_model_id
activation_dataset_id
layer_id
surrogate_architecture
initializer_strategy
distillation_loss_recipe
training_budget
eval_recipe
```

Example:

```yaml
experiment_id: layer_12_router_merge_w2048_mse_cosine
teacher_model_id: poolside/Laguna-XS.2-FP8
activation_dataset_id: coding_smoke_v1
layer_id: 12
surrogate:
  type: gated_mlp
  hidden_size: 2048
  intermediate_size: 2048
initializer:
  type: router_weighted_merge
  top_experts: 8
loss:
  mse_weight: 1.0
  cosine_weight: 0.1
  kl_weight: 0.0
training:
  max_steps: 2000
  batch_tokens: 8192
  lr: 0.0002
eval:
  reconstruction_every_steps: 100
  partial_swap_prompts: data/prompts/python_smoke.jsonl
```

## Proposed Package Layout

Add these focused modules under `src/densify/`:

```text
src/densify/
  activation_store.py
  router_traces.py
  surrogate_architectures.py
  strategy_registry.py
  initializers/
    __init__.py
    random_init.py
    shared_expert.py
    router_weighted_merge.py
    selected_concat.py
    mergemoe_init.py
  losses.py
  train_loop.py
  eval_reconstruction.py
  partial_swap.py
  metrics.py
  experiment_config.py
  experiment_runner.py
  job_manifest.py
```

Add these scripts:

```text
scripts/
  capture_layer_activations.py
  capture_router_traces.py
  make_experiment_grid.py
  run_densification_job.py
  summarize_densification_runs.py
  plot_densification_curves.py
```

Add these configs:

```text
configs/
  activation_capture_laguna.yaml
  grids/
    one_layer_init_sweep.yaml
    one_layer_width_sweep.yaml
    multi_layer_best_strategies.yaml
  jobs/
    generated job configs live here or under runs/
```

## Artifact Layout

Use run IDs aggressively. Never overwrite generated artifacts.

```text
data/
  activations/
    coding_smoke_v1/
      manifest.json
      layer_00/
        shard_00000.pt
        shard_00001.pt
      layer_01/
        shard_00000.pt
  router_traces/
    coding_smoke_v1/
      manifest.json
      layer_00.jsonl
      layer_01.jsonl

runs/
  densification/
    20260529T190001Z_layer12_random_w2048/
      config.yaml
      checkpoints/
        step_000500.pt
        best.pt
      metrics.jsonl
      summary.json
      examples.md
    20260529T190002Z_layer12_router_merge_w2048/
      ...

reports/
  densification/
    init_sweep_layer12.csv
    init_sweep_layer12.png
    width_sweep_layer12.png
```

## Data Model

### Activation Shard

Each shard stores token-level samples for one layer:

```python
{
    "layer_id": int,
    "input": FloatTensor[num_tokens, hidden_size],
    "target": FloatTensor[num_tokens, hidden_size],
    "token_ids": LongTensor[num_tokens],
    "position_ids": LongTensor[num_tokens] | None,
    "metadata": {
        "teacher_model_id": str,
        "activation_dataset_id": str,
        "prompt_ids": list[str],
        "capture_dtype": str,
        "created_at": str,
    },
}
```

The first version can store `input` and `target` in fp16/bf16 on CPU. If storage becomes tight, add token subsampling and shard compression before adding clever formats.

### Router Trace

Router traces are optional but strongly recommended for expert-aware strategies:

```json
{
  "layer_id": 12,
  "prompt_id": "sort_001",
  "token_offset": 142,
  "selected_experts": [7, 19, 34, 88, 101, 120, 199, 244],
  "router_weights": [0.21, 0.18, 0.15, 0.13, 0.11, 0.09, 0.07, 0.06],
  "router_entropy": 1.83,
  "expert_output_norms": [12.1, 10.4, 9.9, 7.3, 6.1, 5.9, 4.8, 3.2]
}
```

These traces support:

- expert frequency histograms,
- router-mass scoring,
- output-contribution scoring,
- batch expert-union simulation,
- selected-expert concat initialization.

## Strategy Interfaces

### Surrogate Architecture

All dense replacements should implement:

```python
class DenseSurrogate(torch.nn.Module):
    hidden_size: int
    intermediate_size: int

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...
```

Initial architectures:

```text
plain_mlp:
  down_proj(act(up_proj(x)))

gated_mlp:
  down_proj(act(gate_proj(x)) * up_proj(x))

low_rank_gated_mlp:
  same interface, lower-rank projections for aggressive compression
```

Use `gated_mlp` first if Laguna's dense/shared expert uses a gated FFN shape. The closer the surrogate matches the teacher expert architecture, the less we ask optimization to invent.

### Initializer Strategy

All initializers should implement:

```python
class InitializerStrategy(Protocol):
    name: str

    def initialize(
        self,
        surrogate: DenseSurrogate,
        teacher_layer: torch.nn.Module,
        router_stats: RouterStats | None,
        activation_sample: ActivationBatch | None,
    ) -> InitReport:
        ...
```

Initial strategies:

1. **Random**
   - Baseline.
   - Tells us whether reconstruction is learnable from scratch.

2. **Shared expert clone**
   - Copy or project the always-on shared expert into the dense surrogate.
   - Strong cheap baseline.

3. **Frequency-weighted merge**
   - Weight experts by selection frequency over coding tokens.
   - Simple and fast.

4. **Router-weighted merge**
   - Weight experts by total router mass over coding tokens.
   - Better aligned with actual contribution.

5. **Contribution-weighted merge**
   - Weight experts by `router_weight * ||expert_output||`.
   - More expensive because it needs expert outputs or approximations.

6. **Selected-expert concat**
   - Select top or diverse experts and concatenate their hidden dimensions into a wider dense FFN.
   - This is closest to the MoE-to-dense paper.

7. **MergeMoE-style output reconstruction**
   - Use activation samples to solve/optimize compression matrices before normal training.
   - Treat as a stretch initializer.

### Loss Recipe

All loss recipes should emit a scalar loss and named components:

```python
class LossRecipe(Protocol):
    name: str

    def __call__(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        batch: ActivationBatch,
        model_outputs: ModelOutputs | None = None,
    ) -> LossReport:
        ...
```

Initial losses:

```text
mse:
  MSE(y_student, y_teacher)

mse_cosine:
  MSE + cosine distance on normalized outputs

variance_normalized_mse:
  MSE normalized by per-dimension teacher variance

router_weighted_mse:
  upweight tokens with high router confidence or high expert output norm

logit_kl:
  only for partial/full swapped model, not pure layer-shard training

on_policy_kd:
  student generates continuations; teacher supplies logits/targets on student distribution
```

## Experiment Grid

The first grid should be small enough to run quickly but broad enough to show trends.

### Grid 1: One-Layer Initializer Sweep

Pick one medium layer and one late layer:

```text
layers: [12, 30]
widths: [2048]
initializers:
  - random
  - shared_expert
  - frequency_weighted_merge
  - router_weighted_merge
  - selected_concat_top4
losses:
  - mse_cosine
steps: 1000
```

Question:

```text
which initializer starts lower and learns faster?
```

Primary curve:

```text
validation MSE / cosine vs training step
```

### Grid 2: Width Sweep

Use the best initializer from Grid 1:

```text
layers: [12, 30]
widths: [512, 1024, 2048, 4096, 4608]
losses:
  - mse_cosine
steps: 1000
```

Question:

```text
what width is enough before quality saturates?
```

Primary curve:

```text
validation MSE vs dense parameter count / estimated dense FLOPs
```

### Grid 3: Loss Sweep

Use best width/init:

```text
losses:
  - mse
  - mse_cosine
  - variance_normalized_mse
  - router_weighted_mse
```

Question:

```text
which offline loss best predicts partial-swap generation quality?
```

Primary curve:

```text
partial-swap smoke score vs reconstruction metrics
```

### Grid 4: Partial-Swap Risk Sweep

Train best recipe on multiple layers:

```text
swap_sets:
  - [single easy layer]
  - [single hard layer]
  - first 4 MoE layers
  - middle 4 MoE layers
  - last 4 MoE layers
```

Question:

```text
does quality degrade smoothly or collapse abruptly?
```

Primary metric:

```text
tiny coding smoke pass rate and next-token KL
```

## Job Execution Model

Keep this boring and robust.

### Single Job Command

```bash
uv run python scripts/run_densification_job.py \
  --config runs/job_configs/layer12_router_merge_w2048.yaml
```

Each job:

1. loads the job config,
2. loads activation shards for one layer,
3. builds the surrogate architecture,
4. applies initializer,
5. trains for configured steps,
6. writes checkpoints and metrics,
7. optionally runs partial-swap smoke eval.

### Parallel Launch

Use a manifest for embarrassingly parallel work:

```json
{
  "manifest_id": "init_sweep_layer12_20260529",
  "jobs": [
    {
      "job_id": "layer12_random_w2048",
      "config": "runs/job_configs/layer12_random_w2048.yaml",
      "gpu": 0
    },
    {
      "job_id": "layer12_router_merge_w2048",
      "config": "runs/job_configs/layer12_router_merge_w2048.yaml",
      "gpu": 0
    }
  ]
}
```

On one VM, we can run sequentially or with simple `tmux` sessions. On multiple VMs, copy the same repo and activation shard subset, then run disjoint job configs.

Do not introduce Ray/Slurm unless simple process-level parallelism becomes the bottleneck. The hackathon risk is method uncertainty, not cluster orchestration.

## Metrics Schema

Every training job appends `metrics.jsonl`:

```json
{
  "step": 500,
  "split": "val",
  "loss": 0.0312,
  "mse": 0.0288,
  "cosine_distance": 0.024,
  "teacher_norm": 18.2,
  "student_norm": 17.9,
  "tokens_seen": 4096000,
  "lr": 0.0002,
  "seconds_elapsed": 812.4,
  "gpu_memory_gb": 34.7
}
```

Partial-swap eval rows use the same run ID:

```json
{
  "step": 1000,
  "eval_name": "python_smoke_partial_swap",
  "num_prompts": 20,
  "non_empty_rate": 1.0,
  "python_like_rate": 0.8,
  "parse_ok_rate": 0.65,
  "tests_ok_rate": 0.35,
  "next_token_kl": 0.42
}
```

Summary files should include:

```json
{
  "best_step_by_val_mse": 900,
  "best_val_mse": 0.027,
  "best_val_cosine": 0.018,
  "surrogate_parameter_count": 12582912,
  "estimated_dense_flops_per_token": 25165824,
  "teacher_active_expert_width": 4608,
  "notes": "router-weighted merge converged faster than random"
}
```

## Curves We Need

The report should produce these plots automatically:

```text
1. val MSE vs step, grouped by initializer
2. val cosine distance vs step, grouped by initializer
3. best val MSE vs surrogate width
4. partial-swap smoke score vs val MSE
5. estimated dense FLOPs/params vs reconstruction quality
6. expert-union growth vs batch size for original MoE
7. predicted dense/MoE crossover point by batch size
```

The most valuable curve is not necessarily the lowest MSE. It is:

```text
quality retained vs serving cost
```

## Implementation Order

### Phase 1: Offline Layer Experiments

Build:

```text
activation_store.py
surrogate_architectures.py
initializers/random_init.py
initializers/shared_expert.py
losses.py
train_loop.py
run_densification_job.py
summarize_densification_runs.py
```

Skip router-aware methods initially if router traces are awkward. A random/shared-expert sweep already tests the basic loop.

### Phase 2: Router-Aware Initialization

Build:

```text
router_traces.py
initializers/frequency_weighted_merge.py
initializers/router_weighted_merge.py
initializers/selected_concat.py
```

This is where RFC 005's paper-inspired methods start paying off.

### Phase 3: Partial Swap and On-Policy Signals

Build:

```text
partial_swap.py
eval_reconstruction.py
logit KL eval
optional on-policy KD loop
```

This phase answers whether offline MSE predicts useful generation.

### Phase 4: Serving Curves

Build:

```text
serving benchmark config
dense checkpoint assembly path
p50/p95/p99 TPOT measurement
batch-mix benchmark
```

This only matters once at least one partial/full dense variant is plausible.

## Codebase Boundary Decisions

1. **No strategy-specific training scripts**
   - Strategies should be config entries, not separate scripts.

2. **No generated artifact imports**
   - Training code reads activation shards and configs.
   - It should not import from `runs/`.

3. **No full-teacher requirement inside every job**
   - Layerwise jobs should train from shards.
   - Full teacher is only needed for activation capture, logit KL, and on-policy KD.

4. **No distributed framework at first**
   - Use independent processes and run directories.
   - Add orchestration only after the simple path becomes painful.

5. **No one true metric**
   - MSE is cheap.
   - KL and on-policy KD are more meaningful.
   - Serving metrics are decisive.
   - The framework should preserve all of them.

## Open Questions

1. Can Laguna's expert weights be shape-adapted cleanly into a dense gated MLP?
2. Does the shared expert use the same projection shapes as routed experts?
3. Can we capture router weights from the HF model without modifying remote code?
4. What activation shard size is practical on the current B300 VM root disk?
5. How many layer jobs can one B300 run concurrently without memory contention?
6. Which layer should be the first probe layer?
7. Does low MSE correlate with partial-swap generation quality?

## Recommendation

Implement the experiment framework before adding clever methods.

The first useful milestone is:

```text
one captured layer
random vs shared-expert init
two widths
same training loop
same metrics schema
one comparison plot
```

Once that works, add router-weighted and selected-concat strategies. This gives us real curves early, lets multiple people own different strategies, and prevents the project from depending on one theoretically pretty but brittle merge method.
