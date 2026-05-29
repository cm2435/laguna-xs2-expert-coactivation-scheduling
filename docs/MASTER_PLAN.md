# Laguna XS.2 MoE Densification Master Plan

## Executive Summary

We are changing direction from expert co-activation scheduling to **MoE densification**: replacing each routed Mixture-of-Experts MLP block in Laguna XS.2 with a single dense surrogate MLP. The goal is to produce a smaller, simpler dense model variant that preserves useful coding capability while improving memory footprint, serving complexity, and potentially tokens/sec on accessible hardware.

The core idea:

```text
teacher: Laguna XS.2 sparse MoE block
  -> router selects 8 / 256 experts plus shared expert

student: dense surrogate block
  -> one dense MLP replaces the routed expert mixture
```

We train the dense replacement blocks from teacher rollouts. During rollouts, we collect hidden representations around each MoE block and optimize the dense surrogate to reconstruct the teacher block outputs. We then evaluate whether the fully swapped model still solves small coding tasks and whether it improves inference memory, throughput, and deployability.

This is a model-compression / architecture-distillation project, not a serving-scheduler project.

## Backend Strategy

The primary backend for densification is **HF/PyTorch**, not a serving engine. The reason is simple: this project needs module introspection, forward hooks, gradients, and layer replacement. Those are natural in PyTorch and awkward inside vLLM/SGLang inference runtimes.

The backend split is:

```text
HF/PyTorch
  -> load teacher
  -> inspect Laguna modules
  -> capture x_layer and y_teacher
  -> train dense surrogate layers
  -> swap modules and assemble densified checkpoint

SGLang / vLLM
  -> optional high-throughput rollout generation
  -> final inference throughput/memory benchmark
  -> not the first training/capture path

pool
  -> optional coding-agent harness for realistic prompts
  -> not the model backend for activation capture
```

The first end-to-end loop should run entirely in HF/PyTorch: prompt batch in, teacher forward/generation with hooks, activation shards out, train one surrogate, swap one layer, run smoke generation.

## Project Thesis

Laguna XS.2 is a 33B-total / 3B-active MoE. The expert weights dominate model size and serving complexity. For a narrow coding distribution, the aggregate behavior of the routed expert mixture may be approximable by a much smaller dense MLP per layer. If so, we can distill the MoE into a dense coding-specialist variant.

The most compelling win is not just tokens/sec. The stronger claim is:

> We can collapse the expert bank into dense per-layer surrogates for coding tasks, trading broad MoE capacity for a smaller, simpler, easier-to-serve coding model.

## Primary Research Questions

1. Can a dense surrogate MLP reconstruct the per-layer MoE outputs on coding rollouts?
2. Which initialization gets closest before training?
3. Does layer-wise reconstruction loss predict downstream coding-task retention?
4. How small can the dense intermediate dimension be before quality collapses?
5. Does densification improve memory footprint, tokens/sec, requests/sec, and device deployability?

## Non-Goals

- We are not trying to preserve full general-purpose Laguna behavior.
- We are not initially replacing attention layers.
- We are not building a production mobile runtime in the first pass.
- We are not relying on expert-routing scheduler tricks.
- We are not training a new model from scratch.

## Target Model Transformation

Each sparse MLP block becomes one dense surrogate:

```text
Original block:
  x
  -> router(x)
  -> top-8 expert MLPs + shared expert
  -> weighted expert sum
  -> y_teacher

Densified block:
  x
  -> dense surrogate MLP
  -> y_student
```

Recommended first surrogate:

```text
DenseSurrogateMLP(
  hidden_size = 2048,
  intermediate_size = sweep over {1024, 2048, 4096},
  activation = SiLU,
  gated = true if matching Laguna MLP shape is easy,
)
```

The first sweep should include a very small option and an active-compute-matched option. Laguna's sparse block activates roughly 8 routed experts plus shared expert. With expert intermediate size 512, active expert intermediate width is approximately:

```text
8 * 512 + 512 = 4608
```

So useful initial widths:

```text
1024: aggressive compression
2048: moderate compression
4096 or 4608: active-compute-like surrogate
```

## Training Signal

The main training signal is layer-wise reconstruction:

```text
loss_layer = MSE(y_student_layer, y_teacher_layer)
```

Enhance with:

```text
cosine loss on normalized outputs
optional hidden-state variance weighting
optional final-logit KL after full-model swap
optional task loss on teacher-generated continuations
```

The first version should train each surrogate independently and in parallel:

```text
for each MoE layer:
  freeze teacher
  collect x_layer and y_teacher_layer
  train dense surrogate f_layer(x_layer) ~= y_teacher_layer
```

Then assemble a fully densified model and run downstream evals.

## Initialization Candidates

We should test initialization explicitly because it may determine whether training converges overnight.

1. **Kaiming/Xavier baseline**
   - Random dense MLP init.
   - Establishes whether reconstruction is learnable from scratch.

2. **Shared-expert clone**
   - Initialize surrogate from Laguna's always-on shared expert where shapes permit.
   - Good because the shared expert is active for every token.

3. **Router-frequency weighted expert merge**
   - Estimate expert usage and router weights from rollouts.
   - Merge expert weights into a dense surrogate initialization using weighted averages.
   - This is the most architecture-aware init.

4. **Activation-regression closed-form head init**
   - Random or copied input projection.
   - Solve the output projection by least squares on collected activations.
   - Useful as a fast, practical intermediate.

5. **SVD / low-rank compressed expert merge**
   - Build a large merged expert matrix, then compress to chosen intermediate width with SVD.
   - More complex, but useful if weighted merge is promising.

## Candidate Coding Tasks

We need tasks that are cheap enough for repeated training/eval and specific enough to reveal coding degradation.

### Smoke Tasks

Use these for the first working loop:

- Write a Python sorting function.
- Explain and implement binary search.
- Parse a unified diff.
- Fix an off-by-one bug in a small function.
- Given failing test output, patch a 20-line Python file.

These are not final evals. They are sanity checks that the densified model can still produce coherent code after layer replacement.

### Core Coding Evals

Use a staged ladder:

1. **Tiny handwritten coding set**
   - 20-50 prompts.
   - Sorting, parsing, tests, small bug fixes.
   - Fast enough for every checkpoint.

2. **HumanEval / MBPP subset**
   - Function synthesis.
   - Easy to run and score.
   - Good for early regression detection.

3. **CruxEval**
   - Code reasoning.
   - Useful for detecting representational damage even when syntax survives.

4. **SWE-bench Lite / Verified tiny subset**
   - Only after smoke tests pass.
   - Expensive but closer to agentic coding.

5. **Agentic replay prompts**
   - Use captured pool or coding-agent turns.
   - Score next-action or continuation quality with teacher comparison.

## Workstreams

The implementation plan should be split into four workstreams:

1. [Internal Setup](rfcs/001-internal-setup.md)
2. [Training](rfcs/002-training.md)
3. [Evals](rfcs/003-evals.md)
4. [Inventory](rfcs/004-inventory.md)

Each workstream is designed to be owned independently but integrated through shared artifacts.

## Shared Artifacts

```text
data/
  prompts/
  rollouts/
  activations/

checkpoints/
  surrogate_layers/
  densified_model/

runs/
  training/
  evals/
  inference/

reports/
  metrics/
  figures/
```

## Minimal End-to-End Milestone

Before activation capture, complete [Teacher Smoke Generation Implementation Plan](eng_plans/001-teacher-smoke-generation.md). This proves that the unmodified HF/PyTorch teacher path can load the model, inspect MoE modules, and generate sane Python coding outputs.

The smallest real success:

1. Load Laguna XS.2 or a smaller architecture-compatible proxy.
2. Identify all sparse MoE layers.
3. Capture `(x_layer, y_teacher_layer)` pairs for one coding prompt batch.
4. Train one dense surrogate for one MoE layer.
5. Swap that one layer into the model.
6. Verify the model still generates plausible text/code.
7. Measure reconstruction loss before and after training.

The next milestone:

1. Train surrogates for all MoE layers.
2. Assemble full densified model.
3. Run smoke coding evals.
4. Measure memory and tokens/sec versus the original.

## Risks

- Full Laguna XS.2 may be too large to iterate on quickly.
- Replacing every MoE block may cause compounding distribution shift.
- Layer-wise MSE may not correlate with generation quality.
- Weighted expert merging may be shape-awkward.
- A dense surrogate may be faster only if it avoids MoE dispatch/offload overhead; if the dense block is too wide, compute may dominate.
- Quality loss may cause longer generations, retries, or failed agent trajectories.

## Go / No-Go Criteria

Continue if:

- One-layer surrogate reconstruction improves strongly after training.
- Several layers can be trained independently without instability.
- A partial swap model can complete simple coding prompts.
- Memory footprint meaningfully drops relative to the expert bank.

Pivot if:

- Even one-layer reconstruction fails.
- Full swap destroys basic language/code ability.
- Dense surrogate compute is too large to improve serving.
- Activation capture is too slow or storage-heavy for the available infra.
