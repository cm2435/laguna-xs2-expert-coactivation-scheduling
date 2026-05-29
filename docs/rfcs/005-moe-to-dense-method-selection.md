# RFC 005: MoE-to-Dense Method Selection

## Purpose

Select the most promising technical path for replacing Laguna XS.2 routed MoE MLP layers with dense surrogate MLP layers, optimized for batched inference latency and throughput.

This RFC is about method choice, not implementation mechanics. It connects the paper landscape to our concrete serving objective:

```text
reduce p50 decode latency
reduce tail decode latency under mixed batches
increase batched tokens/sec and requests/sec
reduce model memory and expert-I/O variance
```

## Motivation

Laguna XS.2 is a sparse MoE model. Per token, only a small subset of experts is activated, but in a batched serving regime the runtime often has to deal with the union of experts activated by many unrelated tokens.

That creates a different cost profile from the clean "3B active parameters" story:

```text
single token:
  few experts active
  sparse compute looks cheap

mixed batch:
  different tokens route to different experts
  expert union grows with batch diversity
  expert weight movement and grouped MoE dispatch become less regular
  latency varies with routing entropy and batch composition
```

Our hypothesis is that a coding-specialized dense surrogate can be smaller than the practical expert footprint exercised across diverse batches, even if it is larger than the theoretical per-token active path. Dense layers also have fixed compute shape and fixed memory access, which should improve batching and tail behavior.

## Success Metrics

The project should be evaluated as a serving and compression project, not only as a reconstruction project.

Primary serving metrics:

```text
p50 time per output token (TPOT)
p95 / p99 TPOT under mixed coding batches
aggregate decode tokens/sec at fixed concurrency
requests/sec at fixed latency target
GPU memory footprint
latency variance across batch compositions
```

Secondary model-quality metrics:

```text
layer reconstruction MSE
layer output cosine similarity
next-token KL against teacher
small Python coding smoke pass rate
HumanEval / MBPP subset score
SWE-bench-style prompt sanity checks
```

The key win condition is not that dense beats MoE for a single isolated token. The key win condition is that dense beats or stabilizes MoE under realistic batched serving, where the sparse expert union becomes expensive and variable.

## Paper and Technique Ranking

### 1. Pruning and Distilling Mixture-of-Experts into Dense Language Models

This is the anchor technique for our project.

Why it is promising:

- It directly studies MoE-to-dense conversion rather than leaving a smaller MoE behind.
- It scores, selects, and groups experts, then concatenates them into dense FFNs.
- It refines the converted dense model with knowledge distillation from the MoE teacher.
- The paper reports that MoE-to-dense compression can outperform dense-to-dense pruning at matched parameter count after distillation.

How we should use it:

```text
use its expert scoring methods as baselines
use its selected-expert grouping / concatenation idea for initialization
use its KD framing for the full-model refinement stage
adapt its recipe to Laguna XS.2 and a coding-specific data distribution
```

Reference:

- https://arxiv.org/abs/2605.28207

### 2. MergeMoE: Efficient Compression of MoE Models via Expert Output Merging

This is the best source for initialization theory.

Why it is promising:

- It argues that expert merging should be understood as output/function approximation, not naive parameter averaging.
- It frames merging as an optimization problem over the forward computation.
- This aligns with our layerwise reconstruction objective: `dense_mlp(x) ~= moe_block(x)`.

How we should use it:

```text
implement a MergeMoE-inspired output-reconstruction initializer if time permits
avoid relying only on parameter averaging
compare output-space initialization against random and shared-expert baselines
```

Reference:

- https://arxiv.org/abs/2510.14436

### 3. MoE-Pruner: Router-Aware Pruning

This is not our final architecture, because it leaves routing and expert dispatch in place. But it gives a strong scoring signal.

Why it is promising:

- It uses router information when estimating importance.
- It combines weight magnitude, input activations, and router weights.
- It includes expert-wise KD as a recovery mechanism.

How we should use it:

```text
use router-weighted activation importance to rank experts and neurons
use it as a baseline against simpler frequency-only scoring
borrow expert-wise / layerwise KD ideas for recovery
```

Reference:

- https://arxiv.org/abs/2410.12013

### 4. Not All Experts Are Equal

This is a useful baseline, but less aligned with our main latency objective.

Why it is useful:

- It shows expert-level sparsification can reduce model size and improve inference speed.
- It provides task-specific and task-agnostic expert pruning baselines.

Why it is not sufficient:

- The model remains an MoE.
- Routing remains dynamic.
- Batch-dependent expert-union variance remains.
- Tail latency may improve less than with a fully dense replacement.

How we should use it:

```text
compare dense conversion against expert-pruned MoE
use pruning results to decide which experts are worth merging
use it as the conservative fallback if full densification fails
```

Reference:

- https://arxiv.org/abs/2402.14800

### 5. DeepSeek FP8 / Quantization and MoE Systems Work

DeepSeek-style quantization is important, but downstream from densification.

Why it matters:

- FP8/low-precision serving can make the dense surrogate significantly faster and smaller.
- DeepSeek demonstrates that MoE and dense GEMMs benefit from careful low-precision system design.

Why it is not the merge method:

- Quantization does not remove routing entropy.
- Quantization does not remove batch-dependent expert unions.
- Quantization helps the original MoE too, so it is not the core differentiator.

How we should use it:

```text
first build a good dense surrogate in bf16/fp16
then benchmark fp8 dense serving if the runtime supports it
compare bf16 dense, fp8 dense, and original FP8 MoE where possible
```

Reference:

- https://arxiv.org/abs/2412.19437

## Recommended Method Stack

The recommended path is a staged MoE-to-dense distillation recipe:

```text
1. Collect coding activations and router decisions.
2. Score experts per MoE layer.
3. Build dense FFN initializers from selected/grouped experts.
4. Train dense surrogates with layerwise reconstruction.
5. Assemble a partially or fully densified model.
6. Refine with teacher KD.
7. Quantize/serve the dense model and benchmark latency/throughput.
```

### Stage 1: Collect Coding Activations and Router Decisions

For each MoE layer, capture:

```text
x_l: input to MoE block
y_l: output of MoE block
router logits
selected expert ids
router weights
optional expert output norms
```

Data should come from coding prompts first, not broad web text. We care about a coding-specialized dense surrogate, so the replacement only needs to approximate the teacher on that distribution.

### Stage 2: Expert Scoring

Score experts per layer using several methods:

```text
frequency score:
  how often expert e is selected

router mass score:
  sum of router weight assigned to expert e

activation-weighted score:
  input activation magnitude times router weight

output contribution score:
  norm(router_weight_e * expert_e(x))

diversity-aware score:
  selected experts should cover different functional behavior, not only the hottest experts
```

The last item should be borrowed from the MoE-to-dense paper if implementation details are available.

### Stage 3: Dense FFN Initialization

Test initializers in this order:

1. **Random baseline**
   - Establishes the difficulty of pure reconstruction.

2. **Shared-expert clone**
   - Copies or shape-adapts the always-on shared expert where possible.
   - Good baseline because the shared expert is active for every token.

3. **Frequency/router-weighted merge**
   - Weighted average or weighted composition of selected experts.
   - Simple but may underperform if experts are functionally diverse.

4. **Selected expert concatenation**
   - Pick top or diverse experts and concatenate their hidden dimensions into one dense FFN.
   - Closest to the MoE-to-dense paper.

5. **MergeMoE-style output reconstruction init**
   - Optimize merge/compression matrices to minimize output error before normal training.
   - More complex, but theoretically cleaner than parameter averaging.

### Stage 4: Layerwise Reconstruction

Train each dense surrogate independently:

```text
freeze teacher
for each target layer l:
  dense_l(x_l) -> y_hat_l
  minimize MSE(y_hat_l, y_l) + cosine_loss(y_hat_l, y_l)
```

Recommended losses:

```text
MSE on block output
cosine loss on normalized block output
optional variance-normalized MSE
optional router-contribution-weighted loss for high-impact tokens
```

### Stage 5: Partial and Full Swap

Do not jump directly to full replacement.

Ablation ladder:

```text
single easy layer swapped
single hard layer swapped
first N MoE layers swapped
last N MoE layers swapped
all MoE layers swapped
```

This tells us whether collapse is caused by one bad layer, accumulated distribution shift, or the method itself.

### Stage 6: Full-Model KD

After layerwise training, refine the assembled model with teacher distillation:

```text
next-token KL(student logits, teacher logits)
teacher continuation imitation
optional hidden-state matching at a few anchor layers
```

If time is short, run short KD only on coding prompts and generated continuations. The goal is to correct compounding distribution shift from swapping many layers.

### Stage 7: Serving Benchmark

Compare:

```text
original Laguna XS.2 MoE
expert-pruned Laguna variant if available
densified bf16/fp16 model
densified fp8 model if feasible
```

Benchmark dimensions:

```text
batch size / concurrency: 1, 4, 8, 16, 32
prompt mix: homogeneous coding prompts vs heterogeneous coding prompts
sequence phase: prefill vs decode
metrics: p50 TPOT, p95 TPOT, p99 TPOT, tokens/sec, requests/sec, memory
```

The dense model should mainly win in mixed, batched decode where MoE expert-union behavior hurts regularity.

## Why Dense Can Win Despite MoE Having Fewer Active Parameters

MoE active parameter count is a per-token abstraction. Serving cost depends on the batch and runtime.

Dense can win when:

```text
batch diversity activates many experts
expert weight movement dominates compute
MoE dispatch/grouping overhead is large
expert parallelism is poorly utilized
latency variance matters more than theoretical FLOPs
```

Dense can lose when:

```text
batch size is one
active MoE experts stay resident and cache-friendly
dense surrogate width is too large
quality drops and generations become longer or require retries
runtime kernels for dense surrogate are worse than MoE kernels
```

Therefore, the dense surrogate width must be chosen for serving, not just reconstruction. An over-wide dense FFN may reconstruct well but lose the latency objective.

## Proposed First Experiments

### Experiment A: Expert Scoring Probe

Goal:

```text
measure whether coding prompts activate a small, stable, or diverse expert set per layer
```

Outputs:

```text
expert frequency histograms
router mass histograms
per-layer entropy
expert overlap across prompts
estimated batch expert-union growth curve
```

This tells us how bad the original MoE tail-latency problem is likely to be.

### Experiment B: One-Layer Dense Surrogate Sweep

Goal:

```text
replace one MoE layer with dense surrogates of different widths and initializers
```

Sweep:

```text
width: 1024, 2048, 4096, active-compute-like width
initializer: random, shared expert, router-weighted merge, selected concat
```

Metrics:

```text
reconstruction MSE
cosine similarity
single-layer swap generation sanity
estimated dense FLOPs and parameter count
```

### Experiment C: Partial-Swap Latency Simulation

Goal:

```text
estimate when dense becomes cheaper than MoE under batch expert-union growth
```

Use captured router traces to estimate:

```text
number of unique experts loaded per batch
expert bytes touched per decode step
expected dense bytes/FLOPs for candidate surrogate widths
```

This can be done before full training and helps pick surrogate width.

### Experiment D: Full Serving Benchmark

Goal:

```text
measure real latency and throughput once a partial/full dense model exists
```

Report:

```text
p50 / p95 / p99 TPOT
requests/sec at target latency
GPU memory
quality score on tiny coding eval
```

## Go / No-Go Criteria

Continue with densification if:

```text
one-layer reconstruction reaches high cosine similarity
partial swaps preserve basic Python generation
router traces show batch expert-union growth is substantial
serving model predicts dense width can beat practical MoE expert footprint
```

Pivot to pruning or mixed precision if:

```text
one-layer dense reconstruction is poor across widths
partial swaps immediately collapse generation
router traces show expert union is already small/stable
surrogate widths required for quality are too large to improve latency
```

## Current Recommendation

Use **Pruning and Distilling MoE into Dense Language Models** as the core recipe, **MergeMoE** as the initialization/theory guide, and **MoE-Pruner / Not All Experts Are Equal** as scoring and baseline references.

DeepSeek-style FP8 should be treated as a later serving multiplier for the dense model, not as the main densification method.

The project should be judged on p50, p95/p99, and throughput under mixed batched coding inference. Single-prompt speed is useful but not decisive.

The codebase should implement these methods as parallel, comparable experiments rather than one-off scripts. See [RFC 006: Parallel Densification Experiment Framework](006-parallel-densification-experiment-framework.md).
