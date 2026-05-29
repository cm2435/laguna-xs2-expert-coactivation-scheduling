# Engineering Plan 005: Dense Midtraining And Post-Training Split

**Status:** Active planning.

**Goal:** Split dense-model recovery into two parallel workstreams:

```text
Team A: midtraining / dense-layer recovery
Team B: post-training / instruction and coding-agent recovery
```

The architecture comes from [Engineering Plan 004](004-dense-placeholder-hf-push.md): Laguna XS.2 with MoE FFNs replaced by K=8 dense routed SwiGLU surrogates, plus the shared expert path retained where possible.

## North Star

We are not trying to run full RL during the hackathon. The practical goal is:

```text
make the dense placeholder produce non-broken code continuations
show reconstruction / loss curves improving
prove the architecture can be served
produce a credible plan and early checkpoint for larger distillation
```

## Training Stages

### Stage 0: Architecture Placeholder

Owner: serving / infra teammate.

Artifact:

```text
cm2435/laguna-xs2-dense-k8-random
```

Purpose:

```text
freeze tensor names
freeze config format
allow serving work to begin immediately
```

### Stage 1: Midtraining Recovery

Owner: model-training team.

Method:

```text
causal-LM SFT on code-heavy data
train dense routed FFNs first
freeze attention/embeddings/norms/lm_head initially
optionally unfreeze norms after loss stabilizes
```

This is the hackathon replacement for the full 4B-token logit-KD plan. It is cheaper and easier to launch.

Loss:

```text
standard causal LM cross entropy
```

Optional if easy:

```text
teacher-generated continuations as targets
teacher logit KL on short sequences
layer reconstruction auxiliary loss for 1-2 layers
```

### Stage 2: Post-Training Recovery

Owner: post-training / agent behavior team.

Method:

```text
instruction SFT on coding tasks
bug-fix and patch-format examples
tool-call trace format preservation
small eval loop over coding harness tasks
```

Optional later:

```text
DPO or preference tuning on teacher-vs-student outputs
rejection-sampled teacher trajectories
```

## Data Mixture

Use a simple first-pass mixture rather than waiting for perfect data.

Recommended midtraining mixture:

```text
40% raw repo-level code / code CLM
25% code instruction SFT
20% Laguna teacher-generated coding answers
10% bug-fix / diff / patch examples
5% chat/tool-call format preservation
```

Candidate datasets:

```text
StarCoder2 / The Stack v2 subset
OpenCodeInstruct
Evol-Instruct-Code or similar permissive code-instruction data
HumanEval/MBPP-style synthetic prompts
our SWE-bench/coding-harness traces once stable
Laguna teacher generations from vLLM
```

Keep the first training run small:

```text
10M-50M tokens: smoke recovery
100M-300M tokens: serious hackathon run
500M-700M tokens: RADLADS-budget MVP if compute allows
```

## Distillation Strategy

The full paper-aligned recipe is:

```text
1. feature reconstruction / representation alignment
2. teacher logit KD
3. SFT / preference recovery
```

For hackathon execution, use:

```text
1. causal-LM SFT on code mixture
2. teacher-generated coding outputs as a large part of SFT
3. optional KL on short cached batches
```

Why this compromise:

```text
SFT is easiest to launch and debug
teacher outputs still transfer behavior
KL requires expensive teacher logits and more plumbing
feature reconstruction requires hooks and layer-specific datasets
```

## What To Train

Initial freeze policy:

```text
train:
  dense routed FFN weights

freeze:
  embeddings
  attention
  shared expert if copied from teacher
  norms
  lm_head
```

If loss plateaus:

```text
unfreeze:
  layer norms
  final norm
  lm_head
```

Avoid unfreezing attention in the first runs. We want to isolate whether dense FFNs can recover.

## Checkpoints

Use clear names:

```text
cm2435/laguna-xs2-dense-k8-random
cm2435/laguna-xs2-dense-k8-midtrain-50m
cm2435/laguna-xs2-dense-k8-midtrain-300m
cm2435/laguna-xs2-dense-k8-sft-preview
```

Every checkpoint card should record:

```text
source placeholder commit
K value
trainable modules
token count
data mixture
loss curve
known limitations
```

## Midtraining Metrics

Minimum:

```text
train loss
validation loss
tokens/sec
GPU hours
checkpoint size
```

Better:

```text
teacher continuation agreement
next-token KL on cached validation slice
simple code completion exact/pass rate
HumanEval/MBPP tiny subset
```

If activation capture lands:

```text
per-layer reconstruction MSE
cosine similarity
relative error
```

## Post-Training Metrics

Minimum:

```text
does the model follow chat template
does it emit syntactically plausible code
does it avoid infinite thinking/tool loops
small coding prompt pass rate
```

Coding harness eval:

```text
5 tiny repo tasks
5 SWE-bench-lite tasks after env setup is hardened
patch produced
tests run
tool-call count
exit success
```

## Immediate Work Split

### Serving / Architecture Owner

```text
implement Plan 004
push random dense K=8 checkpoint
make HF Transformers load
attempt vLLM/SGLang load
report unsupported model hooks if any
```

### Midtraining Owner

```text
build SFT dataset mixer
load placeholder checkpoint
freeze non-FFN weights
run 1M-token smoke
then 10M-50M-token recovery
publish loss curve and preview checkpoint
```

### Post-Training Owner

```text
curate coding instruction and patch examples
stabilize coding harness evals
define small post-training validation suite
prepare SFT examples from teacher outputs and successful traces
```

## Open Decisions

1. K=8 only, or push K=16 placeholder too?
   - Recommendation: K=8 now, K=16 only if serving is easy.

2. Shared expert retained separately or folded?
   - Recommendation: retain separately for research fidelity; consider folded variant only if serving blocks.

3. Teacher logits or teacher text only?
   - Recommendation: text-only SFT first; logits after the first recovery curve.

4. Dataset source for first 50M tokens?
   - Recommendation: OpenCodeInstruct + StarCoder2/The Stack subset + Laguna-generated coding completions.

## Done Definition

This plan is successful if we have:

```text
dense K=8 placeholder on HF
one midtraining run that reduces validation loss
one preview checkpoint pushed
one post-training/eval report showing whether outputs are still broken
clear next token budget estimate for 300M-700M run
```

