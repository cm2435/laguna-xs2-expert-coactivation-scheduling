# Narrative Draft: Sparse-to-Dense Recovery for Laguna XS.2

**Status:** working deck/story plan. This is written to help us cut a clear
submission from a messy but useful hackathon run.

## One-Line Story

We tried to turn Laguna XS.2's MoE into a simpler dense surrogate. The core
finding is not "we solved densification"; it is sharper:

```text
Activation reconstruction can reduce hidden-state error quickly, but agentic
behavior recovery fails in specific, measurable ways unless the reconstruction
and SFT data actually contain the tool-call and recovery distribution.
```

The strongest result is the diagnosis stack:

```text
MoE -> dense reconstruction
  -> tool-call SFT
  -> live rollout
  -> exact-prefix policy probe
  -> data-rendering audit
  -> fixed reconstruction/KL canary
```

This gives us a credible research workflow: a recipe, the failure modes, the
metrics that exposed them, and the next training objective.

## Planned Three-Stage Route

The original plan was:

```text
1. MSE-style dense reconstruction from the MoE.
2. SFT on gold coding-agent rollouts to recover tool-call behavior.
3. Online KD from the MoE teacher on dense-student trajectories.
```

Stage 3 had a hard prerequisite:

```text
The dense model must self-drive enough to generate meaningful tool-use
trajectories: parseable tool calls, grounded first actions, observation use,
and recovery from failed tool outputs.
```

We did not reach that prerequisite. The story is why:

```text
MSE reconstruction reduced hidden-state loss but left the base behaviorally
undercooked; Opus SFT recovered the interface while collapsing first-action
policy; Sonnet metacognitive SFT recovered plausible thinking style but not
grounded tool execution.
```

## Submission Thesis

Laguna XS.2 is efficient at inference because only a subset of experts is active
per token, but serving MoEs still creates expert routing, dispatch, caching, and
batch locality problems. A dense surrogate would have a simpler serving path:

```text
no expert routing
no expert dispatch
predictable dense matmuls
smaller/mobile/offline serving path if compression succeeds
```

The hackathon question:

```text
Can we recover enough behavior from a Laguna MoE into a dense student, and can
we localize what fails when recovery is incomplete?
```

Our answer:

```text
Yes on the workflow and diagnostics. Not yet on final task success.
The first synthetic SFT recovered tool-call mechanics, but collapsed its
conditional tool policy. The metacognitive recovery SFT improved apparent
policy diversity in the text, but the model still made ungrounded/nonsense tool
calls. We found two likely causes: the dense reconstruction was undercooked for
behavior, and our later KL reconstruction corpus initially dropped structured
tool-call content during rendering.
```

## Safe Claims

These are safe to put on slides:

```text
Built a Laguna-compatible dense substitute shell.
Ran teacher-forced layer reconstruction against poolside/Laguna-XS.2.
Reduced reconstruction loss sharply in the first reconstruction run.
Generated and rejected 200 Laguna-generated harness rollouts as weak behavior-cloning exemplars.
Generated 80 Claude Opus synthetic trajectories for the first SFT run.
Recovered Laguna tagged tool-call mechanics after the Opus SFT run.
Ran a dense checkpoint in a live held-out harness loop.
Localized policy failure with an exact-prefix first-action probe.
Measured first-action collapse: generated read_file 20/20 while teacher expected shell 12/20.
Designed a 12-family metacognitive recovery taxonomy and generated Sonnet recovery data.
Ran a Sonnet recovery SFT that produced more diverse-looking reasoning/policy text but still failed grounded tool execution.
Found a reconstruction-data rendering bug where structured tool_calls were not rendered into KL text.
Added strict data/packing audits: EOS-separated packing, pad/boundary warnings, tool-call rendering checks.
Ran a fixed packed canary to test whether KL+activation can recover tool-call format.
```

Avoid claiming:

```text
The dense model solves SWE-bench.
The dense model is faster than the MoE.
The dense model is mobile-ready.
The fixed KL reconstruction fully works on the large corpus.
On-policy KD is ready.
```

## Key Numbers

### Reconstruction v1

Main loss curve:

```text
artifact: artifacts/pretraining_curves/opencode_k8_2048_full2000.png

step 1 loss:     0.2373046875
step 2000 loss:  0.026611328125
best loss:       0.0189208984375 at step 1800
relative drop:   ~88.8%
```

Interpretation:

```text
The dense substitute can fit a large part of the MoE hidden-state map, but
generation/tool behavior did not automatically follow from activation MSE.
```

### SFT Data Lineage

The SFT story has three distinct data phases. Keep them separate on slides:

```text
Phase 0: 200 Laguna-generated rollouts
  Purpose: use the target model's own behavior as SFT exemplars.
  Finding: rejected as behavior targets. The traces contained too much bad
  harness behavior and tool misuse; cloning them would teach the wrong policy.

Phase 1: 80 Claude Opus synthetic trajectories
  Purpose: clean behavior cloning data for tool-call SFT.
  Finding: excellent-looking traces, but too clean. The teacher often solved
  from the start and under-covered exploration, failed actions, and recovery.
  Result: tool-call mechanics recovered, but first-action policy collapsed to
  read_file.

Phase 2: metacognitive recovery data with 12 failure families
  Purpose: force examples of exploration, failed actions, and recovery.
  Process: small prompt-pack smoke, then cheaper teacher attempt, then Sonnet
  4.6 after the cheap lane had too much harness non-adherence.
  Finding: Sonnet produced well-formed data and more diverse-looking
  thinking/policy. The SFT over this data did not produce reliable grounded
  tool execution; it imitated plausible thinking tokens while still making
  nonsense tool calls.
```

Interpretation:

```text
The first SFT failure was a data-distribution failure: Opus was too good and
too clean, so the student learned syntax and a marginal action prior rather
than recovery.

The second SFT failure is stronger evidence of a base/model-capacity problem:
even with metacognitively steered diverse traces, the student could mimic the
shape of reasoning without reliably binding that reasoning to valid tool args.
That sent us back to the reconstruction/pretraining layer.
```

### Opus SFT Mechanics Recovery

Corrected parser result after recognizing Laguna tagged tool-call format:

```text
artifact: runs/sft_sanity_step450_partialparser_20260530T020804Z/summary.json

rollout-prefix tool-call shape:       5/5
rollout-prefix known tool name:       5/5
rollout-prefix expected tool name:    4/5
novel SWE-bench known tool name:      3/3
repetition/run-on on rollout prefix: ~40%
```

Interpretation:

```text
SFT recovered the mechanics of tool use: tags, tool names, parseability.
It did not prove useful policy.
```

### Live Rollout Smoking Gun

Held-out rollout:

```text
artifact:
runs/heldout_step450_rollout_smoke_historyfix_20260530T021529Z/step450_val1/

task: django__django-10097
turns: 6
structured tool calls: 6/6
patch produced: no
stop reason: max_turns
```

Tool-call trace:

```text
turn 1:
  read_file path=django/django/django/pull/10097
  observation: file not found or outside repo

turns 2-6:
  apply_patch to the same bogus path
  observation: error: unrecognized input
```

Interpretation:

```text
The harness and parser worked. The model did not lose the tool-call language.
It failed at grounded first action and observation-conditioned recovery.
```

### Exact-Prefix Policy Collapse

In-distribution first-action probe:

```text
artifact: runs/exact_prefix_first_action_step450_20260530T024012Z/summary.json

num_rows:                       20
parseable_tool_call_rate:        1.0
tool_match_rate:                 0.4
has_expected_required_key_rate:  0.4
primary_overlap_rate:            0.4

expected tools:
  read_file: 8
  shell:    12

generated tools:
  read_file: 20
```

Interpretation:

```text
The 0.4 match rate is exactly the base rate of read_file.
The model learned the marginal first action, not the conditional policy.
```

This is the cleanest "result slide": it converts a messy live failure into a
single quantified mechanism.

### Sonnet Metacognitive Recovery SFT

The follow-up SFT used a deliberately different data recipe:

```text
12 prompt/failure families
examples designed around exploration, tool misuse, failed observations, and recovery
teacher "scheme"/metacognition quarantined from the child-facing target
child-facing traces kept interleaved thinking + tool calls
```

The important qualitative result:

```text
The run improved policy diversity in the generated thinking: the model could
say things like "nothing found, so now I should search the directory" in a
plausible recovery-shaped block.

But the actual tool calls still did not reliably follow the reasoning. The
model could imitate metacognitive text without grounding it into valid,
task-specific tool arguments.
```

Interpretation:

```text
This is no longer just "Opus data was too clean." It suggests the dense base
was behaviorally undercooked: the SFT layer can teach surface style and some
action diversity, but it cannot reliably install grounded tool execution on top
of a weak reconstruction.
```

## Prompt And Rollout Exhibit Pack

Use this section as the source of concrete slide examples. The deck should show
at least one prompt-family snippet, one generated SFT-row snippet, and one
post-SFT student failure.

### Exhibit 1: Metacognitive Prompt Shape

Source:

```text
src/densify/recovery_data/prompts.py
docs/rfcs/mid_post_training/004-metacognitive-recovery-data.md
```

The shared prompt forced a teacher-only plan and a child-facing trajectory:

```text
<example>
  <metadata_json>{...}</metadata_json>
  <scheme>teacher-only data-generation planning</scheme>
  <trajectory_json>{child-facing messages only}</trajectory_json>
</example>
```

The key instruction:

```text
The <scheme> block is teacher-only data-generation planning.
The student will never see <scheme>.

The child-facing <think> blocks should be short and operational, e.g.
"that path was not found, so I should search for the symbol instead."
```

Slide use:

```text
This explains why the generated data can contain deliberate failures without
training the child on hidden teacher metacognition.
```

### Exhibit 2: Prompt Families That Targeted The Observed Failures

Source:

```text
src/densify/recovery_data/schema.py
src/densify/recovery_data/prompts.py
```

Best families to show:

```text
P3: PR URL / issue URL trap
  Lesson: external URLs and issue IDs are not repository paths.
  Required behavior: avoid reading the URL as a path; search symbols/filenames instead.

P5: File-not-found recovery
  Lesson: after file not found, change strategy.
  Required behavior: do not repeat the failed read_file; use shell exploration
  or a visibly different path justified by evidence.

P8: Malformed / no-op patch
  Lesson: header-only patches are failures.
  Required behavior: inspect file context, then emit a corrected non-empty patch.

P11: Correct read_file positives
  Lesson: do not overcorrect to always-shell.
  Required behavior: use read_file first when the issue gives a clear repo path.
```

This is the curriculum slide:

```text
Opus data had clean successes.
The metacognitive prompt pack deliberately covered the failure surface.
```

### Exhibit 3: Generated SFT Row That Actually Teaches Recovery

Source:

```text
data/sft/sonnet46_scale_256_p1_p12_sft_v2_finalmerge_noleaks_seq12288_shuf1337.jsonl
row: sympy__sympy-13372:P5:assistant_0005
family: P5 file-not-found recovery
```

Relevant prefix:

```text
assistant:
  <think>Let me start by reading the evalf.py file...</think>
  <tool_call>read_file
  <arg_key>path</arg_key>
  <arg_value>sympy/core/evalf.py</arg_value>
  </tool_call>

tool:
  Error: file not found: sympy/core/evalf.py

assistant target:
  <think>
  The file was not found at that path. I need to search for the correct
  location of evalf.py in the repository.
  </think>
  <tool_call>shell
  <arg_key>command</arg_key>
  <arg_value>find . -name 'evalf.py' 2>/dev/null</arg_value>
  </tool_call>
```

Slide use:

```text
This is exactly the row shape we wanted: failed action and observation in
context; corrective action as target.
```

### Exhibit 4: Generated SFT Row For Patch-Recovery

Source:

```text
data/sft/sonnet46_scale_256_p1_p12_sft_v2_finalmerge_noleaks_seq12288_shuf1337.jsonl
row: django__django-11149:P8:assistant_0001
family: P8 malformed / no-op patch
```

Relevant prefix and target:

```text
assistant:
  <think>I know the fix involves InlineModelAdmin permission checks...</think>
  <tool_call>apply_patch
  <arg_key>patch</arg_key>
  <arg_value>--- a/django/contrib/admin/options.py
  +++ b/django/contrib/admin/options.py
  </arg_value>
  </tool_call>

tool:
  Error: patch is malformed or contains no hunks. Nothing was applied.

assistant target:
  <think>
  The previous patch had no real edit — it was header-only with no hunk lines.
  I need to find the exact code in options.py before I can write a correct patch.
  </think>
  <tool_call>shell
  <arg_key>command</arg_key>
  <arg_value>grep -n "has_add_permission" django/contrib/admin/options.py | head -40</arg_value>
  </tool_call>
```

Slide use:

```text
This shows the metacognitive data did contain recovery-shaped supervision, not
just clean one-shot success traces.
```

### Exhibit 5: Sonnet Recovery Data Mixture Was Healthy

Source:

```text
data/recovery/sonnet46_scale_256_p1_p12_validation_v2_finalmerge_noleaks.json
```

Metrics:

```text
rollouts: 206
SFT rows: 1623
first actions: shell 121, read_file 66, apply_patch 19
failure_then_recovery rollouts: 72
patch_recovery rollouts: 39
failed-observation prefix rows: 748
parseable tool-call rows: 1623
child-facing think rows: 1623
scheme leaks: 0
hard failures: []
```

Slide use:

```text
The second SFT data was not obviously collapsed. It was balanced, parseable,
and recovery-shaped. The remaining failure is therefore more suspiciously a
student/base issue than a pure data-format issue.
```

### Exhibit 6: Sonnet-Recovery SFT Sanity Was Better But Still Not Correct

Source on remote GPU:

```text
runs/sft_sanity_sonnet46_recovery_epoch_final_20260530/summary.json
runs/sft_sanity_sonnet46_recovery_epoch_final_20260530/examples.md
```

Summary:

```text
num_rows: 12
non_empty_rate: 1.0
emits_tool_call_shape_rate: 0.9167
has_known_tool_name_rate: 0.9167
has_think_tag_rate: 0.6667
uses_expected_tool_name_rate: 0.0
```

Representative failure:

```text
Expected:
  <think>The previous patch had no real edit ... locate the relevant method.</think>
  <tool_call>shell
  <arg_value>grep -n "has_add_permission" django/contrib/admin/options.py | head -40</arg_value>
  </tool_call>

Generated:
  </assistant>
  ... I need to see the exact lines ...
  <tool_call>read_file
  <arg_key>path</arg_key>
  <arg_value>/repo/django/contrib/content.py</arg_value>
  </tool_call>
```

Slide use:

```text
The model learned to produce plausible recovery-flavored text and tool tags,
but it still chose the wrong tool/args. The issue moved from "no tool syntax"
to "ungrounded tool semantics."
```

### Exhibit 7: Real Validation Rollout After Sonnet SFT

Source on remote GPU:

```text
runs/real_val_rollouts_sonnet46_recovery_epoch_final_limit5_tok512_20260530/
```

Example `django__django-10097`:

```text
turn 1:
  read_file path=sklearn/model_selection/_search.py
  observation=file not found or outside repo

turn 2:
  shell command contains django/admin/contenttypes fragments and invalid paths
  observation=grep: no such file or directory

turns 3-7:
  malformed shell commands, missing command args, wrong-repo path fragments
```

Example `django__django-10554`:

```text
turn 1:
  shell command has unterminated quotes / mixed snippets
  observation=syntax error

later turns:
  repeated malformed grep commands and missing command args
```

Slide use:

```text
The Sonnet recovery SFT did improve away from simple read_file-only collapse,
but the student still failed to ground commands in the active repository. That
is the clearest bridge into the reconstruction/base-capacity diagnosis.
```

### Reconstruction v2 / Data Bug

We added a behavior term:

```text
loss = activation MSE + 0.05 * cosine loss + 0.02 * teacher-forced logit KL
```

But early reconstruction-v2 results were confounded by a formatter issue:

```text
structured message["tool_calls"] were not rendered into Laguna tagged text
for the reconstruction/KL corpus.
```

Post-fix audit:

```text
structured tool calls:       59,766
rendered tagged tool calls: 124,425
structured tool calls lost:       0
```

Packed-batch audit:

```text
sampled batches:                  20
sampled packed sequences:        240
sequences containing <tool_call>: 224
sequences containing <think>:     186
pad tokens:                        0
boundary warnings:                 0
```

Canary result after fixed rendering:

```text
artifact:
runs/reconstruction_canary12_packed_kl002_bs12_seq1536_120step_20260530T092035Z/

step 120 canary base probe:
  parseable_tool_call: false

exact-prefix canary probe:
  rows: 12
  parseable_tool_call_rate: 0.25
  tool_match_rate:          0.25
  generated shell:          3
  generated null:           9
```

Interpretation:

```text
The canary improved enough to emit some shell calls, but did not overfit tool
format reliably in 120 steps. This points away from "just launch the full run"
and toward objective/data weighting: tool-name and argument tokens need direct
signal, not just global KL over large vocab logits.
```

## Deck Plan

### Slide 1: Title / Claim

**Title:** Sparse-to-Dense Recovery for Laguna XS.2

Main sentence:

```text
We built and stress-tested a MoE-to-dense recovery pipeline, and localized the
failure from "bad model" to two measurable bottlenecks: conditional tool-policy
collapse and missing tool-call signal in the reconstruction corpus.
```

Visual:

```text
MoE teacher -> dense student -> SFT/KD -> coding harness
```

### Slide 2: Why Dense a MoE?

Message:

```text
MoEs reduce active FLOPs but complicate serving. A dense surrogate is a simpler
deployment target if we can recover enough behavior.
```

Visual:

```text
MoE path: router -> expert dispatch -> scattered expert loads
Dense path: one predictable dense block
```

Do not overclaim speed results; frame this as motivation.

### Slide 3: Method Overview

Pipeline:

```text
1. Replace routed MoE blocks with dense surrogates.
2. Match teacher hidden-state outputs.
3. Recover agent tool-call behavior with trajectory SFT.
4. Diagnose live rollout failures.
5. Add KL/data audits for reconstruction v2.
```

Visual:

```text
stage diagram with checkmarks and warning triangles
```

### Slide 4: Stage 1 Reconstruction Works on Loss

Graph:

```text
artifacts/pretraining_curves/opencode_k8_2048_full2000.png
```

Callout:

```text
~88.8% reconstruction loss reduction in 2k steps.
```

Speaker note:

```text
This is necessary, but later probes show it is not sufficient for agent behavior.
```

### Slide 5: Why We Did Synthetic SFT

Message:

```text
We first generated 200 Laguna rollouts, but they were not good behavior-cloning
exemplars: too much tool misuse and weak harness behavior. The first real SFT
dataset was therefore 80 synthetic Claude Opus trajectories.
```

Visual:

```text
Laguna rollouts -> rejected as bad exemplars
Opus synthetic rollouts -> first clean SFT set
```

Speaker note:

```text
This matters: the SFT result was not "we cloned Laguna rollouts." It was "we
used a stronger synthetic teacher to produce cleaner tool-call trajectories."
```

### Slide 6: Opus SFT Recovers Tool-Call Mechanics

Before/after:

```text
pre-SFT dense sanity:
runs/pre_sft_dense_sanity_20260530T022913Z/summary.json
  tool-call shape: 0/5
  known tool name: 0/5
  repetition: 1.0
```

Graph/table:

```text
rollout-prefix shape:    5/5
known tool name:         5/5
expected tool name:      4/5
novel issue known tool:  3/3
```

Artifact:

```text
runs/sft_sanity_step450_partialparser_20260530T020804Z/summary.json
```

Speaker note:

```text
The first scorer was wrong because Laguna uses tagged tool calls, not JSON
function-call objects. After parser correction, mechanics recovered.
```

### Slide 7: But Clean Opus Data Caused Policy Collapse

Main claim:

```text
Opus was too good. The 80 trajectories were clean, one-shot-ish, and
under-covered exploration, failed actions, and recovery. The student learned
the tool-call surface and the marginal action prior instead of conditional
policy.
```

Graph:

```text
expected first action:  read_file 8, shell 12
generated first action: read_file 20, shell 0
```

Artifact:

```text
runs/exact_prefix_first_action_step450_20260530T024012Z/summary.json
```

Main line:

```text
Tool grammar learned; conditional first-action policy collapsed to read_file.
```

### Slide 8: Live Rollout Separates Mechanics from Policy

Smoking gun:

```text
task: django__django-10097
6/6 turns had one structured tool call
0 patches produced
```

Trace visual:

```text
turn 1: read_file django/django/django/pull/10097 -> file not found
turn 2: apply_patch same bogus path -> error
turn 3: apply_patch same bogus path -> error
...
```

Point:

```text
The model can call tools. It cannot use tool observations to recover.
```

### Slide 9: Metacognitive Recovery Data

Message:

```text
We then built a 12-family metacognitive prompt pack to force the missing
distribution: exploration, tool misuse, failed observations, retry, and
recovery.
```

Process:

```text
1. Generate a few examples per family.
2. QA for harness adherence and target shape.
3. Try a cheaper teacher lane; reject when non-adherence was too high.
4. Use Sonnet 4.6 for the cleaner recovery-shaped data.
```

Speaker note:

```text
The teacher's private "scheme" block was generation scaffolding. The child saw
only child-facing thinking plus tool calls.
```

### Slide 10: Sonnet SFT Exposes A Deeper Problem

Smoking gun to add:

```text
Find a generated example where the child says a plausible recovery thought
("found nothing, now I should search the directory") but then emits an invalid
or nonsensical tool call.
```

Main claim:

```text
Metacognitive data improved policy diversity in the text, but the model did
not reliably bind that thinking to valid grounded tool arguments.
```

Interpretation:

```text
This points back below SFT: the dense base is likely behaviorally undercooked.
SFT can teach style and tool grammar, but not robust grounded execution on a
weak reconstruction.
```

### Slide 11: Why the Next Reconstruction Run Was Confounded

Message:

```text
We added KL to make reconstruction behavioral, but discovered the tool-call
slice was not actually rendered as tool-call text.
```

Visual:

```text
JSON message["tool_calls"] -> formatter bug -> missing <tool_call> tokens
```

Show audit after fix:

```text
structured tool calls lost: 0
packed sequences with <tool_call>: 224/240 sampled
pad tokens: 0
boundary warnings: 0
```

### Slide 12: Canary Says "Do Not Blindly Scale"

Evidence:

```text
12-row packed canary, 120 steps:
parseable exact-prefix rate: 0.25
tool match:                 0.25
base single-shot parseable: false
```

Message:

```text
Even after fixing rendering, global KL+activation is still too weak/indirect
for reliable tool-format overfit at this budget.
```

This is a useful negative result. It saves us from wasting the final hours on
another uninformative full run.

### Slide 13: Diagnosis

Use a two-column table:

```text
Observed failure                         Localized cause
---------------------------------------------------------------------------
No tool calls before SFT                 reconstruction objective not behavioral
Laguna rollouts not cloned               target traces were weak exemplars
Opus SFT tagged calls                     mechanics are recoverable
Opus read_file 20/20 exact-prefix         clean-teacher data caused marginal collapse
Held-out bad path + patch loop            recovery distribution missing
Sonnet plausible thoughts + bad tools     reasoning style learned without grounded execution
v2 KL failed early                       tool-call data rendering bug + weak target weighting
canary only 25% parseable                need targeted token/objective weighting
```

### Slide 14: What We Would Do Next

Next-stage recipe:

```text
1. Reconstruction v2 as the real next step
   Use activation MSE + behavior KL on correctly rendered tool-rich data.

2. Weighted action-token SFT/KL
   Upweight tool name, arg keys, arg values, and patch body tokens.

3. Off-policy KD on teacher trajectories
   Distill read_file vs shell choices on real prefixes.

4. Recovery-shaped data
   Failed action + observation in prefix; corrective action as target.

5. RL only after self-drive
   On-policy is blocked until the student can survive multi-turn feedback.
```

Do not frame this as hand-wavy future work. Frame it as each next step directly
targeting one measured failure.

### Slide 15: Contributions

Claim these:

```text
An end-to-end Laguna-compatible dense recovery harness.
A tagged-tool-call parser/eval path for Laguna.
A live SWE-bench-style rollout diagnostic.
A first-action collapse metric.
A reconstruction formatter audit that catches dropped structured tool_calls.
A packed-batch audit that catches padding/boundary issues.
A clear recipe for the next training objective.
```

### Slide 16: Close

Final sentence:

```text
The dense model did not become a coding agent overnight, but the pipeline made
the failure legible: clean SFT can recover mechanics while collapsing policy,
metacognitive SFT can recover plausible thinking without grounded execution,
and real recovery likely needs a stronger behavior-aware reconstruction before
more policy training.
```

## Figures To Generate

### Required

1. Reconstruction v1 loss curve

```text
source: artifacts/pretraining_curves/opencode_k8_2048_full2000.png
slide: 4
```

2. SFT mechanics recovery bar chart

```text
source: runs/sft_sanity_step450_partialparser_20260530T020804Z/summary.json
metrics:
  rollout-prefix shape 5/5
  known tool 5/5
  expected tool 4/5
  novel known tool 3/3
slide: 6
```

3. First-action collapse chart

```text
source: runs/exact_prefix_first_action_step450_20260530T024012Z/summary.json

expected:
  read_file 8
  shell 12

generated:
  read_file 20
slide: 7
```

4. Live rollout trace table

```text
source:
runs/heldout_step450_rollout_smoke_historyfix_20260530T021529Z/step450_val1/tool_calls.jsonl

show:
  turn
  tool
  arg/path summary
  observation summary
slide: 8
```

5. Sonnet metacognitive SFT example card

```text
source: generated/eval examples from the Sonnet recovery SFT run
show:
  plausible child-facing recovery thought
  actual nonsensical or ungrounded tool call
slide: 10
```

6. Reconstruction-v2 audit card

```text
source: audit output / notes
metrics:
  structured tool calls lost: 0
  packed sequences with <tool_call>: 224/240
  pad tokens: 0
  boundary warnings: 0
slide: 11
```

7. Canary result card

```text
source:
runs/reconstruction_canary12_packed_kl002_bs12_seq1536_120step_20260530T092035Z/

metrics:
  parseable base probe: false
  exact-prefix rows: 12
  exact-prefix parseable: 0.25
  tool match: 0.25
slide: 12
```

### Optional

8. SFT training loss/sequence-length plot

```text
source: artifacts/plots/sonnet46_recovery_sft_loss_seq_len.png
use only if we need a training-progress slide
```

9. Reconstruction v2/canary loss curve

```text
source: canary metrics.jsonl
use only if it visually supports "loss moves but generation still weak"
```

## Smoking Gun Examples To Quote

### Example A: Tool mechanics recovered

Use corrected sanity summary:

```text
rollout-prefix first tool-call shape: 5/5
known tool name: 5/5
expected tool name: 4/5
```

Quote:

```text
The model learned to emit Laguna tagged tool calls, but that was not enough.
```

### Example B: Bad path from issue URL

Use heldout rollout:

```text
read_file path=django/django/django/pull/10097
observation=file not found or outside repo
```

Quote:

```text
It copied the PR URL shape into a repo path instead of exploring.
```

### Example C: Observation ignored

Use turns 2-6:

```text
apply_patch to same bogus path repeated five times
observation=error: unrecognized input
```

Quote:

```text
This is not parser failure; it is missing observation-conditioned recovery.
```

### Example D: Exact-prefix marginal collapse

Use summary:

```text
teacher: shell 12/20, read_file 8/20
student: read_file 20/20
```

Quote:

```text
The model solved the easy scaffold and the marginal action, not the conditional
policy.
```

### Example E: Metacognitive text without grounded action

Use Sonnet recovery SFT eval:

```text
child-facing thought: "found nothing, now I should search the directory" style recovery
tool call: invalid / ungrounded / nonsensical args
```

Quote:

```text
Metacognitive SFT can teach the student to sound like it is recovering without
making the recovery action valid.
```

### Example F: Formatter bug

Use v2 audit:

```text
Before: structured tool calls present in JSON but missing in rendered KL text.
After: structured tool calls lost = 0.
```

Quote:

```text
The behavioral objective only helps if behavior tokens actually reach the
tokenizer.
```

## Final Framing

The deck should not apologize for the model failing. The result is a research
debugging story:

```text
We built the pipeline.
We recovered mechanics.
We showed why the first SFT collapsed: clean Opus data lacked recovery.
We showed why the second SFT was insufficient: metacognitive text did not ground tool args.
We ran the live loop.
We localized policy collapse quantitatively.
We found and fixed a silent data-rendering confound.
We identified the next objective: behavior-aware reconstruction plus action-token weighted KD/SFT.
```

That is a credible hackathon submission because it is useful to anyone else
trying to compress or densify agentic MoE models: the danger is not just lower
perplexity or lower activation MSE; the danger is losing conditional tool policy
and grounded tool arguments while superficial tool-call and reasoning-style
metrics look healthy.
