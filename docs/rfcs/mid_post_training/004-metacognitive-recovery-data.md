# RFC 004: Metacognitive Recovery Data For Dense Student Policy Repair

**Status:** Draft.

## Purpose

Define the next supervised-data repair pass after the first dense-student SFT run.

The current SFT result is not a plumbing failure. It shows a precise policy
failure:

```text
tool-call syntax recovered:
  exact-prefix probe parseable_tool_call_rate = 20/20

first-action policy collapsed:
  teacher first actions:    read_file 8/20, shell 12/20
  student first actions:    read_file 20/20
  tool_match_rate:          0.4, exactly the read_file base rate

live rollout:
  valid tool calls on 6/6 turns
  bad first read_file path copied from PR URL
  no recovery from "file not found"
```

The goal of this RFC is to generate a new SFT/KD-ready dataset that explicitly
teaches:

```text
1. conditional first-action choice:
   when to explore with shell/grep/ls/find vs when to read_file

2. failure-to-recovery behavior:
   when an observation reports failure, choose a corrective action instead of
   repeating the same broken action
```

This RFC is about data shape and validation. It does not replace the existing
rollout-to-SFT pipeline in RFC 001 or the online-KD pipeline in RFC 002.

## Non-Goals

Do not train the child model on teacher metacognition.

Do not generate "bad rollouts" and use the bad actions as SFT targets. SFT
imitates targets. A bad action is useful only as prefix context when the target
is the recovery action.

Do not use raw code-completion data as a substitute for agentic SFT data. If raw
code is useful for a task, wrap it as an agent action such as `write_file` or
`apply_patch` so the child remains in the Laguna tool-call distribution.

## Core Idea

Use a strong teacher model to design recovery examples with a quarantined
metacognitive block:

```xml
<scheme>
The child currently overuses read_file. Construct a trajectory where it first
copies the PR URL as a path, receives "file not found", then recovers by running
grep for the relevant symbol and reading the discovered file.
</scheme>

<trajectory>
...
</trajectory>
```

The `<scheme>` block is data-generation scaffolding only. It must be discarded
before any training row is written. The dense child should train on the
trajectory actions, not the teacher's explanation of why the trajectory was
constructed.

## Example Types

Generate three deliberate example families.

### Type A: Exploration-First Positive Trajectories

The teacher does the correct thing from the first turn:

```text
issue mentions a symbol or behavior but no reliable file path
-> shell grep/rg/ls/find to ground the path
-> read_file on discovered path
-> apply_patch
-> optional test
```

These examples directly counter the `read_file` first-action collapse.

### Type B: Failure-Then-Recovery Trajectories

The trajectory includes a realistic failure and then demonstrates recovery.

Use failures that match the observed dense-student attractors:

```text
read_file path copied from PR URL
read_file path with repeated repo segments, e.g. django/django/django/...
apply_patch with header only and no patch body
grep with an overly narrow query returning no results
read_file with wrong line range or stale path
```

The failure action and failure observation are prefix context. The corrective
action is the SFT target.

### Type C: Ambiguous First-Action Cases

These are first-turn examples where exploration is clearly the right policy:

```text
issue gives an error message but no file path
issue gives a symbol/function/class name but no file path
issue references a PR URL or external link that must not be treated as a repo path
issue describes behavior across multiple possible modules
```

These examples should be over-represented relative to natural sampling because
they are exactly where the dense student currently collapses to `read_file`.

## Teacher Generation Contract

Each synthetic/repaired example should be generated with this structure:

```xml
<example>
  <metadata>
    <task_id>django__django-10097</task_id>
    <example_type>failure_then_recovery</example_type>
    <failure_type>pr_url_as_path</failure_type>
    <intended_first_action>shell</intended_first_action>
    <recovery_action>shell_grep</recovery_action>
  </metadata>

  <scheme>
    Teacher-only plan for constructing the trajectory. This block is discarded.
  </scheme>

  <trajectory>
    <messages>
      ...
    </messages>
  </trajectory>
</example>
```

The parser must reject any output where:

```text
<scheme> appears inside the saved trajectory
the first recoverable failure has no following corrective action
the target action is another repetition of the same failed action
the trajectory contains no parseable Laguna tool call
```

## Training Row Shape

The saved training rows should use the same assistant-action row format as RFC
001:

```json
{
  "id": "recovery_django__django-10097:assistant_0002",
  "task_id": "django__django-10097",
  "source_rollout": "metacognitive_recovery_20260530",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "Repository root: ...\n\nTask:\n..."},
    {
      "role": "assistant",
      "content": "",
      "tool_calls": [
        {
          "type": "function",
          "function": {
            "name": "read_file",
            "arguments": "{\"path\":\"django/django/pull/10097\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "content": "file not found or outside repo: django/django/pull/10097"
    },
    {
      "role": "assistant",
      "content": "",
      "tool_calls": [
        {
          "type": "function",
          "function": {
            "name": "shell",
            "arguments": "{\"cmd\":\"grep -R \\\"standard_duration_re\\\" -n django/\"}"
          }
        }
      ]
    }
  ],
  "quality": "recovery_synthetic",
  "weight": 2.0,
  "metadata": {
    "example_type": "failure_then_recovery",
    "failure_type": "pr_url_as_path",
    "target_role": "assistant",
    "target_tool": "shell",
    "target_is_recovery": true
  }
}
```

Loss masking remains unchanged:

```text
context = all messages before final assistant
target = final assistant message only
loss mask = target assistant tokens only
```

The repair pass should train a thinking-on child. The teacher should generate
two separate kinds of reasoning:

```text
teacher-only scheme:
  lives in <scheme>
  explains how the synthetic/repaired trajectory is constructed
  is discarded before SFT

child-facing steered thinking:
  lives in normal interleaved <think> blocks inside <trajectory>
  is written as if the child never saw the scheme prompt
  explains local recovery decisions, e.g. "the file was not found, so the path is
  probably wrong; search the repo for the symbol"
  is included in the assistant target together with the following tool call
```

The child should never see `<scheme>`, hidden notes, or teacher commentary. It
should see only task-local thinking that would be valid at inference time from
the visible transcript.

This changes the repair target from "tool call only" to:

```text
observation -> short recovery thought -> corrective tool call
```

This is intentional. The live rollout failure was not only a bad first tool; the
student also failed to integrate the observation. A small interleaved thinking
block gives the policy somewhere to represent "that failed; try a different
strategy" before emitting the next action.

## Metacognitive Prompt Family

Do not use one generic recovery prompt. Use a prompt family so the teacher
generates a controlled curriculum instead of a new collapsed behavior.

In this section, one "sample" means one teacher-generated full synthetic
rollout, not one final SFT row. A single rollout should contain a complete
agent lifecycle: initial exploration, one or more recovery moments, patching or
verification when appropriate, and a natural stopping point. It may produce many
SFT rows after splitting by assistant action.

Recommended v1 budget:

```text
phase -1 smoke:    1 full rollout per prompt, 12 full rollouts total
phase 0 pilot:     4 full rollouts per prompt, 48 full rollouts total
phase 1 repair:   10 full rollouts per prompt, 120 full rollouts total
phase 2 scale:    ~20 full rollouts per prompt, ~250 full rollouts total
stretch repair:   ~40 full rollouts per prompt, ~500 full rollouts total
```

Do not start with the 250-rollout scale run. First run phase -1 to prove that
the teacher follows the `<scheme>`/`<trajectory>` contract, that thinking is
child-facing, and that the actual cost per rollout is inside budget.

For the first serious repair run, use phase 1. The goal is not isolated local
fixes; it is diverse rollout-shaped demonstrations of the full policy lifecycle.
Phase 1 should produce roughly:

```text
~120 teacher-generated full synthetic rollouts
~1.5k-4k final SFT rows after splitting by assistant action
```

Only run phase 2 if phase 1 improves the exact-prefix first-action probe and at
least one live rollout metric.

### Prompt Templates And Counts

| ID | Prompt family | Lesson | Phase 2 samples |
| --- | --- | --- | ---: |
| P1 | Symbol but no path | First action should be `shell` grep/rg, not `read_file` | 25 |
| P2 | Error message but no path | Search for the error text before reading files | 20 |
| P3 | PR URL / issue URL trap | Never treat external URLs as repo paths | 20 |
| P4 | Repeated repo path trap | Recover from paths like `django/django/django/...` | 20 |
| P5 | File-not-found recovery | Failed `read_file` observation should trigger exploration | 25 |
| P6 | Empty grep recovery | Broaden or reformulate search after no matches | 20 |
| P7 | Patch context failure | Reread surrounding context, then retry patch | 20 |
| P8 | Malformed/no-op patch | Detect bad patch shape and produce a real patch | 15 |
| P9 | Late-turn recovery | Recover after several prior tool calls, not only turn 1 | 25 |
| P10 | Ambiguous multi-module issue | Explore multiple candidate modules before choosing | 20 |
| P11 | Correct `read_file` positives | Preserve cases where `read_file` is actually the right first action | 20 |
| P12 | Test/verification recovery | Test fails, inspect failure, then patch again | 15 |

Total phase 2 samples: 245.

P11 is intentionally included so the repair pass does not swing from
"always read_file" to "always shell". Keep it smaller than the combined
exploration-first families.

### Phase Gates

Phase -1 smoke gate, 12 rollouts:

```text
1 rollout per prompt family
manual inspect all 12
scheme_leak_count = 0
parseable trajectory rate >= 11/12
child-facing thinking rate >= 11/12
at least 8/12 contain a meaningful recovery or exploration decision
actual cost per rollout measured and written down
```

If phase -1 fails, patch the prompt pack or parser. Do not generate more data.

Phase 0 pilot gate, 48 rollouts:

```text
4 rollouts per prompt family
automatic validation passes hard balance rules
manual spot-check 8 accepted rollouts
train a tiny repair SFT or run target-only validation if time allows
```

If phase 0 shows scheme leakage, non-child-facing thinking, or a new first-tool
collapse, discard and repair the prompt family before continuing.

Phase 1 repair gate, 120 rollouts:

```text
10 rollouts per prompt family
train repair SFT
rerun exact-prefix first-action probe
rerun 3-5 held-out live rollouts
```

Phase 1 is successful if:

```text
generated first-action distribution is not read_file-collapsed
tool_match_rate improves above the 0.4 read_file base-rate result
shell first-actions appear in exact-prefix probes
at least one live rollout changes strategy after a failed observation
```

Phase 2 scale gate, ~250 rollouts:

```text
only run if phase 1 moves the diagnostics
use the same prompt family and validator
increase the weakest prompt families first, not just all families uniformly
```

### Shared Prompt Contract

Every prompt template should force the teacher to produce:

```xml
<scheme>
Teacher-only data-generation plan. Explain what child-model failure this sample
teaches against, which action should be the target, and why the recovery is
correct.
</scheme>

<trajectory>
Actual agent messages, Laguna tool calls, tool observations, and corrective
actions. This is the only block eligible for SFT conversion.
</trajectory>
```

The prompt should explicitly say:

```text
The <scheme> block is for dataset construction only.
The child model will never see <scheme>.
Do not put <scheme>, hidden notes, or teacher commentary inside <trajectory>.
Inside <trajectory>, use short task-local <think> blocks before tool calls.
The <think> blocks must be written as if the model only knows the visible task,
tool observations, and repo context.
The trajectory should contain at least one parseable Laguna tagged tool call.
If the sample is failure-recovery, the recovery action must be different from
the failed action.
```

Good child-facing thinking:

```xml
<think>
The read failed because this looks like a PR URL, not a repo path. I should
search the repository for the symbol mentioned in the issue.
</think>
<tool_call>shell
<arg_key>cmd</arg_key>
<arg_value>grep -R "standard_duration_re" -n django/</arg_value>
</tool_call>
```

Bad child-facing thinking:

```xml
<think>
The data-generation prompt asked me to demonstrate PR URL recovery, so I will
now emit the target corrective action.
</think>
```

Reject samples with the bad pattern. That is scheme leakage, even if it appears
inside `<think>` rather than `<scheme>`.

### Sampling Rules

Enforce these rules while selecting task seeds and accepting samples:

```text
no single repo > 25% of accepted samples
no single failure_type > 20% of accepted samples
no single recovery_action family > 35% of accepted samples
first-action read_file <= 40% of accepted samples
first-action shell/explore >= first-action read_file
at least 25% of accepted samples include a failed observation before the target
at least 20% of accepted samples target turns after turn 2
```

If a prompt family has a high reject rate, resample that family rather than
filling the budget with easier families. The hard part of this dataset is the
distribution, not the raw row count.

### Generation Cost Estimate

The sample counts above are full synthetic rollouts, not SFT rows. Total teacher
API cost depends on:

```text
number of rollouts
average turns per rollout
average input tokens per turn
average output tokens per turn
provider/model token prices
prompt caching behavior, if available
```

Use this formula:

```text
cost = rollouts * average_turns * (
  average_input_tokens_per_turn / 1_000_000 * input_price_per_million
  + average_output_tokens_per_turn / 1_000_000 * output_price_per_million
)
```

The repo's OpenRouter cost script currently uses these defaults:

```text
openai/gpt-5.5:              $5 / 1M input,  $30 / 1M output
openai/gpt-5.5-pro:         $30 / 1M input, $180 / 1M output
anthropic/claude-sonnet-4.5: $3 / 1M input,  $15 / 1M output
```

Anthropic's current public pricing page lists Sonnet 4.6 and Sonnet 4.5 at the
same $3 / 1M input and $15 / 1M output rate. It lists Opus 4.7 at $5 / 1M input
and $25 / 1M output. If "Sonnet 4.7" is an OpenRouter alias, verify the actual
route price before launching the full run and override the script defaults if
needed.

For the default plan, use Sonnet pricing unless we intentionally choose Opus.

### Optional Cheap-Teacher Lane

It is worth testing a cheap GPT-mini route in parallel, but only behind a strict
quality gate. The point is to discover whether the prior GPT/OpenRouter issues
are fixed and whether the cheaper model can follow the rollout contract. It
should not block the Sonnet lane.

OpenAI's public pricing page lists `gpt-5-mini` at roughly:

```text
$0.25 / 1M input tokens
$2.00 / 1M output tokens
```

Use `openai/gpt-5.4-mini` for this lane. A direct smoke test rejected
`openai/gpt-5.5-mini` as an invalid model ID; `openai/gpt-5-mini` also works,
but the intended cheap route for this project is `openai/gpt-5.4-mini`.

Run this lane as:

```text
phase -1 only at first: 12 full rollouts
same 12 prompt families
same parser
same validator
same manual inspection rubric
```

Promote GPT-mini to phase 0 only if:

```text
scheme_leak_count = 0
parseable trajectory rate >= 11/12
child-facing thinking rate >= 11/12
at least 8/12 include meaningful recovery/exploration
no obvious lazy/short/one-shot trajectory collapse
```

If it passes, use GPT-mini for broad draft generation and reserve Sonnet for one
of:

```text
auditing / filtering GPT-mini rollouts
repairing rejected GPT-mini rollouts
generating the hardest prompt families only
```

Do not train on GPT-mini rollouts just because they are cheap. They must pass
the same diversity and leakage validators as Sonnet rollouts.

Full-rollout budget for Sonnet-priced generation:

```text
low:       15 turns/rollout,  8k input tokens/turn, 0.7k output tokens/turn
base:      25 turns/rollout, 15k input tokens/turn, 1.0k output tokens/turn
high:      40 turns/rollout, 25k input tokens/turn, 1.2k output tokens/turn
very high: 40 turns/rollout, 40k input tokens/turn, 1.5k output tokens/turn
```

| Rollouts | Low | Base | High | Very high |
| --- | ---: | ---: | ---: | ---: |
| 12 | ~$6 | ~$18 | ~$45 | ~$68 |
| 48 | ~$25 | ~$72 | ~$179 | ~$274 |
| 120 | ~$62 | ~$180 | ~$446 | ~$684 |
| 250 | ~$129 | ~$375 | ~$930 | ~$1,425 |

This is the table to use for planning the serious repair run.

For Opus 4.7-priced generation, multiply the Sonnet table by about 1.67x.

For `openai/gpt-5.4-mini` generation, the same rollout assumptions are much
cheaper than Sonnet. A live smoke run observed OpenRouter charging roughly
$0.75 / 1M input and $4.50 / 1M output, so use that until the route price
changes.

| Rollouts | Low | Base | High | Very high |
| --- | ---: | ---: | ---: | ---: |
| 12 | ~$1.65 | ~$4.72 | ~$11.59 | ~$17.64 |
| 48 | ~$6.59 | ~$18.90 | ~$46.37 | ~$70.56 |
| 120 | ~$16.47 | ~$47.25 | ~$115.92 | ~$176.40 |
| 250 | ~$34.31 | ~$98.44 | ~$241.50 | ~$367.50 |

This is the economic reason to run the GPT-mini smoke lane. The risk is quality,
not cost.

Empirical sanity check:

```text
if 80 real rollouts cost about $40, then:
  250 similar rollouts ~= $125
  250 rollouts with a 40-turn cap instead of ~25-turn average ~= $200
  250 rollouts with longer thinking/output ~= $250-$500+
```

So the practical budget expectation for 250 Sonnet full rollouts is:

```text
optimistic: $125-$250
realistic:  $300-$600
worst-case: $900+
```

These are planning estimates, not billing truth. After generation, run:

```bash
python scripts/summarize_openrouter_rollout_cost.py \
  --run-dir runs/<recovery_generation_run> \
  --model openai/gpt-5.5
```

If the generation script writes OpenRouter-style response JSON with `usage`
fields, this produces the actual token count and estimated cost for the run.

## Diversity Constraints

Before training, validate the generated dataset with hard histograms.

Minimum required checks:

```text
first_action_tool_counts
target_tool_counts
example_type_counts
failure_type_counts
recovery_action_counts
repo_counts
turn_position_counts
trajectory_length_counts
target_tool_counts
```

Recommended first repair mix:

```text
Type A exploration-first positives: 40%
Type B failure-then-recovery:       40%
Type C ambiguous first-action:      20%
```

Hard balance rule:

```text
shell/exploration first actions >= read_file first actions
```

For failure-then-recovery rollouts, the trajectory may intentionally begin with
a failed `read_file` so that recovery has realistic context. The hard balance
gate should therefore be applied to supervised target tools, not blindly to the
first action in the raw synthetic trajectory.

Useful initial target:

```text
300-500 additional SFT rows
at least 100 first-action rows
at least 100 recovery-target rows
at least 5 repositories
at least 5 failure types
at least 4 recovery action families
```

Do not train if `read_file` is again the dominant first action. That would
reproduce the observed collapse.

## Loss Weighting

Data alone may not be enough because the previous SFT loss was likely dominated
by easy scaffold tokens. The repair pass should support per-token or per-row
weighting.

Initial row weights:

```text
normal clean teacher row:        1.0
exploration-first Type A row:    2.0
failure-recovery Type B row:     2.5
ambiguous first-action Type C:   2.0
```

If token-level weighting is available, upweight:

```text
tool name token span
required argument key/value token span
path/cmd content tokens
```

Do not upweight:

```text
<tool_call>
<arg_key>
<arg_value>
closing tags
assistant boundary tokens
```

The intent is to stop the model from solving the loss with scaffold syntax plus
the `read_file` prior.

## Prompt Pack Appendix

This section defines the first implementation prompt pack. Each prompt family is
a specialization of the same shared contract.

### Shared System Prompt

Use this as the stable system/developer instruction for every teacher call:

```text
You are generating supervised training data for a small dense coding-agent
student model.

The student already knows the Laguna tagged tool-call syntax, but it has two
observed failures:

1. It collapses first actions to read_file even when shell exploration is
   required.
2. When a tool observation reports failure, it repeats the failed strategy
   instead of recovering.

Generate one full synthetic coding-agent rollout that teaches the requested
lesson. The rollout must look like a realistic coding-agent transcript.

You must output exactly:

<example>
  <metadata_json>{...}</metadata_json>
  <scheme>...</scheme>
  <trajectory_json>{...}</trajectory_json>
</example>

The <scheme> block is teacher-only data-generation planning. It should explain
what lesson the rollout teaches and how the failure/recovery arc will be
constructed. The student will never see <scheme>.

The <trajectory_json> block is the only block eligible for supervised training.
It must contain valid JSON with this shape:

{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "<think>...</think>\n<tool_call>...</tool_call>"},
    {"role": "tool", "content": "..."}
  ],
  "stop_reason": "resolved" | "max_turns" | "gave_up",
  "patch_nonempty": true | false,
  "notes": "short audit note"
}

Assistant messages must use Laguna tagged tool-call text:

<think>
Short task-local reasoning based only on visible task text, prior tool calls,
and prior tool observations.
</think>
<tool_call>tool_name
<arg_key>key</arg_key>
<arg_value>value</arg_value>
</tool_call>

Never mention the prompt, data generation, the scheme, labels, the child model,
or training inside <trajectory_json> or inside any <think> block.

The child-facing <think> blocks should be short and operational. They should
sound like normal inference-time recovery reasoning, e.g. "that path was not
found, so I should search for the symbol instead."

Generate a full rollout, not a one-shot answer. Prefer 8-25 assistant turns.
Use up to 40 assistant turns only if the task naturally needs it. Include a
natural stopping point. If a patch is made, include a verification or at least a
reasoned stop.

Use only these tools:

- read_file(path)
- shell(cmd)
- apply_patch(patch)
- exit(message)

Make tool observations realistic and concise. The recovery actions must be
different from the failed actions they respond to.
```

### Shared User Prompt Skeleton

Each prompt-family builder fills this skeleton:

```text
Task seed:
{task_json}

Repository summary:
{repo_summary_or_empty}

Prompt family:
{prompt_family_id}: {prompt_family_name}

Required lesson:
{required_lesson}

Required rollout constraints:
{family_constraints}

Generate exactly one full synthetic rollout following the shared output schema.
```

`task_json` should come from the task registry and include at least:

```text
task_id
repo
issue/title/problem statement
base commit if available
```

`repo_summary_or_empty` can be empty for phase -1. If repo summaries are added
later, they should be factual and brief; do not stuff full files into this
prompt pack unless the cost model is updated.

### P1: Symbol But No Path

Required lesson:

```text
When the issue names a symbol/function/class but gives no reliable file path,
the first useful action is shell exploration, not read_file.
```

Family constraints:

```text
The first assistant action must be shell.
The first shell command should grep/rg for the symbol or a distinctive phrase.
After a successful search observation, the rollout should read the discovered
file, inspect nearby context, and make or outline a plausible patch.
At least one <think> block must explicitly say the symbol is not enough to infer
a path.
Do not include an artificial failed read_file before the first shell action.
```

### P2: Error Message But No Path

Required lesson:

```text
When the issue gives an error message but no file path, search for the error
text or nearby invariant before reading files.
```

Family constraints:

```text
The first assistant action must be shell.
Use grep/rg for the exact error phrase or a stable substring.
If the first search returns too many matches, refine the search once.
The rollout should demonstrate grounding the path from search results before
read_file.
```

### P3: PR URL / Issue URL Trap

Required lesson:

```text
External URLs and issue IDs are not repository paths. Do not pass them to
read_file.
```

Family constraints:

```text
The rollout must include an issue text with a path-like external URL or PR URL.
The <scheme> may plan the trap, but the child-facing <think> must not mention
that it is a synthetic trap.
The assistant should avoid reading the URL as a path and should search for
symbols, filenames, or error text instead.
At least one later turn should read a real discovered repo path.
```

### P4: Repeated Repo Path Trap

Required lesson:

```text
Recover from over-deep hallucinated paths like django/django/django/... by
checking the repository structure or searching from the repo root.
```

Family constraints:

```text
The rollout must include a failed read_file observation for a repeated repo path
in the prefix before the recovery target.
The next assistant action must not be another read_file of the same path.
The recovery should use shell ls/find/grep to ground the correct path.
The child-facing <think> should explicitly cite the failed observation.
```

### P5: File-Not-Found Recovery

Required lesson:

```text
After "file not found", change strategy. Do not repeat the failed read_file.
```

Family constraints:

```text
Include a failed read_file followed by an observation containing "file not found"
or "outside repo".
The immediate next assistant action must be shell exploration or a clearly
different read_file path justified by visible evidence.
The rollout should continue at least three assistant turns after the recovery.
```

### P6: Empty Grep Recovery

Required lesson:

```text
After an empty search, broaden or reformulate the search instead of looping.
```

Family constraints:

```text
Include a shell grep/rg command whose observation has no matches.
The recovery action should broaden the query, search for a related symbol, list
candidate directories, or search tests.
Do not make the recovery another identical grep.
The <think> block should explain why the new query is broader or different.
```

### P7: Patch Context Failure

Required lesson:

```text
When apply_patch fails because context is stale or wrong, reread the relevant
file context before retrying.
```

Family constraints:

```text
Include an apply_patch failure observation such as "hunk failed", "context not
found", or "patch does not apply".
The immediate recovery action should be read_file or shell sed/grep to inspect
nearby context, not another blind patch.
The later retry patch should be non-empty and more specific.
```

### P8: Malformed / No-Op Patch

Required lesson:

```text
Malformed patches and header-only patches are failures; recover by constructing
a real patch body.
```

Family constraints:

```text
Include a failed apply_patch with a malformed, header-only, or no-op patch.
The recovery should inspect file context, then emit a corrected non-empty patch.
The child-facing <think> should identify that the previous patch had no real
edit.
```

### P9: Late-Turn Recovery

Required lesson:

```text
Recovery is needed after several prior turns too, not only on turn 1.
```

Family constraints:

```text
The first 3-6 assistant turns should be plausible exploration or patching.
The failure should occur after turn 3.
The recovery target should be after that late failure.
The rollout should continue after recovery rather than stopping immediately.
```

### P10: Ambiguous Multi-Module Issue

Required lesson:

```text
For issues that could live in multiple modules, inspect multiple candidates
before choosing a file to patch.
```

Family constraints:

```text
The first action should be shell exploration.
The rollout should find at least two plausible candidate files or modules.
The assistant should read enough context to choose one.
Avoid jumping to a single read_file path without evidence.
```

### P11: Correct Read_File Positives

Required lesson:

```text
Preserve cases where read_file is genuinely the right first action because the
issue gives a clear repository path.
```

Family constraints:

```text
The issue text must include a real repo path.
The first assistant action should be read_file on that path.
The rollout should still include thinking that explains why the path is grounded.
Do not include a synthetic failure before the first read_file.
This family prevents the repair pass from overcorrecting to always-shell.
```

### P12: Test / Verification Recovery

Required lesson:

```text
After tests fail, inspect the failure and patch again instead of stopping or
repeating the same patch.
```

Family constraints:

```text
The rollout should include a patch, a test or focused check, a failing
observation, and a recovery action.
The recovery should inspect the failure output or relevant file before applying
another patch.
The rollout should end with either a passing verification observation or a clear
exit explaining the remaining uncertainty.
```

## Implementation Plan

### Task 1: Add Recovery Example Schema

**Files:**

- Create: `src/densify/recovery_data/schema.py`
- Test: `tests/test_recovery_data_schema.py`

Define dataclasses or typed dictionaries for:

```text
RecoveryMetadata
RecoveryExample
RecoveryValidationResult
```

The metadata must include:

```text
task_id
example_type
failure_type
intended_first_action
recovery_action
target_tool
source
```

### Task 2: Add Teacher Prompt Builder

**Files:**

- Create: `src/densify/recovery_data/prompts.py`
- Create: `scripts/generate_recovery_examples.py`
- Test: `tests/test_recovery_data_prompts.py`

The prompt builder should emit explicit instructions:

```text
write <scheme> first
write <trajectory> second
never put <scheme> inside <trajectory>
make failures realistic
make the recovery action correct and executable
```

The generator must support phase-limited runs. Phase -1 should be able to run
exactly one rollout per prompt family and stop.

The script should support:

```bash
python scripts/generate_recovery_examples.py \
  --tasks tasks/registry_balanced_100_train80.jsonl \
  --output data/recovery/metacognitive_recovery_raw.jsonl \
  --phase smoke \
  --samples-per-family 1 \
  --max-turns 40 \
  --provider openrouter \
  --model anthropic/claude-sonnet-4.6
```

### Task 3: Parse And Quarantine Teacher Scheme Blocks

**Files:**

- Create: `src/densify/recovery_data/parse.py`
- Create: `scripts/parse_recovery_examples.py`
- Test: `tests/test_recovery_data_parse.py`

Parser requirements:

```text
extract metadata
extract scheme for audit only
extract trajectory for training conversion
assert scheme is absent from trajectory
assert trajectory contains parseable tool calls
assert child-facing <think> blocks do not mention the scheme or data generation
reject examples without corrective targets
```

The parsed output should write:

```text
data/recovery/metacognitive_recovery_parsed.jsonl
```

with the scheme stored only under an audit-only field:

```json
{
  "id": "...",
  "metadata": {...},
  "scheme_audit": "...",
  "trajectory_messages": [...]
}
```

`scheme_audit` must never be read by the SFT converter.

### Task 4: Convert Recovery Examples To SFT Rows

**Files:**

- Create: `src/densify/recovery_data/to_sft.py`
- Create: `scripts/build_recovery_sft.py`
- Test: `tests/test_recovery_data_to_sft.py`

Conversion rules:

```text
emit one row per corrective assistant action
include the failed action and observation in the prefix
target only the corrective assistant action
preserve metadata.example_type and metadata.failure_type
set row weight from the table in Loss Weighting
```

Output:

```text
data/sft/metacognitive_recovery_sft_<run_id>.jsonl
```

### Task 5: Validate Diversity Before Training

**Files:**

- Create: `src/densify/recovery_data/validate.py`
- Create: `scripts/validate_recovery_sft.py`
- Test: `tests/test_recovery_data_validate.py`

Validation report:

```json
{
  "num_rollouts": 48,
  "num_rows": 720,
  "first_action_tool_counts": {"shell": 132, "read_file": 88, "apply_patch": 20},
  "example_type_counts": {"exploration_first": 96, "failure_then_recovery": 96, "ambiguous_first_action": 48},
  "failure_type_counts": {"pr_url_as_path": 30, "repeated_repo_path": 22, "empty_grep": 18},
  "recovery_action_counts": {"shell_grep": 70, "shell_ls": 32, "read_alternate_file": 41},
  "child_facing_think_rate": 0.96,
  "passes_hard_balance": true,
  "scheme_leak_count": 0
}
```

Hard failures:

```text
scheme_leak_count > 0
think blocks mention scheme/data generation/synthetic prompt
first_action read_file > first_action shell/explore
failure_then_recovery rows with no failed observation in prefix
unknown tool names
empty target assistant actions
```

### Task 6: Train Repair SFT

**Files:**

- Modify: `scripts/train_dense_sft.py`
- Test: `tests/test_train_dense_sft_utils.py`

Add support for row weights if it is not already wired through the loss.

Suggested run:

```bash
python scripts/train_dense_sft.py \
  --model runs/sft/<previous_run>/checkpoint-final \
  --dataset data/sft/metacognitive_recovery_sft_<run_id>.jsonl \
  --output-dir runs/sft_repair/<run_id> \
  --seq-len 16384 \
  --lr 1e-5 \
  --max-steps 300 \
  --enable-thinking
```

Use a low learning rate because the previous run already recovered syntax. This
pass should change policy and observation-conditioned thinking, not destroy
format.

### Task 7: Re-Run The Diagnostic Probes

**Files:**

- Use existing first-action probe scripts and held-out rollout scripts.

Required before/after metrics:

```text
exact-prefix parseable_tool_call_rate
exact-prefix generated_tool_counts
exact-prefix tool_match_rate
exact-prefix primary_overlap_rate
held-out first-turn path_exists_rate
held-out first-turn tool distribution
live rollout parseable turns
live rollout thinking blocks before tool calls
live rollout repeated-action rate after failed observations
```

Success threshold for the repair pass:

```text
parseable_tool_call_rate remains >= 0.95
generated first-action tool distribution no longer collapses to read_file
tool_match_rate improves above read_file base rate
generated thinking cites the visible observation, not the discarded scheme
live rollout changes action after "file not found" at least once
```

## How This Composes With KD

This recovery SFT pass should make the student coherent enough for better
distillation. The recommended sequence is:

```text
1. current dense SFT checkpoint
2. recovery-shaped SFT repair pass from this RFC
3. off-policy KD on teacher trajectories / teacher first-action logits
4. only then reconsider online/on-policy KD
```

Off-policy KD is especially useful for the first-action collapse because it can
teach the teacher's distribution over tool choice, not just a single argmax
target.

## Open Questions

1. Should the child remain no-thinking?

   No. The repair pass should be thinking-on. The teacher-only `<scheme>` is
   still discarded, but short child-facing recovery thinking should remain in
   the trajectory and be trained as part of the assistant target.

2. Should `<scheme>` be saved anywhere?

   Yes, but only as audit metadata in raw/parsed recovery artifacts. It must not
   be rendered into training messages.

3. Should synthetic failures be executable in a real sandbox?

   Prefer yes for high-quality rows, but do not require it for every synthetic
   failure. The required property is that the recovery target is plausible and
   teaches the model to condition on the failed observation.

4. Should natural teacher rollouts still be mixed in?

   Yes. Keep a small amount of previous clean SFT data to preserve syntax and
   normal behavior, but the repair batch should be deliberately balanced rather
   than natural-marginal sampled.
