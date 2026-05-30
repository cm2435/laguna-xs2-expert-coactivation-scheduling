# Reconstruction v2 Preregistered Evaluation

Run under evaluation:

`runs/reconstruction_v2_opencode100k_tool7x_recovery7x_kl002_bs12_seq1536_save250_nocudnn_20260530T080929Z`

This document fixes the checkpoint readout rules before seeing the first model sample.

## Training Run

- Objective: dense routed-FFN activation reconstruction plus teacher logit KL.
- Loss weights: `cosine_weight=0.05`, `logit_kl_weight=0.02`.
- Data: `data/reconstruction/recon_mix_opencode100k_tool7x_recovery7x_seed1337.jsonl`.
- Sequence packing: EOS-separated soft document boundaries.
- Attention isolation: no block-diagonal mask in this v2 probe.
- Sequence length: `1536`.
- Batch size: `12`.
- Checkpoints: every `250` steps.
- Final cap: `5000` steps, unless dataset exhaustion ends earlier.

Known confound:

Because the objective includes KL/next-token behavior, EOS-only packing can still allow cross-document attention. A failure that looks like context bleed or wrong-repo/path bleed is therefore inconclusive between undercooked dense reconstruction and soft-boundary contamination. A strong pass is still meaningful, because it succeeds despite this handicap.

## Checkpoint Cadence

Probe every saved checkpoint until the first clear verdict:

- `checkpoint-step-250`
- `checkpoint-step-500`
- `checkpoint-step-750`
- `checkpoint-step-1000`

After `checkpoint-step-1000`, continue probing only if results are borderline or improving.

Disk babysitting rule:

Keep `checkpoint-step-250` and the two newest `checkpoint-step-*` directories. Delete older intermediate checkpoints after they have been probed or are superseded. Keep `checkpoint-final` if it appears.

## Tier 0: Loadability

Pass if the checkpoint:

- loads with `trust_remote_code=True`;
- can generate without runtime errors using `--disable-cudnn-sdpa`;
- does not emit obvious token soup on a simple single-turn prompt.

Fail if it cannot load or generation is unreadable token soup. A Tier 0 fail is upstream of tool policy and should not be overinterpreted as a tool-calling failure.

## Tier 1: Tool-Call Format Probe

Run the base tool probe against 5 fixed prompts.

Pass threshold:

- at least `4/5` parseable Laguna tagged tool calls;
- at least `4/5` known tool names from the harness tool set;
- at least `3/5` calls contain the required argument key for the chosen tool.

Interpretation:

- `>=4/5` parseable plus known tool names: reconstruction v2 has recovered basic tool-call surface form well enough to justify downstream SFT.
- `2-3/5`: borderline; probe the next checkpoint before deciding.
- `0-1/5`: fail for this checkpoint.

## Tier 2: First-Action Policy Probe

Run exact-prefix first-action eval on the same small held-out/train diagnostic set used earlier, but do not require task success.

Primary pass threshold:

- generated tool distribution must not collapse to one tool;
- `tool_match_rate >= 0.5`;
- `primary_overlap_rate >= 0.4`;
- first action should not repeatedly read PR URLs or obviously bogus repo paths.

Strong pass threshold:

- `tool_match_rate >= 0.65`;
- `primary_overlap_rate >= 0.55`;
- both `shell` and `read_file` appear when expected by the teacher distribution.

Interpretation:

- If Tier 1 passes and Tier 2 improves over the previous collapsed SFT baseline, proceed to recovery/tool SFT from this reconstruction checkpoint.
- If Tier 1 passes but Tier 2 still collapses, use this as a base for targeted SFT but do not claim policy recovery from reconstruction alone.
- If Tier 2 fails specifically via wrong-repo/path bleed, mark the result inconclusive until block-diagonal packing is implemented.

## Tier 3: Live Rollout Smoke

Only run after Tier 1 passes.

Run one held-out SWE-bench task for at most 4-6 turns with conservative generation settings.

Pass threshold:

- first tool call executes;
- at least one second-turn action changes in response to the first observation;
- no identical failed tool call repeated 3 times.

This is not a task-success eval. It is only a closed-loop sanity check.

## Decision Rules

First checkpoint that passes Tier 1:

- Run Tier 2 immediately.
- If Tier 2 is non-collapsed, keep training but mark this checkpoint as the first viable SFT base.

At step 1000:

- If Tier 1 still fails, do not keep grading on a curve. Treat v2 as not yet fixing base format competence.
- Next diagnosis should be KL weighting, block-diagonal packing, or reconstruction objective, not more recovery SFT.

At any checkpoint:

- Do not call a nice training loss curve a success without Tier 1 evidence.
- Do not call a context-bleed failure a capacity failure while this run uses EOS-only soft boundaries.
