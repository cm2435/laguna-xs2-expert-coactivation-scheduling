from __future__ import annotations

# ruff: noqa: E501
import json
from dataclasses import asdict

from densify.recovery_data.schema import PHASE_SAMPLES_PER_FAMILY, PROMPT_FAMILIES, TaskSeed
from densify.tasks.manifest import load_task_manifest

SHARED_SYSTEM_PROMPT = """You are generating supervised training data for a small dense coding-agent student model.

The student already knows the Laguna tagged tool-call syntax, but it has two observed failures:

1. It collapses first actions to read_file even when shell exploration is required.
2. When a tool observation reports failure, it repeats the failed strategy instead of recovering.

Generate one full synthetic coding-agent rollout that teaches the requested lesson. The rollout must look like a realistic coding-agent transcript.

You must output exactly:

<example>
  <metadata_json>{...}</metadata_json>
  <scheme>...</scheme>
  <trajectory_json>{...}</trajectory_json>
</example>

Do not wrap the answer in Markdown or a code fence. Do not output anything before
<example> or after </example>. Both JSON blocks must be complete and parseable.

The <scheme> block is teacher-only data-generation planning. It should explain what lesson the rollout teaches and how the failure/recovery arc will be constructed. The student will never see <scheme>.

The <trajectory_json> block is the only block eligible for supervised training. It must contain valid JSON with this shape:

{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "<think>...</think>\\n<tool_call>...</tool_call>"},
    {"role": "tool", "content": "..."}
  ],
  "stop_reason": "resolved" | "max_turns" | "gave_up",
  "patch_nonempty": true | false,
  "notes": "short audit note"
}

Role strings must be exactly "system", "user", "assistant", or "tool". Do not
invent role strings like "assistant to=tool". Tool observations are separate
messages with role "tool"; tool calls belong inside assistant message content.

Escape every newline inside JSON string values as \\n. This is especially
important for assistant content, tool observations, patch strings, code blocks,
and file excerpts. Do not place raw multiline text inside JSON strings.

Assistant messages must use Laguna tagged tool-call text:

<think>
Short task-local reasoning based only on visible task text, prior tool calls, and prior tool observations.
</think>
<tool_call>tool_name
<arg_key>key</arg_key>
<arg_value>value</arg_value>
</tool_call>

Never mention the prompt, data generation, the scheme, labels, the child model, or training inside <trajectory_json> or inside any <think> block.

The child-facing <think> blocks should be short and operational. They should sound like normal inference-time recovery reasoning, e.g. "that path was not found, so I should search for the symbol instead."

Generate a full rollout, not a one-shot answer. Completeness and valid JSON are more important than trajectory length. Use 3-7 assistant turns. Use more only if the task naturally needs it and you can still close </trajectory_json> and </example>. Do not continue exploratory debugging once the lesson has been demonstrated. Prefer a compact representative trajectory over a fully exhaustive solution transcript. Include a natural stopping point. If a patch is made, include one concise verification or at least a reasoned stop.

Use only these tools:

- read_file(path)
- shell(command)
- apply_patch(patch)
- exit(message)

Prefer read_file after shell has discovered a concrete file path. Avoid long
shell-only streaks; use shell for search, tests, and concise checks, not as a
replacement for every file read.

Make tool observations realistic and concise. Keep each tool observation under
roughly 80 lines or 4000 characters. Summarize long test output, huge file
contents, and repetitive dots instead of pasting them in full.

The recovery actions must be different from the failed actions they respond to.
"""


FAMILY_CONSTRAINTS = {
    "P1": (
        "The first assistant action must be shell.\n"
        "The first shell command should grep/rg for the symbol or a distinctive phrase.\n"
        "After a successful search observation, the rollout should read the discovered file, inspect nearby context, and make or outline a plausible patch.\n"
        "At least one <think> block must explicitly say the symbol is not enough to infer a path.\n"
        "Do not include an artificial failed read_file before the first shell action."
    ),
    "P2": (
        "The first assistant action must be shell.\n"
        "Use grep/rg for the exact error phrase or a stable substring.\n"
        "If the first search returns too many matches, refine the search once.\n"
        "The rollout should demonstrate grounding the path from search results before read_file."
    ),
    "P3": (
        "The rollout must include an issue text with a path-like external URL or PR URL.\n"
        "The <scheme> may plan the trap, but the child-facing <think> must not mention that it is a synthetic trap.\n"
        "The assistant should avoid reading the URL as a path and should search for symbols, filenames, or error text instead.\n"
        "At least one later turn should read a real discovered repo path."
    ),
    "P4": (
        "The rollout must include a failed read_file observation for a repeated repo path in the prefix before the recovery target.\n"
        "The next assistant action must not be another read_file of the same path.\n"
        "The recovery should use shell ls/find/grep to ground the correct path.\n"
        "The child-facing <think> should explicitly cite the failed observation."
    ),
    "P5": (
        "Include a failed read_file followed by an observation containing \"file not found\" or \"outside repo\".\n"
        "The immediate next assistant action must be shell exploration or a clearly different read_file path justified by visible evidence.\n"
        "The rollout should continue at least three assistant turns after the recovery."
    ),
    "P6": (
        "Include a shell grep/rg command whose observation has no matches.\n"
        "The recovery action should broaden the query, search for a related symbol, list candidate directories, or search tests.\n"
        "Do not make the recovery another identical grep.\n"
        "The <think> block should explain why the new query is broader or different."
    ),
    "P7": (
        "Include an apply_patch failure observation such as \"hunk failed\", \"context not found\", or \"patch does not apply\".\n"
        "The immediate recovery action should be read_file or shell sed/grep to inspect nearby context, not another blind patch.\n"
        "The later retry patch should be non-empty and more specific."
    ),
    "P8": (
        "Include a failed apply_patch with a malformed, header-only, or no-op patch.\n"
        "The recovery should inspect file context, then emit a corrected non-empty patch.\n"
        "The child-facing <think> should identify that the previous patch had no real edit."
    ),
    "P9": (
        "The first 3-6 assistant turns should be plausible exploration or patching.\n"
        "The failure should occur after turn 3.\n"
        "The recovery target should be after that late failure.\n"
        "The rollout should continue after recovery rather than stopping immediately."
    ),
    "P10": (
        "The first action should be shell exploration.\n"
        "The rollout should find at least two plausible candidate files or modules.\n"
        "The assistant should read enough context to choose one.\n"
        "Avoid jumping to a single read_file path without evidence."
    ),
    "P11": (
        "The issue text must include a real repo path.\n"
        "The first assistant action should be read_file on that path.\n"
        "The rollout should still include thinking that explains why the path is grounded.\n"
        "Do not include a synthetic failure before the first read_file.\n"
        "This family prevents the repair pass from overcorrecting to always-shell."
    ),
    "P12": (
        "The rollout should include a patch, a test or focused check, a failing observation, and a recovery action.\n"
        "The recovery should inspect the failure output or relevant file before applying another patch.\n"
        "The rollout should end with either a passing verification observation or a clear exit explaining the remaining uncertainty."
    ),
}


REQUIRED_LESSONS = {
    "P1": "When the issue names a symbol/function/class but gives no reliable file path, the first useful action is shell exploration, not read_file.",
    "P2": "When the issue gives an error message but no file path, search for the error text or nearby invariant before reading files.",
    "P3": "External URLs and issue IDs are not repository paths. Do not pass them to read_file.",
    "P4": "Recover from over-deep hallucinated paths like django/django/django/... by checking the repository structure or searching from the repo root.",
    "P5": "After file not found, change strategy. Do not repeat the failed read_file.",
    "P6": "After an empty search, broaden or reformulate the search instead of looping.",
    "P7": "When apply_patch fails because context is stale or wrong, reread the relevant file context before retrying.",
    "P8": "Malformed patches and header-only patches are failures; recover by constructing a real patch body.",
    "P9": "Recovery is needed after several prior turns too, not only on turn 1.",
    "P10": "For issues that could live in multiple modules, inspect multiple candidates before choosing a file to patch.",
    "P11": "Preserve cases where read_file is genuinely the right first action because the issue gives a clear repository path.",
    "P12": "After tests fail, inspect the failure and patch again instead of stopping or repeating the same patch.",
}


def task_seed_from_manifest(path: str) -> TaskSeed:
    manifest = load_task_manifest(path)
    return TaskSeed(
        task_id=manifest.task_id,
        repo=manifest.repo,
        repo_id=manifest.repo_id,
        base_commit=manifest.base_commit,
        problem_statement=manifest.problem_statement,
        manifest_path=str(manifest.path),
    )


def build_teacher_messages(
    task: TaskSeed,
    *,
    prompt_family_id: str,
    repo_summary: str = "",
) -> list[dict[str, str]]:
    user_prompt = build_user_prompt(
        task,
        prompt_family_id=prompt_family_id,
        repo_summary=repo_summary,
    )
    return [
        {"role": "system", "content": SHARED_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_user_prompt(
    task: TaskSeed,
    *,
    prompt_family_id: str,
    repo_summary: str = "",
) -> str:
    family = PROMPT_FAMILIES[prompt_family_id]
    task_json = json.dumps(asdict(task), indent=2)
    return (
        f"Task seed:\n{task_json}\n\n"
        f"Repository summary:\n{repo_summary or '(none provided)'}\n\n"
        f"Prompt family:\n{prompt_family_id}: {family['name']}\n\n"
        "metadata_json must include these exact fields:\n"
        f"{json.dumps(_metadata_template(task, prompt_family_id), indent=2)}\n\n"
        f"Required lesson:\n{REQUIRED_LESSONS[prompt_family_id]}\n\n"
        f"Required rollout constraints:\n{FAMILY_CONSTRAINTS[prompt_family_id]}\n\n"
        "Generate exactly one full synthetic rollout following the shared output schema."
    )


def prompt_family_ids() -> list[str]:
    return list(PROMPT_FAMILIES.keys())


def expand_prompt_schedule(phase: str, *, samples_per_family: int | None = None) -> list[str]:
    count = samples_per_family
    if count is None:
        count = PHASE_SAMPLES_PER_FAMILY[phase]  # type: ignore[index]
    return [family_id for family_id in prompt_family_ids() for _ in range(count)]


def _metadata_template(task: TaskSeed, prompt_family_id: str) -> dict[str, str]:
    family = PROMPT_FAMILIES[prompt_family_id]
    return {
        "id": f"{task.task_id}:{prompt_family_id}",
        "task_id": task.task_id,
        "repo": task.repo,
        "prompt_family_id": prompt_family_id,
        "prompt_family_name": str(family["name"]),
        "example_type": str(family["example_type"]),
        "failure_type": str(family["failure_type"]),
        "intended_first_action": str(family["target_tool"]),
        "recovery_action": str(family["recovery_action"]),
        "target_tool": str(family["target_tool"]),
        "source": "metacognitive_recovery",
    }
