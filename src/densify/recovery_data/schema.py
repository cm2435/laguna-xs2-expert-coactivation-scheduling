from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

EXAMPLE_TYPES = {
    "exploration_first",
    "failure_then_recovery",
    "ambiguous_first_action",
    "read_file_positive",
    "patch_recovery",
    "test_recovery",
}

PROMPT_FAMILIES = {
    "P1": {
        "name": "Symbol but no path",
        "example_type": "exploration_first",
        "failure_type": "none",
        "recovery_action": "shell_grep",
        "target_tool": "shell",
        "phase2_samples": 25,
    },
    "P2": {
        "name": "Error message but no path",
        "example_type": "exploration_first",
        "failure_type": "none",
        "recovery_action": "shell_grep",
        "target_tool": "shell",
        "phase2_samples": 20,
    },
    "P3": {
        "name": "PR URL / issue URL trap",
        "example_type": "ambiguous_first_action",
        "failure_type": "pr_url_as_path",
        "recovery_action": "shell_grep",
        "target_tool": "shell",
        "phase2_samples": 20,
    },
    "P4": {
        "name": "Repeated repo path trap",
        "example_type": "failure_then_recovery",
        "failure_type": "repeated_repo_path",
        "recovery_action": "shell_find_or_grep",
        "target_tool": "shell",
        "phase2_samples": 20,
    },
    "P5": {
        "name": "File-not-found recovery",
        "example_type": "failure_then_recovery",
        "failure_type": "file_not_found",
        "recovery_action": "shell_explore",
        "target_tool": "shell",
        "phase2_samples": 25,
    },
    "P6": {
        "name": "Empty grep recovery",
        "example_type": "failure_then_recovery",
        "failure_type": "empty_grep",
        "recovery_action": "broaden_search",
        "target_tool": "shell",
        "phase2_samples": 20,
    },
    "P7": {
        "name": "Patch context failure",
        "example_type": "patch_recovery",
        "failure_type": "patch_context_failure",
        "recovery_action": "read_context",
        "target_tool": "read_file",
        "phase2_samples": 20,
    },
    "P8": {
        "name": "Malformed / no-op patch",
        "example_type": "patch_recovery",
        "failure_type": "malformed_patch",
        "recovery_action": "construct_real_patch",
        "target_tool": "apply_patch",
        "phase2_samples": 15,
    },
    "P9": {
        "name": "Late-turn recovery",
        "example_type": "failure_then_recovery",
        "failure_type": "late_turn_failure",
        "recovery_action": "late_recovery",
        "target_tool": "shell",
        "phase2_samples": 25,
    },
    "P10": {
        "name": "Ambiguous multi-module issue",
        "example_type": "ambiguous_first_action",
        "failure_type": "ambiguous_module",
        "recovery_action": "multi_candidate_explore",
        "target_tool": "shell",
        "phase2_samples": 20,
    },
    "P11": {
        "name": "Correct read_file positives",
        "example_type": "read_file_positive",
        "failure_type": "none",
        "recovery_action": "none",
        "target_tool": "read_file",
        "phase2_samples": 20,
    },
    "P12": {
        "name": "Test / verification recovery",
        "example_type": "test_recovery",
        "failure_type": "test_failure",
        "recovery_action": "inspect_test_failure",
        "target_tool": "shell",
        "phase2_samples": 15,
    },
}


@dataclass(frozen=True)
class TaskSeed:
    task_id: str
    repo: str
    repo_id: str = ""
    base_commit: str = ""
    problem_statement: str = ""
    manifest_path: str = ""


@dataclass(frozen=True)
class RecoveryMetadata:
    task_id: str
    repo: str
    prompt_family_id: str
    prompt_family_name: str
    example_type: str
    failure_type: str
    intended_first_action: str
    recovery_action: str
    target_tool: str
    source: str = "metacognitive_recovery"


@dataclass(frozen=True)
class RecoveryExample:
    id: str
    metadata: RecoveryMetadata
    scheme_audit: str
    trajectory_messages: list[dict[str, Any]]
    stop_reason: str = ""
    patch_nonempty: bool = False
    notes: str = ""
    raw_text: str = ""


@dataclass(frozen=True)
class RecoveryValidationResult:
    num_rollouts: int
    num_rows: int
    first_action_tool_counts: dict[str, int] = field(default_factory=dict)
    example_type_counts: dict[str, int] = field(default_factory=dict)
    failure_type_counts: dict[str, int] = field(default_factory=dict)
    recovery_action_counts: dict[str, int] = field(default_factory=dict)
    target_tool_counts: dict[str, int] = field(default_factory=dict)
    repo_counts: dict[str, int] = field(default_factory=dict)
    scheme_leak_count: int = 0
    think_leak_count: int = 0
    parseable_tool_call_rows: int = 0
    child_facing_think_rows: int = 0
    failed_observation_prefix_rows: int = 0
    empty_target_rows: int = 0
    unknown_tool_rows: int = 0
    passes_hard_balance: bool = False
    hard_failures: list[str] = field(default_factory=list)


PhaseName = Literal["smoke", "pilot", "repair", "scale", "stretch"]


PHASE_SAMPLES_PER_FAMILY: dict[PhaseName, int] = {
    "smoke": 1,
    "pilot": 4,
    "repair": 10,
    "scale": 20,
    "stretch": 40,
}


ROW_WEIGHTS_BY_EXAMPLE_TYPE = {
    "exploration_first": 2.0,
    "failure_then_recovery": 2.5,
    "ambiguous_first_action": 2.0,
    "read_file_positive": 1.5,
    "patch_recovery": 2.5,
    "test_recovery": 2.5,
}


KNOWN_TOOLS = {"read_file", "shell", "apply_patch", "exit"}
