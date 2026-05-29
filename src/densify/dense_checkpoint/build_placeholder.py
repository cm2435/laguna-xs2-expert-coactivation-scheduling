from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Iterable

from densify.dense_checkpoint.config import LagunaDenseConfig
from densify.dense_checkpoint.modeling_laguna_dense import LagunaDenseForCausalLM


REMOTE_MODEL_IMPORT = "from densify.dense_checkpoint.config import LagunaDenseConfig"
REMOTE_RELATIVE_IMPORT = "from .configuration_laguna_dense import LagunaDenseConfig"


def _copy_remote_code(target_dir: Path) -> None:
    package_dir = Path(__file__).resolve().parent
    config_text = (package_dir / "config.py").read_text()
    model_text = (package_dir / "modeling_laguna_dense.py").read_text().replace(
        REMOTE_MODEL_IMPORT, REMOTE_RELATIVE_IMPORT
    )
    (target_dir / "configuration_laguna_dense.py").write_text(config_text)
    (target_dir / "modeling_laguna_dense.py").write_text(model_text)


def _write_model_card(target_dir: Path, repo_name: str | None = None) -> None:
    title = repo_name or target_dir.name
    (target_dir / "README.md").write_text(
        "\n".join(
            [
                f"# {title}",
                "",
                "This is an architecture placeholder for Laguna XS.2 MoE-to-dense serving integration.",
                "The non-MoE backbone and shared experts are intended to be copied from Laguna XS.2.",
                "The routed dense FFN weights are random or structurally initialized.",
                "",
                "This is not a quality checkpoint.",
                "",
            ]
        )
    )


def _copy_optional_metadata_files(source: Path, target: Path) -> None:
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "generation_config.json",
    ):
        src = source / name
        if src.exists():
            shutil.copy2(src, target / name)


def _dense_sparse_moe_block_source() -> str:
    return '''
class LagunaSparseMoeBlock(nn.Module):
    """Dense replacement for Laguna's routed MoE block.

    The class name is intentionally kept as LagunaSparseMoeBlock so the existing
    LagunaDecoderLayer can instantiate it without changing the rest of the model.
    """

    def __init__(self, config: LagunaConfig):
        super().__init__()
        routed_width = config.num_experts_per_tok * config.moe_intermediate_size
        self.routed_dense = LagunaMLP(config, intermediate_size=routed_width)
        self.shared_experts = LagunaMLP(config, intermediate_size=config.shared_expert_intermediate_size)
        self.routed_scaling_factor = config.moe_routed_scaling_factor

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        routed_output = self.routed_dense(hidden_states) * self.routed_scaling_factor
        shared_output = self.shared_experts(hidden_states)
        return routed_output + shared_output


'''.lstrip()


def _patch_laguna_modeling_source(modeling_text: str) -> str:
    pattern = re.compile(
        r"class LagunaSparseMoeBlock\(nn\.Module\):.*?\n\n(?=def rotate_half)",
        flags=re.DOTALL,
    )
    patched, count = pattern.subn(_dense_sparse_moe_block_source(), modeling_text)
    if count != 1:
        raise ValueError("Could not locate exactly one LagunaSparseMoeBlock in modeling_laguna.py")
    return patched.replace(
        "from .configuration_laguna import LagunaConfig",
        "from .configuration_laguna_dense import LagunaConfig",
    )


def write_laguna_dense_remote_code(source_dir: str | Path, target_dir: str | Path, k_routed: int) -> Path:
    source = Path(source_dir)
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)

    config_source = (source / "configuration_laguna.py").read_text().replace(
        'model_type = "laguna"', 'model_type = "laguna_dense"'
    )
    modeling_source = _patch_laguna_modeling_source((source / "modeling_laguna.py").read_text())
    config_json = json.loads((source / "config.json").read_text())
    config_json["model_type"] = "laguna_dense"
    config_json["architectures"] = ["LagunaForCausalLM"]
    config_json["auto_map"] = {
        "AutoConfig": "configuration_laguna_dense.LagunaConfig",
        "AutoModelForCausalLM": "modeling_laguna_dense.LagunaForCausalLM",
    }
    config_json["num_experts_per_tok"] = k_routed
    config_json["moe_dense_conversion"] = {
        "source_model": "poolside/Laguna-XS.2",
        "kind": "routed_moe_to_dense_swiglu",
        "k_routed": k_routed,
        "expert_intermediate_size": config_json.get("moe_intermediate_size"),
        "dense_routed_intermediate_size": k_routed * config_json.get("moe_intermediate_size", 0),
        "shared_expert": "kept",
        "placeholder_weights": "copied_shell_random_routed",
    }

    (target / "configuration_laguna_dense.py").write_text(config_source)
    (target / "modeling_laguna_dense.py").write_text(modeling_source)
    (target / "config.json").write_text(json.dumps(config_json, indent=2) + "\n")
    _copy_optional_metadata_files(source, target)
    _write_model_card(target)
    return target


def copied_shell_key_report(source_keys: Iterable[str], target_keys: Iterable[str]) -> dict[str, int]:
    source_set = set(source_keys)
    target_set = set(target_keys)
    copied = 0
    routed_dense = 0
    shared = 0
    missing = 0
    for key in target_set:
        if ".routed_dense." in key:
            routed_dense += 1
        elif key in source_set:
            copied += 1
            if ".shared_experts." in key:
                shared += 1
        else:
            missing += 1
    return {
        "copied_keys": copied,
        "copied_shared_expert_keys": shared,
        "random_routed_dense_keys": routed_dense,
        "other_missing_target_keys": missing,
    }


def build_tiny_placeholder(target_dir: str | Path, k_routed: int = 2) -> Path:
    target = Path(target_dir)
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)

    config = LagunaDenseConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=8,
        num_hidden_layers=2,
        k_routed=k_routed,
        expert_intermediate_size=4,
        source_model="tiny/local",
        placeholder_init="tiny_random",
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = LagunaDenseForCausalLM(config)
    model.save_pretrained(target, safe_serialization=True)
    _copy_remote_code(target)
    _write_model_card(target)
    return target
