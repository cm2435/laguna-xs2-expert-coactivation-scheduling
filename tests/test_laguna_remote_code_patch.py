import json
from pathlib import Path

from densify.dense_checkpoint.build_placeholder import write_laguna_dense_remote_code


def test_write_laguna_dense_remote_code_patches_sparse_moe_block(tmp_path: Path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    (source / "configuration_laguna.py").write_text('class LagunaConfig:\n    model_type = "laguna"\n')
    (source / "config.json").write_text(
        json.dumps(
            {
                "model_type": "laguna",
                "architectures": ["LagunaForCausalLM"],
                "auto_map": {
                    "AutoConfig": "configuration_laguna.LagunaConfig",
                    "AutoModelForCausalLM": "modeling_laguna.LagunaForCausalLM",
                },
                "num_experts_per_tok": 8,
                "moe_intermediate_size": 512,
            }
        )
    )
    (source / "modeling_laguna.py").write_text(
        """
class LagunaSparseMoeBlock(nn.Module):
    def __init__(self, config: LagunaConfig):
        super().__init__()
        self.experts = LagunaExperts(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states


def rotate_half(x):
    return x
"""
    )

    write_laguna_dense_remote_code(source, target, k_routed=4)

    config = json.loads((target / "config.json").read_text())
    modeling = (target / "modeling_laguna_dense.py").read_text()

    assert config["model_type"] == "laguna_dense"
    assert config["num_experts_per_tok"] == 4
    assert "routed_dense" in modeling
    assert "LagunaExperts(config)" not in modeling
    assert (target / "configuration_laguna_dense.py").exists()
