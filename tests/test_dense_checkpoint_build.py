from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from densify.dense_checkpoint.build_placeholder import build_tiny_placeholder


def test_build_tiny_placeholder_loads_with_hf_auto_classes(tmp_path: Path):
    out_dir = tmp_path / "tiny-laguna-dense"

    build_tiny_placeholder(out_dir, k_routed=2)

    config = AutoConfig.from_pretrained(out_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(out_dir, trust_remote_code=True)

    assert config.model_type == "laguna_dense"
    assert config.k_routed == 2
    output = model(input_ids=torch.tensor([[1, 2, 3]]))
    assert output.logits.shape == (1, 3, config.vocab_size)
    assert (out_dir / "configuration_laguna_dense.py").exists()
    assert (out_dir / "modeling_laguna_dense.py").exists()
