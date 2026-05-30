import torch

from densify.dense_checkpoint.config import LagunaDenseConfig
from densify.dense_checkpoint.modeling_laguna_dense import (
    LagunaDenseForCausalLM,
    LagunaDenseMoEReplacement,
    LagunaDenseRoutedMLP,
)


def tiny_config() -> LagunaDenseConfig:
    return LagunaDenseConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=8,
        num_hidden_layers=2,
        k_routed=2,
        expert_intermediate_size=4,
    )


def test_dense_routed_mlp_forward_shape():
    mlp = LagunaDenseRoutedMLP(tiny_config())
    x = torch.randn(2, 3, 16)

    out = mlp(x)

    assert out.shape == (2, 3, 16)


def test_dense_moe_replacement_adds_shared_path():
    config = tiny_config()
    module = LagunaDenseMoEReplacement(config)
    x = torch.randn(2, 3, 16)

    with_shared = module(x)
    module.shared_expert = None
    without_shared = module(x)

    assert with_shared.shape == without_shared.shape == (2, 3, 16)
    assert not torch.allclose(with_shared, without_shared)


def test_tiny_dense_model_forward_logits_shape():
    model = LagunaDenseForCausalLM(tiny_config())
    input_ids = torch.tensor([[1, 2, 3]])

    out = model(input_ids=input_ids)

    assert out.logits.shape == (1, 3, 64)


def test_tiny_dense_model_generate_two_tokens():
    model = LagunaDenseForCausalLM(tiny_config())
    input_ids = torch.tensor([[1, 2, 3]])

    out = model.generate(input_ids=input_ids, max_new_tokens=2, do_sample=False)

    assert out.shape == (1, 5)
