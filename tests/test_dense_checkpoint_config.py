from densify.dense_checkpoint.config import LagunaDenseConfig


def test_dense_config_records_conversion_metadata():
    config = LagunaDenseConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=8,
        num_hidden_layers=2,
        k_routed=3,
        expert_intermediate_size=4,
        source_model="poolside/Laguna-XS.2",
        placeholder_init="copied_shell_random_routed",
    )

    assert config.model_type == "laguna_dense"
    assert config.dense_routed_intermediate_size == 12
    assert config.moe_dense_conversion == {
        "source_model": "poolside/Laguna-XS.2",
        "kind": "routed_moe_to_dense_swiglu",
        "k_routed": 3,
        "expert_intermediate_size": 4,
        "dense_routed_intermediate_size": 12,
        "shared_expert": "kept",
        "placeholder_weights": "copied_shell_random_routed",
    }
