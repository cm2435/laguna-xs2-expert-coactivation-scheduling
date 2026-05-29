from densify.dense_checkpoint.build_placeholder import copied_shell_key_report


def test_copied_shell_key_report_counts_random_routed_dense_and_shared_copies():
    report = copied_shell_key_report(
        source_keys=[
            "model.embed_tokens.weight",
            "model.layers.1.mlp.shared_experts.gate_proj.weight",
        ],
        target_keys=[
            "model.embed_tokens.weight",
            "model.layers.1.mlp.shared_experts.gate_proj.weight",
            "model.layers.1.mlp.routed_dense.gate_proj.weight",
            "model.layers.1.mlp.routed_dense.up_proj.weight",
            "model.layers.1.mlp.routed_dense.down_proj.weight",
        ],
    )

    assert report == {
        "copied_keys": 2,
        "copied_shared_expert_keys": 1,
        "random_routed_dense_keys": 3,
        "other_missing_target_keys": 0,
    }
