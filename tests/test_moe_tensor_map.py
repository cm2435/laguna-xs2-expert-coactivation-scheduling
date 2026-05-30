from densify.dense_checkpoint.moe_tensor_map import classify_laguna_tensor_key


def test_tensor_map_classifies_fake_laguna_style_keys():
    assert (
        classify_laguna_tensor_key("model.layers.0.mlp.router.weight").kind == "router"
    )
    assert (
        classify_laguna_tensor_key("model.layers.0.mlp.shared_expert.gate_proj.weight").kind
        == "shared_expert"
    )
    routed = classify_laguna_tensor_key("model.layers.0.mlp.experts.17.up_proj.weight")
    assert routed.kind == "routed_expert"
    assert routed.layer_id == 0
    assert routed.expert_id == 17
    assert routed.proj == "up_proj"


def test_tensor_map_classifies_actual_laguna_packed_keys():
    shared = classify_laguna_tensor_key("model.layers.3.mlp.shared_experts.down_proj.weight")
    assert shared.kind == "shared_expert"
    assert shared.layer_id == 3
    assert shared.proj == "down_proj"

    packed = classify_laguna_tensor_key("model.layers.3.mlp.experts.gate_up_proj")
    assert packed.kind == "packed_routed_experts"
    assert packed.layer_id == 3
    assert packed.proj == "gate_up_proj"
