import torch

from densify.model_introspection import find_candidate_moe_modules


class FakeExpertBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = torch.nn.Linear(4, 2)
        self.shared_expert = torch.nn.Linear(4, 4)


class FakeDecoderLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = FakeExpertBlock()


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([FakeDecoderLayer()])


def test_find_candidate_moe_modules_detects_shared_expert_pattern():
    model = FakeModel()
    found = find_candidate_moe_modules(model)

    assert any(item.name == "layers.0.mlp" for item in found)


def test_architecture_summary_counts_decoder_layers():
    from densify.model_introspection import architecture_summary

    summary = architecture_summary(FakeModel(), "fake/model", "bfloat16")

    assert summary["num_transformer_layers"] == 1
