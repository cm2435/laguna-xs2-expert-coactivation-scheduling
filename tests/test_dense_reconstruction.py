import torch
from torch import nn

from densify.reconstruction import (
    compute_parallel_reconstruction_loss,
    find_reconstruction_layer_ids,
    freeze_for_dense_reconstruction,
)


class FakeMLP(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class FakeDenseMLP(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.routed_dense = nn.Linear(hidden_size, hidden_size, bias=False)
        self.shared_experts = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.routed_dense(x) + self.shared_experts(x)


class FakeLayer(nn.Module):
    def __init__(self, mlp: nn.Module):
        super().__init__()
        self.mlp = mlp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(x)


class FakeBackbone(nn.Module):
    def __init__(self, layers: list[nn.Module]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        x = torch.nn.functional.one_hot(input_ids, num_classes=4).float()
        for layer in self.layers:
            x = layer(x)
        return type("FakeOutput", (), {"logits": x})()


class FakeModel(nn.Module):
    def __init__(self, layers: list[nn.Module]):
        super().__init__()
        self.model = FakeBackbone(layers)
        self.norm = nn.LayerNorm(4)
        self.lm_head = nn.Linear(4, 4, bias=False)
        self.attn = nn.Linear(4, 4, bias=False)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def test_find_reconstruction_layer_ids_uses_student_routed_dense_modules():
    teacher = FakeModel([FakeLayer(FakeMLP(4)), FakeLayer(FakeMLP(4))])
    student = FakeModel([FakeLayer(FakeMLP(4)), FakeLayer(FakeDenseMLP(4))])

    assert find_reconstruction_layer_ids(teacher, student) == [1]


def test_freeze_for_dense_reconstruction_only_trains_routed_dense():
    student = FakeModel([FakeLayer(FakeDenseMLP(4))])

    trainable = freeze_for_dense_reconstruction(student)

    assert trainable == 1
    assert student.model.layers[0].mlp.routed_dense.weight.requires_grad is True
    assert student.model.layers[0].mlp.shared_experts.weight.requires_grad is False
    assert student.norm.weight.requires_grad is False
    assert student.lm_head.weight.requires_grad is False
    assert student.attn.weight.requires_grad is False


def test_freeze_for_dense_reconstruction_can_train_norms_and_lm_head():
    student = FakeModel([FakeLayer(FakeDenseMLP(4))])

    trainable = freeze_for_dense_reconstruction(
        student,
        train_norms=True,
        train_lm_head=True,
    )

    assert trainable == 4
    assert student.model.layers[0].mlp.routed_dense.weight.requires_grad is True
    assert student.model.layers[0].mlp.shared_experts.weight.requires_grad is False
    assert student.norm.weight.requires_grad is True
    assert student.norm.bias.requires_grad is True
    assert student.lm_head.weight.requires_grad is True
    assert student.attn.weight.requires_grad is False


def test_parallel_reconstruction_uses_teacher_inputs_and_masks_padding():
    torch.manual_seed(0)
    teacher_mlp = FakeMLP(4)
    student_mlp = FakeDenseMLP(4)
    teacher = FakeModel([FakeLayer(teacher_mlp)])
    student = FakeModel([FakeLayer(student_mlp)])
    freeze_for_dense_reconstruction(student)

    batch = {
        "input_ids": torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]]),
    }

    result = compute_parallel_reconstruction_loss(
        teacher,
        student,
        batch,
        layer_ids=[0],
        cosine_weight=0.0,
    )

    result.loss.backward()

    assert result.loss.ndim == 0
    assert result.per_layer[0]["token_count"] == 6
    assert student_mlp.routed_dense.weight.grad is not None
    assert student_mlp.shared_experts.weight.grad is None
    assert teacher_mlp.proj.weight.grad is None


def test_reconstruction_can_include_logit_kl():
    teacher = FakeModel([FakeLayer(FakeMLP(4))])
    student = FakeModel([FakeLayer(FakeDenseMLP(4))])
    freeze_for_dense_reconstruction(student)
    batch = {
        "input_ids": torch.ones((1, 4), dtype=torch.long),
        "attention_mask": torch.ones((1, 4), dtype=torch.long),
    }

    result = compute_parallel_reconstruction_loss(
        teacher,
        student,
        batch,
        layer_ids=[0],
        cosine_weight=0.0,
        logit_kl_weight=0.1,
    )

    result.loss.backward()

    assert result.loss.requires_grad
    assert "logit_kl" in result.per_layer[-1]
    assert student.model.layers[0].mlp.routed_dense.weight.grad is not None


def test_reconstruction_can_train_logit_kl_without_structural_loss():
    teacher = FakeModel([FakeLayer(FakeMLP(4))])
    student = FakeModel([FakeLayer(FakeDenseMLP(4))])
    freeze_for_dense_reconstruction(student)
    batch = {
        "input_ids": torch.ones((1, 4), dtype=torch.long),
        "attention_mask": torch.ones((1, 4), dtype=torch.long),
        "kl_attention_mask": torch.tensor([[0, 1, 1, 0]]),
    }

    result = compute_parallel_reconstruction_loss(
        teacher,
        student,
        batch,
        layer_ids=[0],
        cosine_weight=0.0,
        logit_kl_weight=1.0,
        structural_weight=0.0,
    )

    result.loss.backward()

    assert set(result.per_layer) == {-1}
    assert result.per_layer[-1]["token_count"] == 2
    assert student.model.layers[0].mlp.routed_dense.weight.grad is not None
