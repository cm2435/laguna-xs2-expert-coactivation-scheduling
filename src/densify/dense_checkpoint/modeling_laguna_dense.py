from __future__ import annotations

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from densify.dense_checkpoint.config import LagunaDenseConfig


class LagunaDenseRoutedMLP(nn.Module):
    def __init__(self, config: LagunaDenseConfig) -> None:
        super().__init__()
        width = config.dense_routed_intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, width, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, width, bias=False)
        self.down_proj = nn.Linear(width, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class LagunaDenseMoEReplacement(nn.Module):
    def __init__(self, config: LagunaDenseConfig) -> None:
        super().__init__()
        self.routed_dense = LagunaDenseRoutedMLP(config)
        self.shared_expert = LagunaDenseRoutedMLP(config) if config.shared_expert == "kept" else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        routed = self.routed_dense(x)
        if self.shared_expert is None:
            return routed
        return routed + self.shared_expert(x)


class LagunaDenseBlock(nn.Module):
    def __init__(self, config: LagunaDenseConfig) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(config.hidden_size)
        self.self_attn = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size)
        self.mlp = LagunaDenseMoEReplacement(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class LagunaDensePreTrainedModel(PreTrainedModel):
    config_class = LagunaDenseConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class LagunaDenseModel(LagunaDensePreTrainedModel):
    def __init__(self, config: LagunaDenseConfig) -> None:
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([LagunaDenseBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(config.hidden_size)
        self.post_init()

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.norm(hidden_states)


class LagunaDenseForCausalLM(LagunaDensePreTrainedModel, GenerationMixin):
    def __init__(self, config: LagunaDenseConfig) -> None:
        super().__init__(config)
        self.model = LagunaDenseModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(
        self, input_ids: torch.LongTensor, **kwargs: object
    ) -> dict[str, torch.LongTensor]:
        return {"input_ids": input_ids}

    def forward(
        self,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor | None = None,
        **_: object,
    ) -> CausalLMOutputWithPast:
        hidden_states = self.model(input_ids)
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
        return CausalLMOutputWithPast(loss=loss, logits=logits)
