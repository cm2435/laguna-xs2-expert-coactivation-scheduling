from __future__ import annotations

from typing import Any

from transformers import PretrainedConfig


class LagunaDenseConfig(PretrainedConfig):
    model_type = "laguna_dense"

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        intermediate_size: int = 14336,
        num_hidden_layers: int = 40,
        num_attention_heads: int = 32,
        num_key_value_heads: int | None = None,
        hidden_act: str = "silu",
        k_routed: int = 8,
        expert_intermediate_size: int = 512,
        dense_routed_intermediate_size: int | None = None,
        source_model: str = "poolside/Laguna-XS.2",
        shared_expert: str = "kept",
        placeholder_init: str = "copied_shell_random_routed",
        initializer_range: float = 0.02,
        pad_token_id: int | None = None,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        tie_word_embeddings: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.hidden_act = hidden_act
        self.k_routed = k_routed
        self.expert_intermediate_size = expert_intermediate_size
        self.dense_routed_intermediate_size = (
            dense_routed_intermediate_size or k_routed * expert_intermediate_size
        )
        self.source_model = source_model
        self.shared_expert = shared_expert
        self.placeholder_init = placeholder_init
        self.initializer_range = initializer_range
        self.architectures = ["LagunaDenseForCausalLM"]
        self.auto_map = {
            "AutoConfig": "configuration_laguna_dense.LagunaDenseConfig",
            "AutoModelForCausalLM": "modeling_laguna_dense.LagunaDenseForCausalLM",
        }
        self.moe_dense_conversion = {
            "source_model": source_model,
            "kind": "routed_moe_to_dense_swiglu",
            "k_routed": k_routed,
            "expert_intermediate_size": expert_intermediate_size,
            "dense_routed_intermediate_size": self.dense_routed_intermediate_size,
            "shared_expert": shared_expert,
            "placeholder_weights": placeholder_init,
        }
