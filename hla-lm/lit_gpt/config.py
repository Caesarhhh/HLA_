# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# Copyright Lightning AI. Licensed under the Apache License 2.0,
# see LICENSE file at https://github.com/Lightning-AI/litgpt/blob/main/LICENSE
from dataclasses import dataclass
from typing import Any, Literal, Optional, Type
import torch
from typing_extensions import Self
import lit_gpt.model

try:
    from lit_gpt.utils import find_multiple
except ImportError:

    def find_multiple(n: int, k: int) -> int:
        if n % k == 0:
            return n
        return n + k - n % k


@dataclass
class Config:
    org: str = "Lightning-AI"
    name: str = "lit-GPT"
    block_size: int = 4096
    vocab_size: int = 50254
    padding_multiple: int = 512
    padded_vocab_size: Optional[int] = None
    n_layer: int = 16
    n_head: int = 32
    n_embd: int = 4096
    rotary_percentage: float = 0.25
    parallel_residual: bool = True
    bias: bool = True
    local_window: int = -1
    mlp: bool = True
    full_per_layer: int = 1000000
    mb_per_layer: int = -1
    ret_per_layer: int = -1
    gla_per_layer: int = -1
    nope: bool = False
    mamba: bool = False
    sc_attn: bool = False
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    mamba_init: bool = False
    attn_layer_pos: str = None
    gated_delta_per_layer: int = -1
    gated_delta_variant: str = "gdn"
    gated_delta_release: bool = False
    gated_delta_head_dim: int = 128
    gated_delta_num_heads: int = 9
    gated_delta_expand_v: float = 1.5
    gated_delta_conv_size: int = 4
    gated_delta_use_short_conv: bool = True
    gated_delta_use_gate: bool = True
    gated_delta_use_residual: bool = True
    mix_chunk_size: int = 256
    mix_pool_size: int = 16
    mix_apply_gdn_decay: bool = False
    gla_router_route_source: str = "raw_qk"
    gla_router_use_logmean: bool = True
    gla_router_use_rope: bool = False
    gla_router_pool_self_attention: bool = True
    gla_router_gate: str = "affine_sigmoid"
    gla_router_sigmoid_bias: float = 2.2
    n_query_groups: Optional[int] = None
    shared_attention_norm: bool = False
    _norm_class: Literal["LayerNorm", "RMSNorm"] = "LayerNorm"
    norm_eps: float = 1e-05
    _mlp_class: Literal["GptNeoxMLP", "LLaMAMLP"] = "GptNeoxMLP"
    intermediate_size: Optional[int] = None
    condense_ratio: int = 1

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0
        if self.padded_vocab_size is None:
            self.padded_vocab_size = find_multiple(
                self.vocab_size, self.padding_multiple
            )
        if self.n_query_groups is not None:
            assert self.n_head % self.n_query_groups == 0
        else:
            self.n_query_groups = self.n_head
        if self.intermediate_size is None:
            if self._mlp_class == "LLaMAMLP":
                raise ValueError("The config needs to set the `intermediate_size`")
            self.intermediate_size = 4 * self.n_embd

    @property
    def head_size(self) -> int:
        return self.n_embd // self.n_head

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        conf_dict = name_to_config[name].copy()
        conf_dict.update(kwargs)
        return cls(**conf_dict)

    @property
    def mlp_class(self) -> Type:
        return getattr(lit_gpt.model, self._mlp_class)

    @property
    def norm_class(self) -> Type:
        if self._norm_class == "RMSNorm":
            from lit_gpt.rmsnorm import RMSNorm

            return RMSNorm
        elif self._norm_class == "FusedRMSNorm":
            from lit_gpt.rmsnorm import FusedRMSNorm

            return FusedRMSNorm
        elif hasattr(lit_gpt.model, self._norm_class):
            return getattr(lit_gpt.model, self._norm_class)
        return getattr(torch.nn, self._norm_class)


configs = []
GatedDeltaNet = [
    dict(
        org="linear-moe-hub",
        name="GatedDeltaNet_Release_1.3B",
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        gated_delta_per_layer=1,
        gated_delta_variant="gdn",
        gated_delta_release=True,
        gated_delta_head_dim=256,
        gated_delta_num_heads=4,
        gated_delta_expand_v=1.0,
        gated_delta_conv_size=4,
        gated_delta_use_short_conv=False,
        gated_delta_use_gate=True,
        gated_delta_use_residual=False,
        n_layer=24,
        n_head=4,
        n_embd=2048,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        nope=True,
        _norm_class="SimpleRMSNorm",
        norm_eps=1e-06,
        _mlp_class="LLaMAMLP",
        intermediate_size=5632,
        local_window=2048,
        mamba_init=True,
    ),
    dict(
        org="linear-moe-hub",
        name="GatedDeltaNet_GLA_GDN_Release_1.3B",
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        gated_delta_per_layer=1,
        gated_delta_variant="gla_gdn",
        gated_delta_release=True,
        gated_delta_head_dim=256,
        gated_delta_num_heads=4,
        gated_delta_expand_v=1.0,
        gated_delta_conv_size=4,
        gated_delta_use_short_conv=False,
        gated_delta_use_gate=True,
        gated_delta_use_residual=False,
        mix_chunk_size=256,
        mix_pool_size=16,
        n_layer=24,
        n_head=4,
        n_embd=2048,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        nope=True,
        _norm_class="SimpleRMSNorm",
        norm_eps=1e-06,
        _mlp_class="LLaMAMLP",
        intermediate_size=5632,
        local_window=2048,
        mamba_init=True,
    ),
]
configs.extend(GatedDeltaNet)
name_to_config = {config["name"]: config for config in configs}
