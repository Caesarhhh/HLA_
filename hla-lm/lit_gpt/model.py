# Modified by Songlin Yang & Ali Hatamizadeh
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# Copyright Lightning AI. Licensed under the Apache License 2.0,
# see LICENSE file at https://github.com/Lightning-AI/litgpt/blob/main/LICENSE
import math
from typing import Any, List, Optional, Tuple
import torch
import torch.nn as nn

try:
    from lightning_utilities.core.imports import RequirementCache
except ImportError:

    class RequirementCache:

        def __init__(self, requirement: str):
            self.requirement = requirement

        def __bool__(self) -> bool:
            return False


from .gated_delta_net import ReleaseGatedDeltaNet, ReleaseGLAGatedDeltaNet
from typing_extensions import Self
from lit_gpt.config import Config

try:
    from xformers.ops import SwiGLU
except ImportError:

    class SwiGLU(nn.Module):

        def __init__(
            self,
            in_features: int,
            hidden_features: int,
            bias: bool = False,
            _pack_weights: bool = False,
        ):
            super().__init__()
            self.w1 = nn.Linear(in_features, hidden_features, bias=bias)
            self.w2 = nn.Linear(in_features, hidden_features, bias=bias)
            self.w3 = nn.Linear(hidden_features, in_features, bias=bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.w3(F.silu(self.w1(x)) * self.w2(x))


try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = (None, None, None)
from einops import rearrange
import torch.nn.functional as F

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None
RoPECache = Tuple[torch.Tensor, torch.Tensor]
KVCache = Tuple[torch.Tensor, torch.Tensor]
FlashAttention2Available = RequirementCache("flash-attn>=2.0.0.post1")


class GPT(nn.Module):

    def __init__(self, config: Config) -> None:
        super().__init__()
        assert config.padded_vocab_size is not None
        self.config = config
        self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=False)
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
                h=nn.ModuleList((Block(config, i) for i in range(config.n_layer))),
                ln_f=config.norm_class(config.n_embd, eps=config.norm_eps),
            )
        )
        self.rope_cache: Optional[RoPECache] = None
        self.mask_cache: Optional[torch.Tensor] = None
        self.kv_caches: List[KVCache] = []
        self.max_len = self.config.block_size
        self.mamba_init = False or config.mamba_init
        if self.mamba_init:
            self.tie_weights()

    def _init_weights(self, module: nn.Module, n_layer) -> None:
        """Meant to be used with `gpt.apply(gpt._init_weights)`."""
        if isinstance(module, nn.Embedding):
            if self.mamba_init:
                torch.nn.init.normal_(module.weight, std=0.02)
            else:
                torch.nn.init.normal_(
                    module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / self.config.n_embd)
                )
        elif isinstance(module, nn.Linear):
            if self.mamba_init:
                if module.bias is not None:
                    if not getattr(module.bias, "_no_reinit", False):
                        nn.init.zeros_(module.bias)
            else:
                torch.nn.init.normal_(
                    module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / self.config.n_embd)
                )
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
        for name, p in module.named_parameters():
            if (
                name in ["out_proj.weight", "fc2.weight"]
                or (name == "proj.weight" and isinstance(module, LLaMAMLP))
                or (
                    name == "w3.weight"
                    and isinstance(module, SwiGLU)
                    or (
                        name == "proj.weight"
                        and isinstance(module, CausalSelfAttention)
                    )
                )
            ):
                if self.mamba_init:
                    n_residuals_per_layer = 2
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        p /= math.sqrt(n_residuals_per_layer * n_layer)
                else:
                    nn.init.normal_(
                        p, mean=0.0, std=1 / math.sqrt(self.config.n_embd) / n_layer
                    )

    def tie_weights(self):
        self.lm_head.weight = self.transformer.wte.weight

    def reset_cache(self) -> None:
        self.max_len = self.config.block_size
        self.kv_caches.clear()
        if self.mask_cache is not None and self.mask_cache.device.type == "xla":
            self.rope_cache = None
            self.mask_cache = None

    def forward(
        self,
        idx: torch.Tensor,
        max_seq_length: Optional[int] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T = idx.size()
        use_kv_cache = input_pos is not None
        block_size = self.config.block_size
        if max_seq_length is None:
            max_seq_length = block_size
        if use_kv_cache:
            assert (
                max_seq_length >= T
            ), f"Cannot forward sequence of length {T}, max seq length is only {max_seq_length}"
        mask = None
        rope = None
        x = self.transformer.wte(idx)
        if not use_kv_cache:
            for block in self.transformer.h:
                x, *_ = block(x, rope, max_seq_length)
        else:
            self.kv_caches = self.kv_caches or [None for _ in self.transformer.h]
            for i, block in enumerate(self.transformer.h):
                x, self.kv_caches[i] = block(
                    x, rope, max_seq_length, mask, input_pos, self.kv_caches[i]
                )
        x = self.transformer.ln_f(x)
        return self.lm_head(x)

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(Config.from_name(name, **kwargs))

    def build_rope_cache(self, idx: torch.Tensor, seq_len: int) -> RoPECache:
        return build_rope_cache(
            seq_len=seq_len,
            n_elem=int(self.config.rotary_percentage * self.config.head_size),
            dtype=torch.bfloat16,
            device=idx.device,
            condense_ratio=self.config.condense_ratio,
        )

    def build_mask_cache(self, idx: torch.Tensor) -> torch.Tensor:
        ones = torch.ones(
            (self.config.block_size, self.config.block_size),
            device=idx.device,
            dtype=torch.bool,
        )
        return torch.tril(ones).unsqueeze(0).unsqueeze(0)

    def build_kv_caches(
        self, idx: torch.Tensor, max_seq_length: int, rope_cache_length: int
    ) -> List[KVCache]:
        B = idx.size(0)
        heads = 1 if self.config.n_query_groups == 1 else self.config.n_query_groups
        if rope_cache_length is not None:
            k_cache_shape = (
                B,
                max_seq_length,
                heads,
                rope_cache_length
                + self.config.head_size
                - int(self.config.rotary_percentage * self.config.head_size),
            )
        else:
            k_cache_shape = (B, max_seq_length, heads, self.config.head_size)
        v_cache_shape = (B, max_seq_length, heads, self.config.head_size)
        device = idx.device
        return [
            (
                torch.zeros(k_cache_shape, device=device),
                torch.zeros(v_cache_shape, device=device),
            )
            for _ in range(self.config.n_layer)
        ]


class SimpleRMSNorm(nn.Module):

    def __init__(self, hidden_size: int, eps: float = 1e-06) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        y = x.float()
        y = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (y * self.weight.float()).to(dtype)


class ReleaseGatedMLP(nn.Module):

    def __init__(self, config: Config) -> None:
        super().__init__()
        intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(config.n_embd, 2 * intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class Block(nn.Module):

    def __init__(self, config: Config, layer_idx: int) -> None:
        super().__init__()
        self.norm_1 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.use_gated_deltanet = (
            layer_idx % config.gated_delta_per_layer == 0
            if config.gated_delta_per_layer > 0
            else False
        )
        variant = config.gated_delta_variant
        attn_kwargs = dict(
            hidden_size=config.n_embd,
            layer_idx=layer_idx,
            head_dim=config.gated_delta_head_dim,
            num_heads=config.gated_delta_num_heads,
            expand_v=config.gated_delta_expand_v,
            conv_size=config.gated_delta_conv_size,
            use_short_conv=config.gated_delta_use_short_conv,
            norm_eps=config.norm_eps,
            use_gate=config.gated_delta_use_gate,
            use_residual=config.gated_delta_use_residual,
        )
        if variant == "gdn":
            self.attn = ReleaseGatedDeltaNet(**attn_kwargs)
        elif variant == "gla_gdn":
            self.attn = ReleaseGLAGatedDeltaNet(**attn_kwargs)
        else:
            raise ValueError(f"Unsupported gated_delta_variant={variant!r}")
        self.norm_2 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.mlp = ReleaseGatedMLP(config)
        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPECache,
        max_seq_length: int,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        n_1 = self.norm_1(x)
        if input_pos is None:
            h, _, new_kv_cache = self.attn(n_1, attention_mask=mask)
        else:
            h, _, new_kv_cache = self.attn(
                n_1,
                attention_mask=None,
                past_key_values=kv_cache,
                use_cache=True,
                max_seq_length=max_seq_length,
            )
        x = x + h
        n_2 = self.norm_2(x)
        h = self.mlp(n_2)
        x = x + h
        return (x, new_kv_cache)


class CausalSelfAttention(nn.Module):

    def __init__(
        self, config: Config, layer_idx: int, n_embd: int, head_size=None
    ) -> None:
        super().__init__()
        self.local = layer_idx % config.full_per_layer < config.full_per_layer - 1
        if head_size is not None:
            self.head_size = head_size
            self.n_head = n_embd // head_size
            self.n_query_groups = self.n_head
        else:
            self.head_size = config.head_size
            self.n_head = config.n_head
            self.n_query_groups = config.n_query_groups
        shape = (self.n_head + 2 * self.n_query_groups) * self.head_size
        self.attn = nn.Linear(n_embd, shape, bias=config.bias)
        self.proj = nn.Linear(n_embd, n_embd, bias=config.bias)
        self.config = config
        self.sc = config.sc_attn
        if self.sc:
            self.q_dim = self.n_head * self.head_size
            self.kv_dim = self.n_query_groups * self.head_size
            d_conv = 4
            self.q_conv1d = nn.Conv1d(
                in_channels=self.q_dim,
                out_channels=self.q_dim,
                bias=False,
                kernel_size=d_conv,
                groups=self.q_dim,
                padding=d_conv - 1,
            )
            self.k_conv1d = nn.Conv1d(
                in_channels=self.kv_dim,
                out_channels=self.kv_dim,
                bias=False,
                kernel_size=d_conv,
                groups=self.kv_dim,
                padding=d_conv - 1,
            )
            self.v_conv1d = nn.Conv1d(
                in_channels=self.kv_dim,
                out_channels=self.kv_dim,
                bias=False,
                kernel_size=d_conv,
                groups=self.kv_dim,
                padding=d_conv - 1,
            )

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPECache,
        max_seq_length: int,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        B, T, C = x.size()
        qkv = self.attn(x)
        q_per_kv = self.n_head // self.n_query_groups
        total_qkv = q_per_kv + 2
        qkv = qkv.view(B, T, self.n_query_groups, total_qkv, self.head_size)
        q, k, v = qkv.split((q_per_kv, 1, 1), dim=-2)
        q = q.reshape(B, T, -1)
        k = k.reshape(B, T, -1)
        v = v.reshape(B, T, -1)
        if self.sc:
            q = causal_conv1d_fn(
                x=q.transpose(-1, -2),
                weight=rearrange(self.q_conv1d.weight, "d 1 w -> d w"),
                bias=self.q_conv1d.bias,
                activation="silu",
            ).transpose(-1, -2)
            k = causal_conv1d_fn(
                x=k.transpose(-1, -2),
                weight=rearrange(self.k_conv1d.weight, "d 1 w -> d w"),
                bias=self.k_conv1d.bias,
                activation="silu",
            ).transpose(-1, -2)
            v = causal_conv1d_fn(
                x=v.transpose(-1, -2),
                weight=rearrange(self.v_conv1d.weight, "d 1 w -> d w"),
                bias=self.v_conv1d.bias,
                activation="silu",
            ).transpose(-1, -2)
        q = q.reshape(B, T, -1, self.head_size)
        k = k.reshape(B, T, -1, self.head_size)
        v = v.reshape(B, T, -1, self.head_size)
        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            cache_k, cache_v = (cache_k.to(dtype=k.dtype), cache_v.to(dtype=v.dtype))
            if input_pos[-1] >= max_seq_length:
                input_pos = torch.tensor(max_seq_length - 1, device=input_pos.device)
                cache_k = torch.roll(cache_k, -1, dims=1)
                cache_v = torch.roll(cache_v, -1, dims=1)
            k = cache_k.index_copy_(1, input_pos, k)
            v = cache_v.index_copy_(1, input_pos, v)
            kv_cache = (k, v)
        y = self.scaled_dot_product_attention(q, k, v, mask=mask)
        y = y.reshape(B, T, -1)
        y = self.proj(y)
        return (y, kv_cache)

    def scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        scale = 1.0 / math.sqrt(self.head_size)
        if (
            FlashAttention2Available
            and mask is None
            and (q.device.type == "cuda")
            and (q.dtype in (torch.float16, torch.bfloat16))
        ):
            from flash_attn import flash_attn_func

            if self.local and self.config.local_window > -1:
                win_tuple = (self.config.local_window - 1, 0)
            else:
                win_tuple = (-1, -1)
            return flash_attn_func(
                q,
                k,
                v,
                dropout_p=0.0,
                softmax_scale=scale,
                causal=True,
                window_size=win_tuple,
            )
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if q.size() != k.size():
            k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
            v = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, scale=scale, is_causal=mask is None
        )
        return y.transpose(1, 2)


class LLaMAMLP(nn.Module):

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.swiglu = SwiGLU(
            config.n_embd,
            config.intermediate_size,
            bias=config.bias,
            _pack_weights=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.swiglu(x)
        return x


def build_rope_cache(
    seq_len: int,
    n_elem: int,
    dtype: torch.dtype,
    device: torch.device,
    base: int = 10000,
    condense_ratio: int = 1,
) -> RoPECache:
    """Enhanced Transformer with Rotary Position Embedding.

    Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
    transformers/rope/__init__.py. MIT License:
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
    """
    theta = 1.0 / base ** (torch.arange(0, n_elem, 2, device=device) / n_elem)
    seq_idx = torch.arange(seq_len, device=device) / condense_ratio
    idx_theta = torch.outer(seq_idx, theta)
    cos, sin = (torch.cos(idx_theta), torch.sin(idx_theta))
    if dtype == torch.bfloat16:
        return (cos.bfloat16(), sin.bfloat16())
    if dtype in (torch.float16, torch.bfloat16, torch.int8):
        return (cos.half(), sin.half())
    return (cos, sin)
