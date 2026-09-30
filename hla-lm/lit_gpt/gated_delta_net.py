# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
from __future__ import annotations
import os
from typing import TYPE_CHECKING, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange

try:
    from fla.modules import FusedRMSNormSwishGate, RMSNorm, ShortConvolution
    from fla.modules.activations import ACT2FN
except ImportError:
    ACT2FN = {"swish": F.silu, "silu": F.silu, "sigmoid": torch.sigmoid}

    class RMSNorm(nn.Module):

        def __init__(
            self, hidden_size: int, elementwise_affine: bool = True, eps: float = 1e-05
        ):
            super().__init__()
            self.weight = (
                nn.Parameter(torch.ones(hidden_size)) if elementwise_affine else None
            )
            self.eps = eps

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            dtype = x.dtype
            y = x.float()
            y = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + self.eps)
            if self.weight is not None:
                y = y * self.weight.float()
            return y.to(dtype)

    class FusedRMSNormSwishGate(nn.Module):

        def __init__(
            self, hidden_size: int, elementwise_affine: bool = True, eps: float = 1e-05
        ):
            super().__init__()
            self.weight = (
                nn.Parameter(torch.ones(hidden_size)) if elementwise_affine else None
            )
            self.eps = eps

        def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
            dtype = x.dtype
            y = x.float()
            y = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + self.eps)
            if self.weight is not None:
                y = y * self.weight.float()
            return (y * F.silu(gate.float())).to(dtype)

    class ShortConvolution(nn.Conv1d):

        def __init__(
            self,
            hidden_size: int,
            kernel_size: int = 4,
            bias: bool = False,
            activation: Optional[str] = "silu",
        ):
            super().__init__(
                in_channels=hidden_size,
                out_channels=hidden_size,
                kernel_size=kernel_size,
                groups=hidden_size,
                bias=bias,
                padding=kernel_size - 1,
            )
            self.activation = activation
            self.state_size = hidden_size * kernel_size

        def forward(
            self,
            x: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            cache: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            seq_len = x.shape[1]
            y = F.conv1d(
                x.transpose(1, 2),
                self.weight,
                self.bias,
                padding=self.padding[0],
                groups=self.groups,
            )[:, :, :seq_len].transpose(1, 2)
            if self.activation in ("silu", "swish"):
                y = F.silu(y)
            if attention_mask is not None and attention_mask.dim() == 2:
                y = y * attention_mask.unsqueeze(-1)
            return y


try:
    from fla.ops.simple_gla import chunk_simple_gla
except ImportError:
    chunk_simple_gla = None
from .gated_delta_rule_ops import chunk_gated_delta_rule

try:
    from .triton_pool_prefix_attention import (
        can_use_fused_pool_prefix_attention,
        fused_pool_prefix_attention,
    )
except Exception:
    can_use_fused_pool_prefix_attention = None
    fused_pool_prefix_attention = None
try:
    from .triton_affine_prefix_readout import (
        can_use_fused_affine_prefix_readout,
        fused_affine_prefix_readout,
    )
except Exception:
    can_use_fused_affine_prefix_readout = None
    fused_affine_prefix_readout = None
try:
    from .triton_affine_chunk_summary import (
        can_use_fused_affine_chunk_a,
        fused_affine_chunk_a,
    )
except Exception:
    can_use_fused_affine_chunk_a = None
    fused_affine_chunk_a = None
try:
    from .triton_affine_decode_readout import (
        can_use_fused_affine_decode_readout,
        fused_affine_decode_readout,
    )
except Exception:
    can_use_fused_affine_decode_readout = None
    fused_affine_decode_readout = None
try:
    from .triton_affine_decode_update import (
        can_use_fused_affine_decode_update,
        can_use_fused_affine_decode_joint_update,
        fused_affine_decode_update_,
        fused_affine_decode_joint_update_,
        fused_affine_decode_joint_update_route_,
    )
except Exception:
    can_use_fused_affine_decode_update = None
    can_use_fused_affine_decode_joint_update = None
    fused_affine_decode_update_ = None
    fused_affine_decode_joint_update_ = None
    fused_affine_decode_joint_update_route_ = None
try:
    from .triton_affine_history_mix import (
        can_use_fused_affine_history_mix,
        can_use_fused_affine_history_route_mix_wide,
        fused_affine_history_mix,
        fused_affine_history_route_mix_wide,
    )
except Exception:
    can_use_fused_affine_history_mix = None
    can_use_fused_affine_history_route_mix_wide = None
    fused_affine_history_mix = None
    fused_affine_history_route_mix_wide = None
try:
    from .affine_history_compiled import compiled_affine_history_mix
except Exception:
    compiled_affine_history_mix = None
try:
    from .triton_gdn_affine_decode_mix import (
        can_use_fused_gdn_affine_decode_mix,
        fused_gdn_affine_decode_mix,
    )
except Exception:
    can_use_fused_gdn_affine_decode_mix = None
    fused_gdn_affine_decode_mix = None
try:
    from .triton_headwise_decode_linear import (
        can_use_fused_headwise_decode_linear,
        fused_headwise_decode_linear,
    )
except Exception:
    can_use_fused_headwise_decode_linear = None
    fused_headwise_decode_linear = None
try:
    from .triton_hla_router import (
        fused_hla_router_sigmoid,
        fused_hla_router_sigmoid_projected,
    )
except Exception:
    fused_hla_router_sigmoid = None
    fused_hla_router_sigmoid_projected = None
import math

if TYPE_CHECKING:
    from fla.models.utils import Cache
try:
    from fla.modules.l2norm import l2_norm as l2_norm_fn
except Exception:
    try:
        from fla.modules.l2norm import l2_norm_fn
    except Exception:

        def l2_norm_fn(x: torch.Tensor) -> torch.Tensor:
            return F.normalize(x, p=2, dim=-1)


class ReleaseShortConvolution(nn.Conv1d):

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int = 4,
        bias: bool = False,
        activation: str = "silu",
    ):
        super().__init__(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,
            bias=bias,
            padding=kernel_size - 1,
        )
        self.activation = activation

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        seq_len = x.shape[1]
        y = F.conv1d(
            x.transpose(1, 2),
            self.weight,
            self.bias,
            padding=self.padding[0],
            groups=self.groups,
        )[:, :, :seq_len].transpose(1, 2)
        if self.activation in ("silu", "swish"):
            y = F.silu(y)
        if attention_mask is not None and attention_mask.dim() == 2:
            y = y * attention_mask.unsqueeze(-1)
        return y


class SimpleRMSNormGated(nn.Module):

    def __init__(self, hidden_size: int, eps: float = 1e-06, activation: str = "swish"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self.activation = activation

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        y = x.float()
        y = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        y = y * self.weight.float()
        if self.activation in ("silu", "swish"):
            gate = F.silu(gate.float())
        elif self.activation == "sigmoid":
            gate = gate.float().sigmoid()
        else:
            raise ValueError(f"Unsupported activation={self.activation!r}")
        return (y * gate).to(dtype)


class HeadwiseLinear(nn.Module):

    def __init__(
        self, num_heads: int, in_features: int, out_features: int, bias: bool = False
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(num_heads, out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(num_heads, out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        use_fused_decode = (
            self.bias is None
            and (not torch.is_grad_enabled())
            and (fused_headwise_decode_linear is not None)
            and (can_use_fused_headwise_decode_linear is not None)
            and can_use_fused_headwise_decode_linear(x, self.weight)
        )
        if use_fused_decode:
            y = fused_headwise_decode_linear(
                x,
                self.weight,
                num_warps=16 if max(self.in_features, self.out_features) >= 256 else 8,
            )
        else:
            y = torch.einsum("b h l i, h o i -> b h l o", x, self.weight)
        if self.bias is not None:
            y = y + self.bias.view(1, self.num_heads, 1, self.out_features)
        return y


class HLAAffineMixin:

    def _pad_tokens(self, x: torch.Tensor, pad_len: int) -> torch.Tensor:
        if pad_len == 0:
            return x
        return torch.cat([x, x.new_zeros(*x.shape[:2], pad_len, *x.shape[3:])], dim=2)

    def _pad_gate(self, x: torch.Tensor, pad_len: int) -> torch.Tensor:
        if pad_len == 0:
            return x
        return torch.cat([x, x.new_zeros(*x.shape[:2], pad_len)], dim=2)

    def _route_qk(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_route = self.route_q_proj(q.to(self.route_q_proj.weight.dtype)).to(q.dtype)
        if getattr(self, "route_k_proj", None) is None:
            k_route = k
        else:
            k_route = self.route_k_proj(k.to(self.route_k_proj.weight.dtype)).to(
                k.dtype
            )
        return (q_route, k_route)

    def _apply_route_pool_linear(
        self, module: HeadwiseLinear, x: torch.Tensor
    ) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(shape[0], shape[1], -1, shape[-1])
        out = module(flat.to(module.weight.dtype)).to(x.dtype)
        return out.reshape(*shape[:-1], out.shape[-1])

    def _route_pool_prefix_representatives(
        self, k_bucket: torch.Tensor
    ) -> torch.Tensor:
        fused = self._route_pool_self_attention_prefix_representatives_fused(k_bucket)
        if fused is not None:
            return fused
        prefix_reps = []
        for prefix_len in range(1, k_bucket.shape[4] + 1):
            prefix_reps.append(
                self._route_pool_self_attention_representative(
                    k_bucket[..., :prefix_len, :]
                )
            )
        return torch.stack(prefix_reps, dim=4)

    def _route_pool_self_attention_prefix_representatives_fused(
        self, k_bucket: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if (
            fused_pool_prefix_attention is None
            or can_use_fused_pool_prefix_attention is None
        ):
            return None
        pool_key = self._apply_route_pool_linear(self.route_pool_attn_k_proj, k_bucket)
        pool_value = self._apply_route_pool_linear(
            self.route_pool_attn_v_proj, k_bucket
        )
        flat_key = pool_key.reshape(-1, pool_key.shape[-2], pool_key.shape[-1])
        flat_value = pool_value.reshape(-1, pool_value.shape[-2], pool_value.shape[-1])
        if not can_use_fused_pool_prefix_attention(flat_key, flat_value):
            return None
        prefix = fused_pool_prefix_attention(flat_key, flat_value)
        prefix = prefix.reshape(*pool_value.shape)
        return self._apply_route_pool_linear(
            self.route_pool_attn_o_proj, prefix
        ).float()

    def _route_pool_self_attention_representative(
        self, k_bucket: torch.Tensor
    ) -> torch.Tensor:
        pool_key = self._apply_route_pool_linear(self.route_pool_attn_k_proj, k_bucket)
        scale = pool_key.shape[-1] ** (-0.5)
        attn_scores = (
            torch.einsum(
                "b h c p i d, b h c p j d -> b h c p i j",
                pool_key.float(),
                pool_key.float(),
            )
            * scale
        )
        attn = torch.softmax(attn_scores, dim=-1).to(k_bucket.dtype)
        pool_value = self._apply_route_pool_linear(
            self.route_pool_attn_v_proj, k_bucket
        )
        attended = torch.einsum(
            "b h c p i j, b h c p j d -> b h c p i d", attn, pool_value
        )
        attended = self._apply_route_pool_linear(self.route_pool_attn_o_proj, attended)
        return attended.float().mean(dim=4)

    def _route_pool_representatives(self, k_bucket: torch.Tensor) -> torch.Tensor:
        return self._route_pool_self_attention_representative(k_bucket)

    def _current_chunk_scores(
        self,
        q_route: torch.Tensor,
        k_padded: torch.Tensor,
        pooled: torch.Tensor,
        num_tokens: int,
        num_chunks: int,
    ) -> torch.Tensor:
        chunk_size = 256
        pool_size = 16
        buckets = chunk_size // pool_size
        pad_len = num_chunks * chunk_size - num_tokens
        q_padded = self._pad_tokens(q_route, pad_len)
        q_chunk = rearrange(
            q_padded, "b h (c l) d -> b h c l d", c=num_chunks, l=chunk_size
        )
        k_chunk = rearrange(
            k_padded, "b h (c l) d -> b h c l d", c=num_chunks, l=chunk_size
        )
        k_bucket = rearrange(
            k_chunk, "b h c (p s) d -> b h c p s d", p=buckets, s=pool_size
        )
        prefix_mean = self._route_pool_prefix_representatives(k_bucket)
        prefix_mean = rearrange(prefix_mean, "b h c p s d -> b h c (p s) d")
        pooled_for_scores = pooled
        prefix_mean = prefix_mean
        scale = q_route.shape[-1] ** (-0.5) * 1.0
        bucket_scores = (
            torch.einsum(
                "b h c l d, b h c p d -> b h c l p",
                q_chunk.float(),
                pooled_for_scores.float(),
            )
            * scale
        )
        partial_scores = (q_chunk.float() * prefix_mean).sum(dim=-1) * scale
        token_offset = torch.arange(chunk_size, device=q_route.device)
        bucket_idx = torch.div(token_offset, pool_size, rounding_mode="floor")
        bucket_scores = bucket_scores.scatter(
            4,
            bucket_idx.view(1, 1, 1, chunk_size, 1).expand(
                bucket_scores.shape[0],
                bucket_scores.shape[1],
                num_chunks,
                chunk_size,
                1,
            ),
            partial_scores.unsqueeze(-1).to(bucket_scores.dtype),
        )
        visible_bucket = torch.arange(buckets, device=q_route.device).view(
            1, buckets
        ) <= bucket_idx.view(chunk_size, 1)
        bucket_scores = bucket_scores.masked_fill(
            ~visible_bucket.view(1, 1, 1, chunk_size, buckets), -1e309
        )
        visible_count = (bucket_idx + 1).to(bucket_scores.dtype)
        current_scores = torch.logsumexp(bucket_scores, dim=-1)
        current_scores = current_scores - visible_count.log().view(1, 1, 1, chunk_size)
        return rearrange(current_scores, "b h c l -> b h (c l)")[:, :, :num_tokens]

    def _use_affine_composition(self) -> bool:
        gate = "affine_sigmoid"
        return gate == "affine_sigmoid"

    def _local_chunks(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        gk: torch.Tensor,
        num_chunks: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk_size = 256
        qf = rearrange(q, "b h (c l) d -> (b c) h l d", c=num_chunks, l=chunk_size)
        kf = rearrange(k, "b h (c l) d -> (b c) h l d", c=num_chunks, l=chunk_size)
        vf = rearrange(v, "b h (c l) d -> (b c) h l d", c=num_chunks, l=chunk_size)
        bf = rearrange(beta, "b h (c l) -> (b c) h l", c=num_chunks, l=chunk_size)
        gf = rearrange(gk, "b h (c l) -> (b c) h l", c=num_chunks, l=chunk_size)
        out, final_state = chunk_gated_delta_rule(
            qf, kf, vf, bf, gf, output_final_state=True
        )
        out = rearrange(out, "(b c) h l d -> b h c l d", c=num_chunks)
        final_state = rearrange(final_state, "(b c) h k v -> b h c k v", c=num_chunks)
        return (out, final_state)

    def _mix_weights(
        self, q: torch.Tensor, k: torch.Tensor, num_tokens: int, num_chunks: int
    ) -> torch.Tensor:
        chunk_size = 256
        pool_size = 16
        buckets = chunk_size // pool_size
        fuse_raw_q_projection = False
        if fuse_raw_q_projection:
            k_input = k[:, :, :num_tokens]
            if getattr(self, "route_k_proj", None) is None:
                k_route = k_input
            else:
                k_route = self.route_k_proj(
                    k_input.to(self.route_k_proj.weight.dtype)
                ).to(k_input.dtype)
            pad_len = num_chunks * chunk_size - num_tokens
            k_padded = self._pad_tokens(k_route, pad_len)
            k_chunk = rearrange(
                k_padded, "b h (c l) d -> b h c l d", c=num_chunks, l=chunk_size
            )
            pooled = rearrange(
                k_chunk, "b h c (p s) d -> b h c p s d", p=buckets, s=pool_size
            ).mean(dim=4)
            return fused_hla_router_sigmoid(
                rearrange(q[:, :, :num_tokens], "b h l d -> b l h d"),
                self.route_q_proj.weight,
                pooled,
                chunk_size=chunk_size,
                router_logit_scale=1.0,
                sigmoid_bias=2.2,
                sigmoid_temperature=1.0,
                use_logmean=True,
                current_always_on=True,
            )
        q_route, k_route = self._route_qk(q[:, :, :num_tokens], k[:, :, :num_tokens])
        q_route = q_route
        pad_len = num_chunks * chunk_size - num_tokens
        k_padded = self._pad_tokens(k_route, pad_len)
        k_chunk = rearrange(
            k_padded, "b h (c l) d -> b h c l d", c=num_chunks, l=chunk_size
        )
        k_bucket = rearrange(
            k_chunk, "b h c (p s) d -> b h c p s d", p=buckets, s=pool_size
        )
        pooled = self._route_pool_representatives(k_bucket)
        pooled_for_scores = pooled
        use_fused_inference_router = (
            not self.training
            and (not torch.is_grad_enabled())
            and (not os.environ.get("GDN_DISABLE_TRITON_HLA_ROUTER", ""))
            and (fused_hla_router_sigmoid_projected is not None)
            and (num_chunks <= 32)
            and (buckets <= 32)
            and (num_chunks * buckets <= 512)
        )
        if use_fused_inference_router:
            return fused_hla_router_sigmoid_projected(
                rearrange(q_route, "b h l d -> b l h d"),
                pooled_for_scores,
                chunk_size=chunk_size,
                router_logit_scale=1.0,
                sigmoid_bias=2.2,
                sigmoid_temperature=1.0,
                use_logmean=True,
                current_always_on=True,
            )
        bucket_scores = torch.einsum(
            "b h l d, b h c p d -> b h l c p",
            q_route.float(),
            pooled_for_scores.float(),
        )
        scale = q_route.shape[-1] ** (-0.5) * 1.0
        scores = torch.logsumexp(bucket_scores * scale, dim=-1)
        scores = scores - math.log(buckets)
        token_chunk_idx = torch.div(
            torch.arange(num_tokens, device=q.device), chunk_size, rounding_mode="floor"
        )
        chunk_idx = torch.arange(num_chunks, device=q.device)
        causal_chunk_mask = chunk_idx.view(1, num_chunks) <= token_chunk_idx.view(
            num_tokens, 1
        )
        scores = scores.masked_fill(
            ~causal_chunk_mask.view(1, 1, num_tokens, num_chunks), -1e309
        )
        current_idx = token_chunk_idx.view(1, 1, num_tokens, 1).expand(
            scores.shape[0], scores.shape[1], num_tokens, 1
        )
        skip_current_scores = (
            os.environ.get("GDN_SKIP_SELFATTN_POOL_CURRENT_SCORES", "0") == "1"
        )
        if not skip_current_scores:
            current_scores = self._current_chunk_scores(
                q_route, k_padded, pooled, num_tokens, num_chunks
            )
            current_bias = 0.0
            if current_bias:
                current_scores = current_scores + current_bias
            scores = scores.scatter(
                3, current_idx, current_scores.unsqueeze(-1).to(scores.dtype)
            )
        visible_mask = causal_chunk_mask.view(1, 1, num_tokens, num_chunks)
        sigmoid_scores = scores + 2.2
        sigmoid_temperature = 1.0
        weights = torch.sigmoid(sigmoid_scores / max(sigmoid_temperature, 0.0001))
        weights = weights.masked_fill(~visible_mask, 0)
        weights = weights.scatter(3, current_idx, 1.0)
        weights = weights
        return weights.to(q.dtype)

    def _chunk_affine_summaries(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        gk: torch.Tensor,
        num_chunks: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk_size = 256
        k_chunk = rearrange(k, "b h (c l) d -> b h c l d", c=num_chunks, l=chunk_size)
        v_chunk = rearrange(v, "b h (c l) d -> b h c l d", c=num_chunks, l=chunk_size)
        beta_chunk = rearrange(beta, "b h (c l) -> b h c l", c=num_chunks, l=chunk_size)
        gk_chunk = rearrange(gk, "b h (c l) -> b h c l", c=num_chunks, l=chunk_size)
        batch_size, num_heads, _, _, d_k = k_chunk.shape
        d_v = v_chunk.shape[-1]
        dtype = torch.float32
        eye = torch.eye(d_k, device=k.device, dtype=dtype).view(1, 1, 1, d_k, d_k)
        a = eye.expand(batch_size, num_heads, num_chunks, d_k, d_k).clone()
        use_fused_b = k.is_cuda and d_k >= 16 and (d_v >= 16)
        if use_fused_b:
            b = None
        else:
            b = torch.zeros(
                batch_size,
                num_heads,
                num_chunks,
                d_k,
                d_v,
                device=k.device,
                dtype=dtype,
            )
        kf = k_chunk.float()
        vf = v_chunk.float()
        bf = beta_chunk.float()
        gf = gk_chunk.float()
        use_fused_a = (
            k.is_cuda
            and fused_affine_chunk_a is not None
            and (can_use_fused_affine_chunk_a is not None)
        )
        if use_fused_a:
            kf_a_flat = rearrange(kf, "b h c l d -> (b h c) l d").contiguous()
            bf_a_flat = rearrange(bf, "b h c l -> (b h c) l").contiguous()
            gf_a_flat = rearrange(gf, "b h c l -> (b h c) l").contiguous()
            if can_use_fused_affine_chunk_a(kf_a_flat, bf_a_flat, gf_a_flat):
                a_flat = fused_affine_chunk_a(kf_a_flat, bf_a_flat, gf_a_flat)
                a = rearrange(
                    a_flat,
                    "(b h c) d e -> b h c d e",
                    b=batch_size,
                    h=num_heads,
                    c=num_chunks,
                )
            else:
                use_fused_a = False
        if not use_fused_a:
            for t in range(chunk_size):
                kt = kf[:, :, :, t]
                vt = vf[:, :, :, t]
                bt = bf[:, :, :, t]
                decay = gf[:, :, :, t].exp()
                outer_kk = kt[..., :, None] * kt[..., None, :]
                a_t = decay[..., None, None] * (eye - bt[..., None, None] * outer_kk)
                if not use_fused_b:
                    vt = vf[:, :, :, t]
                    b_t = bt[..., None, None] * (kt[..., :, None] * vt[..., None, :])
                    b = torch.matmul(a_t, b) + b_t
                a = torch.matmul(a_t, a)
        elif not use_fused_b:
            for t in range(chunk_size):
                kt = kf[:, :, :, t]
                vt = vf[:, :, :, t]
                bt = bf[:, :, :, t]
                decay = gf[:, :, :, t].exp()
                outer_kk = kt[..., :, None] * kt[..., None, :]
                a_t = decay[..., None, None] * (eye - bt[..., None, None] * outer_kk)
                b_t = bt[..., None, None] * (kt[..., :, None] * vt[..., None, :])
                b = torch.matmul(a_t, b) + b_t
        if use_fused_b:
            kf_flat = rearrange(kf, "b h c l d -> (b c) h l d").contiguous()
            vf_flat = rearrange(vf, "b h c l d -> (b c) h l d").contiguous()
            bf_flat = rearrange(bf, "b h c l -> (b c) h l").contiguous()
            gf_flat = rearrange(gf, "b h c l -> (b c) h l").contiguous()
            q_flat = torch.zeros_like(kf_flat)
            _, b_flat = chunk_gated_delta_rule(
                q_flat, kf_flat, vf_flat, bf_flat, gf_flat, output_final_state=True
            )
            b = rearrange(
                b_flat, "(b c) h d e -> b h c d e", b=batch_size, c=num_chunks
            ).float()
        return (a, b)

    def _fast_affine_chunk_data(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        gk: torch.Tensor,
        num_chunks: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute local output/readout and chunk B/A in one GDN pass.

        Appending zero value columns and seeding those columns with an identity
        state lets the native GDN kernel propagate ``[B | A]`` jointly.  Its
        token outputs are correspondingly ``[local_output | local_readout]``.
        When the recurrent projections are frozen, all four results remain
        detached as in the original router-only fast path.  When any recurrent
        state parameter is trainable, append identity queries to each chunk so
        the final affine state is exposed as ordinary token outputs.  The GDN
        kernel's backward supports output gradients but (intentionally) does
        not consume a final-state gradient, hence this readout construction is
        required for end-to-end q/k/v/beta/decay training.
        """
        chunk_size = 256
        batch_size, num_heads, _, d_k = q.shape
        d_v = v.shape[-1]
        differentiable_state = self._affine_state_grad_enabled()
        maybe_detach = (lambda x: x) if differentiable_state else lambda x: x.detach()
        qf = rearrange(
            maybe_detach(q), "b h (c l) d -> (b c) h l d", c=num_chunks, l=chunk_size
        ).contiguous()
        kf = rearrange(
            maybe_detach(k), "b h (c l) d -> (b c) h l d", c=num_chunks, l=chunk_size
        ).contiguous()
        vf = rearrange(
            maybe_detach(v), "b h (c l) d -> (b c) h l d", c=num_chunks, l=chunk_size
        ).contiguous()
        bf = rearrange(
            maybe_detach(beta), "b h (c l) -> (b c) h l", c=num_chunks, l=chunk_size
        ).contiguous()
        gf = rearrange(
            maybe_detach(gk), "b h (c l) -> (b c) h l", c=num_chunks, l=chunk_size
        ).contiguous()
        native_inference = (
            not self.training
            and (not torch.is_grad_enabled())
            and bool(os.environ.get("GDN_GLA_NATIVE_AFFINE_PREFILL", ""))
        )
        native_bf16_training = (
            differentiable_state
            and os.environ.get("GDN_AFFINE_FINAL_STATE_BACKWARD") == "1"
            and (os.environ.get("GDN_AFFINE_NATIVE_BF16") == "1")
            and (qf.dtype == torch.bfloat16)
        )
        if not native_inference:
            if not native_bf16_training:
                qf = qf.float()
                kf = kf.float()
                vf = vf.float()
            bf = bf.float()
            gf = gf.float()
        joint_v = torch.cat((vf, torch.zeros_like(kf)), dim=-1)
        initial = torch.zeros(
            batch_size * num_chunks,
            num_heads,
            d_k,
            d_v + d_k,
            device=q.device,
            dtype=torch.float32,
        )
        eye = torch.eye(d_k, device=q.device, dtype=torch.float32)
        initial[..., d_v:].copy_(eye)
        if (
            differentiable_state
            and os.environ.get("GDN_AFFINE_FINAL_STATE_BACKWARD") == "1"
        ):
            if os.environ.get("GDN_AFFINE_TF32X3") == "1":
                from .gated_delta_rule_ops.chunk_final_state_tf32x3 import (
                    chunk_gated_delta_rule_final_state,
                )
            else:
                from .gated_delta_rule_ops.chunk_final_state import (
                    chunk_gated_delta_rule_final_state,
                )
            local_joint, chunk_joint = chunk_gated_delta_rule_final_state(
                qf, kf, joint_v, bf, gf, initial_state=initial, output_final_state=True
            )
        elif differentiable_state:
            read_q = eye.mul(d_k**0.5).to(qf.dtype).view(1, 1, d_k, d_k)
            read_q = read_q.expand(batch_size * num_chunks, num_heads, -1, -1)
            zero_k = torch.zeros_like(read_q)
            zero_v = torch.zeros(
                batch_size * num_chunks,
                num_heads,
                d_k,
                d_v + d_k,
                device=q.device,
                dtype=joint_v.dtype,
            )
            zero_gate = torch.zeros(
                batch_size * num_chunks, num_heads, d_k, device=q.device, dtype=bf.dtype
            )
            q_run = torch.cat((qf, read_q), dim=2)
            k_run = torch.cat((kf, zero_k), dim=2)
            v_run = torch.cat((joint_v, zero_v), dim=2)
            b_run = torch.cat((bf, zero_gate), dim=2)
            g_run = torch.cat((gf, zero_gate), dim=2)
            local_joint, chunk_joint = chunk_gated_delta_rule(
                q_run,
                k_run,
                v_run,
                b_run,
                g_run,
                initial_state=initial,
                output_final_state=False,
            )
            chunk_joint = local_joint[:, :, chunk_size:]
            local_joint = local_joint[:, :, :chunk_size]
        else:
            with torch.no_grad():
                local_joint, chunk_joint = chunk_gated_delta_rule(
                    qf,
                    kf,
                    joint_v,
                    bf,
                    gf,
                    initial_state=initial,
                    output_final_state=True,
                )
        local_joint = rearrange(
            local_joint, "(b c) h l d -> b h c l d", b=batch_size, c=num_chunks
        )
        chunk_joint = rearrange(
            chunk_joint, "(b c) h d e -> b h c d e", b=batch_size, c=num_chunks
        )
        native_bf16_history = native_inference and (
            not os.environ.get("GDN_GLA_FP32_AFFINE_HISTORY", "")
        )
        chunk_joint = (
            chunk_joint.to(v.dtype) if native_bf16_history else chunk_joint.float()
        )
        return (
            local_joint[..., :d_v],
            local_joint[..., d_v:],
            chunk_joint[..., d_v:],
            chunk_joint[..., :d_v],
        )

    def _affine_state_grad_enabled(self) -> bool:
        """Whether this module must retain gradients through affine states."""
        if not (self.training and torch.is_grad_enabled()):
            return False
        decay_proj = getattr(self, "a_proj", None)
        if decay_proj is None:
            decay_proj = getattr(self, "gk_proj", None)
        modules = tuple(
            (
                module
                for module in (
                    self.q_proj,
                    self.k_proj,
                    self.v_proj,
                    decay_proj,
                    self.b_proj,
                )
                if module is not None
            )
        )
        if any((p.requires_grad for module in modules for p in module.parameters())):
            return True
        return bool(
            getattr(getattr(self, "A_log", None), "requires_grad", False)
        ) or bool(getattr(getattr(self, "dt_bias", None), "requires_grad", False))

    def _mix_outputs_affine(
        self,
        local_out: torch.Tensor,
        chunk_a: torch.Tensor,
        chunk_b: torch.Tensor,
        q: torch.Tensor,
        weights: torch.Tensor,
        num_tokens: int,
        num_chunks: int,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        gk: torch.Tensor,
        local_readout: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self._mix_outputs_affine_q_parallel(
            local_out,
            chunk_a,
            chunk_b,
            q,
            weights,
            num_tokens,
            num_chunks,
            k,
            v,
            beta,
            gk,
            local_readout=local_readout,
        )

    def _mix_outputs_affine_q_parallel(
        self,
        local_out: torch.Tensor,
        chunk_a: torch.Tensor,
        chunk_b: torch.Tensor,
        q: torch.Tensor,
        weights: torch.Tensor,
        num_tokens: int,
        num_chunks: int,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        gk: torch.Tensor,
        local_readout: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        chunk_size = 256
        total_tokens = num_chunks * chunk_size
        q_padded = self._pad_tokens(q, total_tokens - num_tokens)
        if weights.shape[2] != total_tokens:
            weights = F.pad(weights, (0, 0, 0, total_tokens - weights.shape[2]))
        differentiable_state = self._affine_state_grad_enabled()
        maybe_detach = (lambda x: x) if differentiable_state else lambda x: x.detach()
        q_chunk = rearrange(
            maybe_detach(q_padded),
            "b h (c l) d -> b h c l d",
            c=num_chunks,
            l=chunk_size,
        )
        weight_chunk = rearrange(
            weights, "b h (c l) k -> b h c l k", c=num_chunks, l=chunk_size
        )
        native_inference = (
            not self.training
            and (not torch.is_grad_enabled())
            and bool(os.environ.get("GDN_GLA_NATIVE_AFFINE_PREFILL", ""))
        )
        native_bf16_history = native_inference and (
            not os.environ.get("GDN_GLA_FP32_AFFINE_HISTORY", "")
        )
        if differentiable_state:
            chunk_a = chunk_a.float()
            chunk_b = chunk_b.float()
        elif native_bf16_history:
            chunk_a = chunk_a.detach().to(q.dtype)
            chunk_b = chunk_b.detach().to(q.dtype)
        else:
            chunk_a = chunk_a.detach().float()
            chunk_b = chunk_b.detach().float()
        k_chunk = rearrange(
            maybe_detach(k), "b h (c l) d -> b h c l d", c=num_chunks, l=chunk_size
        ).float()
        v_chunk = rearrange(
            maybe_detach(v), "b h (c l) d -> b h c l d", c=num_chunks, l=chunk_size
        ).float()
        beta_chunk = rearrange(
            maybe_detach(beta), "b h (c l) -> b h c l", c=num_chunks, l=chunk_size
        ).float()
        gk_chunk = rearrange(
            maybe_detach(gk), "b h (c l) -> b h c l", c=num_chunks, l=chunk_size
        ).float()
        batch_size, num_heads, _, d_k, d_v = chunk_b.shape
        scale = q.shape[-1] ** (-0.5)
        token_offsets = torch.arange(chunk_size, device=q.device).view(1, 1, chunk_size)

        def compose_one_query_chunk(
            gates_c: torch.Tensor, q_c: torch.Tensor, query_chunk_idx: int
        ) -> torch.Tensor:
            q_scaled = q_c.float() * scale
            current_gate = gates_c[:, :, :, query_chunk_idx].float()
            prefix_gate = current_gate
            use_fused_prefix = False
            if (
                fused_affine_prefix_readout is not None
                and can_use_fused_affine_prefix_readout is not None
            ):
                q_flat = q_scaled.reshape(
                    batch_size * num_heads, chunk_size, d_k
                ).contiguous()
                gate_flat = prefix_gate.reshape(
                    batch_size * num_heads, chunk_size
                ).contiguous()
                k_flat = (
                    k_chunk[:, :, query_chunk_idx]
                    .reshape(batch_size * num_heads, chunk_size, d_k)
                    .contiguous()
                )
                v_flat = (
                    v_chunk[:, :, query_chunk_idx]
                    .reshape(batch_size * num_heads, chunk_size, d_v)
                    .contiguous()
                )
                beta_flat = (
                    beta_chunk[:, :, query_chunk_idx]
                    .reshape(batch_size * num_heads, chunk_size)
                    .contiguous()
                )
                gk_flat = (
                    gk_chunk[:, :, query_chunk_idx]
                    .reshape(batch_size * num_heads, chunk_size)
                    .contiguous()
                )
                if can_use_fused_affine_prefix_readout(
                    q_flat, gate_flat, k_flat, v_flat, beta_flat, gk_flat
                ):
                    out_flat, r_flat = fused_affine_prefix_readout(
                        q_flat, gate_flat, k_flat, v_flat, beta_flat, gk_flat
                    )
                    out = out_flat.reshape(batch_size, num_heads, chunk_size, d_v)
                    r = r_flat.reshape(batch_size, num_heads, chunk_size, d_k)
                    use_fused_prefix = True
            if not use_fused_prefix:
                r = prefix_gate[..., None] * q_scaled
                out = torch.zeros(
                    batch_size,
                    num_heads,
                    chunk_size,
                    d_v,
                    device=q.device,
                    dtype=torch.float32,
                )
                for u in range(chunk_size - 1, -1, -1):
                    active = token_offsets >= u
                    kt = k_chunk[:, :, query_chunk_idx, u].unsqueeze(2)
                    vt = v_chunk[:, :, query_chunk_idx, u].unsqueeze(2)
                    bt = beta_chunk[:, :, query_chunk_idx, u].view(
                        batch_size, num_heads, 1
                    )
                    decay = (
                        gk_chunk[:, :, query_chunk_idx, u]
                        .exp()
                        .view(batch_size, num_heads, 1, 1)
                    )
                    dot = torch.sum(r * kt, dim=-1)
                    coeff = bt * dot
                    out = out + active.to(out.dtype)[..., None] * coeff[..., None] * vt
                    next_r = decay * (r - coeff[..., None] * kt)
                    r = torch.where(active[..., None], next_r, r)
                r = (1.0 - current_gate)[..., None] * q_scaled + r
            for i in range(query_chunk_idx - 1, -1, -1):
                s_i = gates_c[:, :, :, i].float()
                contribution = torch.einsum(
                    "b h l d, b h d e -> b h l e", r, chunk_b[:, :, i]
                )
                out = out + s_i[..., None] * contribution
                transformed_r = torch.einsum(
                    "b h d e, b h l d -> b h l e", chunk_a[:, :, i], r
                )
                r = (1.0 - s_i)[..., None] * r + s_i[..., None] * transformed_r
            return out.to(q.dtype)

        if local_readout is not None:
            local_output = local_out
            readout = local_readout
            output = local_output
            readout = local_readout
            use_fused_history = (
                not self.training
                and (not torch.is_grad_enabled())
                and (fused_affine_history_mix is not None)
                and (can_use_fused_affine_history_mix is not None)
                and can_use_fused_affine_history_mix(
                    output, readout, chunk_a, chunk_b, weight_chunk
                )
            )
            if use_fused_history:
                output = fused_affine_history_mix(
                    output, readout, chunk_a, chunk_b, weight_chunk
                )
            elif (
                self.training
                and torch.is_grad_enabled()
                and os.environ.get("GDN_COMPILE_AFFINE_HISTORY_TRAINING", "")
                and (compiled_affine_history_mix is not None)
            ):
                output = compiled_affine_history_mix(
                    output, readout, chunk_a, chunk_b, weight_chunk
                )
            else:
                differentiable = self.training and torch.is_grad_enabled()
                for source_idx in range(num_chunks - 2, -1, -1):
                    query_slice = slice(source_idx + 1, None)
                    gate = weight_chunk[:, :, query_slice, :, source_idx]
                    readout_tail = readout[:, :, query_slice]
                    source_b = chunk_b[:, :, source_idx]
                    source_a = chunk_a[:, :, source_idx]
                    if not differentiable:
                        source_b = source_b.to(readout_tail.dtype)
                        source_a = source_a.to(readout_tail.dtype)
                        gate = gate.to(readout_tail.dtype)
                    contribution = torch.einsum(
                        "bhcld,bhde->bhcle", readout_tail, source_b
                    )
                    transformed = torch.einsum(
                        "bhde,bhcld->bhcle", source_a, readout_tail
                    )
                    output_tail = (
                        output[:, :, query_slice] + gate[..., None] * contribution
                    )
                    readout_tail = (1.0 - gate)[..., None] * readout_tail + gate[
                        ..., None
                    ] * transformed
                    if differentiable:
                        output = torch.cat(
                            (output[:, :, : source_idx + 1], output_tail), dim=2
                        )
                        readout = torch.cat(
                            (readout[:, :, : source_idx + 1], readout_tail), dim=2
                        )
                    else:
                        output[:, :, query_slice].copy_(output_tail)
                        readout[:, :, query_slice].copy_(readout_tail)
            return rearrange(output.to(q.dtype), "b h c l d -> b h (c l) d")[
                :, :, :num_tokens
            ]
        out_chunks = []
        for c in range(num_chunks):
            gates_c = weight_chunk[:, :, c]
            q_c = q_chunk[:, :, c]
            out_c = compose_one_query_chunk(gates_c, q_c, c)
            out_chunks.append(out_c)
        out = torch.stack(out_chunks, dim=2)
        return rearrange(out, "b h c l d -> b h (c l) d")[:, :, :num_tokens]


class ReleaseGatedDeltaNet(nn.Module):

    def __init__(
        self,
        hidden_size: int = 1024,
        head_dim: int = 256,
        num_heads: int = 4,
        expand_v: float = 1.0,
        conv_size: int = 4,
        conv_bias: bool = False,
        use_short_conv: bool = True,
        layer_idx: int = None,
        norm_eps: float = 1e-06,
        use_gate: bool = True,
        use_residual: bool = True,
    ) -> ReleaseGatedDeltaNet:
        super().__init__()
        self.hidden_size = hidden_size
        self.head_k_dim = head_dim
        self.head_v_dim = int(head_dim * expand_v)
        self.num_heads = num_heads
        self.key_dim = num_heads * self.head_k_dim
        self.value_dim = num_heads * self.head_v_dim
        self.conv_size = conv_size
        self.use_short_conv = use_short_conv
        self.layer_idx = layer_idx
        self.use_gate = use_gate
        self.use_residual = use_residual
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.a_proj = nn.Linear(hidden_size, self.num_heads, bias=False)
        self.b_proj = nn.Linear(hidden_size, self.num_heads, bias=False)
        if self.use_short_conv:
            self.q_conv1d = ReleaseShortConvolution(
                self.key_dim, conv_size, bias=conv_bias, activation="silu"
            )
            self.k_conv1d = ReleaseShortConvolution(
                self.key_dim, conv_size, bias=conv_bias, activation="silu"
            )
            self.v_conv1d = ReleaseShortConvolution(
                self.value_dim, conv_size, bias=conv_bias, activation="silu"
            )
        A = torch.empty(self.num_heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        dt_min = 0.001
        dt_max = 0.1
        dt_init_floor = 0.0001
        dt = torch.exp(
            torch.rand(self.num_heads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.dt_bias._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.num_heads))
        self.D._no_weight_decay = True
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = SimpleRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(
                hidden_size=self.head_v_dim, elementwise_affine=True, eps=norm_eps
            )
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def _project_qkvg_with_route_inputs(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
    ):
        if use_cache or past_key_values is not None:
            raise NotImplementedError(
                "ReleaseGatedDeltaNet cache path is not implemented in lit_gpt training."
            )
        q_preconv = self.q_proj(hidden_states)
        k_preconv = self.k_proj(hidden_states)
        q = q_preconv
        k = k_preconv
        v = self.v_proj(hidden_states)
        if self.use_short_conv:
            q = self.q_conv1d(q, attention_mask)
            k = self.k_conv1d(k, attention_mask)
            v = self.v_conv1d(v, attention_mask)
        else:
            q = F.silu(q)
            k = F.silu(k)
            v = F.silu(v)
        if attention_mask is not None and attention_mask.dim() == 2:
            v = v * attention_mask.unsqueeze(-1)
        q_raw_route = rearrange(
            q_preconv, "b l (h d) -> b h l d", h=self.num_heads, d=self.head_k_dim
        )
        k_raw_route = rearrange(
            k_preconv, "b l (h d) -> b h l d", h=self.num_heads, d=self.head_k_dim
        )
        q = rearrange(q, "b l (h d) -> b h l d", h=self.num_heads, d=self.head_k_dim)
        k = rearrange(k, "b l (h d) -> b h l d", h=self.num_heads, d=self.head_k_dim)
        v = rearrange(v, "b l (h d) -> b h l d", h=self.num_heads, d=self.head_v_dim)
        q = l2_norm_fn(q).to(v)
        k = l2_norm_fn(k).to(v)
        q_route_base = q_raw_route.to(v)
        k_route_base = k_raw_route.to(v)
        beta = self.b_proj(hidden_states).float().sigmoid().transpose(1, 2)
        g = -self.A_log.float().exp() * F.softplus(
            self.a_proj(hidden_states).float() + self.dt_bias
        )
        g = g.transpose(1, 2)
        return (q, k, v, beta, g, q_route_base, k_route_base)

    def _project_qkvg(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
    ):
        q, k, v, beta, g, *_ = self._project_qkvg_with_route_inputs(
            hidden_states, attention_mask, past_key_values, use_cache
        )
        return (q, k, v, beta, g)

    def _finalize(
        self, hidden_states: torch.Tensor, o: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        if self.use_residual:
            o = o + self.D[None, :, None, None] * v
        o = rearrange(o, "b h l d -> b l h d")
        if self.use_gate:
            gate = rearrange(
                self.g_proj(hidden_states),
                "b l (h d) -> b l h d",
                h=self.num_heads,
                d=self.head_v_dim,
            )
            o = self.o_norm(o, gate)
        else:
            o = self.o_norm(o)
        o = rearrange(o, "b l h d -> b l (h d)")
        return self.o_proj(o)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        if use_cache:
            if self.use_short_conv:
                raise NotImplementedError(
                    "ReleaseGatedDeltaNet cache requires use_short_conv=False."
                )
            q, k, v, beta, g = self._project_qkvg(
                hidden_states, attention_mask, None, False
            )
            o, recurrent_state = chunk_gated_delta_rule(
                q, k, v, beta, g, initial_state=past_key_values, output_final_state=True
            )
            return (self._finalize(hidden_states, o, v), None, recurrent_state)
        q, k, v, beta, g = self._project_qkvg(
            hidden_states, attention_mask, past_key_values, use_cache
        )
        o, _ = chunk_gated_delta_rule(q, k, v, beta, g, output_final_state=False)
        return (self._finalize(hidden_states, o, v), None, past_key_values)


class ReleaseGLAGatedDeltaNet(ReleaseGatedDeltaNet):

    def __init__(self, *args, **kwargs) -> ReleaseGLAGatedDeltaNet:
        super().__init__(*args, **kwargs)
        self.mix_chunk_size = 256
        self.mix_pool_size = 16
        self.mix_apply_gdn_decay = False
        self.router_route_source = "raw_qk"
        self.router_use_logmean = True
        self.router_use_rope = False
        self.router_pool_self_attention = True
        self.router_gate = "affine_sigmoid"
        self.router_sigmoid_bias = 2.2
        self.route_head_dim = self.head_k_dim
        self.route_q_proj = HeadwiseLinear(
            self.num_heads, self.head_k_dim, self.head_k_dim, bias=False
        )
        self.route_k_proj = None
        self.route_pool_input_dim = self.head_k_dim
        self.route_pool_attn_k_proj = HeadwiseLinear(
            self.num_heads, self.route_pool_input_dim, self.route_head_dim, bias=False
        )
        self.route_pool_attn_v_proj = HeadwiseLinear(
            self.num_heads,
            self.route_pool_input_dim,
            self.route_pool_input_dim,
            bias=False,
        )
        self.route_pool_attn_o_proj = HeadwiseLinear(
            self.num_heads, self.route_pool_input_dim, self.route_head_dim, bias=False
        )

    _pad_tokens = HLAAffineMixin._pad_tokens
    _pad_gate = HLAAffineMixin._pad_gate
    _route_qk = HLAAffineMixin._route_qk
    _use_affine_composition = HLAAffineMixin._use_affine_composition
    _apply_route_pool_linear = HLAAffineMixin._apply_route_pool_linear
    _route_pool_self_attention_prefix_representatives_fused = (
        HLAAffineMixin._route_pool_self_attention_prefix_representatives_fused
    )
    _route_pool_self_attention_representative = (
        HLAAffineMixin._route_pool_self_attention_representative
    )
    _route_pool_prefix_representatives = (
        HLAAffineMixin._route_pool_prefix_representatives
    )
    _route_pool_representatives = HLAAffineMixin._route_pool_representatives
    _local_chunks = HLAAffineMixin._local_chunks
    _current_chunk_scores = HLAAffineMixin._current_chunk_scores
    _mix_weights = HLAAffineMixin._mix_weights
    _chunk_affine_summaries = HLAAffineMixin._chunk_affine_summaries
    _fast_affine_chunk_data = HLAAffineMixin._fast_affine_chunk_data
    _affine_state_grad_enabled = HLAAffineMixin._affine_state_grad_enabled
    _mix_outputs_affine = HLAAffineMixin._mix_outputs_affine
    _mix_outputs_affine_q_parallel = HLAAffineMixin._mix_outputs_affine_q_parallel

    def _init_stream_state(
        self, batch_size: int, capacity: int, device: torch.device, dtype: torch.dtype
    ) -> dict:
        buckets = 256 // 16
        capacity = max(int(capacity), 1)
        state = {
            "pos": torch.zeros(batch_size, device=device, dtype=torch.long),
            "pos_int": 0 if batch_size == 1 else None,
            "local_state": torch.zeros(
                batch_size,
                self.num_heads,
                self.head_k_dim,
                self.head_v_dim,
                device=device,
                dtype=dtype,
            ),
            "final_states": torch.zeros(
                batch_size,
                self.num_heads,
                capacity,
                self.head_k_dim,
                self.head_v_dim,
                device=device,
                dtype=dtype,
            ),
            "chunk_end_decay": torch.zeros(
                batch_size, self.num_heads, capacity, device=device, dtype=torch.float32
            ),
            "g_cumsum": torch.zeros(
                batch_size, self.num_heads, device=device, dtype=torch.float32
            ),
            "route_sums": torch.zeros(
                batch_size,
                self.num_heads,
                capacity,
                buckets,
                self.route_head_dim,
                device=device,
                dtype=dtype,
            ),
            "route_counts": torch.zeros(
                batch_size, capacity, buckets, device=device, dtype=torch.float32
            ),
            "route_current_bucket_tokens": torch.zeros(
                batch_size,
                self.num_heads,
                16,
                self.route_head_dim,
                device=device,
                dtype=dtype,
            ),
        }
        joint_dim = self.head_v_dim + self.head_k_dim
        state["chunk_joint"] = torch.zeros(
            batch_size,
            self.num_heads,
            capacity,
            self.head_k_dim,
            joint_dim,
            device=device,
            dtype=torch.float32,
        )
        state["chunk_b"] = state["chunk_joint"][..., : self.head_v_dim]
        state["chunk_a"] = state["chunk_joint"][..., self.head_v_dim :]
        state["current_joint"] = torch.zeros(
            batch_size,
            self.num_heads,
            self.head_k_dim,
            joint_dim,
            device=device,
            dtype=torch.float32,
        )
        state["current_b"] = state["current_joint"][..., : self.head_v_dim]
        state["current_a"] = state["current_joint"][..., self.head_v_dim :]
        eye = torch.eye(self.head_k_dim, device=device, dtype=torch.float32)
        state["current_a"].copy_(eye.view(1, 1, self.head_k_dim, self.head_k_dim))
        return state

    def _grow_stream_state(self, state: dict, min_capacity: int) -> dict:
        old_capacity = state["final_states"].shape[2]
        if old_capacity >= min_capacity:
            return state
        new_capacity = max(int(min_capacity), old_capacity * 2)

        def grow_dim2(tensor: torch.Tensor) -> torch.Tensor:
            new_shape = list(tensor.shape)
            new_shape[2] = new_capacity
            grown = tensor.new_zeros(*new_shape)
            grown[:, :, :old_capacity] = tensor
            return grown

        final_states = state["final_states"].new_zeros(
            state["final_states"].shape[0],
            state["final_states"].shape[1],
            new_capacity,
            state["final_states"].shape[3],
            state["final_states"].shape[4],
        )
        final_states[:, :, :old_capacity] = state["final_states"]
        state["final_states"] = final_states
        state["chunk_end_decay"] = grow_dim2(state["chunk_end_decay"])
        state["route_sums"] = grow_dim2(state["route_sums"])
        if state.get("chunk_joint") is not None:
            state["chunk_joint"] = grow_dim2(state["chunk_joint"])
            state["chunk_b"] = state["chunk_joint"][..., : self.head_v_dim]
            state["chunk_a"] = state["chunk_joint"][..., self.head_v_dim :]
        else:
            if state.get("chunk_a") is not None:
                state["chunk_a"] = grow_dim2(state["chunk_a"])
            if state.get("chunk_b") is not None:
                state["chunk_b"] = grow_dim2(state["chunk_b"])
        route_counts = state["route_counts"].new_zeros(
            state["route_counts"].shape[0], new_capacity, state["route_counts"].shape[2]
        )
        route_counts[:, :old_capacity] = state["route_counts"]
        state["route_counts"] = route_counts
        return state

    def _route_bucket_stats(
        self, k_route: torch.Tensor, num_tokens: int, num_chunks: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pad_len = num_chunks * 256 - num_tokens
        k_padded = self._pad_tokens(k_route, pad_len)
        buckets = 256 // 16
        k_bucket = rearrange(
            k_padded, "b h (c p s) d -> b h c p s d", c=num_chunks, p=buckets, s=16
        )
        token_idx = torch.arange(num_chunks * 256, device=k_route.device)
        real = token_idx < num_tokens
        counts = rearrange(real, "(c p s) -> c p s", c=num_chunks, p=buckets, s=16)
        route_counts = (
            counts.sum(dim=2)
            .to(torch.float32)
            .unsqueeze(0)
            .expand(k_route.shape[0], -1, -1)
            .contiguous()
        )
        prefix_reps = self._route_pool_prefix_representatives(k_bucket)
        gather_idx = (
            route_counts[0]
            .long()
            .sub(1)
            .clamp_min(0)
            .view(1, 1, num_chunks, buckets, 1, 1)
        )
        gather_idx = gather_idx.expand(
            k_route.shape[0],
            k_route.shape[1],
            num_chunks,
            buckets,
            1,
            self.route_head_dim,
        )
        route_sums = prefix_reps.gather(4, gather_idx).squeeze(4).to(k_route.dtype)
        route_sums = route_sums.masked_fill(
            route_counts.view(k_route.shape[0], 1, num_chunks, buckets, 1) == 0, 0
        )
        return (route_sums, route_counts)

    def _state_from_prefill(
        self,
        final_state: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        g: torch.Tensor,
        k_route: torch.Tensor,
        k_amp: Optional[torch.Tensor],
        num_tokens: int,
        num_chunks: int,
        dtype: torch.dtype,
        capacity: Optional[int] = None,
        chunk_a: Optional[torch.Tensor] = None,
        chunk_b: Optional[torch.Tensor] = None,
    ) -> dict:
        batch_size = final_state.shape[0]
        state = self._init_stream_state(
            batch_size, capacity or num_chunks, g.device, dtype
        )
        state["pos"].fill_(num_tokens)
        state["pos_int"] = int(num_tokens) if batch_size == 1 else None
        state["g_cumsum"] = g.float().sum(dim=2)
        completed_chunks = num_tokens // 256
        if completed_chunks > 0:
            state["final_states"][:, :, :completed_chunks] = final_state[
                :, :, :completed_chunks
            ]
            pad_len = num_chunks * 256 - num_tokens
            g_pad = self._pad_gate(g, pad_len)
            g_cumsum = g_pad.float().cumsum(dim=2)
            chunk_end_idx = (torch.arange(num_chunks, device=g.device) + 1) * 256 - 1
            chunk_end_decay = g_cumsum.index_select(2, chunk_end_idx)
            state["chunk_end_decay"][:, :, :completed_chunks] = chunk_end_decay[
                :, :, :completed_chunks
            ]
        if num_tokens % 256 != 0:
            current_chunk = num_tokens // 256
            current_start = current_chunk * 256
            _, current_state = chunk_gated_delta_rule(
                q[:, :, current_start:num_tokens],
                k[:, :, current_start:num_tokens],
                v[:, :, current_start:num_tokens],
                beta[:, :, current_start:num_tokens],
                g[:, :, current_start:num_tokens],
                output_final_state=True,
            )
            state["local_state"] = current_state
        route_sums, route_counts = self._route_bucket_stats(
            k_route, num_tokens, num_chunks
        )
        state["route_sums"][:, :, :num_chunks] = route_sums
        state["route_counts"][:, :num_chunks] = route_counts
        if chunk_a is not None and chunk_b is not None:
            state["chunk_a"][:, :, :num_chunks] = chunk_a[:, :, :num_chunks].float()
            state["chunk_b"][:, :, :num_chunks] = chunk_b[:, :, :num_chunks].float()
            eye = torch.eye(self.head_k_dim, device=g.device, dtype=torch.float32)
            state["current_a"].copy_(eye.view(1, 1, self.head_k_dim, self.head_k_dim))
            state["current_b"].zero_()
            if num_tokens % 256 != 0:
                current_chunk = num_tokens // 256
                state["current_a"].copy_(chunk_a[:, :, current_chunk].float())
                state["current_b"].copy_(chunk_b[:, :, current_chunk].float())
        if num_tokens % 16 != 0:
            current_pool_start = num_tokens // 16 * 16
            current_pool_tokens = k_route[:, :, current_pool_start:num_tokens]
            state["route_current_bucket_tokens"].zero_()
            state["route_current_bucket_tokens"][
                :, :, : current_pool_tokens.shape[2]
            ] = current_pool_tokens
        return state

    def _forward_prefill(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        cache_capacity: Optional[int] = None,
    ) -> Tuple[torch.Tensor, dict]:
        q, k, v, beta, g, q_route_base, k_route_base = (
            self._project_qkvg_with_route_inputs(
                hidden_states, attention_mask, None, False
            )
        )
        num_tokens = q.shape[2]
        pad_len = -num_tokens % 256
        num_chunks = (num_tokens + pad_len) // 256
        q_pad = self._pad_tokens(q, pad_len)
        k_pad = self._pad_tokens(k, pad_len)
        v_pad = self._pad_tokens(v, pad_len)
        beta_pad = self._pad_gate(beta, pad_len)
        g_pad = self._pad_gate(g, pad_len)
        skip_local_chunks = True
        local_out = None
        local_readout = None
        final_state = None
        if not skip_local_chunks:
            local_out, final_state = self._local_chunks(
                q_pad, k_pad, v_pad, beta_pad, g_pad, num_chunks
            )
        use_fused_route_history = False
        route_q_prefill = None
        route_pooled_prefill = None
        k_route = None
        if use_fused_route_history:
            route_q_prefill, k_route = self._route_qk(
                q_route_base[:, :, :num_tokens], k_route_base[:, :, :num_tokens]
            )
            k_route_pad = self._pad_tokens(k_route, pad_len)
            route_pooled_prefill = rearrange(
                k_route_pad,
                "b h (c p s) d -> b h c p s d",
                c=num_chunks,
                p=256 // 16,
                s=16,
            ).mean(dim=4)
            weights = None
        else:
            weights = self._mix_weights(
                q_route_base, k_route_base, num_tokens, num_chunks
            )
        chunk_a = None
        chunk_b = None
        if not os.environ.get("GDN_DISABLE_FAST_AFFINE_CHUNK_DATA", ""):
            local_out, local_readout, chunk_a, chunk_b = self._fast_affine_chunk_data(
                q_pad, k_pad, v_pad, beta_pad, g_pad, num_chunks
            )
            final_state = chunk_b
        else:
            with torch.no_grad():
                chunk_a, chunk_b = self._chunk_affine_summaries(
                    k_pad.detach(),
                    v_pad.detach(),
                    beta_pad.detach(),
                    g_pad.detach(),
                    num_chunks,
                )
        if final_state is None:
            final_state = chunk_b.new_zeros(
                q.shape[0], q.shape[1], num_chunks, self.head_k_dim, self.head_v_dim
            )
        if (
            use_fused_route_history
            and local_out is not None
            and (local_readout is not None)
            and can_use_fused_affine_history_route_mix_wide(
                local_out,
                local_readout,
                chunk_a,
                chunk_b,
                route_q_prefill,
                route_pooled_prefill,
            )
        ):
            fused_output = fused_affine_history_route_mix_wide(
                local_out,
                local_readout,
                chunk_a,
                chunk_b,
                route_q_prefill,
                route_pooled_prefill,
                router_logit_scale=1.0,
                sigmoid_bias=2.2,
                sigmoid_temperature=1.0,
                use_logmean=True,
            )
            o = rearrange(fused_output, "b h c l d -> b h (c l) d")[:, :, :num_tokens]
        else:
            if weights is None:
                weights = self._mix_weights(
                    q_route_base, k_route_base, num_tokens, num_chunks
                )
            o = self._mix_outputs_affine(
                local_out if local_out is not None else v_pad.new_empty(0),
                chunk_a,
                chunk_b,
                q,
                weights,
                num_tokens,
                num_chunks,
                k_pad,
                v_pad,
                beta_pad,
                g_pad,
                local_readout=local_readout,
            )
        if k_route is None:
            _, k_route = self._route_qk(
                q_route_base[:, :, :num_tokens], k_route_base[:, :, :num_tokens]
            )
        _, k_amp = (None, None)
        state = self._state_from_prefill(
            final_state,
            q,
            k,
            v,
            beta,
            g,
            k_route,
            k_amp,
            num_tokens,
            num_chunks,
            q.dtype,
            cache_capacity,
            chunk_a=chunk_a,
            chunk_b=chunk_b,
        )
        return (self._finalize(hidden_states, o, v), state)

    def _commit_stream_chunk_boundary(
        self,
        state: dict,
        pos: torch.Tensor,
        chunk_idx: torch.Tensor,
        batch_idx: torch.Tensor,
        pos_int: Optional[int] = None,
    ) -> None:
        """Commit completed chunks without a host-visible ``Tensor.any`` sync."""
        if pos_int is not None:
            if (pos_int + 1) % 256 != 0:
                return
            completed = pos_int // 256
            state["final_states"][:, :, completed] = state["local_state"]
            state["chunk_end_decay"][:, :, completed] = state["g_cumsum"]
            if state.get("chunk_a") is not None:
                state["chunk_a"][:, :, completed] = state["current_a"]
                state["chunk_b"][:, :, completed] = state["current_b"]
            return
        end_chunk = (pos + 1) % 256 == 0
        mask_state = end_chunk[:, None, None, None]
        mask_decay = end_chunk[:, None]
        old_final = state["final_states"][batch_idx, :, chunk_idx]
        state["final_states"][batch_idx, :, chunk_idx] = torch.where(
            mask_state, state["local_state"], old_final
        )
        old_decay = state["chunk_end_decay"][batch_idx, :, chunk_idx]
        state["chunk_end_decay"][batch_idx, :, chunk_idx] = torch.where(
            mask_decay, state["g_cumsum"], old_decay
        )
        if state.get("chunk_a") is not None:
            old_a = state["chunk_a"][batch_idx, :, chunk_idx]
            old_b = state["chunk_b"][batch_idx, :, chunk_idx]
            state["chunk_a"][batch_idx, :, chunk_idx] = torch.where(
                mask_state, state["current_a"], old_a
            )
            state["chunk_b"][batch_idx, :, chunk_idx] = torch.where(
                mask_state, state["current_b"], old_b
            )

    def _forward_stream(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        state: Optional[dict],
        cache_capacity: Optional[int] = None,
    ) -> Tuple[torch.Tensor, dict]:
        if self.use_short_conv:
            raise NotImplementedError(
                "ReleaseGLAGatedDeltaNet streaming cache requires use_short_conv=False."
            )
        q, k, v, beta, g, q_route_base, k_route_base = (
            self._project_qkvg_with_route_inputs(
                hidden_states, attention_mask, None, False
            )
        )
        q_route, k_route = self._route_qk(q_route_base, k_route_base)
        batch_size, _, seq_len, _ = q.shape
        if state is None:
            state = self._init_stream_state(
                batch_size, True, hidden_states.device, q.dtype
            )
        elif state["pos"].shape[0] != batch_size:
            raise ValueError(
                "Cached HLA+GDN state batch size does not match input batch size."
            )
        outputs = []
        readout_scale = q.shape[-1] ** (-0.5)
        router_scale = q_route.shape[-1] ** (-0.5) * 1.0
        batch_idx_all = torch.arange(batch_size, device=hidden_states.device)
        for token_idx in range(seq_len):
            pos = state["pos"]
            pos_int = state.get("pos_int")
            if pos_int is not None:
                chunk_idx_value = pos_int // 256
                chunk_idx = torch.full_like(pos, chunk_idx_value)
            else:
                chunk_idx = torch.div(pos, 256, rounding_mode="floor")
            if pos_int is not None:
                state = self._grow_stream_state(state, chunk_idx_value + 1)
            elif cache_capacity is None:
                state = self._grow_stream_state(state, int(chunk_idx.max().item()) + 1)
            new_chunk = pos_int % 256 == 0 if pos_int is not None else None
            if new_chunk is True:
                state["local_state"].zero_()
                if state.get("current_a") is not None:
                    eye = torch.eye(
                        self.head_k_dim,
                        device=hidden_states.device,
                        dtype=torch.float32,
                    )
                    state["current_a"].copy_(
                        eye.view(1, 1, self.head_k_dim, self.head_k_dim)
                    )
                    state["current_b"].zero_()
            elif new_chunk is None:
                new_chunk_mask = pos % 256 == 0
                state["local_state"] = torch.where(
                    new_chunk_mask[:, None, None, None],
                    torch.zeros_like(state["local_state"]),
                    state["local_state"],
                )
                if state.get("current_a") is not None:
                    eye = torch.eye(
                        self.head_k_dim,
                        device=hidden_states.device,
                        dtype=torch.float32,
                    ).view(1, 1, self.head_k_dim, self.head_k_dim)
                    state["current_a"].copy_(
                        torch.where(
                            new_chunk_mask[:, None, None, None], eye, state["current_a"]
                        )
                    )
                    state["current_b"].masked_fill_(
                        new_chunk_mask[:, None, None, None], 0.0
                    )
            q_t = q[:, :, token_idx : token_idx + 1]
            k_t = k[:, :, token_idx : token_idx + 1]
            v_t = v[:, :, token_idx : token_idx + 1]
            beta_t = beta[:, :, token_idx : token_idx + 1]
            g_t = g[:, :, token_idx : token_idx + 1]
            k_route_t = k_route[:, :, token_idx]
            if pos_int is not None:
                bucket_idx_value = pos_int % 256 // 16
                bucket_offset_value = pos_int % 16
            else:
                bucket_idx_value = None
                bucket_offset_value = None
            affine_decode_only = (
                state.get("current_a") is not None
                and (not torch.is_grad_enabled())
                and (not os.environ.get("GDN_DISABLE_AFFINE_DECODE_SKIP_LOCAL", ""))
            )
            local_out = None
            if not affine_decode_only:
                local_out, local_state = chunk_gated_delta_rule(
                    q_t,
                    k_t,
                    v_t,
                    beta_t,
                    g_t,
                    initial_state=state["local_state"],
                    output_final_state=True,
                )
                state["local_state"] = local_state
            state["g_cumsum"] = state["g_cumsum"] + g_t.squeeze(2).float()
            use_fused_joint_route_update = False
            if state.get("current_a") is not None:
                kt = k_t.squeeze(2).float().contiguous()
                vt = v_t.squeeze(2).float().contiguous()
                bt = beta_t.squeeze(2).float().contiguous()
                gt = g_t.squeeze(2).float().contiguous()
                fused_update_enabled = bool(
                    os.environ.get("GDN_ENABLE_TRITON_AFFINE_DECODE_UPDATE", "")
                ) and (not torch.is_grad_enabled())
                use_fused_joint_update = (
                    fused_update_enabled
                    and state.get("current_joint") is not None
                    and (fused_affine_decode_joint_update_ is not None)
                    and (can_use_fused_affine_decode_joint_update is not None)
                    and can_use_fused_affine_decode_joint_update(
                        kt, vt, bt, gt, state["current_joint"]
                    )
                )
                use_fused_joint_route_update = False
                use_fused_update = (
                    fused_update_enabled
                    and fused_affine_decode_update_ is not None
                    and (can_use_fused_affine_decode_update is not None)
                    and can_use_fused_affine_decode_update(
                        kt, vt, bt, gt, state["current_a"], state["current_b"]
                    )
                )
                if use_fused_joint_route_update:
                    fused_affine_decode_joint_update_route_(
                        kt,
                        vt,
                        bt,
                        gt,
                        state["current_joint"],
                        k_route_t,
                        state["route_sums"],
                        state["route_counts"],
                        chunk_idx_value,
                        bucket_idx_value,
                        normalize_key=False,
                    )
                elif use_fused_joint_update:
                    fused_affine_decode_joint_update_(
                        kt, vt, bt, gt, state["current_joint"], normalize_key=False
                    )
                elif use_fused_update:
                    fused_affine_decode_update_(
                        kt,
                        vt,
                        bt,
                        gt,
                        state["current_a"],
                        state["current_b"],
                        normalize_key=False,
                    )
                else:
                    decay = gt.exp()
                    eye = torch.eye(
                        self.head_k_dim,
                        device=hidden_states.device,
                        dtype=torch.float32,
                    ).view(1, 1, self.head_k_dim, self.head_k_dim)
                    outer_kk = kt[..., :, None] * kt[..., None, :]
                    a_t = decay[..., None, None] * (
                        eye - bt[..., None, None] * outer_kk
                    )
                    b_t = bt[..., None, None] * (kt[..., :, None] * vt[..., None, :])
                    next_b = torch.matmul(a_t, state["current_b"].float()) + b_t
                    next_a = torch.matmul(a_t, state["current_a"].float())
                    state["current_b"].copy_(next_b)
                    state["current_a"].copy_(next_a)
            if pos_int is not None:
                bucket_idx = None
                bucket_offset = None
            else:
                bucket_idx = torch.div(pos % 256, 16, rounding_mode="floor")
                bucket_offset = pos % 16
            if pos_int is not None:
                if bucket_offset_value == 0:
                    state["route_current_bucket_tokens"].zero_()
                state["route_current_bucket_tokens"][
                    :, :, bucket_offset_value, :
                ] = k_route_t
            else:
                new_bucket = bucket_offset == 0
                if new_bucket.any():
                    state["route_current_bucket_tokens"][new_bucket] = 0
                state["route_current_bucket_tokens"][
                    batch_idx_all, :, bucket_offset, :
                ] = k_route_t
            current_bucket = (
                state["route_current_bucket_tokens"].unsqueeze(2).unsqueeze(3)
            )
            prefix_reps = (
                self._route_pool_prefix_representatives(current_bucket)
                .squeeze(2)
                .squeeze(2)
            )
            if pos_int is not None:
                current_rep = prefix_reps[:, :, bucket_offset_value, :]
                state["route_sums"][:, :, chunk_idx_value, bucket_idx_value, :] = (
                    current_rep.to(state["route_sums"].dtype)
                )
                state["route_counts"][:, chunk_idx_value, bucket_idx_value].fill_(
                    bucket_offset_value + 1
                )
            else:
                current_rep = prefix_reps[batch_idx_all, :, bucket_offset, :]
                state["route_sums"][batch_idx_all, :, chunk_idx, bucket_idx, :] = (
                    current_rep.to(state["route_sums"].dtype)
                )
                state["route_counts"][batch_idx_all, chunk_idx, bucket_idx] = (
                    bucket_offset + 1
                ).to(state["route_counts"].dtype)
            route_counts = state["route_counts"]
            pooled_are_sums = False
            if pooled_are_sums:
                route_values = state["route_sums"]
            else:
                route_means = state["route_sums"].float()
            q_route_t = q_route[:, :, token_idx]
            if not pooled_are_sums:
                route_values = route_means
            use_fused_decode_mix = False
            if use_fused_decode_mix:
                out_t = fused_gdn_affine_decode_mix(
                    q_route_t,
                    route_values,
                    route_counts,
                    q_t.squeeze(2).float() * readout_scale,
                    chunk_idx,
                    state["current_a"],
                    state["current_b"],
                    state["chunk_a"],
                    state["chunk_b"],
                    router_logit_scale=1.0,
                    sigmoid_bias=2.2,
                    sigmoid_temperature=1.0,
                    use_logmean=True,
                    pooled_are_sums=pooled_are_sums,
                    num_warps=16 if self.head_k_dim >= 256 else 8,
                ).to(q.dtype)
                outputs.append(out_t.unsqueeze(2))
                self._commit_stream_chunk_boundary(
                    state, pos, chunk_idx, batch_idx_all, pos_int
                )
                state["pos"] = pos + 1
                if pos_int is not None:
                    state["pos_int"] = pos_int + 1
                continue
            if pooled_are_sums:
                route_values = state["route_sums"].float() / route_counts.clamp_min(
                    1
                ).view(batch_size, 1, route_counts.shape[1], route_counts.shape[2], 1)
            bucket_scores = (
                torch.einsum(
                    "b h d, b h c p d -> b h c p",
                    q_route_t.float(),
                    route_values.float(),
                )
                * router_scale
            )
            bucket_scores = bucket_scores.masked_fill(
                ~(route_counts > 0).view(batch_size, 1, *route_counts.shape[1:]), -1e309
            )
            visible_counts = (
                (route_counts > 0).sum(dim=2).clamp_min(1).to(bucket_scores.dtype)
            )
            scores = torch.logsumexp(bucket_scores, dim=-1)
            scores = scores - visible_counts.log().view(
                batch_size, 1, route_counts.shape[1]
            )
            chunk_ids = torch.arange(
                state["final_states"].shape[2], device=hidden_states.device
            )
            chunk_mask = chunk_ids.view(1, -1) <= chunk_idx.view(batch_size, 1)
            scores = scores.masked_fill(~chunk_mask.view(batch_size, 1, -1), -1e309)
            current_bias = 0.0
            if current_bias:
                current_idx = chunk_idx.view(batch_size, 1, 1).expand(
                    batch_size, self.num_heads, 1
                )
                scores = scores.scatter(
                    2, current_idx, scores.gather(2, current_idx) + current_bias
                )
            sigmoid_scores = scores + 2.2
            sigmoid_temperature = 1.0
            weights = torch.sigmoid(sigmoid_scores / max(sigmoid_temperature, 0.0001))
            weights = weights.masked_fill(~chunk_mask.view(batch_size, 1, -1), 0)
            current_idx = chunk_idx.view(batch_size, 1, 1).expand(
                batch_size, self.num_heads, 1
            )
            weights = weights.scatter(2, current_idx, 1.0)
            weights = weights
            current_weight = weights.gather(
                2,
                chunk_idx.view(batch_size, 1, 1).expand(batch_size, self.num_heads, 1),
            ).view(batch_size, self.num_heads, 1)
            q_read = q_t.squeeze(2).float() * readout_scale
            current_gate = current_weight.float()
            out_t_float = None
            if (
                fused_affine_decode_readout is not None
                and can_use_fused_affine_decode_readout is not None
                and can_use_fused_affine_decode_readout(
                    q_read,
                    current_gate.squeeze(-1),
                    weights.float(),
                    chunk_idx,
                    state["current_a"].float(),
                    state["current_b"].float(),
                    state["chunk_a"].float(),
                    state["chunk_b"].float(),
                )
            ):
                out_t_float = fused_affine_decode_readout(
                    q_read,
                    current_gate.squeeze(-1),
                    weights.float(),
                    chunk_idx,
                    state["current_a"].float(),
                    state["current_b"].float(),
                    state["chunk_a"].float(),
                    state["chunk_b"].float(),
                    num_warps=16 if self.head_k_dim >= 256 else 8,
                )
            if out_t_float is None:
                out_t_float = current_gate * torch.einsum(
                    "b h d, b h d e -> b h e", q_read, state["current_b"].float()
                )
                transformed_current = torch.einsum(
                    "b h d e, b h d -> b h e", state["current_a"].float(), q_read
                )
                r = (1.0 - current_gate) * q_read + current_gate * transformed_current
                max_prev_chunks = int(chunk_idx.max().item())
                for prev_idx in range(max_prev_chunks - 1, -1, -1):
                    active = (chunk_idx > prev_idx).view(batch_size, 1)
                    gate_i = weights[:, :, prev_idx].float()
                    b_contribution = torch.einsum(
                        "b h d, b h d e -> b h e",
                        r,
                        state["chunk_b"][:, :, prev_idx].float(),
                    )
                    out_update = gate_i[..., None] * b_contribution
                    transformed_r = torch.einsum(
                        "b h d e, b h d -> b h e",
                        state["chunk_a"][:, :, prev_idx].float(),
                        r,
                    )
                    r_update = (1.0 - gate_i[..., None]) * r + gate_i[
                        ..., None
                    ] * transformed_r
                    out_t_float = torch.where(
                        active[..., None], out_t_float + out_update, out_t_float
                    )
                    r = torch.where(active[..., None], r_update, r)
            out_t = out_t_float.to(q.dtype)
            outputs.append(out_t.unsqueeze(2))
            self._commit_stream_chunk_boundary(
                state, pos, chunk_idx, batch_idx_all, pos_int
            )
            state["pos"] = pos + 1
            if pos_int is not None:
                state["pos_int"] = pos_int + 1
        o = torch.cat(outputs, dim=2)
        return (self._finalize(hidden_states, o, v), state)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        if use_cache:
            max_seq_length = kwargs.get("max_seq_length")
            cache_capacity = None
            if max_seq_length is not None:
                cache_capacity = math.ceil(int(max_seq_length) / 256)
            if past_key_values is None and hidden_states.shape[1] > 1:
                o, state = self._forward_prefill(
                    hidden_states, attention_mask, cache_capacity
                )
            else:
                o, state = self._forward_stream(
                    hidden_states, attention_mask, past_key_values, cache_capacity
                )
            return (o, None, state)
        q, k, v, beta, g, q_route_base, k_route_base = (
            self._project_qkvg_with_route_inputs(
                hidden_states, attention_mask, past_key_values, False
            )
        )
        num_tokens = q.shape[2]
        pad_len = -num_tokens % 256
        num_chunks = (num_tokens + pad_len) // 256
        q_pad = self._pad_tokens(q, pad_len)
        k_pad = self._pad_tokens(k, pad_len)
        v_pad = self._pad_tokens(v, pad_len)
        beta_pad = self._pad_gate(beta, pad_len)
        g_pad = self._pad_gate(g, pad_len)
        local_readout = None
        if not os.environ.get("GDN_DISABLE_FAST_AFFINE_CHUNK_DATA", ""):
            local_out, local_readout, chunk_a, chunk_b = self._fast_affine_chunk_data(
                q_pad, k_pad, v_pad, beta_pad, g_pad, num_chunks
            )
            final_state = chunk_b
        else:
            local_out, final_state = self._local_chunks(
                q_pad, k_pad, v_pad, beta_pad, g_pad, num_chunks
            )
            chunk_a = None
            chunk_b = None
        weights = self._mix_weights(q_route_base, k_route_base, num_tokens, num_chunks)
        if True and chunk_a is None:
            with torch.no_grad():
                chunk_a, chunk_b = self._chunk_affine_summaries(
                    k_pad.detach(),
                    v_pad.detach(),
                    beta_pad.detach(),
                    g_pad.detach(),
                    num_chunks,
                )
        o = self._mix_outputs_affine(
            local_out,
            chunk_a,
            chunk_b,
            q,
            weights,
            num_tokens,
            num_chunks,
            k_pad,
            v_pad,
            beta_pad,
            g_pad,
            local_readout=local_readout,
        )
        return (self._finalize(hidden_states, o, v), None, past_key_values)
