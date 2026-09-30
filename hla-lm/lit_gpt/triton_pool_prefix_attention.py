from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_last_dim_fp32(scores):
    row_max = tl.max(scores, axis=1)
    scores = scores - row_max[:, None]
    numerator = tl.exp(scores)
    denominator = tl.sum(numerator, axis=1)
    return numerator / denominator[:, None]


@triton.jit
def _pool_prefix_attn_fwd_kernel(
    key,
    value,
    out,
    seq_len: tl.constexpr,
    key_dim: tl.constexpr,
    value_dim: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_s = tl.arange(0, BLOCK_S)
    offs_k = tl.arange(0, BLOCK_K)
    offs_v = tl.arange(0, BLOCK_V)
    key_base = pid * seq_len * key_dim
    value_base = pid * seq_len * value_dim

    key_ptrs = key_base + offs_s[:, None] * key_dim + offs_k[None, :]
    value_ptrs = value_base + offs_s[:, None] * value_dim + offs_v[None, :]
    k = tl.load(key + key_ptrs, mask=(offs_s[:, None] < seq_len) & (offs_k[None, :] < key_dim), other=0.0).to(tl.float32)
    v = tl.load(value + value_ptrs, mask=(offs_s[:, None] < seq_len) & (offs_v[None, :] < value_dim), other=0.0).to(tl.float32)

    scores = tl.dot(k, tl.trans(k), input_precision="ieee") * scale
    rows = offs_s[:, None]
    cols = offs_s[None, :]

    for prefix_len in tl.static_range(1, seq_len + 1):
        col_mask = cols < prefix_len
        row_mask = rows < prefix_len
        prefix_scores = tl.where(row_mask, tl.where(col_mask, scores, -float("inf")), 0.0)
        attn = _softmax_last_dim_fp32(prefix_scores)
        attn = tl.where(row_mask, attn, 0.0)
        attended = tl.dot(attn.to(tl.float32), v, input_precision="ieee")
        rep = tl.sum(attended, axis=0) / prefix_len
        tl.store(
            out + value_base + (prefix_len - 1) * value_dim + offs_v,
            rep,
            mask=offs_v < value_dim,
        )


@triton.jit
def _pool_prefix_attn_bwd_kernel(
    key,
    value,
    grad_out,
    grad_key,
    grad_value,
    seq_len: tl.constexpr,
    key_dim: tl.constexpr,
    value_dim: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_s = tl.arange(0, BLOCK_S)
    offs_k = tl.arange(0, BLOCK_K)
    offs_v = tl.arange(0, BLOCK_V)
    key_base = pid * seq_len * key_dim
    value_base = pid * seq_len * value_dim

    key_ptrs = key_base + offs_s[:, None] * key_dim + offs_k[None, :]
    value_ptrs = value_base + offs_s[:, None] * value_dim + offs_v[None, :]
    k = tl.load(key + key_ptrs, mask=(offs_s[:, None] < seq_len) & (offs_k[None, :] < key_dim), other=0.0).to(tl.float32)
    v = tl.load(value + value_ptrs, mask=(offs_s[:, None] < seq_len) & (offs_v[None, :] < value_dim), other=0.0).to(tl.float32)

    dk = tl.zeros((BLOCK_S, BLOCK_K), dtype=tl.float32)
    dv = tl.zeros((BLOCK_S, BLOCK_V), dtype=tl.float32)
    scores = tl.dot(k, tl.trans(k), input_precision="ieee") * scale
    rows = offs_s[:, None]
    cols = offs_s[None, :]

    for prefix_len in tl.static_range(1, seq_len + 1):
        go = tl.load(
            grad_out + value_base + (prefix_len - 1) * value_dim + offs_v,
            mask=offs_v < value_dim,
            other=0.0,
        ).to(tl.float32)
        col_mask = cols < prefix_len
        row_mask = rows < prefix_len
        active = row_mask & col_mask
        prefix_scores = tl.where(row_mask, tl.where(col_mask, scores, -float("inf")), 0.0)
        attn = _softmax_last_dim_fp32(prefix_scores)
        attn = tl.where(row_mask, attn, 0.0)

        coeff_v = tl.sum(attn, axis=0) / prefix_len
        dv += coeff_v[:, None] * go[None, :]

        grad_attn_col = tl.sum(v * (go[None, :] / prefix_len), axis=1)
        row_center = tl.sum(attn * grad_attn_col[None, :], axis=1)
        ds = attn * (grad_attn_col[None, :] - row_center[:, None])
        ds = tl.where(active, ds, 0.0)
        dk += (tl.dot(ds, k, input_precision="ieee") + tl.dot(tl.trans(ds), k, input_precision="ieee")) * scale

    tl.store(grad_key + key_ptrs, dk, mask=(offs_s[:, None] < seq_len) & (offs_k[None, :] < key_dim))
    tl.store(grad_value + value_ptrs, dv, mask=(offs_s[:, None] < seq_len) & (offs_v[None, :] < value_dim))


class _FusedPoolPrefixAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        if not can_use_fused_pool_prefix_attention(key, value):
            raise ValueError(
                "fused_pool_prefix_attention expects CUDA contiguous-compatible [N, S, D] "
                "fp16/bf16/fp32 tensors with S <= 64 and D <= 256."
            )
        key = key.contiguous()
        value = value.contiguous()
        rows, seq_len, key_dim = key.shape
        value_dim = value.shape[-1]
        block_s = max(16, triton.next_power_of_2(seq_len))
        block_k = max(16, triton.next_power_of_2(key_dim))
        block_v = triton.next_power_of_2(value_dim)
        out = torch.empty_like(value)
        scale = 1.0 / math.sqrt(key_dim)
        _pool_prefix_attn_fwd_kernel[(rows,)](
            key,
            value,
            out,
            seq_len,
            key_dim,
            value_dim,
            scale,
            BLOCK_S=block_s,
            BLOCK_K=block_k,
            BLOCK_V=block_v,
            num_warps=8,
        )
        ctx.save_for_backward(key, value)
        ctx.seq_len = seq_len
        ctx.key_dim = key_dim
        ctx.value_dim = value_dim
        ctx.block_s = block_s
        ctx.block_k = block_k
        ctx.block_v = block_v
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        key, value = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        grad_key = torch.empty_like(key)
        grad_value = torch.empty_like(value)
        _pool_prefix_attn_bwd_kernel[(key.shape[0],)](
            key,
            value,
            grad_out,
            grad_key,
            grad_value,
            ctx.seq_len,
            ctx.key_dim,
            ctx.value_dim,
            ctx.scale,
            BLOCK_S=ctx.block_s,
            BLOCK_K=ctx.block_k,
            BLOCK_V=ctx.block_v,
            num_warps=8,
        )
        return grad_key, grad_value


def can_use_fused_pool_prefix_attention(key: torch.Tensor, value: torch.Tensor) -> bool:
    if not key.is_cuda or not value.is_cuda:
        return False
    if key.dim() != 3 or value.dim() != 3:
        return False
    if key.shape[:2] != value.shape[:2]:
        return False
    if key.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if value.dtype != key.dtype:
        return False
    return key.shape[1] <= 64 and key.shape[2] <= 256 and value.shape[2] <= 256


def fused_pool_prefix_attention(key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    return _FusedPoolPrefixAttention.apply(key, value)


def reference_pool_prefix_attention(key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    scale = key.shape[-1] ** -0.5
    scores = torch.matmul(key.float(), key.float().transpose(-1, -2)) * scale
    outs = []
    for prefix_len in range(1, key.shape[1] + 1):
        attn = torch.softmax(scores[:, :prefix_len, :prefix_len], dim=-1).to(value.dtype)
        attended = torch.matmul(attn, value[:, :prefix_len])
        outs.append(attended.float().mean(dim=1).to(value.dtype))
    return torch.stack(outs, dim=1)
