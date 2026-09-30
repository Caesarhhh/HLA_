from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


@triton.jit
def _headwise_decode_linear_kernel(
    x,
    weight,
    out,
    stride_xb,
    stride_xh,
    stride_wh,
    stride_wo,
    stride_wi,
    stride_ob,
    stride_oh,
    IN_DIM: tl.constexpr,
    OUT_DIM: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_O: tl.constexpr,
):
    row = tl.program_id(0)
    batch_idx = tl.program_id(1)
    head_idx = row
    offs_i = tl.arange(0, BLOCK_I)
    offs_o = tl.arange(0, BLOCK_O)
    xv = tl.load(
        x + batch_idx * stride_xb + head_idx * stride_xh + offs_i,
        mask=offs_i < IN_DIM,
        other=0.0,
    ).to(tl.float32)
    w = tl.load(
        weight
        + head_idx * stride_wh
        + offs_o[:, None] * stride_wo
        + offs_i[None, :] * stride_wi,
        mask=(offs_o[:, None] < OUT_DIM) & (offs_i[None, :] < IN_DIM),
        other=0.0,
    ).to(tl.float32)
    y = tl.sum(w * xv[None, :], axis=1)
    tl.store(
        out + batch_idx * stride_ob + head_idx * stride_oh + offs_o,
        y,
        mask=offs_o < OUT_DIM,
    )


def can_use_fused_headwise_decode_linear(x: torch.Tensor, weight: torch.Tensor) -> bool:
    if os.environ.get("GDN_DISABLE_TRITON_HEADWISE_DECODE_LINEAR", ""):
        return False
    return (
        x.is_cuda
        and weight.is_cuda
        and x.dim() == 4
        and x.shape[2] == 1
        and weight.dim() == 3
        and x.shape[1] == weight.shape[0]
        and x.shape[-1] == weight.shape[-1]
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and weight.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and x.shape[-1] <= 256
        and weight.shape[-2] <= 256
        and x.stride(-1) == weight.stride(-1) == 1
    )


def fused_headwise_decode_linear(
    x: torch.Tensor, weight: torch.Tensor, *, num_warps: int = 16
) -> torch.Tensor:
    if not can_use_fused_headwise_decode_linear(x, weight):
        raise ValueError("unsupported headwise decode linear inputs")
    if num_warps not in (4, 8, 16):
        raise ValueError("num_warps must be 4, 8, or 16")
    batch, heads, _, in_dim = x.shape
    out_dim = weight.shape[-2]
    out = torch.empty(batch, heads, 1, out_dim, device=x.device, dtype=x.dtype)
    _headwise_decode_linear_kernel[(heads, batch)](
        x, weight, out,
        x.stride(0), x.stride(1), weight.stride(0), weight.stride(1), weight.stride(2),
        out.stride(0), out.stride(1), in_dim, out_dim,
        BLOCK_I=triton.next_power_of_2(in_dim), BLOCK_O=triton.next_power_of_2(out_dim),
        num_warps=num_warps,
    )
    return out
