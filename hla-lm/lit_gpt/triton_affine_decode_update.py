from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


@triton.jit
def _affine_decode_rank1_update_kernel(
    key,
    value,
    beta,
    g,
    chunk_a,
    chunk_b,
    stride_sb,
    stride_sh,
    stride_sd,
    num_heads: tl.constexpr,
    key_dim: tl.constexpr,
    value_dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_B: tl.constexpr,
    NORMALIZE_KEY: tl.constexpr,
):
    """Update the active affine state without materializing the token matrix.

    The token transition is ``decay * (I - beta * k k^T)``.  Applying that
    rank-one matrix to every A/B column only needs one dot product per column.
    Programs tile output columns so a batch-one decode still exposes enough
    independent CTAs to occupy the GPU.
    """

    row = tl.program_id(0)
    block = tl.program_id(1)
    offs_d = tl.arange(0, BLOCK_D)
    offs_n = block * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_d = offs_d < key_dim
    width: tl.constexpr = value_dim if IS_B else key_dim
    mask_n = offs_n < width

    kv = tl.load(key + row * key_dim + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    if NORMALIZE_KEY:
        kv *= tl.rsqrt(tl.sum(kv * kv, axis=0) + 1e-6)
    bt = tl.load(beta + row).to(tl.float32)
    dt = tl.exp(tl.load(g + row).to(tl.float32))

    state = chunk_b if IS_B else chunk_a
    state_width: tl.constexpr = value_dim if IS_B else key_dim
    # key/value are contiguous flattened [batch, heads, dim], whereas the
    # active A/B views can retain the prefill tensor's non-contiguous head
    # stride.  Recover batch/head from the shared row index and honor it.
    batch_idx = row // num_heads
    head_idx = row - batch_idx * num_heads
    state_ptrs = (
        state
        + batch_idx * stride_sb
        + head_idx * stride_sh
        + offs_d[:, None] * stride_sd
        + offs_n[None, :]
    )
    old = tl.load(
        state_ptrs,
        mask=mask_d[:, None] & mask_n[None, :],
        other=0.0,
    ).to(tl.float32)
    projected = tl.sum(kv[:, None] * old, axis=0)
    updated = dt * (old - bt * kv[:, None] * projected[None, :])
    if IS_B:
        vv = tl.load(
            value + row * value_dim + offs_n,
            mask=mask_n,
            other=0.0,
        ).to(tl.float32)
        updated += bt * kv[:, None] * vv[None, :]
    tl.store(
        state_ptrs,
        updated,
        mask=mask_d[:, None] & mask_n[None, :],
    )


@triton.jit
def _affine_decode_joint_rank1_update_kernel(
    key,
    value,
    beta,
    g,
    joint,
    stride_sb,
    stride_sh,
    stride_sd,
    num_heads: tl.constexpr,
    key_dim: tl.constexpr,
    value_dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NORMALIZE_KEY: tl.constexpr,
):
    """Update joint [B | A] columns in one launch."""

    row = tl.program_id(0)
    block = tl.program_id(1)
    offs_d = tl.arange(0, BLOCK_D)
    offs_n = block * BLOCK_N + tl.arange(0, BLOCK_N)
    joint_dim: tl.constexpr = value_dim + key_dim
    mask_d = offs_d < key_dim
    mask_n = offs_n < joint_dim

    kv = tl.load(key + row * key_dim + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    if NORMALIZE_KEY:
        kv *= tl.rsqrt(tl.sum(kv * kv, axis=0) + 1e-6)
    bt = tl.load(beta + row).to(tl.float32)
    dt = tl.exp(tl.load(g + row).to(tl.float32))
    batch_idx = row // num_heads
    head_idx = row - batch_idx * num_heads
    state_ptrs = (
        joint
        + batch_idx * stride_sb
        + head_idx * stride_sh
        + offs_d[:, None] * stride_sd
        + offs_n[None, :]
    )
    old = tl.load(
        state_ptrs,
        mask=mask_d[:, None] & mask_n[None, :],
        other=0.0,
    ).to(tl.float32)
    projected = tl.sum(kv[:, None] * old, axis=0)
    updated = dt * (old - bt * kv[:, None] * projected[None, :])
    is_b = offs_n < value_dim
    vv = tl.load(
        value + row * value_dim + offs_n,
        mask=mask_n & is_b,
        other=0.0,
    ).to(tl.float32)
    updated += tl.where(
        is_b[None, :], bt * kv[:, None] * vv[None, :], 0.0
    )
    tl.store(
        state_ptrs,
        updated,
        mask=mask_d[:, None] & mask_n[None, :],
    )


@triton.jit
def _affine_decode_joint_rank1_update_route_kernel(
    key,
    value,
    beta,
    g,
    joint,
    route_key,
    route_sums,
    route_counts,
    current_chunk,
    current_bucket,
    stride_sb,
    stride_sh,
    stride_sd,
    stride_rsb,
    stride_rsh,
    stride_rsc,
    stride_rsp,
    stride_rcb,
    stride_rcc,
    stride_rcp,
    num_heads: tl.constexpr,
    key_dim: tl.constexpr,
    value_dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NORMALIZE_KEY: tl.constexpr,
):
    """Joint affine update plus direct-pool cache append in one launch."""

    row = tl.program_id(0)
    block = tl.program_id(1)
    offs_d = tl.arange(0, BLOCK_D)
    offs_n = block * BLOCK_N + tl.arange(0, BLOCK_N)
    joint_dim: tl.constexpr = value_dim + key_dim
    mask_d = offs_d < key_dim
    mask_n = offs_n < joint_dim
    batch_idx = row // num_heads
    head_idx = row - batch_idx * num_heads

    kv = tl.load(key + row * key_dim + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    if NORMALIZE_KEY:
        kv *= tl.rsqrt(tl.sum(kv * kv, axis=0) + 1e-6)
    bt = tl.load(beta + row).to(tl.float32)
    dt = tl.exp(tl.load(g + row).to(tl.float32))
    state_ptrs = (
        joint
        + batch_idx * stride_sb
        + head_idx * stride_sh
        + offs_d[:, None] * stride_sd
        + offs_n[None, :]
    )
    old = tl.load(
        state_ptrs,
        mask=mask_d[:, None] & mask_n[None, :],
        other=0.0,
    ).to(tl.float32)
    projected = tl.sum(kv[:, None] * old, axis=0)
    updated = dt * (old - bt * kv[:, None] * projected[None, :])
    is_b = offs_n < value_dim
    vv = tl.load(
        value + row * value_dim + offs_n,
        mask=mask_n & is_b,
        other=0.0,
    ).to(tl.float32)
    updated += tl.where(is_b[None, :], bt * kv[:, None] * vv[None, :], 0.0)
    tl.store(state_ptrs, updated, mask=mask_d[:, None] & mask_n[None, :])

    route_mask = offs_n < key_dim
    route_ptrs = (
        route_sums
        + batch_idx * stride_rsb
        + head_idx * stride_rsh
        + current_chunk * stride_rsc
        + current_bucket * stride_rsp
        + offs_n
    )
    old_route = tl.load(route_ptrs, mask=route_mask, other=0.0).to(tl.float32)
    new_route = tl.load(
        route_key + row * key_dim + offs_n,
        mask=route_mask,
        other=0.0,
    ).to(tl.float32)
    tl.store(route_ptrs, old_route + new_route, mask=route_mask)

    count_ptr = (
        route_counts
        + batch_idx * stride_rcb
        + current_chunk * stride_rcc
        + current_bucket * stride_rcp
    )
    update_count = (head_idx == 0) & (block == 0)
    count = tl.load(count_ptr, mask=update_count, other=0.0).to(tl.float32)
    tl.store(count_ptr, count + 1.0, mask=update_count)


def can_use_fused_affine_decode_update(
    key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    chunk_a: torch.Tensor,
    chunk_b: torch.Tensor,
) -> bool:
    if os.environ.get("GDN_DISABLE_TRITON_AFFINE_DECODE_UPDATE", ""):
        return False
    if not all(x.is_cuda for x in (key, value, beta, g, chunk_a, chunk_b)):
        return False
    if key.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if value.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if beta.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if g.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if chunk_a.dtype != torch.float32 or chunk_b.dtype != torch.float32:
        return False
    if key.dim() != 3 or value.dim() != 3:
        return False
    if beta.shape != key.shape[:2] or g.shape != key.shape[:2]:
        return False
    batch, heads, key_dim = key.shape
    value_dim = value.shape[-1]
    if value.shape[:2] != (batch, heads):
        return False
    if chunk_a.shape != (batch, heads, key_dim, key_dim):
        return False
    if chunk_b.shape != (batch, heads, key_dim, value_dim):
        return False
    if key_dim <= 0 or key_dim > 256 or value_dim <= 0 or value_dim > 256:
        return False
    return (
        key.is_contiguous()
        and value.is_contiguous()
        and beta.is_contiguous()
        and g.is_contiguous()
        and chunk_a.stride(-1) == 1
        and chunk_b.stride(-1) == 1
    )


def fused_affine_decode_update_(
    key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    chunk_a: torch.Tensor,
    chunk_b: torch.Tensor,
    *,
    normalize_key: bool = True,
) -> None:
    """In-place exact rank-one update of the active chunk's A/B summaries."""

    if not can_use_fused_affine_decode_update(
        key, value, beta, g, chunk_a, chunk_b
    ):
        raise ValueError("unsupported fused affine decode update inputs")
    batch, heads, key_dim = key.shape
    value_dim = value.shape[-1]
    block_d = triton.next_power_of_2(key_dim)
    block_n = 32
    _affine_decode_rank1_update_kernel[
        (batch * heads, triton.cdiv(key_dim, block_n))
    ](
        key, value, beta, g, chunk_a, chunk_b,
        chunk_a.stride(0), chunk_a.stride(1), chunk_a.stride(2),
        heads, key_dim, value_dim,
        BLOCK_D=block_d,
        BLOCK_N=block_n,
        IS_B=False,
        NORMALIZE_KEY=bool(normalize_key),
        num_warps=4,
    )
    _affine_decode_rank1_update_kernel[
        (batch * heads, triton.cdiv(value_dim, block_n))
    ](
        key, value, beta, g, chunk_a, chunk_b,
        chunk_b.stride(0), chunk_b.stride(1), chunk_b.stride(2),
        heads, key_dim, value_dim,
        BLOCK_D=block_d,
        BLOCK_N=block_n,
        IS_B=True,
        NORMALIZE_KEY=bool(normalize_key),
        num_warps=4,
    )


def can_use_fused_affine_decode_joint_update(
    key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    joint: torch.Tensor,
) -> bool:
    if os.environ.get("GDN_DISABLE_TRITON_AFFINE_DECODE_UPDATE", ""):
        return False
    if not all(x.is_cuda for x in (key, value, beta, g, joint)):
        return False
    if key.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if value.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if beta.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if g.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if joint.dtype != torch.float32 or key.dim() != 3 or value.dim() != 3:
        return False
    batch, heads, key_dim = key.shape
    value_dim = value.shape[-1]
    if value.shape[:2] != (batch, heads):
        return False
    if beta.shape != (batch, heads) or g.shape != (batch, heads):
        return False
    if joint.shape != (batch, heads, key_dim, value_dim + key_dim):
        return False
    if key_dim <= 0 or key_dim > 256 or value_dim <= 0 or value_dim > 256:
        return False
    return (
        key.is_contiguous()
        and value.is_contiguous()
        and beta.is_contiguous()
        and g.is_contiguous()
        and joint.stride(-1) == 1
    )


def fused_affine_decode_joint_update_(
    key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    joint: torch.Tensor,
    *,
    normalize_key: bool = True,
) -> None:
    if not can_use_fused_affine_decode_joint_update(key, value, beta, g, joint):
        raise ValueError("unsupported fused joint affine decode update inputs")
    batch, heads, key_dim = key.shape
    value_dim = value.shape[-1]
    block_n = 32
    _affine_decode_joint_rank1_update_kernel[
        (batch * heads, triton.cdiv(value_dim + key_dim, block_n))
    ](
        key,
        value,
        beta,
        g,
        joint,
        joint.stride(0),
        joint.stride(1),
        joint.stride(2),
        heads,
        key_dim,
        value_dim,
        BLOCK_D=triton.next_power_of_2(key_dim),
        BLOCK_N=block_n,
        NORMALIZE_KEY=bool(normalize_key),
        num_warps=4,
    )


def fused_affine_decode_joint_update_route_(
    key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    joint: torch.Tensor,
    route_key: torch.Tensor,
    route_sums: torch.Tensor,
    route_counts: torch.Tensor,
    current_chunk: int,
    current_bucket: int,
    *,
    normalize_key: bool = True,
) -> None:
    if not can_use_fused_affine_decode_joint_update(key, value, beta, g, joint):
        raise ValueError("unsupported fused joint affine decode update inputs")
    batch, heads, key_dim = key.shape
    value_dim = value.shape[-1]
    if (
        batch != 1
        or route_key.shape != (batch, heads, key_dim)
        or route_sums.dim() != 5
        or route_sums.shape[:2] != (batch, heads)
        or route_sums.shape[-1] != key_dim
        or route_counts.shape
        != (batch, route_sums.shape[2], route_sums.shape[3])
        or not all(x.is_cuda for x in (route_key, route_sums, route_counts))
        or route_sums.dtype not in (torch.bfloat16, torch.float16, torch.float32)
        or route_counts.dtype != torch.float32
        or not (0 <= current_chunk < route_sums.shape[2])
        or not (0 <= current_bucket < route_sums.shape[3])
    ):
        raise ValueError("unsupported fused affine/route cache inputs")
    block_n = 32
    _affine_decode_joint_rank1_update_route_kernel[
        (batch * heads, triton.cdiv(value_dim + key_dim, block_n))
    ](
        key,
        value,
        beta,
        g,
        joint,
        route_key.contiguous(),
        route_sums,
        route_counts,
        int(current_chunk),
        int(current_bucket),
        joint.stride(0),
        joint.stride(1),
        joint.stride(2),
        route_sums.stride(0),
        route_sums.stride(1),
        route_sums.stride(2),
        route_sums.stride(3),
        route_counts.stride(0),
        route_counts.stride(1),
        route_counts.stride(2),
        heads,
        key_dim,
        value_dim,
        BLOCK_D=triton.next_power_of_2(key_dim),
        BLOCK_N=block_n,
        NORMALIZE_KEY=bool(normalize_key),
        num_warps=4,
    )
