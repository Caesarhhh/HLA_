from __future__ import annotations
import os
import torch
import triton
import triton.language as tl


@triton.jit
def _affine_decode_readout_kernel(
    q,
    key,
    value,
    beta,
    g,
    current_gate,
    weights,
    chunk_idx,
    current_a,
    current_b,
    chunk_a,
    chunk_b,
    out,
    stride_cab,
    stride_cah,
    stride_cad,
    stride_cbb,
    stride_cbh,
    stride_cbd,
    stride_ab,
    stride_ah,
    stride_ac,
    stride_ad,
    stride_bb,
    stride_bh,
    stride_bc,
    stride_bd,
    q_scale: tl.constexpr,
    normalize_q: tl.constexpr,
    use_chunk_idx: tl.constexpr,
    skip_zero_weights: tl.constexpr,
    num_heads: tl.constexpr,
    num_chunks,
    key_dim: tl.constexpr,
    value_dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_N: tl.constexpr,
    UPDATE_CURRENT: tl.constexpr,
):
    row = tl.program_id(0)
    b = row // num_heads
    h = row - b * num_heads
    offs_d = tl.arange(0, BLOCK_D)
    offs_v = tl.arange(0, BLOCK_V)
    mask_d = offs_d < key_dim
    mask_v = offs_v < value_dim
    q_base = row * key_dim
    qv = tl.load(q + q_base + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    if normalize_q:
        qv *= tl.rsqrt(tl.sum(qv * qv, axis=0) + 1e-06) * q_scale
    cg = tl.load(current_gate + row).to(tl.float32)
    if UPDATE_CURRENT:
        kv = tl.load(key + row * key_dim + offs_d, mask=mask_d, other=0.0).to(
            tl.float32
        )
        kv *= tl.rsqrt(tl.sum(kv * kv, axis=0) + 1e-06)
        bt = tl.load(beta + row).to(tl.float32)
        dt = tl.exp(tl.load(g + row).to(tl.float32))
        offs_n = tl.arange(0, BLOCK_N)
        block = 0
        while block < tl.cdiv(value_dim + key_dim, BLOCK_N):
            joint_n = block * BLOCK_N + offs_n
            is_b = joint_n < value_dim
            local_n = tl.where(is_b, joint_n, joint_n - value_dim)
            valid_n = joint_n < value_dim + key_dim
            state_ptrs = tl.where(
                is_b[None, :],
                current_b
                + b * stride_cbb
                + h * stride_cbh
                + offs_d[:, None] * stride_cbd
                + local_n[None, :],
                current_a
                + b * stride_cab
                + h * stride_cah
                + offs_d[:, None] * stride_cad
                + local_n[None, :],
            )
            old = tl.load(
                state_ptrs, mask=mask_d[:, None] & valid_n[None, :], other=0.0
            ).to(tl.float32)
            projected = tl.sum(kv[:, None] * old, axis=0)
            updated = dt * (old - bt * kv[:, None] * projected[None, :])
            vv = tl.load(
                value + row * value_dim + local_n, mask=valid_n & is_b, other=0.0
            ).to(tl.float32)
            updated += tl.where(is_b[None, :], bt * kv[:, None] * vv[None, :], 0.0)
            tl.store(state_ptrs, updated, mask=mask_d[:, None] & valid_n[None, :])
            block += 1
        tl.debug_barrier()
    a_ptrs = (
        b * stride_cab + h * stride_cah + offs_d[:, None] * stride_cad + offs_d[None, :]
    )
    cur_a = tl.load(
        current_a + a_ptrs, mask=mask_d[:, None] & mask_d[None, :], other=0.0
    ).to(tl.float32)
    b_ptrs = (
        b * stride_cbb + h * stride_cbh + offs_d[:, None] * stride_cbd + offs_v[None, :]
    )
    cur_b = tl.load(
        current_b + b_ptrs, mask=mask_d[:, None] & mask_v[None, :], other=0.0
    ).to(tl.float32)
    outv = cg * tl.sum(qv[:, None] * cur_b, axis=0)
    transformed_current = tl.sum(cur_a * qv[:, None], axis=0)
    r = (1.0 - cg) * qv + cg * transformed_current
    if use_chunk_idx:
        limit = tl.load(chunk_idx + b)
    else:
        limit = num_chunks - 1
    if skip_zero_weights:
        offs_c = tl.arange(0, BLOCK_C)
        valid_c = offs_c < limit
        history = tl.load(
            weights + row * num_chunks + offs_c, mask=valid_c, other=0.0
        ).to(tl.float32)
        remaining = valid_c & (history != 0.0)
        remaining_count = tl.sum(remaining.to(tl.int32), axis=0)
        while remaining_count > 0:
            prev = tl.max(tl.where(remaining, offs_c, -1), axis=0)
            gate = tl.sum(tl.where(offs_c == prev, history, 0.0), axis=0)
            cb_ptrs = (
                b * stride_bb
                + h * stride_bh
                + prev * stride_bc
                + offs_d[:, None] * stride_bd
                + offs_v[None, :]
            )
            cb = tl.load(
                chunk_b + cb_ptrs, mask=mask_d[:, None] & mask_v[None, :], other=0.0
            ).to(tl.float32)
            outv += gate * tl.sum(r[:, None] * cb, axis=0)
            ca_ptrs = (
                b * stride_ab
                + h * stride_ah
                + prev * stride_ac
                + offs_d[:, None] * stride_ad
                + offs_d[None, :]
            )
            ca = tl.load(
                chunk_a + ca_ptrs, mask=mask_d[:, None] & mask_d[None, :], other=0.0
            ).to(tl.float32)
            transformed_r = tl.sum(ca * r[:, None], axis=0)
            r = (1.0 - gate) * r + gate * transformed_r
            remaining = remaining & (offs_c != prev)
            remaining_count -= 1
    else:
        prev = limit - 1
        while prev >= 0:
            active = prev < limit
            gate = tl.load(weights + row * num_chunks + prev).to(tl.float32)
            cb_ptrs = (
                b * stride_bb
                + h * stride_bh
                + prev * stride_bc
                + offs_d[:, None] * stride_bd
                + offs_v[None, :]
            )
            cb = tl.load(
                chunk_b + cb_ptrs, mask=mask_d[:, None] & mask_v[None, :], other=0.0
            ).to(tl.float32)
            out_update = gate * tl.sum(r[:, None] * cb, axis=0)
            ca_ptrs = (
                b * stride_ab
                + h * stride_ah
                + prev * stride_ac
                + offs_d[:, None] * stride_ad
                + offs_d[None, :]
            )
            ca = tl.load(
                chunk_a + ca_ptrs, mask=mask_d[:, None] & mask_d[None, :], other=0.0
            ).to(tl.float32)
            transformed_r = tl.sum(ca * r[:, None], axis=0)
            r_update = (1.0 - gate) * r + gate * transformed_r
            outv = tl.where(active, outv + out_update, outv)
            r = tl.where(active, r_update, r)
            prev -= 1
    out_base = row * value_dim
    tl.store(out + out_base + offs_v, outv, mask=mask_v)


def can_use_fused_affine_decode_readout(
    q: torch.Tensor,
    current_gate: torch.Tensor,
    weights: torch.Tensor,
    chunk_idx: torch.Tensor | None,
    current_a: torch.Tensor,
    current_b: torch.Tensor,
    chunk_a: torch.Tensor,
    chunk_b: torch.Tensor,
) -> bool:
    if os.environ.get("GDN_DISABLE_TRITON_AFFINE_DECODE_READOUT", ""):
        return False
    tensors = (q, current_gate, weights, current_a, current_b, chunk_a, chunk_b)
    if not all((t.is_cuda for t in tensors)):
        return False
    if chunk_idx is not None and (not chunk_idx.is_cuda):
        return False
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if current_gate.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if weights.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if (
        q.dim() != 3
        or current_gate.shape != q.shape[:2]
        or weights.shape[:2] != q.shape[:2]
    ):
        return False
    if chunk_idx is not None and (
        chunk_idx.dim() != 1 or chunk_idx.shape[0] != q.shape[0]
    ):
        return False
    if current_a.shape != (*q.shape[:2], q.shape[-1], q.shape[-1]):
        return False
    if current_b.shape[:3] != (*q.shape[:2], q.shape[-1]):
        return False
    if chunk_a.shape != (*q.shape[:2], weights.shape[-1], q.shape[-1], q.shape[-1]):
        return False
    if chunk_b.shape != (
        *q.shape[:2],
        weights.shape[-1],
        q.shape[-1],
        current_b.shape[-1],
    ):
        return False
    max_chunks = (
        256 if os.environ.get("GDN_ENABLE_WIDE_AFFINE_DECODE_READOUT") == "1" else 128
    )
    return (
        q.shape[-1] <= 256
        and current_b.shape[-1] <= 256
        and (weights.shape[-1] <= max_chunks)
    )


def fused_affine_decode_readout(
    q: torch.Tensor,
    current_gate: torch.Tensor,
    weights: torch.Tensor,
    chunk_idx: torch.Tensor | None,
    current_a: torch.Tensor,
    current_b: torch.Tensor,
    chunk_a: torch.Tensor,
    chunk_b: torch.Tensor,
    *,
    normalize_q: bool = False,
    q_scale: float = 1.0,
    skip_zero_weights: bool = False,
    num_warps: int = 4,
    out: torch.Tensor | None = None,
    update_key: torch.Tensor | None = None,
    update_value: torch.Tensor | None = None,
    update_beta: torch.Tensor | None = None,
    update_g: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if not can_use_fused_affine_decode_readout(
        q, current_gate, weights, chunk_idx, current_a, current_b, chunk_a, chunk_b
    ):
        raise ValueError("unsupported affine decode readout inputs")
    q = q.contiguous()
    if num_warps not in (4, 8, 16):
        raise ValueError("num_warps must be 4, 8, or 16")
    current_gate = current_gate.contiguous()
    weights = weights.contiguous()
    if chunk_idx is not None:
        chunk_idx = chunk_idx.contiguous()
    batch_size, num_heads, key_dim = q.shape
    num_chunks = weights.shape[-1]
    value_dim = current_b.shape[-1]
    if output_dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise ValueError("unsupported affine decode-readout output dtype")
    update_tensors = (update_key, update_value, update_beta, update_g)
    update_current = any((tensor is not None for tensor in update_tensors))
    if update_current:
        if not all((isinstance(tensor, torch.Tensor) for tensor in update_tensors)):
            raise ValueError("all affine update tensors must be provided together")
        if update_key.shape != (batch_size, num_heads, key_dim):
            raise ValueError("update_key has incompatible shape")
        if update_value.shape != (batch_size, num_heads, value_dim):
            raise ValueError("update_value has incompatible shape")
        if update_beta.shape != (batch_size, num_heads) or update_g.shape != (
            batch_size,
            num_heads,
        ):
            raise ValueError("update beta/g have incompatible shape")
        if not all(
            (tensor.is_cuda and tensor.is_contiguous() for tensor in update_tensors)
        ):
            raise ValueError("affine update tensors must be contiguous CUDA tensors")
        key_arg, value_arg, beta_arg, g_arg = update_tensors
    else:
        key_arg = value_arg = q
        beta_arg = g_arg = current_gate
    expected_shape = (batch_size, num_heads, value_dim)
    if out is None:
        out = torch.empty(expected_shape, device=q.device, dtype=output_dtype)
    elif (
        out.shape != expected_shape
        or out.device != q.device
        or out.dtype != output_dtype
        or (not out.is_contiguous())
    ):
        raise ValueError(
            "affine decode-readout output buffer has incompatible metadata"
        )
    _affine_decode_readout_kernel[batch_size * num_heads,](
        q,
        key_arg,
        value_arg,
        beta_arg,
        g_arg,
        current_gate,
        weights,
        q if chunk_idx is None else chunk_idx,
        current_a,
        current_b,
        chunk_a,
        chunk_b,
        out,
        current_a.stride(0),
        current_a.stride(1),
        current_a.stride(2),
        current_b.stride(0),
        current_b.stride(1),
        current_b.stride(2),
        chunk_a.stride(0),
        chunk_a.stride(1),
        chunk_a.stride(2),
        chunk_a.stride(3),
        chunk_b.stride(0),
        chunk_b.stride(1),
        chunk_b.stride(2),
        chunk_b.stride(3),
        float(q_scale),
        bool(normalize_q),
        chunk_idx is not None,
        bool(skip_zero_weights),
        num_heads,
        num_chunks,
        key_dim,
        value_dim,
        BLOCK_D=triton.next_power_of_2(key_dim),
        BLOCK_V=triton.next_power_of_2(value_dim),
        BLOCK_C=triton.next_power_of_2(num_chunks),
        BLOCK_N=32,
        UPDATE_CURRENT=update_current,
        num_warps=num_warps,
    )
    return out
