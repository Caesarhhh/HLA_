from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


@triton.jit
def _gdn_decode_router_gates_kernel(
    route_q,
    pooled,
    route_counts,
    gates,
    stride_rqb,
    stride_rqh,
    stride_pb,
    stride_ph,
    stride_pc,
    stride_pp,
    router_scale: tl.constexpr,
    sigmoid_bias: tl.constexpr,
    inv_temperature: tl.constexpr,
    use_logmean: tl.constexpr,
    POOLED_ARE_SUMS: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    NUM_BUCKETS: tl.constexpr,
    KEY_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    program = tl.program_id(0)
    source = program % NUM_CHUNKS
    row = program // NUM_CHUNKS
    batch_idx = row // NUM_HEADS
    head_idx = row - batch_idx * NUM_HEADS
    offs_d = tl.arange(0, BLOCK_D)
    offs_p = tl.arange(0, BLOCK_P)
    mask_d = offs_d < KEY_DIM
    counts = tl.load(
        route_counts + (batch_idx * NUM_CHUNKS + source) * NUM_BUCKETS + offs_p,
        mask=offs_p < NUM_BUCKETS,
        other=0.0,
    )
    valid_p = (offs_p < NUM_BUCKETS) & (counts > 0)
    route_query = tl.load(
        route_q + batch_idx * stride_rqb + head_idx * stride_rqh + offs_d,
        mask=mask_d,
        other=0.0,
    ).to(tl.float32)
    pooled_values = tl.load(
        pooled
        + batch_idx * stride_pb
        + head_idx * stride_ph
        + source * stride_pc
        + offs_p[None, :] * stride_pp
        + offs_d[:, None],
        mask=mask_d[:, None] & valid_p[None, :],
        other=0.0,
    ).to(tl.float32)
    if POOLED_ARE_SUMS:
        pooled_values /= tl.where(valid_p, counts, 1.0)[None, :]
    bucket_scores = tl.sum(route_query[:, None] * pooled_values, axis=0) * router_scale
    bucket_scores = tl.where(valid_p, bucket_scores, -float("inf"))
    score_max = tl.max(bucket_scores, axis=0)
    score = score_max + tl.log(tl.sum(tl.exp(bucket_scores - score_max), axis=0))
    if use_logmean:
        score -= tl.log(tl.sum(valid_p.to(tl.float32), axis=0))
    tl.store(gates + row * NUM_CHUNKS + source, tl.sigmoid((score + sigmoid_bias) * inv_temperature))


@triton.jit
def _gdn_affine_decode_current_tiled_kernel(
    q,
    current_a,
    current_b,
    readout_out,
    out,
    stride_cb,
    stride_ch,
    stride_cd,
    NUM_HEADS: tl.constexpr,
    KEY_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_O: tl.constexpr,
):
    tile = tl.program_id(0)
    tiles_per_row: tl.constexpr = tl.cdiv(KEY_DIM, BLOCK_O)
    row = tile // tiles_per_row
    output_tile = tile - row * tiles_per_row
    offs_d = tl.arange(0, BLOCK_D)
    offs_o = output_tile * BLOCK_O + tl.arange(0, BLOCK_O)
    mask_d = offs_d < KEY_DIM
    mask_o = offs_o < VALUE_DIM
    batch_idx = row // NUM_HEADS
    head_idx = row - batch_idx * NUM_HEADS
    readout = tl.load(q + row * KEY_DIM + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    cur_b = tl.load(
        current_b
        + batch_idx * stride_cb
        + head_idx * stride_ch
        + offs_d[:, None] * stride_cd
        + offs_o[None, :],
        mask=mask_d[:, None] & mask_o[None, :],
        other=0.0,
    ).to(tl.float32)
    cur_a = tl.load(
        current_a
        + batch_idx * stride_cb
        + head_idx * stride_ch
        + offs_d[:, None] * stride_cd
        + offs_o[None, :],
        mask=mask_d[:, None] & mask_o[None, :],
        other=0.0,
    ).to(tl.float32)
    tl.store(out + row * VALUE_DIM + offs_o, tl.sum(readout[:, None] * cur_b, axis=0), mask=mask_o)
    tl.store(readout_out + row * KEY_DIM + offs_o, tl.sum(readout[:, None] * cur_a, axis=0), mask=mask_o)


@triton.jit
def _gdn_affine_decode_source_tiled_kernel(
    readout_in,
    readout_out,
    gates,
    chunk_idx,
    chunk_a,
    chunk_b,
    out,
    stride_ab,
    stride_ah,
    stride_ac,
    stride_ad,
    stride_bb,
    stride_bh,
    stride_bc,
    stride_bd,
    SOURCE: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    KEY_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_O: tl.constexpr,
):
    tile = tl.program_id(0)
    tiles_per_row: tl.constexpr = tl.cdiv(KEY_DIM, BLOCK_O)
    row = tile // tiles_per_row
    output_tile = tile - row * tiles_per_row
    batch_idx = row // NUM_HEADS
    head_idx = row - batch_idx * NUM_HEADS
    offs_d = tl.arange(0, BLOCK_D)
    offs_o = output_tile * BLOCK_O + tl.arange(0, BLOCK_O)
    mask_d = offs_d < KEY_DIM
    mask_o = offs_o < VALUE_DIM
    active = SOURCE < tl.load(chunk_idx + batch_idx)
    readout = tl.load(readout_in + row * KEY_DIM + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    old_readout = tl.load(readout_in + row * KEY_DIM + offs_o, mask=mask_o, other=0.0).to(tl.float32)
    source_b = tl.load(
        chunk_b
        + batch_idx * stride_bb
        + head_idx * stride_bh
        + SOURCE * stride_bc
        + offs_d[:, None] * stride_bd
        + offs_o[None, :],
        mask=mask_d[:, None] & mask_o[None, :],
        other=0.0,
    ).to(tl.float32)
    contribution = tl.sum(readout[:, None] * source_b, axis=0)
    source_a = tl.load(
        chunk_a
        + batch_idx * stride_ab
        + head_idx * stride_ah
        + SOURCE * stride_ac
        + offs_d[:, None] * stride_ad
        + offs_o[None, :],
        mask=mask_d[:, None] & mask_o[None, :],
        other=0.0,
    ).to(tl.float32)
    transformed = tl.sum(readout[:, None] * source_a, axis=0)
    gate = tl.load(gates + row * NUM_CHUNKS + SOURCE)
    old_output = tl.load(out + row * VALUE_DIM + offs_o, mask=mask_o, other=0.0).to(tl.float32)
    tl.store(
        out + row * VALUE_DIM + offs_o,
        tl.where(active, old_output + gate * contribution, old_output),
        mask=mask_o,
    )
    tl.store(
        readout_out + row * KEY_DIM + offs_o,
        tl.where(active, old_readout + gate * (transformed - old_readout), old_readout),
        mask=mask_o,
    )






@triton.jit
def _gdn_affine_decode_mix_kernel(
    route_q,
    pooled,
    route_counts,
    q,
    chunk_idx,
    current_a,
    current_b,
    chunk_a,
    chunk_b,
    out,
    stride_rqb,
    stride_rqh,
    stride_pb,
    stride_ph,
    stride_pc,
    stride_pp,
    stride_cb,
    stride_ch,
    stride_cd,
    stride_ab,
    stride_ah,
    stride_ac,
    stride_ad,
    stride_bb,
    stride_bh,
    stride_bc,
    stride_bd,
    router_scale: tl.constexpr,
    sigmoid_bias: tl.constexpr,
    inv_temperature: tl.constexpr,
    use_logmean: tl.constexpr,
    POOLED_ARE_SUMS: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    NUM_BUCKETS: tl.constexpr,
    KEY_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    row = tl.program_id(0)
    batch_idx = row // NUM_HEADS
    head_idx = row - batch_idx * NUM_HEADS
    offs_d = tl.arange(0, BLOCK_D)
    offs_v = tl.arange(0, BLOCK_V)
    offs_p = tl.arange(0, BLOCK_P)
    mask_d = offs_d < KEY_DIM
    mask_v = offs_v < VALUE_DIM
    current = tl.load(chunk_idx + batch_idx)

    route_query = tl.load(
        route_q + batch_idx * stride_rqb + head_idx * stride_rqh + offs_d,
        mask=mask_d,
        other=0.0,
    ).to(tl.float32)
    readout = tl.load(q + row * KEY_DIM + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    cur_b = tl.load(
        current_b
        + batch_idx * stride_cb
        + head_idx * stride_ch
        + offs_d[:, None] * stride_cd
        + offs_v[None, :],
        mask=mask_d[:, None] & mask_v[None, :],
        other=0.0,
    ).to(tl.float32)
    cur_a = tl.load(
        current_a
        + batch_idx * stride_cb
        + head_idx * stride_ch
        + offs_d[:, None] * stride_cd
        + offs_d[None, :],
        mask=mask_d[:, None] & mask_d[None, :],
        other=0.0,
    ).to(tl.float32)
    output = tl.sum(readout[:, None] * cur_b, axis=0)
    readout = tl.sum(cur_a * readout[:, None], axis=0)

    source = NUM_CHUNKS - 1
    while source >= 0:
        active = source < current
        counts = tl.load(
            route_counts + (batch_idx * NUM_CHUNKS + source) * NUM_BUCKETS + offs_p,
            mask=offs_p < NUM_BUCKETS,
            other=0.0,
        )
        valid_p = (offs_p < NUM_BUCKETS) & (counts > 0)
        pooled_values = tl.load(
            pooled
            + batch_idx * stride_pb
            + head_idx * stride_ph
            + source * stride_pc
            + offs_p[None, :] * stride_pp
            + offs_d[:, None],
            mask=mask_d[:, None] & valid_p[None, :],
            other=0.0,
        ).to(tl.float32)
        if POOLED_ARE_SUMS:
            pooled_values /= tl.where(valid_p, counts, 1.0)[None, :]
        bucket_scores = tl.sum(route_query[:, None] * pooled_values, axis=0) * router_scale
        bucket_scores = tl.where(valid_p, bucket_scores, -float("inf"))
        score_max = tl.max(bucket_scores, axis=0)
        score = score_max + tl.log(tl.sum(tl.exp(bucket_scores - score_max), axis=0))
        if use_logmean:
            score -= tl.log(tl.sum(valid_p.to(tl.float32), axis=0))
        gate = tl.sigmoid((score + sigmoid_bias) * inv_temperature)

        source_b = tl.load(
            chunk_b
            + batch_idx * stride_bb
            + head_idx * stride_bh
            + source * stride_bc
            + offs_d[:, None] * stride_bd
            + offs_v[None, :],
            mask=mask_d[:, None] & mask_v[None, :],
            other=0.0,
        ).to(tl.float32)
        source_a = tl.load(
            chunk_a
            + batch_idx * stride_ab
            + head_idx * stride_ah
            + source * stride_ac
            + offs_d[:, None] * stride_ad
            + offs_d[None, :],
            mask=mask_d[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)
        contribution = tl.sum(readout[:, None] * source_b, axis=0)
        transformed = tl.sum(source_a * readout[:, None], axis=0)
        output = tl.where(active, output + gate * contribution, output)
        readout = tl.where(active, readout + gate * (transformed - readout), readout)
        source -= 1
    tl.store(out + row * VALUE_DIM + offs_v, output, mask=mask_v)


@triton.jit
def _gdn_affine_decode_mix_persistent_kernel(
    route_q,
    pooled,
    route_counts,
    q,
    chunk_idx,
    current_a,
    current_b,
    chunk_a,
    chunk_b,
    out,
    readout_scratch,
    stride_rqb,
    stride_rqh,
    stride_pb,
    stride_ph,
    stride_pc,
    stride_pp,
    stride_cb,
    stride_ch,
    stride_cd,
    stride_ab,
    stride_ah,
    stride_ac,
    stride_ad,
    stride_bb,
    stride_bh,
    stride_bc,
    stride_bd,
    router_scale: tl.constexpr,
    sigmoid_bias: tl.constexpr,
    inv_temperature: tl.constexpr,
    use_logmean: tl.constexpr,
    POOLED_ARE_SUMS: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    NUM_BUCKETS: tl.constexpr,
    KEY_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """One-launch decode route/readout for wide GDN heads.

    Loading a complete 256x256 A/B pair at once exceeds H200 shared memory.
    The previous fallback therefore launched a router kernel, a current-state
    kernel, and one kernel per history chunk.  This kernel keeps the two
    256-element accumulators resident and streams A/B in 32-row tiles, so the
    exact source-major recurrence completes in one launch per layer.
    """

    row = tl.program_id(0)
    batch_idx = row // NUM_HEADS
    head_idx = row - batch_idx * NUM_HEADS
    offs_d = tl.arange(0, BLOCK_D)
    offs_p = tl.arange(0, BLOCK_P)
    mask_d = offs_d < KEY_DIM
    mask_v = offs_d < VALUE_DIM
    current = tl.load(chunk_idx + batch_idx)

    route_query = tl.load(
        route_q + batch_idx * stride_rqb + head_idx * stride_rqh + offs_d,
        mask=mask_d,
        other=0.0,
    ).to(tl.float32)
    output = tl.zeros((BLOCK_D,), tl.float32)
    readout = tl.zeros((BLOCK_D,), tl.float32)
    k_offset = 0
    while k_offset < KEY_DIM:
        offs_k = k_offset + tl.arange(0, BLOCK_K)
        mask_k = offs_k < KEY_DIM
        qk = tl.load(
            q + row * KEY_DIM + offs_k, mask=mask_k, other=0.0
        ).to(tl.float32)
        cur_b = tl.load(
            current_b
            + batch_idx * stride_cb
            + head_idx * stride_ch
            + offs_k[:, None] * stride_cd
            + offs_d[None, :],
            mask=mask_k[:, None] & mask_v[None, :],
            other=0.0,
        ).to(tl.float32)
        cur_a = tl.load(
            current_a
            + batch_idx * stride_cb
            + head_idx * stride_ch
            + offs_k[:, None] * stride_cd
            + offs_d[None, :],
            mask=mask_k[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)
        output += tl.sum(qk[:, None] * cur_b, axis=0)
        readout += tl.sum(qk[:, None] * cur_a, axis=0)
        k_offset += BLOCK_K

    source = NUM_CHUNKS - 1
    while source >= 0:
        counts = tl.load(
            route_counts
            + (batch_idx * NUM_CHUNKS + source) * NUM_BUCKETS
            + offs_p,
            mask=offs_p < NUM_BUCKETS,
            other=0.0,
        )
        valid_p = (offs_p < NUM_BUCKETS) & (counts > 0)
        pooled_values = tl.load(
            pooled
            + batch_idx * stride_pb
            + head_idx * stride_ph
            + source * stride_pc
            + offs_p[None, :] * stride_pp
            + offs_d[:, None],
            mask=mask_d[:, None] & valid_p[None, :],
            other=0.0,
        ).to(tl.float32)
        if POOLED_ARE_SUMS:
            pooled_values /= tl.where(valid_p, counts, 1.0)[None, :]
        bucket_scores = (
            tl.sum(route_query[:, None] * pooled_values, axis=0) * router_scale
        )
        bucket_scores = tl.where(valid_p, bucket_scores, -float("inf"))
        score_max = tl.max(bucket_scores, axis=0)
        score = score_max + tl.log(
            tl.sum(tl.exp(bucket_scores - score_max), axis=0)
        )
        if use_logmean:
            score -= tl.log(tl.sum(valid_p.to(tl.float32), axis=0))
        gate = tl.sigmoid((score + sigmoid_bias) * inv_temperature)

        contribution = tl.zeros((BLOCK_D,), tl.float32)
        transformed = tl.zeros((BLOCK_D,), tl.float32)
        # Triton values cannot be dynamically sliced.  A row-private scratch
        # round-trip makes the updated readout addressable in BLOCK_K tiles
        # without introducing another kernel launch or cross-program sync.
        tl.store(
            readout_scratch + row * KEY_DIM + offs_d,
            readout,
            mask=mask_d,
        )
        k_offset = 0
        while k_offset < KEY_DIM:
            offs_k = k_offset + tl.arange(0, BLOCK_K)
            mask_k = offs_k < KEY_DIM
            rk = tl.load(
                readout_scratch + row * KEY_DIM + offs_k,
                mask=mask_k,
                other=0.0,
            ).to(tl.float32)
            source_b = tl.load(
                chunk_b
                + batch_idx * stride_bb
                + head_idx * stride_bh
                + source * stride_bc
                + offs_k[:, None] * stride_bd
                + offs_d[None, :],
                mask=mask_k[:, None] & mask_v[None, :],
                other=0.0,
            ).to(tl.float32)
            source_a = tl.load(
                chunk_a
                + batch_idx * stride_ab
                + head_idx * stride_ah
                + source * stride_ac
                + offs_k[:, None] * stride_ad
                + offs_d[None, :],
                mask=mask_k[:, None] & mask_d[None, :],
                other=0.0,
            ).to(tl.float32)
            contribution += tl.sum(rk[:, None] * source_b, axis=0)
            transformed += tl.sum(rk[:, None] * source_a, axis=0)
            k_offset += BLOCK_K

        active = source < current
        output = tl.where(active, output + gate * contribution, output)
        readout = tl.where(
            active, readout + gate * (transformed - readout), readout
        )
        source -= 1

    tl.store(out + row * VALUE_DIM + offs_d, output, mask=mask_v)


def can_use_fused_gdn_affine_decode_mix(
    route_q: torch.Tensor,
    pooled: torch.Tensor,
    route_counts: torch.Tensor,
    q: torch.Tensor,
    chunk_idx: torch.Tensor,
    current_a: torch.Tensor,
    current_b: torch.Tensor,
    chunk_a: torch.Tensor,
    chunk_b: torch.Tensor,
) -> bool:
    if os.environ.get("GDN_DISABLE_TRITON_GDN_DECODE_MIX", ""):
        return False
    tensors = (route_q, pooled, route_counts, q, chunk_idx, current_a, current_b, chunk_a, chunk_b)
    if not all(x.is_cuda for x in tensors):
        return False
    if route_q.dim() != 3 or q.shape != route_q.shape or pooled.dim() != 5:
        return False
    batch, heads, key_dim = q.shape
    chunks, buckets = pooled.shape[2:4]
    value_dim = current_b.shape[-1]
    return (
        route_q.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and q.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and pooled.shape == (batch, heads, chunks, buckets, key_dim)
        and route_counts.shape == (batch, chunks, buckets)
        and chunk_idx.shape == (batch,)
        and current_a.shape == (batch, heads, key_dim, key_dim)
        and current_b.shape == (batch, heads, key_dim, value_dim)
        and chunk_a.shape == (batch, heads, chunks, key_dim, key_dim)
        and chunk_b.shape == (batch, heads, chunks, key_dim, value_dim)
        and current_a.dtype == current_b.dtype == chunk_a.dtype == chunk_b.dtype == torch.float32
        and 1 < chunks <= 128
        and 0 < buckets <= 32
        and 0 < key_dim <= 256
        and value_dim == key_dim
        and all(x.stride(-1) == 1 for x in tensors)
    )


def fused_gdn_affine_decode_mix(
    route_q: torch.Tensor,
    pooled: torch.Tensor,
    route_counts: torch.Tensor,
    q: torch.Tensor,
    chunk_idx: torch.Tensor,
    current_a: torch.Tensor,
    current_b: torch.Tensor,
    chunk_a: torch.Tensor,
    chunk_b: torch.Tensor,
    *,
    router_logit_scale: float,
    sigmoid_bias: float,
    sigmoid_temperature: float,
    use_logmean: bool,
    pooled_are_sums: bool = False,
    num_warps: int = 4,
) -> torch.Tensor:
    if not can_use_fused_gdn_affine_decode_mix(
        route_q, pooled, route_counts, q, chunk_idx, current_a, current_b, chunk_a, chunk_b
    ):
        raise ValueError("unsupported GDN affine decode mix inputs")
    batch, heads, key_dim = q.shape
    if num_warps not in (4, 8, 16):
        raise ValueError("num_warps must be 4, 8, or 16")
    chunks, buckets = pooled.shape[2:4]
    value_dim = current_b.shape[-1]
    out = torch.empty(batch, heads, value_dim, device=q.device, dtype=torch.float32)
    route_q = route_q.contiguous()
    q = q.contiguous()
    chunk_idx = chunk_idx.contiguous()
    if key_dim >= 192 and os.environ.get(
        "GDN_ENABLE_PERSISTENT_GDN_DECODE_MIX", ""
    ):
        block_k = int(os.environ.get("GDN_PERSISTENT_DECODE_BLOCK_K", "1"))
        if block_k not in (1, 2, 4, 8, 16, 32, 64):
            raise ValueError("GDN_PERSISTENT_DECODE_BLOCK_K must be a power of two from 1 to 64")
        readout_scratch = torch.empty(
            batch, heads, key_dim, device=q.device, dtype=torch.float32
        )
        _gdn_affine_decode_mix_persistent_kernel[(batch * heads,)](
            route_q,
            pooled,
            route_counts,
            q,
            chunk_idx,
            current_a,
            current_b,
            chunk_a,
            chunk_b,
            out,
            readout_scratch,
            route_q.stride(0),
            route_q.stride(1),
            pooled.stride(0),
            pooled.stride(1),
            pooled.stride(2),
            pooled.stride(3),
            current_a.stride(0),
            current_a.stride(1),
            current_a.stride(2),
            chunk_a.stride(0),
            chunk_a.stride(1),
            chunk_a.stride(2),
            chunk_a.stride(3),
            chunk_b.stride(0),
            chunk_b.stride(1),
            chunk_b.stride(2),
            chunk_b.stride(3),
            float(key_dim**-0.5 * router_logit_scale),
            float(sigmoid_bias),
            float(1.0 / max(sigmoid_temperature, 1e-4)),
            bool(use_logmean),
            bool(pooled_are_sums),
            heads,
            chunks,
            buckets,
            key_dim,
            value_dim,
            BLOCK_D=triton.next_power_of_2(key_dim),
            BLOCK_P=triton.next_power_of_2(buckets),
            BLOCK_K=block_k,
            num_warps=8,
            num_stages=1,
        )
        return out
    if key_dim >= 192 and not os.environ.get("GDN_DISABLE_TILED_GDN_DECODE_MIX", ""):
        block_d = triton.next_power_of_2(key_dim)
        block_p = triton.next_power_of_2(buckets)
        block_o = 64
        rows = batch * heads
        gates = torch.empty(batch, heads, chunks, device=q.device, dtype=torch.float32)
        _gdn_decode_router_gates_kernel[(rows * chunks,)](
            route_q,
            pooled,
            route_counts,
            gates,
            route_q.stride(0),
            route_q.stride(1),
            pooled.stride(0),
            pooled.stride(1),
            pooled.stride(2),
            pooled.stride(3),
            float(key_dim**-0.5 * router_logit_scale),
            float(sigmoid_bias),
            float(1.0 / max(sigmoid_temperature, 1e-4)),
            bool(use_logmean),
            bool(pooled_are_sums),
            heads,
            chunks,
            buckets,
            key_dim,
            BLOCK_D=block_d,
            BLOCK_P=block_p,
            num_warps=4,
        )
        readouts = torch.empty(2, batch, heads, key_dim, device=q.device, dtype=torch.float32)
        tiles_per_row = triton.cdiv(key_dim, block_o)
        _gdn_affine_decode_current_tiled_kernel[(rows * tiles_per_row,)](
            q,
            current_a,
            current_b,
            readouts[0],
            out,
            current_a.stride(0),
            current_a.stride(1),
            current_a.stride(2),
            heads,
            key_dim,
            value_dim,
            BLOCK_D=block_d,
            BLOCK_O=block_o,
            num_warps=4,
        )
        read_idx = 0
        for source in range(chunks - 2, -1, -1):
            write_idx = 1 - read_idx
            _gdn_affine_decode_source_tiled_kernel[(rows * tiles_per_row,)](
                readouts[read_idx],
                readouts[write_idx],
                gates,
                chunk_idx,
                chunk_a,
                chunk_b,
                out,
                chunk_a.stride(0),
                chunk_a.stride(1),
                chunk_a.stride(2),
                chunk_a.stride(3),
                chunk_b.stride(0),
                chunk_b.stride(1),
                chunk_b.stride(2),
                chunk_b.stride(3),
                SOURCE=source,
                NUM_HEADS=heads,
                NUM_CHUNKS=chunks,
                KEY_DIM=key_dim,
                VALUE_DIM=value_dim,
                BLOCK_D=block_d,
                BLOCK_O=block_o,
                num_warps=4,
            )
            read_idx = write_idx
        return out
    _gdn_affine_decode_mix_kernel[(batch * heads,)](
        route_q, pooled, route_counts, q, chunk_idx,
        current_a, current_b, chunk_a, chunk_b, out,
        route_q.stride(0), route_q.stride(1), pooled.stride(0), pooled.stride(1),
        pooled.stride(2), pooled.stride(3), current_a.stride(0), current_a.stride(1),
        current_a.stride(2), chunk_a.stride(0), chunk_a.stride(1), chunk_a.stride(2),
        chunk_a.stride(3), chunk_b.stride(0), chunk_b.stride(1), chunk_b.stride(2),
        chunk_b.stride(3), float(key_dim**-0.5 * router_logit_scale), float(sigmoid_bias),
        float(1.0 / max(sigmoid_temperature, 1e-4)), bool(use_logmean),
        bool(pooled_are_sums), heads, chunks,
        buckets, key_dim, value_dim, BLOCK_D=triton.next_power_of_2(key_dim),
        BLOCK_V=triton.next_power_of_2(value_dim), BLOCK_P=triton.next_power_of_2(buckets),
        num_warps=num_warps,
    )
    return out
