from __future__ import annotations
import math
import os
import torch
import triton
import triton.language as tl


@triton.jit
def _affine_history_mix_kernel(
    local_output,
    local_readout,
    chunk_a,
    chunk_b,
    weights,
    route_q,
    route_q_weight,
    pooled,
    out,
    stride_ob,
    stride_oh,
    stride_oc,
    stride_os,
    stride_outb,
    stride_outh,
    stride_outc,
    stride_outs,
    stride_rb,
    stride_rh,
    stride_rc,
    stride_rs,
    stride_ab,
    stride_ah,
    stride_ac,
    stride_ad,
    stride_bb,
    stride_bh,
    stride_bc,
    stride_bd,
    stride_wb,
    stride_wh,
    stride_wc,
    stride_ws,
    stride_wsrc,
    stride_qb,
    stride_qs,
    stride_qh,
    stride_qwh,
    stride_pb,
    stride_ph,
    stride_pc,
    stride_pp,
    num_heads: tl.constexpr,
    num_chunks: tl.constexpr,
    chunk_size: tl.constexpr,
    key_dim: tl.constexpr,
    value_dim: tl.constexpr,
    num_buckets: tl.constexpr,
    route_scale: tl.constexpr,
    sigmoid_bias: tl.constexpr,
    inv_temperature: tl.constexpr,
    use_logmean: tl.constexpr,
    FUSE_ROUTE: tl.constexpr,
    Q_BF16: tl.constexpr,
    Q_FP16: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    tile = tl.program_id(0)
    tiles_per_chunk: tl.constexpr = tl.cdiv(chunk_size, BLOCK_S)
    sequence_tile = tile % tiles_per_chunk
    head_chunk = tile // tiles_per_chunk
    query_chunk = head_chunk % num_chunks
    row = head_chunk // num_chunks
    batch_idx = row // num_heads
    head_idx = row - batch_idx * num_heads
    offs_s = sequence_tile * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_d = tl.arange(0, BLOCK_D)
    offs_v = tl.arange(0, BLOCK_V)
    mask_s = offs_s < chunk_size
    mask_d = offs_d < key_dim
    mask_v = offs_v < value_dim
    o_ptrs = (
        local_output
        + batch_idx * stride_ob
        + head_idx * stride_oh
        + query_chunk * stride_oc
        + offs_s[:, None] * stride_os
        + offs_v[None, :]
    )
    r_ptrs = (
        local_readout
        + batch_idx * stride_rb
        + head_idx * stride_rh
        + query_chunk * stride_rc
        + offs_s[:, None] * stride_rs
        + offs_d[None, :]
    )
    output = tl.load(o_ptrs, mask=mask_s[:, None] & mask_v[None, :], other=0.0).to(
        tl.float32
    )
    readout = tl.load(r_ptrs, mask=mask_s[:, None] & mask_d[None, :], other=0.0).to(
        tl.float32
    )
    if FUSE_ROUTE:
        global_tokens = query_chunk * chunk_size + offs_s
        q_ptrs = (
            route_q
            + batch_idx * stride_qb
            + global_tokens[:, None] * stride_qs
            + head_idx * stride_qh
            + offs_d[None, :]
        )
        route_query = tl.load(
            q_ptrs, mask=mask_s[:, None] & mask_d[None, :], other=0.0
        ).to(tl.float32)
        qw_ptrs = (
            route_q_weight
            + head_idx * stride_qwh
            + offs_d[:, None] * key_dim
            + offs_d[None, :]
        )
        qw = tl.load(qw_ptrs, mask=mask_d[:, None] & mask_d[None, :], other=0.0).to(
            tl.float32
        )
        route_query = tl.dot(route_query, qw, input_precision="tf32")
        if Q_BF16:
            route_query = route_query.to(tl.bfloat16).to(tl.float32)
        elif Q_FP16:
            route_query = route_query.to(tl.float16).to(tl.float32)
    source = query_chunk - 1
    while source >= 0:
        if FUSE_ROUTE:
            offs_p = tl.arange(0, 32)
            valid_p = offs_p < num_buckets
            pooled_ptrs = (
                pooled
                + batch_idx * stride_pb
                + head_idx * stride_ph
                + source * stride_pc
                + offs_p[None, :] * stride_pp
                + offs_d[:, None]
            )
            pooled_values = tl.load(
                pooled_ptrs, mask=mask_d[:, None] & valid_p[None, :], other=0.0
            ).to(tl.float32)
            bucket_scores = (
                tl.dot(route_query, pooled_values, input_precision="tf32") * route_scale
            )
            bucket_scores = tl.where(valid_p[None, :], bucket_scores, -float("inf"))
            score_max = tl.max(bucket_scores, axis=1)
            score = score_max + tl.log(
                tl.sum(tl.exp(bucket_scores - score_max[:, None]), axis=1)
            )
            if use_logmean:
                score -= math.log(num_buckets)
            gate = tl.sigmoid((score + sigmoid_bias) * inv_temperature)
        else:
            gate = tl.load(
                weights
                + batch_idx * stride_wb
                + head_idx * stride_wh
                + query_chunk * stride_wc
                + offs_s * stride_ws
                + source * stride_wsrc,
                mask=mask_s,
                other=0.0,
            ).to(tl.float32)
        b_ptrs = (
            chunk_b
            + batch_idx * stride_bb
            + head_idx * stride_bh
            + source * stride_bc
            + offs_d[:, None] * stride_bd
            + offs_v[None, :]
        )
        a_ptrs = (
            chunk_a
            + batch_idx * stride_ab
            + head_idx * stride_ah
            + source * stride_ac
            + offs_d[:, None] * stride_ad
            + offs_d[None, :]
        )
        source_b = tl.load(
            b_ptrs, mask=mask_d[:, None] & mask_v[None, :], other=0.0
        ).to(tl.float32)
        source_a = tl.load(
            a_ptrs, mask=mask_d[:, None] & mask_d[None, :], other=0.0
        ).to(tl.float32)
        contribution = tl.dot(readout, source_b, input_precision="tf32")
        transformed = tl.dot(readout, source_a, input_precision="tf32")
        output += gate[:, None] * contribution
        readout += gate[:, None] * (transformed - readout)
        source -= 1
    out_ptrs = (
        out
        + batch_idx * stride_outb
        + head_idx * stride_outh
        + query_chunk * stride_outc
        + offs_s[:, None] * stride_outs
        + offs_v[None, :]
    )
    tl.store(out_ptrs, output, mask=mask_s[:, None] & mask_v[None, :])


@triton.jit
def _affine_history_mix_joint_kernel(
    local_output,
    local_readout,
    chunk_joint,
    weights,
    out,
    stride_ob,
    stride_oh,
    stride_oc,
    stride_os,
    stride_rb,
    stride_rh,
    stride_rc,
    stride_rs,
    stride_jb,
    stride_jh,
    stride_jc,
    stride_jd,
    stride_wb,
    stride_wh,
    stride_wc,
    stride_ws,
    stride_wsrc,
    stride_outb,
    stride_outh,
    stride_outc,
    stride_outs,
    num_heads: tl.constexpr,
    num_chunks: tl.constexpr,
    chunk_size: tl.constexpr,
    key_dim: tl.constexpr,
    value_dim: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_J: tl.constexpr,
    DOT_FP16: tl.constexpr,
    DOT_BF16: tl.constexpr,
    SKIP_ZERO_TILES: tl.constexpr,
):
    tile = tl.program_id(0)
    tiles_per_chunk: tl.constexpr = tl.cdiv(chunk_size, BLOCK_S)
    sequence_tile = tile % tiles_per_chunk
    head_chunk = tile // tiles_per_chunk
    query_chunk = head_chunk % num_chunks
    row = head_chunk // num_chunks
    batch_idx = row // num_heads
    head_idx = row - batch_idx * num_heads
    offs_s = sequence_tile * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_d = tl.arange(0, BLOCK_D)
    offs_v = tl.arange(0, BLOCK_V)
    offs_j = tl.arange(0, BLOCK_J)
    mask_s = offs_s < chunk_size
    mask_d = offs_d < key_dim
    mask_v = offs_v < value_dim
    mask_j = offs_j < value_dim + key_dim
    output = tl.load(
        local_output
        + batch_idx * stride_ob
        + head_idx * stride_oh
        + query_chunk * stride_oc
        + offs_s[:, None] * stride_os
        + offs_v[None, :],
        mask=mask_s[:, None] & mask_v[None, :],
        other=0.0,
    ).to(tl.float32)
    readout = tl.load(
        local_readout
        + batch_idx * stride_rb
        + head_idx * stride_rh
        + query_chunk * stride_rc
        + offs_s[:, None] * stride_rs
        + offs_d[None, :],
        mask=mask_s[:, None] & mask_d[None, :],
        other=0.0,
    ).to(tl.float32)
    source = query_chunk - 1
    while source >= 0:
        gate = tl.load(
            weights
            + batch_idx * stride_wb
            + head_idx * stride_wh
            + query_chunk * stride_wc
            + offs_s * stride_ws
            + source * stride_wsrc,
            mask=mask_s,
            other=0.0,
        ).to(tl.float32)
        tile_is_active = tl.sum(tl.abs(gate), axis=0) != 0.0
        if not SKIP_ZERO_TILES or tile_is_active:
            source_joint = tl.load(
                chunk_joint
                + batch_idx * stride_jb
                + head_idx * stride_jh
                + source * stride_jc
                + offs_d[:, None] * stride_jd
                + offs_j[None, :],
                mask=mask_d[:, None] & mask_j[None, :],
                other=0.0,
            )
            if DOT_FP16:
                mixed = tl.dot(readout.to(tl.float16), source_joint.to(tl.float16))
            elif DOT_BF16:
                mixed = tl.dot(readout.to(tl.bfloat16), source_joint.to(tl.bfloat16))
            else:
                mixed = tl.dot(
                    readout, source_joint.to(tl.float32), input_precision="tf32"
                )
            mixed = tl.reshape(mixed, (BLOCK_S, 2, BLOCK_D))
            mixed = tl.permute(mixed, (0, 2, 1))
            contribution, transformed = tl.split(mixed)
            output += gate[:, None] * contribution
            readout += gate[:, None] * (transformed - readout)
        source -= 1
    tl.store(
        out
        + batch_idx * stride_outb
        + head_idx * stride_outh
        + query_chunk * stride_outc
        + offs_s[:, None] * stride_outs
        + offs_v[None, :],
        output,
        mask=mask_s[:, None] & mask_v[None, :],
    )


@triton.jit
def _compact_top1_affine_history_mix_joint_kernel(
    local_output,
    local_readout,
    chunk_joint,
    selected_gates,
    selected_sources,
    out,
    stride_ob,
    stride_oh,
    stride_oc,
    stride_os,
    stride_rb,
    stride_rh,
    stride_rc,
    stride_rs,
    stride_jb,
    stride_jh,
    stride_jc,
    stride_jd,
    stride_gb,
    stride_gh,
    stride_gs,
    stride_sb,
    stride_sh,
    stride_st,
    stride_outb,
    stride_outh,
    stride_outc,
    stride_outs,
    num_heads: tl.constexpr,
    num_chunks: tl.constexpr,
    chunk_size: tl.constexpr,
    key_dim: tl.constexpr,
    value_dim: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_V: tl.constexpr,
    DOT_FP16: tl.constexpr,
    DOT_BF16: tl.constexpr,
):
    """Apply preselected tile Top-1 history without a dense gate tensor."""
    tile = tl.program_id(0)
    tiles_per_chunk: tl.constexpr = tl.cdiv(chunk_size, BLOCK_S)
    sequence_tile = tile % tiles_per_chunk
    head_chunk = tile // tiles_per_chunk
    query_chunk = head_chunk % num_chunks
    row = head_chunk // num_chunks
    batch_idx = row // num_heads
    head_idx = row - batch_idx * num_heads
    offs_s = sequence_tile * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_d = tl.arange(0, BLOCK_D)
    offs_v = tl.arange(0, BLOCK_V)
    mask_s = offs_s < chunk_size
    mask_d = offs_d < key_dim
    mask_v = offs_v < value_dim
    output = tl.load(
        local_output
        + batch_idx * stride_ob
        + head_idx * stride_oh
        + query_chunk * stride_oc
        + offs_s[:, None] * stride_os
        + offs_v[None, :],
        mask=mask_s[:, None] & mask_v[None, :],
        other=0.0,
    ).to(tl.float32)
    readout = tl.load(
        local_readout
        + batch_idx * stride_rb
        + head_idx * stride_rh
        + query_chunk * stride_rc
        + offs_s[:, None] * stride_rs
        + offs_d[None, :],
        mask=mask_s[:, None] & mask_d[None, :],
        other=0.0,
    ).to(tl.float32)
    global_tile = query_chunk * tiles_per_chunk + sequence_tile
    source = tl.load(
        selected_sources
        + batch_idx * stride_sb
        + head_idx * stride_sh
        + global_tile * stride_st
    )
    active = query_chunk > 0
    global_token = query_chunk * chunk_size + offs_s
    gate = tl.load(
        selected_gates
        + batch_idx * stride_gb
        + head_idx * stride_gh
        + global_token * stride_gs,
        mask=mask_s & active,
        other=0.0,
    ).to(tl.float32)
    source_b = tl.load(
        chunk_joint
        + batch_idx * stride_jb
        + head_idx * stride_jh
        + source * stride_jc
        + offs_d[:, None] * stride_jd
        + offs_v[None, :],
        mask=active & mask_d[:, None] & mask_v[None, :],
        other=0.0,
    )
    if DOT_FP16:
        contribution = tl.dot(readout.to(tl.float16), source_b.to(tl.float16))
    elif DOT_BF16:
        contribution = tl.dot(readout.to(tl.bfloat16), source_b.to(tl.bfloat16))
    else:
        contribution = tl.dot(readout, source_b.to(tl.float32), input_precision="tf32")
    output += gate[:, None] * contribution
    tl.store(
        out
        + batch_idx * stride_outb
        + head_idx * stride_outh
        + query_chunk * stride_outc
        + offs_s[:, None] * stride_outs
        + offs_v[None, :],
        output,
        mask=mask_s[:, None] & mask_v[None, :],
    )


@triton.jit
def _sparse_top2_affine_history_mix_joint_kernel(
    local_output,
    local_readout,
    chunk_joint,
    weights,
    out,
    stride_ob,
    stride_oh,
    stride_oc,
    stride_os,
    stride_rb,
    stride_rh,
    stride_rc,
    stride_rs,
    stride_jb,
    stride_jh,
    stride_jc,
    stride_jd,
    stride_wb,
    stride_wh,
    stride_wc,
    stride_ws,
    stride_wsrc,
    stride_outb,
    stride_outh,
    stride_outc,
    stride_outs,
    num_heads: tl.constexpr,
    num_chunks: tl.constexpr,
    chunk_size: tl.constexpr,
    key_dim: tl.constexpr,
    value_dim: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_J: tl.constexpr,
    BLOCK_C: tl.constexpr,
    DOT_FP16: tl.constexpr,
    DOT_BF16: tl.constexpr,
    TOP_K: tl.constexpr,
    TOP1_RECENT_OF_TOP2: tl.constexpr,
):
    """Apply the two strongest historical affine states per token tile.

    Selecting one source for the whole small token tile preserves Tensor-Core
    matrix reuse.  Gates remain token-specific after selection, and selected
    sources are applied newest-to-oldest to preserve affine composition order.
    """
    tile = tl.program_id(0)
    tiles_per_chunk: tl.constexpr = tl.cdiv(chunk_size, BLOCK_S)
    sequence_tile = tile % tiles_per_chunk
    head_chunk = tile // tiles_per_chunk
    query_chunk = head_chunk % num_chunks
    row = head_chunk // num_chunks
    batch_idx = row // num_heads
    head_idx = row - batch_idx * num_heads
    offs_s = sequence_tile * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_d = tl.arange(0, BLOCK_D)
    offs_v = tl.arange(0, BLOCK_V)
    offs_j = tl.arange(0, BLOCK_J)
    offs_c = tl.arange(0, BLOCK_C)
    mask_s = offs_s < chunk_size
    mask_d = offs_d < key_dim
    mask_v = offs_v < value_dim
    mask_j = offs_j < value_dim + key_dim
    output = tl.load(
        local_output
        + batch_idx * stride_ob
        + head_idx * stride_oh
        + query_chunk * stride_oc
        + offs_s[:, None] * stride_os
        + offs_v[None, :],
        mask=mask_s[:, None] & mask_v[None, :],
        other=0.0,
    ).to(tl.float32)
    readout = tl.load(
        local_readout
        + batch_idx * stride_rb
        + head_idx * stride_rh
        + query_chunk * stride_rc
        + offs_s[:, None] * stride_rs
        + offs_d[None, :],
        mask=mask_s[:, None] & mask_d[None, :],
        other=0.0,
    ).to(tl.float32)
    tile_gates = tl.load(
        weights
        + batch_idx * stride_wb
        + head_idx * stride_wh
        + query_chunk * stride_wc
        + offs_s[:, None] * stride_ws
        + offs_c[None, :] * stride_wsrc,
        mask=mask_s[:, None] & (offs_c[None, :] < query_chunk),
        other=0.0,
    ).to(tl.float32)
    source_scores = tl.sum(tile_gates, axis=0)
    source_scores = tl.where(offs_c < query_chunk, source_scores, -float("inf"))
    selected0 = tl.argmax(source_scores, axis=0)
    source_scores = tl.where(offs_c == selected0, -float("inf"), source_scores)
    selected1 = tl.argmax(source_scores, axis=0)
    newest = tl.maximum(selected0, selected1)
    oldest = tl.minimum(selected0, selected1)
    source = tl.where((TOP_K == 1) & (TOP1_RECENT_OF_TOP2 == 0), selected0, newest)
    active = query_chunk > 0
    gate = tl.load(
        weights
        + batch_idx * stride_wb
        + head_idx * stride_wh
        + query_chunk * stride_wc
        + offs_s * stride_ws
        + source * stride_wsrc,
        mask=mask_s & active,
        other=0.0,
    ).to(tl.float32)
    if TOP_K == 1:
        source_b = tl.load(
            chunk_joint
            + batch_idx * stride_jb
            + head_idx * stride_jh
            + source * stride_jc
            + offs_d[:, None] * stride_jd
            + offs_v[None, :],
            mask=active & mask_d[:, None] & mask_v[None, :],
            other=0.0,
        )
        if DOT_FP16:
            contribution = tl.dot(readout.to(tl.float16), source_b.to(tl.float16))
        elif DOT_BF16:
            contribution = tl.dot(readout.to(tl.bfloat16), source_b.to(tl.bfloat16))
        else:
            contribution = tl.dot(
                readout, source_b.to(tl.float32), input_precision="tf32"
            )
        output += gate[:, None] * contribution
    else:
        source_joint = tl.load(
            chunk_joint
            + batch_idx * stride_jb
            + head_idx * stride_jh
            + source * stride_jc
            + offs_d[:, None] * stride_jd
            + offs_j[None, :],
            mask=active & mask_d[:, None] & mask_j[None, :],
            other=0.0,
        )
        if DOT_FP16:
            mixed = tl.dot(readout.to(tl.float16), source_joint.to(tl.float16))
        elif DOT_BF16:
            mixed = tl.dot(readout.to(tl.bfloat16), source_joint.to(tl.bfloat16))
        else:
            mixed = tl.dot(readout, source_joint.to(tl.float32), input_precision="tf32")
        mixed = tl.reshape(mixed, (BLOCK_S, 2, BLOCK_D))
        mixed = tl.permute(mixed, (0, 2, 1))
        contribution, transformed = tl.split(mixed)
        output += gate[:, None] * contribution
        readout += gate[:, None] * (transformed - readout)
    source = oldest
    active = (query_chunk > 1) & (TOP_K > 1)
    gate = tl.load(
        weights
        + batch_idx * stride_wb
        + head_idx * stride_wh
        + query_chunk * stride_wc
        + offs_s * stride_ws
        + source * stride_wsrc,
        mask=mask_s & active,
        other=0.0,
    ).to(tl.float32)
    source_joint = tl.load(
        chunk_joint
        + batch_idx * stride_jb
        + head_idx * stride_jh
        + source * stride_jc
        + offs_d[:, None] * stride_jd
        + offs_j[None, :],
        mask=active & mask_d[:, None] & mask_j[None, :],
        other=0.0,
    )
    if DOT_FP16:
        mixed = tl.dot(readout.to(tl.float16), source_joint.to(tl.float16))
    elif DOT_BF16:
        mixed = tl.dot(readout.to(tl.bfloat16), source_joint.to(tl.bfloat16))
    else:
        mixed = tl.dot(readout, source_joint.to(tl.float32), input_precision="tf32")
    mixed = tl.reshape(mixed, (BLOCK_S, 2, BLOCK_D))
    mixed = tl.permute(mixed, (0, 2, 1))
    contribution, transformed = tl.split(mixed)
    output += gate[:, None] * contribution
    tl.store(
        out
        + batch_idx * stride_outb
        + head_idx * stride_outh
        + query_chunk * stride_outc
        + offs_s[:, None] * stride_outs
        + offs_v[None, :],
        output,
        mask=mask_s[:, None] & mask_v[None, :],
    )


@triton.jit
def _sparse_top2_bonly_history_mix_kernel(
    local_output,
    query,
    chunk_state,
    weights,
    out,
    stride_ob,
    stride_oh,
    stride_oc,
    stride_os,
    stride_qb,
    stride_qh,
    stride_qc,
    stride_qs,
    stride_qd,
    stride_jb,
    stride_jh,
    stride_jc,
    stride_jd,
    stride_wb,
    stride_wh,
    stride_wc,
    stride_ws,
    stride_wsrc,
    stride_outb,
    stride_outh,
    stride_outc,
    stride_outs,
    num_heads: tl.constexpr,
    num_chunks: tl.constexpr,
    chunk_size: tl.constexpr,
    key_dim: tl.constexpr,
    value_dim: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Top-2 additive history without materializing a source-token tensor."""
    tile = tl.program_id(0)
    tiles_per_chunk: tl.constexpr = tl.cdiv(chunk_size, BLOCK_S)
    sequence_tile = tile % tiles_per_chunk
    head_chunk = tile // tiles_per_chunk
    query_chunk = head_chunk % num_chunks
    row = head_chunk // num_chunks
    batch_idx = row // num_heads
    head_idx = row - batch_idx * num_heads
    offs_s = sequence_tile * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_d = tl.arange(0, BLOCK_D)
    offs_v = tl.arange(0, BLOCK_V)
    offs_c = tl.arange(0, BLOCK_C)
    mask_s = offs_s < chunk_size
    mask_d = offs_d < key_dim
    mask_v = offs_v < value_dim
    output = tl.load(
        local_output
        + batch_idx * stride_ob
        + head_idx * stride_oh
        + query_chunk * stride_oc
        + offs_s[:, None] * stride_os
        + offs_v[None, :],
        mask=mask_s[:, None] & mask_v[None, :],
        other=0.0,
    ).to(tl.float32)
    readout = (
        tl.load(
            query
            + batch_idx * stride_qb
            + head_idx * stride_qh
            + query_chunk * stride_qc
            + offs_s[:, None] * stride_qs
            + offs_d[None, :] * stride_qd,
            mask=mask_s[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)
        * scale
    )
    tile_gates = tl.load(
        weights
        + batch_idx * stride_wb
        + head_idx * stride_wh
        + query_chunk * stride_wc
        + offs_s[:, None] * stride_ws
        + offs_c[None, :] * stride_wsrc,
        mask=mask_s[:, None] & (offs_c[None, :] < query_chunk),
        other=0.0,
    ).to(tl.float32)
    source_scores = tl.sum(tile_gates, axis=0)
    source_scores = tl.where(offs_c < query_chunk, source_scores, -float("inf"))
    selected0 = tl.argmax(source_scores, axis=0)
    source_scores = tl.where(offs_c == selected0, -float("inf"), source_scores)
    selected1 = tl.argmax(source_scores, axis=0)
    newest = tl.maximum(selected0, selected1)
    oldest = tl.minimum(selected0, selected1)
    source = newest
    active = query_chunk > 0
    gate = tl.load(
        weights
        + batch_idx * stride_wb
        + head_idx * stride_wh
        + query_chunk * stride_wc
        + offs_s * stride_ws
        + source * stride_wsrc,
        mask=mask_s & active,
        other=0.0,
    ).to(tl.float32)
    state = tl.load(
        chunk_state
        + batch_idx * stride_jb
        + head_idx * stride_jh
        + source * stride_jc
        + offs_d[:, None] * stride_jd
        + offs_v[None, :],
        mask=active & mask_d[:, None] & mask_v[None, :],
        other=0.0,
    ).to(tl.float32)
    contribution = tl.dot(readout.to(tl.float16), state.to(tl.float16))
    output += gate[:, None] * contribution
    source = oldest
    active = query_chunk > 1
    gate = tl.load(
        weights
        + batch_idx * stride_wb
        + head_idx * stride_wh
        + query_chunk * stride_wc
        + offs_s * stride_ws
        + source * stride_wsrc,
        mask=mask_s & active,
        other=0.0,
    ).to(tl.float32)
    state = tl.load(
        chunk_state
        + batch_idx * stride_jb
        + head_idx * stride_jh
        + source * stride_jc
        + offs_d[:, None] * stride_jd
        + offs_v[None, :],
        mask=active & mask_d[:, None] & mask_v[None, :],
        other=0.0,
    ).to(tl.float32)
    contribution = tl.dot(readout.to(tl.float16), state.to(tl.float16))
    output += gate[:, None] * contribution
    tl.store(
        out
        + batch_idx * stride_outb
        + head_idx * stride_outh
        + query_chunk * stride_outc
        + offs_s[:, None] * stride_outs
        + offs_v[None, :],
        output,
        mask=mask_s[:, None] & mask_v[None, :],
    )


@triton.jit
def _affine_history_route_mix_joint_kernel(
    local_output,
    local_readout,
    chunk_joint,
    route_q,
    route_q_weight,
    pooled,
    out,
    stride_ob,
    stride_oh,
    stride_oc,
    stride_os,
    stride_rb,
    stride_rh,
    stride_rc,
    stride_rs,
    stride_jb,
    stride_jh,
    stride_jc,
    stride_jd,
    stride_qb,
    stride_qs,
    stride_qh,
    stride_qwh,
    stride_pb,
    stride_ph,
    stride_pc,
    stride_pp,
    stride_outb,
    stride_outh,
    stride_outc,
    stride_outs,
    num_heads: tl.constexpr,
    num_chunks: tl.constexpr,
    chunk_size: tl.constexpr,
    key_dim: tl.constexpr,
    value_dim: tl.constexpr,
    num_buckets: tl.constexpr,
    route_scale: tl.constexpr,
    sigmoid_bias: tl.constexpr,
    inv_temperature: tl.constexpr,
    use_logmean: tl.constexpr,
    Q_BF16: tl.constexpr,
    Q_FP16: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_J: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    tile = tl.program_id(0)
    tiles_per_chunk: tl.constexpr = tl.cdiv(chunk_size, BLOCK_S)
    sequence_tile = tile % tiles_per_chunk
    head_chunk = tile // tiles_per_chunk
    query_chunk = head_chunk % num_chunks
    row = head_chunk // num_chunks
    batch_idx = row // num_heads
    head_idx = row - batch_idx * num_heads
    offs_s = sequence_tile * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_d = tl.arange(0, BLOCK_D)
    offs_v = tl.arange(0, BLOCK_V)
    offs_j = tl.arange(0, BLOCK_J)
    offs_p = tl.arange(0, BLOCK_P)
    mask_s = offs_s < chunk_size
    mask_d = offs_d < key_dim
    mask_v = offs_v < value_dim
    mask_j = offs_j < value_dim + key_dim
    valid_p = offs_p < num_buckets
    output = tl.load(
        local_output
        + batch_idx * stride_ob
        + head_idx * stride_oh
        + query_chunk * stride_oc
        + offs_s[:, None] * stride_os
        + offs_v[None, :],
        mask=mask_s[:, None] & mask_v[None, :],
        other=0.0,
    ).to(tl.float32)
    readout = tl.load(
        local_readout
        + batch_idx * stride_rb
        + head_idx * stride_rh
        + query_chunk * stride_rc
        + offs_s[:, None] * stride_rs
        + offs_d[None, :],
        mask=mask_s[:, None] & mask_d[None, :],
        other=0.0,
    ).to(tl.float32)
    global_tokens = query_chunk * chunk_size + offs_s
    route_query = tl.load(
        route_q
        + batch_idx * stride_qb
        + global_tokens[:, None] * stride_qs
        + head_idx * stride_qh
        + offs_d[None, :],
        mask=mask_s[:, None] & mask_d[None, :],
        other=0.0,
    ).to(tl.float32)
    qw = tl.load(
        route_q_weight
        + head_idx * stride_qwh
        + offs_d[:, None] * key_dim
        + offs_d[None, :],
        mask=mask_d[:, None] & mask_d[None, :],
        other=0.0,
    ).to(tl.float32)
    route_query = tl.dot(route_query, qw, input_precision="tf32")
    if Q_BF16:
        route_query = route_query.to(tl.bfloat16).to(tl.float32)
    elif Q_FP16:
        route_query = route_query.to(tl.float16).to(tl.float32)
    source = query_chunk - 1
    while source >= 0:
        bucket_offset = 0
        score_max = tl.full((BLOCK_S,), -float("inf"), tl.float32)
        score_sum = tl.zeros((BLOCK_S,), tl.float32)
        while bucket_offset < num_buckets:
            bucket_indices = bucket_offset + offs_p
            valid_bucket = bucket_indices < num_buckets
            pooled_values = tl.load(
                pooled
                + batch_idx * stride_pb
                + head_idx * stride_ph
                + source * stride_pc
                + bucket_indices[None, :] * stride_pp
                + offs_d[:, None],
                mask=mask_d[:, None] & valid_bucket[None, :],
                other=0.0,
            ).to(tl.float32)
            bucket_scores = (
                tl.dot(route_query, pooled_values, input_precision="tf32") * route_scale
            )
            bucket_scores = tl.where(
                valid_bucket[None, :], bucket_scores, -float("inf")
            )
            tile_max = tl.max(bucket_scores, axis=1)
            next_max = tl.maximum(score_max, tile_max)
            score_sum = score_sum * tl.exp(score_max - next_max) + tl.sum(
                tl.exp(bucket_scores - next_max[:, None]), axis=1
            )
            score_max = next_max
            bucket_offset += BLOCK_P
        score = score_max + tl.log(score_sum)
        if use_logmean:
            score -= math.log(num_buckets)
        gate = tl.sigmoid((score + sigmoid_bias) * inv_temperature)
        source_joint = tl.load(
            chunk_joint
            + batch_idx * stride_jb
            + head_idx * stride_jh
            + source * stride_jc
            + offs_d[:, None] * stride_jd
            + offs_j[None, :],
            mask=mask_d[:, None] & mask_j[None, :],
            other=0.0,
        ).to(tl.float32)
        mixed = tl.dot(readout, source_joint, input_precision="tf32")
        mixed = tl.reshape(mixed, (BLOCK_S, 2, BLOCK_D))
        mixed = tl.permute(mixed, (0, 2, 1))
        contribution, transformed = tl.split(mixed)
        output += gate[:, None] * contribution
        readout += gate[:, None] * (transformed - readout)
        source -= 1
    tl.store(
        out
        + batch_idx * stride_outb
        + head_idx * stride_outh
        + query_chunk * stride_outc
        + offs_s[:, None] * stride_outs
        + offs_v[None, :],
        output,
        mask=mask_s[:, None] & mask_v[None, :],
    )


@triton.jit
def _affine_history_mix_wide_tiled_k_kernel(
    local_output,
    local_readout,
    chunk_a,
    chunk_b,
    weights,
    route_q,
    pooled,
    readout_scratch,
    out,
    stride_ob,
    stride_oh,
    stride_oc,
    stride_os,
    stride_outb,
    stride_outh,
    stride_outc,
    stride_outs,
    stride_rb,
    stride_rh,
    stride_rc,
    stride_rs,
    stride_ab,
    stride_ah,
    stride_ac,
    stride_ad,
    stride_bb,
    stride_bh,
    stride_bc,
    stride_bd,
    stride_wb,
    stride_wh,
    stride_wc,
    stride_ws,
    stride_wsrc,
    stride_qb,
    stride_qh,
    stride_qs,
    stride_pb,
    stride_ph,
    stride_pc,
    stride_pp,
    num_heads: tl.constexpr,
    num_chunks: tl.constexpr,
    chunk_size: tl.constexpr,
    key_dim: tl.constexpr,
    value_dim: tl.constexpr,
    num_buckets: tl.constexpr,
    route_scale: tl.constexpr,
    sigmoid_bias: tl.constexpr,
    inv_temperature: tl.constexpr,
    use_logmean: tl.constexpr,
    FUSE_ROUTE: tl.constexpr,
    STATE_BF16: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Tensor-Core history scan for GDN's D=256 inference state.

    The old D=128 kernel loads a complete A/B matrix for each source.  At
    D=256 that exceeds H200 shared memory.  This variant retains 16 queries
    per program but streams the contraction dimension in 64-row tiles.  A
    private FP32 scratch tensor makes the recurrent readout addressable by K
    tile while preserving the exact descending source order.
    """
    tile = tl.program_id(0)
    tiles_per_chunk: tl.constexpr = tl.cdiv(chunk_size, BLOCK_S)
    sequence_tile = tile % tiles_per_chunk
    head_chunk = tile // tiles_per_chunk
    query_chunk = head_chunk % num_chunks
    row = head_chunk // num_chunks
    batch_idx = row // num_heads
    head_idx = row - batch_idx * num_heads
    offs_s = sequence_tile * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_d = tl.arange(0, BLOCK_D)
    mask_s = offs_s < chunk_size
    mask_d = offs_d < key_dim
    output = tl.load(
        local_output
        + batch_idx * stride_ob
        + head_idx * stride_oh
        + query_chunk * stride_oc
        + offs_s[:, None] * stride_os
        + offs_d[None, :],
        mask=mask_s[:, None] & mask_d[None, :],
        other=0.0,
    ).to(tl.float32)
    readout = tl.load(
        local_readout
        + batch_idx * stride_rb
        + head_idx * stride_rh
        + query_chunk * stride_rc
        + offs_s[:, None] * stride_rs
        + offs_d[None, :],
        mask=mask_s[:, None] & mask_d[None, :],
        other=0.0,
    ).to(tl.float32)
    if FUSE_ROUTE:
        route_query = tl.load(
            route_q
            + batch_idx * stride_qb
            + head_idx * stride_qh
            + (query_chunk * chunk_size + offs_s)[:, None] * stride_qs
            + offs_d[None, :],
            mask=mask_s[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)
    scratch_base = (
        ((batch_idx * num_heads + head_idx) * num_chunks + query_chunk) * chunk_size
        + offs_s[:, None]
    ) * key_dim
    source = query_chunk - 1
    while source >= 0:
        if FUSE_ROUTE:
            offs_p = tl.arange(0, 32)
            valid_p = offs_p < num_buckets
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
            bucket_scores = (
                tl.dot(route_query, pooled_values, input_precision="tf32") * route_scale
            )
            bucket_scores = tl.where(valid_p[None, :], bucket_scores, -float("inf"))
            score_max = tl.max(bucket_scores, axis=1)
            score = score_max + tl.log(
                tl.sum(tl.exp(bucket_scores - score_max[:, None]), axis=1)
            )
            if use_logmean:
                score -= math.log(num_buckets)
            gate = tl.sigmoid((score + sigmoid_bias) * inv_temperature)
        else:
            gate = tl.load(
                weights
                + batch_idx * stride_wb
                + head_idx * stride_wh
                + query_chunk * stride_wc
                + offs_s * stride_ws
                + source * stride_wsrc,
                mask=mask_s,
                other=0.0,
            ).to(tl.float32)
        tl.store(
            readout_scratch + scratch_base + offs_d[None, :],
            readout,
            mask=mask_s[:, None] & mask_d[None, :],
        )
        tl.debug_barrier()
        contribution = tl.zeros((BLOCK_S, BLOCK_D), tl.float32)
        transformed = tl.zeros((BLOCK_S, BLOCK_D), tl.float32)
        k_offset = 0
        while k_offset < key_dim:
            offs_k = k_offset + tl.arange(0, BLOCK_K)
            mask_k = offs_k < key_dim
            r_tile = tl.load(
                readout_scratch + scratch_base + offs_k[None, :],
                mask=mask_s[:, None] & mask_k[None, :],
                other=0.0,
            )
            source_b = tl.load(
                chunk_b
                + batch_idx * stride_bb
                + head_idx * stride_bh
                + source * stride_bc
                + offs_k[:, None] * stride_bd
                + offs_d[None, :],
                mask=mask_k[:, None] & mask_d[None, :],
                other=0.0,
            )
            source_a = tl.load(
                chunk_a
                + batch_idx * stride_ab
                + head_idx * stride_ah
                + source * stride_ac
                + offs_k[:, None] * stride_ad
                + offs_d[None, :],
                mask=mask_k[:, None] & mask_d[None, :],
                other=0.0,
            )
            contribution += tl.dot(r_tile.to(source_b.dtype), source_b)
            transformed += tl.dot(r_tile.to(source_a.dtype), source_a)
            k_offset += BLOCK_K
        state_dtype: tl.constexpr = tl.bfloat16 if STATE_BF16 else tl.float16
        contribution = contribution.to(state_dtype).to(tl.float32)
        transformed = transformed.to(state_dtype).to(tl.float32)
        gate = gate.to(state_dtype).to(tl.float32)
        output = (
            (output.to(state_dtype).to(tl.float32) + gate[:, None] * contribution)
            .to(state_dtype)
            .to(tl.float32)
        )
        readout = (
            (
                readout.to(state_dtype).to(tl.float32)
                + gate[:, None] * (transformed - readout.to(state_dtype).to(tl.float32))
            )
            .to(state_dtype)
            .to(tl.float32)
        )
        source -= 1
    tl.store(
        out
        + batch_idx * stride_outb
        + head_idx * stride_outh
        + query_chunk * stride_outc
        + offs_s[:, None] * stride_outs
        + offs_d[None, :],
        output,
        mask=mask_s[:, None] & mask_d[None, :],
    )


def can_use_fused_affine_history_mix(
    local_output: torch.Tensor,
    local_readout: torch.Tensor,
    chunk_a: torch.Tensor,
    chunk_b: torch.Tensor,
    weights: torch.Tensor,
) -> bool:
    if os.environ.get("GDN_DISABLE_TRITON_AFFINE_HISTORY_MIX", ""):
        return False
    tensors = (local_output, local_readout, chunk_a, chunk_b, weights)
    if not all((t.is_cuda for t in tensors)):
        return False
    if local_output.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return False
    if local_readout.dtype != local_output.dtype:
        return False
    state_dtype_supported = (
        chunk_a.dtype == torch.float32
        and chunk_b.dtype == torch.float32
        or (
            chunk_a.dtype == local_output.dtype
            and chunk_b.dtype == local_output.dtype
            and (chunk_a.dtype in (torch.bfloat16, torch.float16))
        )
    )
    if not state_dtype_supported:
        return False
    if weights.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if local_output.dim() != 5 or local_readout.dim() != 5:
        return False
    batch, heads, chunks, seq_len, value_dim = local_output.shape
    key_dim = local_readout.shape[-1]
    if local_readout.shape[:4] != (batch, heads, chunks, seq_len):
        return False
    if chunk_a.shape != (batch, heads, chunks, key_dim, key_dim):
        return False
    if chunk_b.shape != (batch, heads, chunks, key_dim, value_dim):
        return False
    if weights.shape != (batch, heads, chunks, seq_len, chunks):
        return False
    return (
        0 < chunks <= 128
        and 0 < seq_len <= 2048
        and (key_dim in (64, 128, 256))
        and (64 <= value_dim <= 256)
        and (
            key_dim <= 128
            or (key_dim == value_dim == 256 and chunk_a.dtype != torch.float32)
        )
        and all((t.stride(-1) == 1 for t in tensors))
    )


def fused_affine_history_mix(
    local_output: torch.Tensor,
    local_readout: torch.Tensor,
    chunk_a: torch.Tensor,
    chunk_b: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    if not can_use_fused_affine_history_mix(
        local_output, local_readout, chunk_a, chunk_b, weights
    ):
        raise ValueError("unsupported affine history mix inputs")
    batch, heads, chunks, seq_len, value_dim = local_output.shape
    key_dim = local_readout.shape[-1]
    if chunks == 1:
        return local_output
    out = torch.empty_like(local_output)
    if key_dim == 256:
        block_s = 16
        readout_scratch = torch.empty(
            local_readout.shape, device=local_readout.device, dtype=torch.float32
        )
        grid = (batch * heads * chunks * triton.cdiv(seq_len, block_s),)
        _affine_history_mix_wide_tiled_k_kernel[grid](
            local_output,
            local_readout,
            chunk_a,
            chunk_b,
            weights,
            weights,
            weights,
            readout_scratch,
            out,
            local_output.stride(0),
            local_output.stride(1),
            local_output.stride(2),
            local_output.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            local_readout.stride(0),
            local_readout.stride(1),
            local_readout.stride(2),
            local_readout.stride(3),
            chunk_a.stride(0),
            chunk_a.stride(1),
            chunk_a.stride(2),
            chunk_a.stride(3),
            chunk_b.stride(0),
            chunk_b.stride(1),
            chunk_b.stride(2),
            chunk_b.stride(3),
            weights.stride(0),
            weights.stride(1),
            weights.stride(2),
            weights.stride(3),
            weights.stride(4),
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            heads,
            chunks,
            seq_len,
            key_dim,
            value_dim,
            1,
            1.0,
            0.0,
            1.0,
            False,
            FUSE_ROUTE=False,
            STATE_BF16=chunk_a.dtype == torch.bfloat16,
            BLOCK_S=block_s,
            BLOCK_D=256,
            BLOCK_K=64,
            num_warps=8,
            num_stages=1,
        )
        return out
    wide_state = key_dim > 128
    block_s = 16 if wide_state else 32
    grid = (batch * heads * chunks * triton.cdiv(seq_len, block_s),)
    _affine_history_mix_kernel[grid](
        local_output,
        local_readout,
        chunk_a,
        chunk_b,
        weights,
        weights,
        weights,
        weights,
        out,
        local_output.stride(0),
        local_output.stride(1),
        local_output.stride(2),
        local_output.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        local_readout.stride(0),
        local_readout.stride(1),
        local_readout.stride(2),
        local_readout.stride(3),
        chunk_a.stride(0),
        chunk_a.stride(1),
        chunk_a.stride(2),
        chunk_a.stride(3),
        chunk_b.stride(0),
        chunk_b.stride(1),
        chunk_b.stride(2),
        chunk_b.stride(3),
        weights.stride(0),
        weights.stride(1),
        weights.stride(2),
        weights.stride(3),
        weights.stride(4),
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        heads,
        chunks,
        seq_len,
        key_dim,
        value_dim,
        1,
        1.0,
        0.0,
        1.0,
        False,
        FUSE_ROUTE=False,
        Q_BF16=False,
        Q_FP16=False,
        BLOCK_S=block_s,
        BLOCK_D=triton.next_power_of_2(key_dim),
        BLOCK_V=triton.next_power_of_2(value_dim),
        num_warps=4 if wide_state else 8,
        num_stages=1 if wide_state else 3,
    )
    return out


def can_use_fused_affine_history_route_mix_wide(
    local_output: torch.Tensor,
    local_readout: torch.Tensor,
    chunk_a: torch.Tensor,
    chunk_b: torch.Tensor,
    route_q: torch.Tensor,
    pooled: torch.Tensor,
) -> bool:
    if os.environ.get("GDN_DISABLE_TRITON_AFFINE_HISTORY_ROUTE_MIX", ""):
        return False
    tensors = (local_output, local_readout, chunk_a, chunk_b, route_q, pooled)
    if not all((x.is_cuda for x in tensors)):
        return False
    if local_output.dim() != 5 or local_readout.shape != local_output.shape:
        return False
    batch, heads, chunks, chunk_size, dim = local_output.shape
    return (
        dim == 256
        and chunks > 1
        and (chunk_size <= 2048)
        and (local_output.dtype in (torch.bfloat16, torch.float16))
        and (local_readout.dtype == local_output.dtype)
        and (chunk_a.dtype == chunk_b.dtype == local_output.dtype)
        and (chunk_a.shape == (batch, heads, chunks, dim, dim))
        and (chunk_b.shape == (batch, heads, chunks, dim, dim))
        and (route_q.shape == (batch, heads, chunks * chunk_size, dim))
        and (pooled.dim() == 5)
        and (pooled.shape[:3] == (batch, heads, chunks))
        and (pooled.shape[-1] == dim)
        and (0 < pooled.shape[3] <= 32)
        and all((x.stride(-1) == 1 for x in tensors))
    )


def fused_affine_history_route_mix_wide(
    local_output: torch.Tensor,
    local_readout: torch.Tensor,
    chunk_a: torch.Tensor,
    chunk_b: torch.Tensor,
    route_q: torch.Tensor,
    pooled: torch.Tensor,
    *,
    router_logit_scale: float,
    sigmoid_bias: float,
    sigmoid_temperature: float,
    use_logmean: bool,
) -> torch.Tensor:
    if not can_use_fused_affine_history_route_mix_wide(
        local_output, local_readout, chunk_a, chunk_b, route_q, pooled
    ):
        raise ValueError("unsupported wide fused route/history inputs")
    if sigmoid_temperature <= 0:
        raise ValueError("sigmoid_temperature must be positive")
    batch, heads, chunks, seq_len, dim = local_output.shape
    buckets = pooled.shape[3]
    out = torch.empty_like(local_output)
    scratch = torch.empty_like(local_readout, dtype=torch.float32)
    block_s = 16
    grid = (batch * heads * chunks * triton.cdiv(seq_len, block_s),)
    _affine_history_mix_wide_tiled_k_kernel[grid](
        local_output,
        local_readout,
        chunk_a,
        chunk_b,
        local_output,
        route_q,
        pooled,
        scratch,
        out,
        local_output.stride(0),
        local_output.stride(1),
        local_output.stride(2),
        local_output.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        local_readout.stride(0),
        local_readout.stride(1),
        local_readout.stride(2),
        local_readout.stride(3),
        chunk_a.stride(0),
        chunk_a.stride(1),
        chunk_a.stride(2),
        chunk_a.stride(3),
        chunk_b.stride(0),
        chunk_b.stride(1),
        chunk_b.stride(2),
        chunk_b.stride(3),
        0,
        0,
        0,
        0,
        0,
        route_q.stride(0),
        route_q.stride(1),
        route_q.stride(2),
        pooled.stride(0),
        pooled.stride(1),
        pooled.stride(2),
        pooled.stride(3),
        heads,
        chunks,
        seq_len,
        dim,
        dim,
        buckets,
        float(dim ** (-0.5) * router_logit_scale),
        float(sigmoid_bias),
        float(1.0 / sigmoid_temperature),
        bool(use_logmean),
        FUSE_ROUTE=True,
        STATE_BF16=local_output.dtype == torch.bfloat16,
        BLOCK_S=block_s,
        BLOCK_D=256,
        BLOCK_K=64,
        num_warps=8,
        num_stages=1,
    )
    return out


def fused_affine_history_mix_token_major(
    local_output: torch.Tensor,
    local_readout: torch.Tensor,
    chunk_a: torch.Tensor,
    chunk_b: torch.Tensor,
    weights: torch.Tensor,
    *,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Mix FLA local outputs and write directly as contiguous ``[B, T, H, V]``.

    Besides accepting FLA's native BF16/FP16 local output, this folds the
    former FP32 materialization, final dtype conversion, and BHCSV-to-BTHV
    layout conversion into the history kernel's loads/stores.
    """
    if not can_use_fused_affine_history_mix(
        local_output, local_readout, chunk_a, chunk_b, weights
    ):
        raise ValueError("unsupported affine history mix inputs")
    if output_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(f"unsupported output dtype: {output_dtype}")
    batch, heads, chunks, seq_len, value_dim = local_output.shape
    key_dim = local_readout.shape[-1]
    out = torch.empty(
        batch,
        chunks * seq_len,
        heads,
        value_dim,
        device=local_output.device,
        dtype=output_dtype,
    )
    wide_state = key_dim > 128
    block_s = 16 if wide_state else 32
    grid = (batch * heads * chunks * triton.cdiv(seq_len, block_s),)
    _affine_history_mix_kernel[grid](
        local_output,
        local_readout,
        chunk_a,
        chunk_b,
        weights,
        weights,
        weights,
        weights,
        out,
        local_output.stride(0),
        local_output.stride(1),
        local_output.stride(2),
        local_output.stride(3),
        out.stride(0),
        out.stride(2),
        seq_len * out.stride(1),
        out.stride(1),
        local_readout.stride(0),
        local_readout.stride(1),
        local_readout.stride(2),
        local_readout.stride(3),
        chunk_a.stride(0),
        chunk_a.stride(1),
        chunk_a.stride(2),
        chunk_a.stride(3),
        chunk_b.stride(0),
        chunk_b.stride(1),
        chunk_b.stride(2),
        chunk_b.stride(3),
        weights.stride(0),
        weights.stride(1),
        weights.stride(2),
        weights.stride(3),
        weights.stride(4),
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        heads,
        chunks,
        seq_len,
        key_dim,
        value_dim,
        1,
        1.0,
        0.0,
        1.0,
        False,
        FUSE_ROUTE=False,
        Q_BF16=False,
        Q_FP16=False,
        BLOCK_S=block_s,
        BLOCK_D=triton.next_power_of_2(key_dim),
        BLOCK_V=triton.next_power_of_2(value_dim),
        num_warps=4 if wide_state else 8,
        num_stages=1 if wide_state else 3,
    )
    return out


def fused_affine_history_mix_token_major_joint(
    local_output: torch.Tensor,
    local_readout: torch.Tensor,
    chunk_joint: torch.Tensor,
    weights: torch.Tensor,
    *,
    output_dtype: torch.dtype,
    block_s: int = 32,
    num_warps: int = 8,
    dot_precision: str = "tf32",
    skip_zero_tiles: bool = False,
) -> torch.Tensor:
    """Use FLA's contiguous ``[B|A]`` state in one wide history matmul."""
    if output_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(f"unsupported output dtype: {output_dtype}")
    if block_s not in (16, 32) or num_warps not in (4, 8):
        raise ValueError("joint history tuning supports block_s=16/32 and warps=4/8")
    if dot_precision not in ("tf32", "fp16_tf32", "fp16", "bf16"):
        raise ValueError("dot_precision must be 'tf32', 'fp16_tf32', 'fp16', or 'bf16'")
    if not all(
        (x.is_cuda for x in (local_output, local_readout, chunk_joint, weights))
    ):
        raise ValueError("joint history mix requires CUDA tensors")
    if local_output.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(f"unsupported local output dtype: {local_output.dtype}")
    expected_state_dtype = (
        torch.float16
        if dot_precision in ("fp16_tf32", "fp16")
        else torch.bfloat16 if dot_precision == "bf16" else torch.float32
    )
    if (
        local_readout.dtype != local_output.dtype
        or chunk_joint.dtype != expected_state_dtype
    ):
        raise ValueError("unsupported joint history input dtypes")
    batch, heads, chunks, seq_len, value_dim = local_output.shape
    key_dim = local_readout.shape[-1]
    if key_dim != value_dim or key_dim not in (64, 128, 256):
        raise ValueError("joint history kernel requires equal power-of-two K/V dims")
    if local_readout.shape[:4] != (batch, heads, chunks, seq_len):
        raise ValueError("local readout shape mismatch")
    if chunk_joint.shape != (batch, heads, chunks, key_dim, value_dim + key_dim):
        raise ValueError("joint chunk-state shape mismatch")
    if weights.shape != (batch, heads, chunks, seq_len, chunks):
        raise ValueError("history weight shape mismatch")
    if not all(
        (x.stride(-1) == 1 for x in (local_output, local_readout, chunk_joint, weights))
    ):
        raise ValueError("joint history inputs require contiguous innermost dimensions")
    out = torch.empty(
        batch,
        chunks * seq_len,
        heads,
        value_dim,
        device=local_output.device,
        dtype=output_dtype,
    )
    block_d = triton.next_power_of_2(key_dim)
    block_v = triton.next_power_of_2(value_dim)
    grid = (batch * heads * chunks * triton.cdiv(seq_len, block_s),)
    _affine_history_mix_joint_kernel[grid](
        local_output,
        local_readout,
        chunk_joint,
        weights,
        out,
        local_output.stride(0),
        local_output.stride(1),
        local_output.stride(2),
        local_output.stride(3),
        local_readout.stride(0),
        local_readout.stride(1),
        local_readout.stride(2),
        local_readout.stride(3),
        chunk_joint.stride(0),
        chunk_joint.stride(1),
        chunk_joint.stride(2),
        chunk_joint.stride(3),
        weights.stride(0),
        weights.stride(1),
        weights.stride(2),
        weights.stride(3),
        weights.stride(4),
        out.stride(0),
        out.stride(2),
        seq_len * out.stride(1),
        out.stride(1),
        heads,
        chunks,
        seq_len,
        key_dim,
        value_dim,
        BLOCK_S=block_s,
        BLOCK_D=block_d,
        BLOCK_V=block_v,
        BLOCK_J=block_d + block_v,
        DOT_FP16=dot_precision == "fp16",
        DOT_BF16=dot_precision == "bf16",
        SKIP_ZERO_TILES=bool(skip_zero_tiles),
        num_warps=num_warps,
    )
    return out


def can_use_fused_sparse_affine_history_mix_token_major_joint(
    local_output: torch.Tensor,
    local_readout: torch.Tensor,
    chunk_joint: torch.Tensor,
    weights: torch.Tensor,
) -> bool:
    if os.environ.get("GDN_DISABLE_TRITON_SPARSE_AFFINE_HISTORY_MIX", ""):
        return False
    tensors = (local_output, local_readout, chunk_joint, weights)
    if not all((x.is_cuda and x.stride(-1) == 1 for x in tensors)):
        return False
    if local_output.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return False
    if local_readout.dtype != local_output.dtype:
        return False
    if chunk_joint.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        return False
    if weights.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        return False
    if local_output.dim() != 5 or local_readout.dim() != 5:
        return False
    batch, heads, chunks, seq_len, value_dim = local_output.shape
    key_dim = local_readout.shape[-1]
    return (
        1 <= chunks <= 32
        and 16 <= seq_len <= 2048
        and (key_dim == value_dim)
        and (key_dim in (64, 128, 256))
        and (local_readout.shape == (batch, heads, chunks, seq_len, key_dim))
        and (chunk_joint.shape == (batch, heads, chunks, key_dim, value_dim + key_dim))
        and (weights.shape == (batch, heads, chunks, seq_len, chunks))
    )


def can_use_fused_compact_top1_affine_history_mix_token_major_joint(
    local_output: torch.Tensor,
    local_readout: torch.Tensor,
    chunk_joint: torch.Tensor,
    selected_gates: torch.Tensor,
    selected_sources: torch.Tensor,
    *,
    block_s: int = 32,
) -> bool:
    tensors = (
        local_output,
        local_readout,
        chunk_joint,
        selected_gates,
        selected_sources,
    )
    if not all((x.is_cuda and x.stride(-1) == 1 for x in tensors)):
        return False
    if local_output.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return False
    if local_readout.dtype != local_output.dtype:
        return False
    if selected_gates.dtype != local_output.dtype:
        return False
    if selected_sources.dtype != torch.int32:
        return False
    if local_output.dim() != 5 or local_readout.dim() != 5:
        return False
    batch, heads, chunks, seq_len, value_dim = local_output.shape
    key_dim = local_readout.shape[-1]
    return (
        block_s == 32
        and seq_len % block_s == 0
        and (1 <= chunks <= 32)
        and (key_dim == value_dim)
        and (key_dim in (64, 128, 256))
        and (local_readout.shape == (batch, heads, chunks, seq_len, key_dim))
        and (
            chunk_joint.shape
            in (
                (batch, heads, chunks, key_dim, value_dim),
                (batch, heads, chunks, key_dim, value_dim + key_dim),
            )
        )
        and (selected_gates.shape == (batch, heads, chunks * seq_len))
        and (selected_sources.shape == (batch, heads, chunks * (seq_len // block_s)))
    )


def fused_compact_top1_affine_history_mix_token_major_joint(
    local_output: torch.Tensor,
    local_readout: torch.Tensor,
    chunk_joint: torch.Tensor,
    selected_gates: torch.Tensor,
    selected_sources: torch.Tensor,
    *,
    output_dtype: torch.dtype,
    block_s: int = 32,
    num_warps: int = 8,
    dot_precision: str = "fp16",
) -> torch.Tensor:
    if not can_use_fused_compact_top1_affine_history_mix_token_major_joint(
        local_output,
        local_readout,
        chunk_joint,
        selected_gates,
        selected_sources,
        block_s=block_s,
    ):
        raise ValueError("unsupported compact Top-1 joint history inputs")
    if output_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(f"unsupported output dtype: {output_dtype}")
    if num_warps not in (4, 8):
        raise ValueError("num_warps must be 4 or 8")
    if dot_precision not in ("tf32", "fp16_tf32", "fp16", "bf16"):
        raise ValueError(f"unsupported dot precision: {dot_precision}")
    expected_state_dtype = (
        torch.float16
        if dot_precision in ("fp16_tf32", "fp16")
        else torch.bfloat16 if dot_precision == "bf16" else torch.float32
    )
    if chunk_joint.dtype != expected_state_dtype:
        raise ValueError(
            f"dot precision {dot_precision} requires {expected_state_dtype} state"
        )
    batch, heads, chunks, seq_len, value_dim = local_output.shape
    key_dim = local_readout.shape[-1]
    out = torch.empty(
        batch,
        chunks * seq_len,
        heads,
        value_dim,
        device=local_output.device,
        dtype=output_dtype,
    )
    grid = (batch * heads * chunks * triton.cdiv(seq_len, block_s),)
    _compact_top1_affine_history_mix_joint_kernel[grid](
        local_output,
        local_readout,
        chunk_joint,
        selected_gates,
        selected_sources,
        out,
        local_output.stride(0),
        local_output.stride(1),
        local_output.stride(2),
        local_output.stride(3),
        local_readout.stride(0),
        local_readout.stride(1),
        local_readout.stride(2),
        local_readout.stride(3),
        chunk_joint.stride(0),
        chunk_joint.stride(1),
        chunk_joint.stride(2),
        chunk_joint.stride(3),
        selected_gates.stride(0),
        selected_gates.stride(1),
        selected_gates.stride(2),
        selected_sources.stride(0),
        selected_sources.stride(1),
        selected_sources.stride(2),
        out.stride(0),
        out.stride(2),
        seq_len * out.stride(1),
        out.stride(1),
        heads,
        chunks,
        seq_len,
        key_dim,
        value_dim,
        BLOCK_S=block_s,
        BLOCK_D=triton.next_power_of_2(key_dim),
        BLOCK_V=triton.next_power_of_2(value_dim),
        DOT_FP16=dot_precision == "fp16",
        DOT_BF16=dot_precision == "bf16",
        num_warps=num_warps,
    )
    return out


def fused_sparse_affine_history_mix_token_major_joint(
    local_output: torch.Tensor,
    local_readout: torch.Tensor,
    chunk_joint: torch.Tensor,
    weights: torch.Tensor,
    *,
    output_dtype: torch.dtype,
    block_s: int = 32,
    num_warps: int = 8,
    dot_precision: str = "tf32",
    top_k: int = 2,
    top1_recent_of_top2: bool = False,
) -> torch.Tensor:
    """Tile-wise historical Top-2 affine scan for sparse-router inference."""
    if not can_use_fused_sparse_affine_history_mix_token_major_joint(
        local_output, local_readout, chunk_joint, weights
    ):
        raise ValueError("unsupported sparse joint history inputs")
    if block_s not in (16, 32, 64, 128) or num_warps not in (4, 8):
        raise ValueError(
            "sparse joint history supports block_s=16/32/64/128 and warps=4/8"
        )
    if output_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(f"unsupported output dtype: {output_dtype}")
    if dot_precision not in ("tf32", "fp16_tf32", "fp16", "bf16"):
        raise ValueError(f"unsupported dot precision: {dot_precision}")
    if top_k not in (1, 2):
        raise ValueError("sparse joint history supports top_k=1 or 2")
    expected_state_dtype = (
        torch.float16
        if dot_precision in ("fp16_tf32", "fp16")
        else torch.bfloat16 if dot_precision == "bf16" else torch.float32
    )
    if chunk_joint.dtype != expected_state_dtype:
        raise ValueError(
            f"dot precision {dot_precision} requires {expected_state_dtype} state"
        )
    batch, heads, chunks, seq_len, value_dim = local_output.shape
    key_dim = local_readout.shape[-1]
    out = torch.empty(
        batch,
        chunks * seq_len,
        heads,
        value_dim,
        device=local_output.device,
        dtype=output_dtype,
    )
    grid = (batch * heads * chunks * triton.cdiv(seq_len, block_s),)
    block_d = triton.next_power_of_2(key_dim)
    _sparse_top2_affine_history_mix_joint_kernel[grid](
        local_output,
        local_readout,
        chunk_joint,
        weights,
        out,
        local_output.stride(0),
        local_output.stride(1),
        local_output.stride(2),
        local_output.stride(3),
        local_readout.stride(0),
        local_readout.stride(1),
        local_readout.stride(2),
        local_readout.stride(3),
        chunk_joint.stride(0),
        chunk_joint.stride(1),
        chunk_joint.stride(2),
        chunk_joint.stride(3),
        weights.stride(0),
        weights.stride(1),
        weights.stride(2),
        weights.stride(3),
        weights.stride(4),
        out.stride(0),
        out.stride(2),
        seq_len * out.stride(1),
        out.stride(1),
        heads,
        chunks,
        seq_len,
        key_dim,
        value_dim,
        BLOCK_S=block_s,
        BLOCK_D=block_d,
        BLOCK_V=triton.next_power_of_2(value_dim),
        BLOCK_J=2 * block_d,
        BLOCK_C=triton.next_power_of_2(chunks),
        DOT_FP16=dot_precision == "fp16",
        DOT_BF16=dot_precision == "bf16",
        TOP_K=top_k,
        TOP1_RECENT_OF_TOP2=top1_recent_of_top2,
        num_warps=num_warps,
    )
    return out


def can_use_fused_sparse_bonly_history_mix_token_major(
    local_output: torch.Tensor,
    query: torch.Tensor,
    chunk_state: torch.Tensor,
    weights: torch.Tensor,
) -> bool:
    if os.environ.get("GDN_DISABLE_TRITON_SPARSE_BONLY_HISTORY_MIX", ""):
        return False
    tensors = (local_output, query, chunk_state, weights)
    if not all((x.is_cuda for x in tensors)):
        return False
    if not all((x.stride(-1) == 1 for x in (local_output, chunk_state, weights))):
        return False
    if local_output.dim() != 5 or query.dim() != 5 or chunk_state.dim() != 5:
        return False
    batch, heads, chunks, seq_len, value_dim = local_output.shape
    key_dim = query.shape[-1]
    return (
        1 <= chunks <= 32
        and 16 <= seq_len <= 2048
        and (key_dim == value_dim)
        and (key_dim in (64, 128, 256))
        and (query.shape == (batch, heads, chunks, seq_len, key_dim))
        and (chunk_state.shape == (batch, heads, chunks, key_dim, value_dim))
        and (weights.shape == (batch, heads, chunks, seq_len, chunks))
        and (local_output.dtype in (torch.bfloat16, torch.float16, torch.float32))
        and (query.dtype in (torch.bfloat16, torch.float16, torch.float32))
        and (chunk_state.dtype in (torch.bfloat16, torch.float16, torch.float32))
        and (weights.dtype in (torch.bfloat16, torch.float16, torch.float32))
    )


def fused_sparse_bonly_history_mix_token_major(
    local_output: torch.Tensor,
    query: torch.Tensor,
    chunk_state: torch.Tensor,
    weights: torch.Tensor,
    *,
    output_dtype: torch.dtype,
    block_s: int = 32,
    num_warps: int = 8,
) -> torch.Tensor:
    if not can_use_fused_sparse_bonly_history_mix_token_major(
        local_output, query, chunk_state, weights
    ):
        raise ValueError("unsupported sparse B-only history inputs")
    if block_s not in (16, 32) or num_warps not in (4, 8):
        raise ValueError("sparse B-only history supports block_s=16/32 and warps=4/8")
    if output_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(f"unsupported output dtype: {output_dtype}")
    batch, heads, chunks, seq_len, value_dim = local_output.shape
    key_dim = query.shape[-1]
    out = torch.empty(
        batch,
        chunks * seq_len,
        heads,
        value_dim,
        device=local_output.device,
        dtype=output_dtype,
    )
    grid = (batch * heads * chunks * triton.cdiv(seq_len, block_s),)
    _sparse_top2_bonly_history_mix_kernel[grid](
        local_output,
        query,
        chunk_state,
        weights,
        out,
        local_output.stride(0),
        local_output.stride(1),
        local_output.stride(2),
        local_output.stride(3),
        query.stride(0),
        query.stride(1),
        query.stride(2),
        query.stride(3),
        query.stride(4),
        chunk_state.stride(0),
        chunk_state.stride(1),
        chunk_state.stride(2),
        chunk_state.stride(3),
        weights.stride(0),
        weights.stride(1),
        weights.stride(2),
        weights.stride(3),
        weights.stride(4),
        out.stride(0),
        out.stride(2),
        seq_len * out.stride(1),
        out.stride(1),
        heads,
        chunks,
        seq_len,
        key_dim,
        value_dim,
        float(key_dim ** (-0.5)),
        BLOCK_S=block_s,
        BLOCK_D=triton.next_power_of_2(key_dim),
        BLOCK_V=triton.next_power_of_2(value_dim),
        BLOCK_C=triton.next_power_of_2(chunks),
        num_warps=num_warps,
    )
    return out


def can_use_fused_affine_history_route_mix_token_major_joint(
    local_output: torch.Tensor,
    local_readout: torch.Tensor,
    chunk_joint: torch.Tensor,
    route_q: torch.Tensor,
    route_q_weight: torch.Tensor,
    pooled: torch.Tensor,
) -> bool:
    if os.environ.get("GDN_DISABLE_TRITON_AFFINE_HISTORY_ROUTE_MIX", ""):
        return False
    tensors = (
        local_output,
        local_readout,
        chunk_joint,
        route_q,
        route_q_weight,
        pooled,
    )
    if not all((x.is_cuda for x in tensors)):
        return False
    if local_output.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return False
    if local_readout.dtype != local_output.dtype or chunk_joint.dtype != torch.float32:
        return False
    batch, heads, chunks, seq_len, value_dim = local_output.shape
    key_dim = local_readout.shape[-1]
    if key_dim != value_dim or key_dim not in (64, 128, 256):
        return False
    if local_readout.shape[:4] != (batch, heads, chunks, seq_len):
        return False
    if chunk_joint.shape != (batch, heads, chunks, key_dim, value_dim + key_dim):
        return False
    if route_q.shape != (batch, chunks * seq_len, heads, key_dim):
        return False
    if route_q_weight.shape != (heads, key_dim, key_dim):
        return False
    if pooled.shape[:3] != (batch, heads, chunks):
        return False
    if pooled.shape[-1] != key_dim or not 0 < pooled.shape[3] <= 256:
        return False
    return all((x.stride(-1) == 1 for x in tensors))


def fused_affine_history_route_mix_token_major_joint(
    local_output: torch.Tensor,
    local_readout: torch.Tensor,
    chunk_joint: torch.Tensor,
    route_q: torch.Tensor,
    route_q_weight: torch.Tensor,
    pooled: torch.Tensor,
    *,
    router_logit_scale: float,
    sigmoid_bias: float,
    sigmoid_temperature: float,
    use_logmean: bool,
) -> torch.Tensor:
    if not can_use_fused_affine_history_route_mix_token_major_joint(
        local_output, local_readout, chunk_joint, route_q, route_q_weight, pooled
    ):
        raise ValueError("unsupported joint affine history-route inputs")
    if sigmoid_temperature <= 0:
        raise ValueError("sigmoid_temperature must be positive")
    batch, heads, chunks, seq_len, value_dim = local_output.shape
    key_dim = local_readout.shape[-1]
    buckets = pooled.shape[3]
    out = torch.empty(
        batch,
        chunks * seq_len,
        heads,
        value_dim,
        device=local_output.device,
        dtype=route_q.dtype,
    )
    block_s = 32
    block_d = triton.next_power_of_2(key_dim)
    block_v = triton.next_power_of_2(value_dim)
    grid = (batch * heads * chunks * triton.cdiv(seq_len, block_s),)
    _affine_history_route_mix_joint_kernel[grid](
        local_output,
        local_readout,
        chunk_joint,
        route_q,
        route_q_weight,
        pooled,
        out,
        local_output.stride(0),
        local_output.stride(1),
        local_output.stride(2),
        local_output.stride(3),
        local_readout.stride(0),
        local_readout.stride(1),
        local_readout.stride(2),
        local_readout.stride(3),
        chunk_joint.stride(0),
        chunk_joint.stride(1),
        chunk_joint.stride(2),
        chunk_joint.stride(3),
        route_q.stride(0),
        route_q.stride(1),
        route_q.stride(2),
        route_q_weight.stride(0),
        pooled.stride(0),
        pooled.stride(1),
        pooled.stride(2),
        pooled.stride(3),
        out.stride(0),
        out.stride(2),
        seq_len * out.stride(1),
        out.stride(1),
        heads,
        chunks,
        seq_len,
        key_dim,
        value_dim,
        buckets,
        float(key_dim ** (-0.5) * router_logit_scale),
        float(sigmoid_bias),
        float(1.0 / sigmoid_temperature),
        bool(use_logmean),
        Q_BF16=route_q.dtype == torch.bfloat16,
        Q_FP16=route_q.dtype == torch.float16,
        BLOCK_S=block_s,
        BLOCK_D=block_d,
        BLOCK_V=block_v,
        BLOCK_J=block_d + block_v,
        BLOCK_P=min(32, triton.next_power_of_2(buckets)),
        num_warps=8,
    )
    return out


def can_use_fused_affine_history_route_mix(
    local_output: torch.Tensor,
    local_readout: torch.Tensor,
    chunk_a: torch.Tensor,
    chunk_b: torch.Tensor,
    route_q: torch.Tensor,
    route_q_weight: torch.Tensor,
    pooled: torch.Tensor,
) -> bool:
    if os.environ.get("GDN_DISABLE_TRITON_AFFINE_HISTORY_ROUTE_MIX", ""):
        return False
    tensors = (local_output, local_readout, chunk_a, chunk_b)
    if not all((t.is_cuda for t in tensors)):
        return False
    if local_output.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        return False
    if local_readout.dtype != local_output.dtype:
        return False
    if chunk_a.dtype != torch.float32 or chunk_b.dtype != torch.float32:
        return False
    if local_output.dim() != 5 or local_readout.dim() != 5:
        return False
    batch, heads, chunks, seq_len, value_dim = local_output.shape
    key_dim = local_readout.shape[-1]
    if local_readout.shape[:4] != (batch, heads, chunks, seq_len):
        return False
    if chunk_a.shape != (batch, heads, chunks, key_dim, key_dim):
        return False
    if chunk_b.shape != (batch, heads, chunks, key_dim, value_dim):
        return False
    if not (
        0 < chunks <= 128
        and 0 < seq_len <= 2048
        and (64 <= key_dim <= 256)
        and (64 <= value_dim <= 256)
        and all((t.stride(-1) == 1 for t in tensors))
    ):
        return False
    if not all((x.is_cuda for x in (route_q, route_q_weight, pooled))):
        return False
    if route_q.shape != (batch, chunks * seq_len, heads, key_dim):
        return False
    if route_q_weight.shape != (heads, key_dim, key_dim):
        return False
    if pooled.shape[:3] != (batch, heads, chunks):
        return False
    if pooled.shape[-1] != key_dim or not 0 < pooled.shape[3] <= 32:
        return False
    return all((x.stride(-1) == 1 for x in (route_q, route_q_weight, pooled)))


def fused_affine_history_route_mix(
    local_output: torch.Tensor,
    local_readout: torch.Tensor,
    chunk_a: torch.Tensor,
    chunk_b: torch.Tensor,
    route_q: torch.Tensor,
    route_q_weight: torch.Tensor,
    pooled: torch.Tensor,
    *,
    router_logit_scale: float,
    sigmoid_bias: float,
    sigmoid_temperature: float,
    use_logmean: bool,
) -> torch.Tensor:
    if not can_use_fused_affine_history_route_mix(
        local_output, local_readout, chunk_a, chunk_b, route_q, route_q_weight, pooled
    ):
        raise ValueError("unsupported affine history-route mix inputs")
    if sigmoid_temperature <= 0:
        raise ValueError("sigmoid_temperature must be positive")
    batch, heads, chunks, seq_len, value_dim = local_output.shape
    key_dim = local_readout.shape[-1]
    buckets = pooled.shape[3]
    out = torch.empty_like(local_output)
    block_s = 32
    grid = (batch * heads * chunks * triton.cdiv(seq_len, block_s),)
    _affine_history_mix_kernel[grid](
        local_output,
        local_readout,
        chunk_a,
        chunk_b,
        route_q,
        route_q,
        route_q_weight,
        pooled,
        out,
        local_output.stride(0),
        local_output.stride(1),
        local_output.stride(2),
        local_output.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        local_readout.stride(0),
        local_readout.stride(1),
        local_readout.stride(2),
        local_readout.stride(3),
        chunk_a.stride(0),
        chunk_a.stride(1),
        chunk_a.stride(2),
        chunk_a.stride(3),
        chunk_b.stride(0),
        chunk_b.stride(1),
        chunk_b.stride(2),
        chunk_b.stride(3),
        0,
        0,
        0,
        0,
        0,
        route_q.stride(0),
        route_q.stride(1),
        route_q.stride(2),
        route_q_weight.stride(0),
        pooled.stride(0),
        pooled.stride(1),
        pooled.stride(2),
        pooled.stride(3),
        heads,
        chunks,
        seq_len,
        key_dim,
        value_dim,
        buckets,
        float(key_dim ** (-0.5) * router_logit_scale),
        float(sigmoid_bias),
        float(1.0 / sigmoid_temperature),
        bool(use_logmean),
        FUSE_ROUTE=True,
        Q_BF16=route_q.dtype == torch.bfloat16,
        Q_FP16=route_q.dtype == torch.float16,
        BLOCK_S=block_s,
        BLOCK_D=triton.next_power_of_2(key_dim),
        BLOCK_V=triton.next_power_of_2(value_dim),
        num_warps=8,
    )
    return out
