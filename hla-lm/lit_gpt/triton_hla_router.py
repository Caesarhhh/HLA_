from __future__ import annotations

import math
import os

import torch
import triton
import triton.language as tl


@triton.jit
def _hla_router_sigmoid_fwd_kernel(
    query,
    query_weight,
    pooled,
    weights,
    stride_qb: tl.constexpr,
    stride_qs: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qwh: tl.constexpr,
    stride_pb: tl.constexpr,
    stride_ph: tl.constexpr,
    stride_pc: tl.constexpr,
    stride_pp: tl.constexpr,
    num_tokens: tl.constexpr,
    head_dim: tl.constexpr,
    chunk_size: tl.constexpr,
    scale: tl.constexpr,
    sigmoid_bias: tl.constexpr,
    inv_temperature: tl.constexpr,
    use_logmean: tl.constexpr,
    current_always_on: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    NUM_BUCKETS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_D: tl.constexpr,
    Q_BF16: tl.constexpr,
    Q_FP16: tl.constexpr,
    PROJECT_Q: tl.constexpr,
):
    bh = tl.program_id(0)
    token_block = tl.program_id(1)
    batch_idx = bh // NUM_HEADS
    head_idx = bh % NUM_HEADS

    offs_m = token_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_C * BLOCK_P)
    chunk_idx = offs_n // BLOCK_P
    bucket_idx = offs_n % BLOCK_P

    q_ptrs = (
        query
        + batch_idx * stride_qb
        + offs_m[:, None] * stride_qs
        + head_idx * stride_qh
        + offs_d[None, :]
    )
    q = tl.load(
        q_ptrs,
        mask=(offs_m[:, None] < num_tokens) & (offs_d[None, :] < head_dim),
        other=0.0,
    ).to(tl.float32)
    if PROJECT_Q:
        qw_ptrs = (
            query_weight
            + head_idx * stride_qwh
            + offs_d[:, None] * head_dim
            + offs_d[None, :]
        )
        qw = tl.load(
            qw_ptrs,
            mask=(offs_d[:, None] < head_dim) & (offs_d[None, :] < head_dim),
            other=0.0,
        ).to(tl.float32)
        q = tl.dot(q, qw, input_precision="tf32")
        # Match _headwise_linear's output cast before the fp32 routing score.
        if Q_BF16:
            q = q.to(tl.bfloat16).to(tl.float32)
        elif Q_FP16:
            q = q.to(tl.float16).to(tl.float32)
    p_ptrs = (
        pooled
        + batch_idx * stride_pb
        + head_idx * stride_ph
        + chunk_idx[None, :] * stride_pc
        + bucket_idx[None, :] * stride_pp
        + offs_d[:, None]
    )
    valid_pool = (chunk_idx[None, :] < NUM_CHUNKS) & (
        bucket_idx[None, :] < NUM_BUCKETS
    )
    p = tl.load(
        p_ptrs,
        mask=(offs_d[:, None] < head_dim) & valid_pool,
        other=0.0,
    ).to(tl.float32)

    # Match PyTorch's high-precision CUDA matmul mode used by the Qwen
    # benchmark.  TF32 is materially faster for these fp32 router operands and
    # has the same precision contract as the reference einsum on Ampere+.
    bucket_scores = tl.dot(q, p, input_precision="tf32") * scale
    bucket_scores = tl.where(valid_pool, bucket_scores, -float("inf"))
    bucket_scores = tl.reshape(
        bucket_scores, (BLOCK_M, BLOCK_C, BLOCK_P)
    )
    score_max = tl.max(bucket_scores, axis=2)
    score = score_max + tl.log(
        tl.sum(tl.exp(bucket_scores - score_max[:, :, None]), axis=2)
    )
    if use_logmean:
        score -= math.log(NUM_BUCKETS)

    chunks = tl.arange(0, BLOCK_C)
    current_chunk = offs_m // chunk_size
    visible = (chunks[None, :] <= current_chunk[:, None]) & (
        chunks[None, :] < NUM_CHUNKS
    )
    gates = tl.sigmoid((score + sigmoid_bias) * inv_temperature)
    gates = tl.where(visible, gates, 0.0)
    if current_always_on:
        gates = tl.where(chunks[None, :] == current_chunk[:, None], 1.0, gates)

    out_ptrs = (
        weights
        + ((batch_idx * NUM_HEADS + head_idx) * num_tokens + offs_m[:, None])
        * NUM_CHUNKS
        + chunks[None, :]
    )
    tl.store(
        out_ptrs,
        gates,
        mask=(offs_m[:, None] < num_tokens) & (chunks[None, :] < NUM_CHUNKS),
    )


@triton.jit
def _hla_router_sigmoid_top1_tiles_fwd_kernel(
    query,
    query_weight,
    pooled,
    selected_gates,
    selected_sources,
    stride_qb: tl.constexpr,
    stride_qs: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qwh: tl.constexpr,
    stride_pb: tl.constexpr,
    stride_ph: tl.constexpr,
    stride_pc: tl.constexpr,
    stride_pp: tl.constexpr,
    num_tokens: tl.constexpr,
    head_dim: tl.constexpr,
    chunk_size: tl.constexpr,
    scale: tl.constexpr,
    sigmoid_bias: tl.constexpr,
    inv_temperature: tl.constexpr,
    use_logmean: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    NUM_BUCKETS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_D: tl.constexpr,
    Q_BF16: tl.constexpr,
    Q_FP16: tl.constexpr,
    TOP1_RECENT_OF_TOP2: tl.constexpr,
):
    """Compute only the tile Top-1 source and its token-specific gates.

    This preserves the exact arithmetic and BLOCK_M=32 selection performed by
    ``fused_hla_router_sigmoid`` followed by the sparse affine-history kernel,
    but avoids writing and rereading the unused [token, chunk] gates.
    """

    bh = tl.program_id(0)
    token_block = tl.program_id(1)
    batch_idx = bh // NUM_HEADS
    head_idx = bh % NUM_HEADS

    offs_m = token_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_C * BLOCK_P)
    chunk_idx = offs_n // BLOCK_P
    bucket_idx = offs_n % BLOCK_P

    q = tl.load(
        query
        + batch_idx * stride_qb
        + offs_m[:, None] * stride_qs
        + head_idx * stride_qh
        + offs_d[None, :],
        mask=(offs_m[:, None] < num_tokens) & (offs_d[None, :] < head_dim),
        other=0.0,
    ).to(tl.float32)
    qw = tl.load(
        query_weight
        + head_idx * stride_qwh
        + offs_d[:, None] * head_dim
        + offs_d[None, :],
        mask=(offs_d[:, None] < head_dim) & (offs_d[None, :] < head_dim),
        other=0.0,
    ).to(tl.float32)
    q = tl.dot(q, qw, input_precision="tf32")
    if Q_BF16:
        q = q.to(tl.bfloat16).to(tl.float32)
    elif Q_FP16:
        q = q.to(tl.float16).to(tl.float32)

    valid_pool = (chunk_idx[None, :] < NUM_CHUNKS) & (
        bucket_idx[None, :] < NUM_BUCKETS
    )
    p = tl.load(
        pooled
        + batch_idx * stride_pb
        + head_idx * stride_ph
        + chunk_idx[None, :] * stride_pc
        + bucket_idx[None, :] * stride_pp
        + offs_d[:, None],
        mask=(offs_d[:, None] < head_dim) & valid_pool,
        other=0.0,
    ).to(tl.float32)
    bucket_scores = tl.dot(q, p, input_precision="tf32") * scale
    bucket_scores = tl.where(valid_pool, bucket_scores, -float("inf"))
    bucket_scores = tl.reshape(
        bucket_scores, (BLOCK_M, BLOCK_C, BLOCK_P)
    )
    score_max = tl.max(bucket_scores, axis=2)
    score = score_max + tl.log(
        tl.sum(tl.exp(bucket_scores - score_max[:, :, None]), axis=2)
    )
    if use_logmean:
        score -= math.log(NUM_BUCKETS)

    chunks = tl.arange(0, BLOCK_C)
    # chunk_size is required to be a multiple of BLOCK_M, so a tile never
    # crosses a query-chunk boundary.
    query_chunk = (token_block * BLOCK_M) // chunk_size
    historical = (chunks < query_chunk) & (chunks < NUM_CHUNKS)
    gates = tl.sigmoid((score + sigmoid_bias) * inv_temperature)
    gates = tl.where(historical[None, :], gates, 0.0)
    # Match the materialized router tensor's store/load rounding before the
    # sparse kernel sums gates and performs argmax.
    if Q_BF16:
        gates = gates.to(tl.bfloat16).to(tl.float32)
    elif Q_FP16:
        gates = gates.to(tl.float16).to(tl.float32)

    source_scores = tl.sum(gates, axis=0)
    source_scores = tl.where(historical, source_scores, -float("inf"))
    selected0 = tl.argmax(source_scores, axis=0)
    selected = selected0
    if TOP1_RECENT_OF_TOP2:
        source_scores = tl.where(
            chunks == selected0, -float("inf"), source_scores
        )
        selected1 = tl.argmax(source_scores, axis=0)
        selected = tl.maximum(selected0, selected1)
    selected_gate = tl.sum(
        tl.where(chunks[None, :] == selected, gates, 0.0), axis=1
    )
    selected_gate = tl.where(query_chunk > 0, selected_gate, 0.0)

    tl.store(
        selected_gates + bh * num_tokens + offs_m,
        selected_gate,
        mask=offs_m < num_tokens,
    )
    num_tiles: tl.constexpr = tl.cdiv(num_tokens, BLOCK_M)
    tl.store(selected_sources + bh * num_tiles + token_block, selected)


@triton.jit
def _hla_router_scores_fwd_kernel(
    query,
    query_weight,
    pooled,
    scores,
    stride_qb: tl.constexpr,
    stride_qs: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qwh: tl.constexpr,
    stride_pb: tl.constexpr,
    stride_ph: tl.constexpr,
    stride_pc: tl.constexpr,
    stride_pp: tl.constexpr,
    num_tokens: tl.constexpr,
    head_dim: tl.constexpr,
    scale: tl.constexpr,
    use_logmean: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    NUM_BUCKETS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_D: tl.constexpr,
    Q_BF16: tl.constexpr,
    Q_FP16: tl.constexpr,
    PROJECT_Q: tl.constexpr,
):
    """Compute logsumexp bucket scores without materializing [S,C,P]."""
    bh = tl.program_id(0)
    token_block = tl.program_id(1)
    chunk_block = tl.program_id(2)
    batch_idx = bh // NUM_HEADS
    head_idx = bh % NUM_HEADS

    offs_m = token_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_C * BLOCK_P)
    chunk_idx = chunk_block * BLOCK_C + offs_n // BLOCK_P
    bucket_idx = offs_n % BLOCK_P

    q = tl.load(
        query
        + batch_idx * stride_qb
        + offs_m[:, None] * stride_qs
        + head_idx * stride_qh
        + offs_d[None, :],
        mask=(offs_m[:, None] < num_tokens) & (offs_d[None, :] < head_dim),
        other=0.0,
    ).to(tl.float32)
    if PROJECT_Q:
        qw = tl.load(
            query_weight
            + head_idx * stride_qwh
            + offs_d[:, None] * head_dim
            + offs_d[None, :],
            mask=(offs_d[:, None] < head_dim) & (offs_d[None, :] < head_dim),
            other=0.0,
        ).to(tl.float32)
        q = tl.dot(q, qw, input_precision="tf32")
        if Q_BF16:
            q = q.to(tl.bfloat16).to(tl.float32)
        elif Q_FP16:
            q = q.to(tl.float16).to(tl.float32)

    valid_pool = (chunk_idx[None, :] < NUM_CHUNKS) & (
        bucket_idx[None, :] < NUM_BUCKETS
    )
    p = tl.load(
        pooled
        + batch_idx * stride_pb
        + head_idx * stride_ph
        + chunk_idx[None, :] * stride_pc
        + bucket_idx[None, :] * stride_pp
        + offs_d[:, None],
        mask=(offs_d[:, None] < head_dim) & valid_pool,
        other=0.0,
    ).to(tl.float32)
    bucket_scores = tl.dot(q, p, input_precision="tf32") * scale
    bucket_scores = tl.where(valid_pool, bucket_scores, -float("inf"))
    bucket_scores = tl.reshape(bucket_scores, (BLOCK_M, BLOCK_C, BLOCK_P))
    score_max = tl.max(bucket_scores, axis=2)
    score = score_max + tl.log(
        tl.sum(tl.exp(bucket_scores - score_max[:, :, None]), axis=2)
    )
    if use_logmean:
        score -= math.log(NUM_BUCKETS)

    chunks = chunk_block * BLOCK_C + tl.arange(0, BLOCK_C)
    tl.store(
        scores
        + (bh * num_tokens + offs_m[:, None]) * NUM_CHUNKS
        + chunks[None, :],
        score,
        mask=(offs_m[:, None] < num_tokens) & (chunks[None, :] < NUM_CHUNKS),
    )


@triton.jit
def _hla_router_sigmoid_from_scores_fwd_kernel(
    scores,
    weights,
    num_tokens: tl.constexpr,
    chunk_size: tl.constexpr,
    sigmoid_bias: tl.constexpr,
    inv_temperature: tl.constexpr,
    current_always_on: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Apply the causal sigmoid gate to precomputed [BH, S, C] scores."""
    bh = tl.program_id(0)
    token_block = tl.program_id(1)
    offs_m = token_block * BLOCK_M + tl.arange(0, BLOCK_M)
    chunks = tl.arange(0, BLOCK_C)
    valid_token = offs_m < num_tokens
    valid_chunk = chunks < NUM_CHUNKS
    current_chunk = offs_m // chunk_size
    visible = valid_chunk[None, :] & (
        chunks[None, :] <= current_chunk[:, None]
    )
    score = tl.load(
        scores
        + (bh * num_tokens + offs_m[:, None]) * NUM_CHUNKS
        + chunks[None, :],
        mask=valid_token[:, None] & valid_chunk[None, :],
        other=0.0,
    )
    gates = tl.sigmoid((score + sigmoid_bias) * inv_temperature)
    gates = tl.where(visible, gates, 0.0)
    if current_always_on:
        gates = tl.where(chunks[None, :] == current_chunk[:, None], 1.0, gates)
    tl.store(
        weights
        + (bh * num_tokens + offs_m[:, None]) * NUM_CHUNKS
        + chunks[None, :],
        gates,
        mask=valid_token[:, None] & valid_chunk[None, :],
    )


@triton.jit
def _hla_router_budget_softmax_fwd_kernel(
    scores,
    weights,
    num_tokens: tl.constexpr,
    chunk_size: tl.constexpr,
    inv_temperature: tl.constexpr,
    softmax_budget: tl.constexpr,
    current_always_on: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    bh = tl.program_id(0)
    token_block = tl.program_id(1)
    offs_m = token_block * BLOCK_M + tl.arange(0, BLOCK_M)
    chunks = tl.arange(0, BLOCK_C)
    valid_token = offs_m < num_tokens
    valid_chunk = chunks < NUM_CHUNKS
    current_chunk = offs_m // chunk_size
    historical = valid_chunk[None, :] & (
        chunks[None, :] < current_chunk[:, None]
    )
    logits = tl.load(
        scores
        + (bh * num_tokens + offs_m[:, None]) * NUM_CHUNKS
        + chunks[None, :],
        mask=valid_token[:, None] & valid_chunk[None, :],
        other=-float("inf"),
    ) * inv_temperature
    logits = tl.where(historical, logits, -float("inf"))
    has_history = current_chunk > 0
    logits_max = tl.max(logits, axis=1)
    logits_max = tl.where(has_history, logits_max, 0.0)
    numerator = tl.where(
        historical, tl.exp(logits - logits_max[:, None]), 0.0
    )
    denominator = tl.sum(numerator, axis=1)
    denominator = tl.where(has_history, denominator, 1.0)
    budget = tl.minimum(current_chunk.to(tl.float32), softmax_budget)
    gates = numerator / denominator[:, None] * budget[:, None]
    if current_always_on:
        gates = tl.where(chunks[None, :] == current_chunk[:, None], 1.0, gates)
    tl.store(
        weights
        + (bh * num_tokens + offs_m[:, None]) * NUM_CHUNKS
        + chunks[None, :],
        gates,
        mask=valid_token[:, None] & valid_chunk[None, :],
    )


def can_use_fused_hla_router_sigmoid(
    query: torch.Tensor,
    query_weight: torch.Tensor,
    pooled: torch.Tensor,
    *,
    chunk_size: int,
) -> bool:
    if os.environ.get("GDN_DISABLE_TRITON_HLA_ROUTER", ""):
        return False
    if not (query.is_cuda and query_weight.is_cuda and pooled.is_cuda):
        return False
    if query.dim() != 4 or pooled.dim() != 5:
        return False
    if query.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if pooled.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    batch, num_tokens, heads, head_dim = query.shape
    if query_weight.shape != (heads, head_dim, head_dim):
        return False
    if query_weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if pooled.shape[0] != batch or pooled.shape[1] != heads:
        return False
    if pooled.shape[-1] != head_dim:
        return False
    chunks, buckets = pooled.shape[2:4]
    if not (0 < chunks <= 128 and 0 < buckets <= 64):
        return False
    if not (0 < head_dim <= 256):
        return False
    if num_tokens <= 0 or chunk_size <= 0:
        return False
    if chunks != (num_tokens + chunk_size - 1) // chunk_size:
        return False
    return (
        query.stride(-1) == 1
        and query_weight.stride(-1) == 1
        and pooled.stride(-1) == 1
    )


def fused_hla_router_sigmoid(
    query: torch.Tensor,
    query_weight: torch.Tensor,
    pooled: torch.Tensor,
    *,
    chunk_size: int,
    router_logit_scale: float,
    sigmoid_bias: float,
    sigmoid_temperature: float,
    use_logmean: bool,
    current_always_on: bool,
    block_m: int | None = None,
    num_warps: int | None = None,
) -> torch.Tensor:
    if not can_use_fused_hla_router_sigmoid(
        query, query_weight, pooled, chunk_size=chunk_size
    ):
        raise ValueError("unsupported fused HLA router inputs")
    if sigmoid_temperature <= 0:
        raise ValueError("sigmoid_temperature must be positive")
    batch, num_tokens, heads, head_dim = query.shape
    if block_m is None and num_warps is None:
        # Preserve the original small-shape reduction order used by exact
        # cache regressions; the larger tile is ~18% faster at 4K prefill.
        block_m, num_warps = (
            (32, 4) if num_tokens >= 1024 else (16, 8)
        )
    if block_m not in (16, 32, 64) or num_warps not in (4, 8):
        raise ValueError("router tuning supports block_m=16/32/64 and warps=4/8")
    chunks, buckets = pooled.shape[2:4]
    block_c = triton.next_power_of_2(chunks)
    block_p = triton.next_power_of_2(buckets)
    block_d = triton.next_power_of_2(head_dim)
    # The original single-kernel path is fastest for the short sequences used
    # in training and normal LongBench evaluation.  At 262K, however, 128
    # chunks x 64 meta-query buckets cannot fit in one Triton tile.  Reuse the
    # memory-bounded tiled score kernel from budget-softmax and apply sigmoid
    # in a second small kernel instead of falling back to a gigantic PyTorch
    # [B,H,S,C,P] intermediate.
    # The single kernel's shared-memory use grows with the rounded C x P tile.
    # A 32 x 16 tile already needs 272 KiB at the Qwen production shape, above
    # H200's 227 KiB/block limit.  Route larger tiles through the existing
    # memory-bounded score kernel instead.
    use_single_kernel = chunks <= 32 and buckets <= 32 and chunks * buckets <= 256
    weights = torch.empty(
        batch,
        heads,
        num_tokens,
        chunks,
        device=query.device,
        dtype=query.dtype,
    )
    if use_single_kernel:
        _hla_router_sigmoid_fwd_kernel[
            (batch * heads, triton.cdiv(num_tokens, block_m))
        ](
            query,
            query_weight,
            pooled,
            weights,
            query.stride(0),
            query.stride(1),
            query.stride(2),
            query_weight.stride(0),
            pooled.stride(0),
            pooled.stride(1),
            pooled.stride(2),
            pooled.stride(3),
            num_tokens,
            head_dim,
            int(chunk_size),
            float(head_dim**-0.5 * router_logit_scale),
            float(sigmoid_bias),
            float(1.0 / sigmoid_temperature),
            bool(use_logmean),
            bool(current_always_on),
            NUM_HEADS=heads,
            NUM_CHUNKS=chunks,
            NUM_BUCKETS=buckets,
            BLOCK_M=block_m,
            BLOCK_C=block_c,
            BLOCK_P=block_p,
            BLOCK_D=block_d,
            Q_BF16=query.dtype == torch.bfloat16,
            Q_FP16=query.dtype == torch.float16,
            PROJECT_Q=True,
            num_warps=num_warps,
        )
        return weights

    score_block_m = 8
    score_block_c = min(
        block_c, max(1, 32768 // (block_d * block_p))
    )
    scores = torch.empty(
        batch, heads, num_tokens, chunks, device=query.device, dtype=torch.float32
    )
    _hla_router_scores_fwd_kernel[
        (
            batch * heads,
            triton.cdiv(num_tokens, score_block_m),
            triton.cdiv(chunks, score_block_c),
        )
    ](
        query,
        query_weight,
        pooled,
        scores,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        query_weight.stride(0),
        pooled.stride(0),
        pooled.stride(1),
        pooled.stride(2),
        pooled.stride(3),
        num_tokens,
        head_dim,
        float(head_dim**-0.5 * router_logit_scale),
        bool(use_logmean),
        NUM_HEADS=heads,
        NUM_CHUNKS=chunks,
        NUM_BUCKETS=buckets,
        BLOCK_M=score_block_m,
        BLOCK_C=score_block_c,
        BLOCK_P=block_p,
        BLOCK_D=block_d,
        Q_BF16=query.dtype == torch.bfloat16,
        Q_FP16=query.dtype == torch.float16,
        PROJECT_Q=True,
        num_warps=8,
    )
    _hla_router_sigmoid_from_scores_fwd_kernel[
        (batch * heads, triton.cdiv(num_tokens, score_block_m))
    ](
        scores,
        weights,
        num_tokens,
        int(chunk_size),
        float(sigmoid_bias),
        float(1.0 / sigmoid_temperature),
        bool(current_always_on),
        NUM_CHUNKS=chunks,
        BLOCK_M=score_block_m,
        BLOCK_C=block_c,
        num_warps=8,
    )
    return weights


def can_use_fused_hla_router_sigmoid_top1_tiles(
    query: torch.Tensor,
    query_weight: torch.Tensor,
    pooled: torch.Tensor,
    *,
    chunk_size: int,
    block_m: int = 32,
) -> bool:
    if os.environ.get("GDN_DISABLE_COMPACT_TOP1_PREFILL", ""):
        return False
    if not can_use_fused_hla_router_sigmoid(
        query, query_weight, pooled, chunk_size=chunk_size
    ):
        return False
    chunks, buckets = pooled.shape[2:4]
    return (
        block_m == 32
        and chunk_size % block_m == 0
        and chunks <= 32
        and buckets <= 32
        and chunks * buckets <= 256
    )


def fused_hla_router_sigmoid_top1_tiles(
    query: torch.Tensor,
    query_weight: torch.Tensor,
    pooled: torch.Tensor,
    *,
    chunk_size: int,
    router_logit_scale: float,
    sigmoid_bias: float,
    sigmoid_temperature: float,
    use_logmean: bool,
    top1_recent_of_top2: bool = False,
    block_m: int = 32,
    num_warps: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return selected gates [B,H,S] and sources [B,H,ceil(S/32)]."""

    if not can_use_fused_hla_router_sigmoid_top1_tiles(
        query,
        query_weight,
        pooled,
        chunk_size=chunk_size,
        block_m=block_m,
    ):
        raise ValueError("unsupported compact Top-1 HLA router inputs")
    if sigmoid_temperature <= 0:
        raise ValueError("sigmoid_temperature must be positive")
    if num_warps not in (4, 8):
        raise ValueError("num_warps must be 4 or 8")
    batch, num_tokens, heads, head_dim = query.shape
    chunks, buckets = pooled.shape[2:4]
    block_c = triton.next_power_of_2(chunks)
    block_p = triton.next_power_of_2(buckets)
    block_d = triton.next_power_of_2(head_dim)
    num_tiles = triton.cdiv(num_tokens, block_m)
    selected_gates = torch.empty(
        batch, heads, num_tokens, device=query.device, dtype=query.dtype
    )
    selected_sources = torch.empty(
        batch, heads, num_tiles, device=query.device, dtype=torch.int32
    )
    _hla_router_sigmoid_top1_tiles_fwd_kernel[
        (batch * heads, num_tiles)
    ](
        query,
        query_weight,
        pooled,
        selected_gates,
        selected_sources,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        query_weight.stride(0),
        pooled.stride(0),
        pooled.stride(1),
        pooled.stride(2),
        pooled.stride(3),
        num_tokens,
        head_dim,
        int(chunk_size),
        float(head_dim**-0.5 * router_logit_scale),
        float(sigmoid_bias),
        float(1.0 / sigmoid_temperature),
        bool(use_logmean),
        NUM_HEADS=heads,
        NUM_CHUNKS=chunks,
        NUM_BUCKETS=buckets,
        BLOCK_M=block_m,
        BLOCK_C=block_c,
        BLOCK_P=block_p,
        BLOCK_D=block_d,
        Q_BF16=query.dtype == torch.bfloat16,
        Q_FP16=query.dtype == torch.float16,
        TOP1_RECENT_OF_TOP2=bool(top1_recent_of_top2),
        num_warps=num_warps,
    )
    return selected_gates, selected_sources


def fused_hla_router_sigmoid_projected(
    query: torch.Tensor,
    pooled: torch.Tensor,
    *,
    chunk_size: int,
    router_logit_scale: float,
    sigmoid_bias: float,
    sigmoid_temperature: float,
    use_logmean: bool,
    current_always_on: bool,
    block_m: int = 32,
    num_warps: int = 4,
) -> torch.Tensor:
    """Fused sigmoid router when the headwise Q projection is already done."""
    if not (query.is_cuda and pooled.is_cuda and query.dim() == 4 and pooled.dim() == 5):
        raise ValueError("unsupported projected HLA router inputs")
    batch, num_tokens, heads, head_dim = query.shape
    chunks, buckets = pooled.shape[2:4]
    if (
        pooled.shape[:2] != (batch, heads)
        or pooled.shape[-1] != head_dim
        or chunks != (num_tokens + chunk_size - 1) // chunk_size
        or chunks > 32
        or buckets > 32
        or chunks * buckets > 512
        or sigmoid_temperature <= 0
        or query.stride(-1) != 1
        or pooled.stride(-1) != 1
    ):
        raise ValueError("unsupported projected HLA router shape")
    block_c = triton.next_power_of_2(chunks)
    block_p = triton.next_power_of_2(buckets)
    block_d = triton.next_power_of_2(head_dim)
    weights = torch.empty(
        batch, heads, num_tokens, chunks, device=query.device, dtype=query.dtype
    )
    # Large projected tiles are dominated by the resident pooled D x C x P
    # operand, so shrinking BLOCK_M does not reduce shared memory enough.  Use
    # the same chunk-tiled score kernel as the unprojected path; query is
    # already projected here, hence PROJECT_Q=False.
    if block_d * block_c * block_p >= 32768:
        score_block_m = 8
        score_block_c = min(
            block_c, max(1, 32768 // (block_d * block_p))
        )
        scores = torch.empty(
            batch, heads, num_tokens, chunks, device=query.device, dtype=torch.float32
        )
        _hla_router_scores_fwd_kernel[
            (
                batch * heads,
                triton.cdiv(num_tokens, score_block_m),
                triton.cdiv(chunks, score_block_c),
            )
        ](
            query,
            query,
            pooled,
            scores,
            query.stride(0),
            query.stride(1),
            query.stride(2),
            0,
            pooled.stride(0),
            pooled.stride(1),
            pooled.stride(2),
            pooled.stride(3),
            num_tokens,
            head_dim,
            float(head_dim**-0.5 * router_logit_scale),
            bool(use_logmean),
            NUM_HEADS=heads,
            NUM_CHUNKS=chunks,
            NUM_BUCKETS=buckets,
            BLOCK_M=score_block_m,
            BLOCK_C=score_block_c,
            BLOCK_P=block_p,
            BLOCK_D=block_d,
            Q_BF16=False,
            Q_FP16=False,
            PROJECT_Q=False,
            num_warps=8,
        )
        _hla_router_sigmoid_from_scores_fwd_kernel[
            (batch * heads, triton.cdiv(num_tokens, score_block_m))
        ](
            scores,
            weights,
            num_tokens,
            int(chunk_size),
            float(sigmoid_bias),
            float(1.0 / sigmoid_temperature),
            bool(current_always_on),
            NUM_CHUNKS=chunks,
            BLOCK_M=score_block_m,
            BLOCK_C=block_c,
            num_warps=8,
        )
        return weights
    # query_weight is not dereferenced when PROJECT_Q=False.
    _hla_router_sigmoid_fwd_kernel[(batch * heads, triton.cdiv(num_tokens, block_m))](
        query,
        query,
        pooled,
        weights,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        0,
        pooled.stride(0),
        pooled.stride(1),
        pooled.stride(2),
        pooled.stride(3),
        num_tokens,
        head_dim,
        int(chunk_size),
        float(head_dim**-0.5 * router_logit_scale),
        float(sigmoid_bias),
        float(1.0 / sigmoid_temperature),
        bool(use_logmean),
        bool(current_always_on),
        NUM_HEADS=heads,
        NUM_CHUNKS=chunks,
        NUM_BUCKETS=buckets,
        BLOCK_M=block_m,
        BLOCK_C=block_c,
        BLOCK_P=block_p,
        BLOCK_D=block_d,
        Q_BF16=False,
        Q_FP16=False,
        PROJECT_Q=False,
        num_warps=num_warps,
    )
    return weights


def can_use_fused_hla_router_budget_softmax(
    query: torch.Tensor,
    query_weight: torch.Tensor,
    pooled: torch.Tensor,
    *,
    chunk_size: int,
) -> bool:
    if os.environ.get("GDN_DISABLE_TRITON_HLA_ROUTER", ""):
        return False
    if not (query.is_cuda and query_weight.is_cuda and pooled.is_cuda):
        return False
    if query.dim() != 4 or pooled.dim() != 5:
        return False
    if query.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if query_weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if pooled.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    batch, num_tokens, heads, head_dim = query.shape
    if query_weight.shape != (heads, head_dim, head_dim):
        return False
    if pooled.shape[0] != batch or pooled.shape[1] != heads:
        return False
    if pooled.shape[-1] != head_dim:
        return False
    chunks, buckets = pooled.shape[2:4]
    if not (0 < chunks <= 128 and 0 < buckets <= 64 and 0 < head_dim <= 256):
        return False
    if num_tokens <= 0 or chunk_size <= 0:
        return False
    if chunks != (num_tokens + chunk_size - 1) // chunk_size:
        return False
    return (
        query.stride(-1) == 1
        and query_weight.stride(-1) == 1
        and pooled.stride(-1) == 1
    )


def fused_hla_router_budget_softmax(
    query: torch.Tensor,
    query_weight: torch.Tensor,
    pooled: torch.Tensor,
    *,
    chunk_size: int,
    router_logit_scale: float,
    softmax_temperature: float,
    softmax_budget: float,
    use_logmean: bool,
    current_always_on: bool,
) -> torch.Tensor:
    """Memory-bounded HLA meta-query router for budget softmax inference."""
    if not can_use_fused_hla_router_budget_softmax(
        query, query_weight, pooled, chunk_size=chunk_size
    ):
        raise ValueError("unsupported fused budget-softmax HLA router inputs")
    if softmax_temperature <= 0 or softmax_budget < 0:
        raise ValueError("softmax temperature must be positive and budget nonnegative")
    batch, num_tokens, heads, head_dim = query.shape
    chunks, buckets = pooled.shape[2:4]
    block_m = 8
    block_p = triton.next_power_of_2(buckets)
    # Keep the pooled K tile at or below 256 columns.  A 128-dim head with
    # 64 meta queries and eight chunks exceeds H200's shared-memory limit.
    block_c = min(triton.next_power_of_2(chunks), max(1, 256 // block_p))
    block_d = triton.next_power_of_2(head_dim)
    scores = torch.empty(
        batch, heads, num_tokens, chunks, device=query.device, dtype=torch.float32
    )
    _hla_router_scores_fwd_kernel[
        (
            batch * heads,
            triton.cdiv(num_tokens, block_m),
            triton.cdiv(chunks, block_c),
        )
    ](
        query,
        query_weight,
        pooled,
        scores,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        query_weight.stride(0),
        pooled.stride(0),
        pooled.stride(1),
        pooled.stride(2),
        pooled.stride(3),
        num_tokens,
        head_dim,
        float(head_dim**-0.5 * router_logit_scale),
        bool(use_logmean),
        NUM_HEADS=heads,
        NUM_CHUNKS=chunks,
        NUM_BUCKETS=buckets,
        BLOCK_M=block_m,
        BLOCK_C=block_c,
        BLOCK_P=block_p,
        BLOCK_D=block_d,
        Q_BF16=query.dtype == torch.bfloat16,
        Q_FP16=query.dtype == torch.float16,
        PROJECT_Q=True,
        num_warps=8,
    )
    weights = torch.empty(
        batch, heads, num_tokens, chunks, device=query.device, dtype=query.dtype
    )
    softmax_block_c = triton.next_power_of_2(chunks)
    _hla_router_budget_softmax_fwd_kernel[
        (batch * heads, triton.cdiv(num_tokens, block_m))
    ](
        scores,
        weights,
        num_tokens,
        int(chunk_size),
        float(1.0 / softmax_temperature),
        float(softmax_budget),
        bool(current_always_on),
        NUM_CHUNKS=chunks,
        BLOCK_M=block_m,
        BLOCK_C=softmax_block_c,
        num_warps=8,
    )
    return weights


def reference_hla_router_sigmoid(
    query: torch.Tensor,
    pooled: torch.Tensor,
    *,
    chunk_size: int,
    router_logit_scale: float,
    sigmoid_bias: float,
    sigmoid_temperature: float,
    use_logmean: bool,
    current_always_on: bool,
) -> torch.Tensor:
    buckets = pooled.shape[3]
    bucket_scores = torch.einsum(
        "bshd,bhcpd->bhscp", query.float(), pooled.float()
    )
    bucket_scores *= query.shape[-1] ** -0.5 * router_logit_scale
    scores = torch.logsumexp(bucket_scores, dim=-1)
    if use_logmean:
        scores -= math.log(buckets)
    token_chunk = torch.arange(query.shape[1], device=query.device) // chunk_size
    chunk_ids = torch.arange(pooled.shape[2], device=query.device)
    visible = chunk_ids.view(1, pooled.shape[2]) <= token_chunk.view(-1, 1)
    weights = torch.sigmoid((scores + sigmoid_bias) / sigmoid_temperature)
    weights = weights.masked_fill(~visible.view(1, 1, query.shape[1], -1), 0.0)
    if current_always_on:
        current_idx = token_chunk.view(1, 1, -1, 1).expand(
            query.shape[0], query.shape[2], query.shape[1], 1
        )
        weights = weights.scatter(3, current_idx, 1.0)
    return weights.to(query.dtype)
