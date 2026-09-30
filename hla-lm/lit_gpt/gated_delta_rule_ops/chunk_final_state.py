# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Opt-in GDN kernel with differentiable final and initial recurrent states.

Reuses the existing native forward and WY kernels. The state adjoint scan starts
from the caller's terminal gradient instead of zero, removing the need to append
identity-query tokens just to make the final state differentiable. The default
operator exported by this package is intentionally unchanged.
"""
import os
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.amp import custom_fwd, custom_bwd
from .chunk import (
    contiguous, chunk_fwd_h_fn, chunk_fwd_o_fn, fwd_prepare_du,
    chunk_bwd_dqkw_fn,
)
from .wy_fast import fwd_recompute_w_u, fwd_prepare_wy_repr, bwd_prepare_wy_repr

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ],
    key=["BT", "BK", "BV"],
)
@triton.jit
def _bwd_dhu_with_final_state_kernel(
    q,
    k,
    w,
    g,
    do,
    dh,
    dv,
    dv2,
    dht,
    dh0,
    s_qk_h,
    s_qk_t,
    s_qk_d,
    s_vo_h,
    s_vo_t,
    s_vo_d,
    s_h_h,
    s_h_t,
    scale,
    H: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
    USE_DHT: tl.constexpr,
    STORE_DH0: tl.constexpr,
):
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    # [BK, BV]
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_DHT:
        p_dht = tl.make_block_ptr(dht + i_bh * K * V, (K, V), (V, 1),
                                 (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_dh += tl.load(p_dht, boundary_check=(0, 1)).to(tl.float32)
    for i_t in range(NT - 1, -1, -1):
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_dh, b_dh.to(p_dh.dtype.element_ty), boundary_check=(0, 1))
        b_dh_tmp = tl.zeros([BK, BV], dtype=tl.float32)

        bg_last = tl.load(g + i_bh * T + i_t * BT + BT - 1)
        for i_c in range(tl.cdiv(BT, BC) - 1, -1, -1):
            p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i_t * BT + i_c * BC), (BK, BC), (0, 1))
            p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT + i_c * BC, i_k * BK), (BC, BK), (1, 0))
            p_w = tl.make_block_ptr(w + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i_t * BT + i_c * BC), (BK, BC), (0, 1))
            p_dv = tl.make_block_ptr(dv + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT + i_c * BC, i_v * BV), (BC, BV), (1, 0))
            p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT + i_c * BC, i_v * BV), (BC, BV), (1, 0))
            b_g = tl.load(g + i_bh * T + i_t * BT + i_c * BC + tl.arange(0, BC))
            # [BK, BT]
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_q = (b_q * scale * tl.math.exp2(b_g)[None, :]).to(b_q.dtype)
            b_do = tl.load(p_do, boundary_check=(0, 1))
            b_dh_tmp += tl.dot(b_q, b_do.to(b_q.dtype), allow_tf32=False)
            # b_q = b_do = None
            # [BT, BK]

            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_k = (b_k * tl.math.exp2(bg_last - b_g)[:, None]).to(b_k.dtype)
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_w = (b_w * tl.math.exp2(b_g)[None, :]).to(b_w.dtype)
            # [BT, V]
            b_dv = tl.load(p_dv, boundary_check=(0, 1))
            b_dv += tl.dot(b_k, b_dh.to(b_k.dtype), allow_tf32=False)
            p_dv2 = tl.make_block_ptr(dv2 + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT + i_c * BC, i_v * BV), (BC, BV), (1, 0))
            tl.store(p_dv2, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
            # [BK, BV]
            b_dh_tmp -= tl.dot(b_w, b_dv.to(b_q.dtype), allow_tf32=False)

        b_dh *= tl.math.exp2(bg_last)
        b_dh += b_dh_tmp

    if STORE_DH0:
        p_dh0 = tl.make_block_ptr(dh0 + i_bh * K * V, (K, V), (V, 1),
                                 (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_dh0, b_dh.to(p_dh0.dtype.element_ty), boundary_check=(0, 1))


def _bwd_dhu(q, k, w, g, do, dv, dht, initial_state, bt):
    b, h, t, k_dim = q.shape
    v_dim = do.shape[-1]
    bk = triton.next_power_of_2(k_dim)
    if bk > 256:
        raise ValueError("The native state kernel supports key dimension <= 256")
    bv = 64 if bk <= 64 else (16 if bk > 128 else 32)
    bc = min(bt, 64 if bk <= 64 else (16 if bk > 128 else 32))
    nt = triton.cdiv(t, bt)
    dh = q.new_empty(b, h, nt * k_dim, v_dim, dtype=torch.float32)
    dh0 = None if initial_state is None else torch.empty_like(initial_state)
    dv2 = torch.empty_like(dv)
    _bwd_dhu_with_final_state_kernel[(1, triton.cdiv(v_dim, bv), b * h)](
        q, k, w, g, do, dh, dv, dv2, dht, dh0,
        q.stride(1), q.stride(2), q.stride(3),
        do.stride(1), do.stride(2), do.stride(3),
        dh.stride(1), dh.stride(2), k_dim ** -0.5,
        H=h, T=t, K=k_dim, V=v_dim, BT=bt, BC=bc, BK=bk, BV=bv, NT=nt,
        USE_DHT=dht is not None, STORE_DH0=dh0 is not None,
    )
    return dh, dv2, dh0


class _ChunkFinalStateFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type="cuda")
    @contiguous
    def forward(ctx, q, k, v, beta, g, bt, initial_state, output_final_state):
        b, h, t, _ = q.shape
        cumulative_g = g.float().reshape(b, h, -1, bt).cumsum(-1)
        cumulative_g = (cumulative_g * 1.44269504).reshape(b, h, t)
        w, u, aw, au, aw_original, au_original = fwd_prepare_wy_repr(
            k, v, beta, cumulative_g, bt
        )
        # Preserve the native recurrent accumulator precision even if token
        # projections use BF16. HLA repeatedly composes the returned A/B state.
        ht = q.new_empty(b, h, q.shape[-1], v.shape[-1], dtype=torch.float32) if output_final_state else None
        states, v_new = chunk_fwd_h_fn(
            k, w, u, cumulative_g, bt, initial_state, ht, state_in_fp32=False
        )
        out = chunk_fwd_o_fn(q, k, v_new, cumulative_g, states, bt)
        ctx.save_for_backward(
            q, k, v, beta, cumulative_g, aw, au, aw_original, au_original, initial_state
        )
        ctx.bt = bt
        ctx.g_dtype = g.dtype
        return out.to(q.dtype), ht

    @staticmethod
    @custom_bwd(device_type="cuda")
    @contiguous
    def backward(ctx, do, dht):
        q, k, v, beta, g, aw, au, aw_original, au_original, h0 = ctx.saved_tensors
        bt = ctx.bt
        if do is None:
            do = torch.zeros_like(v)
        w, u = fwd_recompute_w_u(k, v, beta, aw, au, bt)
        states, v_new = chunk_fwd_h_fn(k, w, u, g, bt, h0, None, state_in_fp32=True)
        du = fwd_prepare_du(q, k, g, do, bt)
        dh, du, dh0 = _bwd_dhu(
            q, k, w, g, do, du, dht, h0 if ctx.needs_input_grad[6] else None, bt
        )
        if q.dtype == torch.float32 and os.environ.get("GDN_AFFINE_BATCHED_DQKW", "0") == "1":
            # Opt-in standard tiled GEMMs avoid the monolithic kernel's
            # register pressure. FP32 tensors and accumulation are retained;
            # TF32x3 products measured < 6e-7 relative L2 derivative error at
            # the production shape. The unchanged IEEE path remains default.
            from .chunk_dqkw_batched import chunk_bwd_dqkw_batched
            dq, dk, dw, dg = chunk_bwd_dqkw_batched(
                q, k, v_new, w, g, states, du, do, dh, bt,
                precision=os.environ.get("GDN_AFFINE_BATCHED_PRECISION", "tf32x3"),
            )
        else:
            dq, dk, dw, dg = chunk_bwd_dqkw_fn(q, k, v_new, w, g, states, du, do, dh, bt)
        if k.dtype == torch.float32 and os.environ.get("GDN_AFFINE_BATCHED_WY", "0") == "1":
            # Preserve the native triangular recurrence; decompose its two
            # GEMM-heavy derivative stages into bounded standard tiles.
            from .wy_batched import bwd_prepare_wy_batched
            dk2, dv, db, dg2 = bwd_prepare_wy_batched(
                k, v, beta, g, aw, au, aw_original, au_original, dw, du, bt,
                precision=os.environ.get("GDN_AFFINE_BATCHED_PRECISION", "tf32x3"),
            )
        else:
            dk2, dv, db, dg2 = bwd_prepare_wy_repr(
                k, v, beta, g, aw, au, aw_original, au_original, dw, du, bt
            )
        dk.add_(dk2)
        dg.add_(dg2)
        # The native WY derivative is expressed in natural-log coordinates;
        # match its reverse-prefix reduction exactly (no additional log2 factor).
        grouped = dg.reshape(*g.shape[:2], -1, bt)
        cumulative = grouped.cumsum(-1)
        dg = (cumulative[..., -1:] - cumulative + grouped).reshape_as(g)
        return dq.to(q), dk.to(k), dv.to(v), db.to(beta), dg.to(ctx.g_dtype), None, dh0, None


def chunk_gated_delta_rule_final_state(
    q, k, v, beta, g, BT=64, initial_state=None, output_final_state=False
):
    """Native-compatible GDN call with gradients through returned final state.

    Inputs use [batch, head, time, feature] layout. Unlike the legacy wrapper,
    a supplied initial state is not detached. Final state is returned in FP32.
    Padding uses beta=0 and g=0, so
    the returned final state remains the true state after the last real token.
    """
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k and v must have the same dtype")
    if q.ndim != 4 or k.shape != q.shape or v.shape[:3] != q.shape[:3]:
        raise ValueError("Expected matching head-first q/k/v tensors")
    if q.shape[-2] == 0:
        raise ValueError("The sequence must contain at least one token")
    if BT not in (16, 32, 64):
        raise ValueError("This opt-in wrapper supports BT in (16, 32, 64)")
    if beta.shape != q.shape[:3] or g.shape != q.shape[:3]:
        raise ValueError("beta and g must have shape [batch, head, time]")
    expected_state_shape = (*q.shape[:2], q.shape[-1], v.shape[-1])
    if initial_state is not None and initial_state.shape != expected_state_shape:
        raise ValueError(f"initial_state must have shape {expected_state_shape}")
    length = q.shape[-2]
    padding = (-length) % BT
    if padding:
        q, k, v = (F.pad(x, (0, 0, 0, padding)) for x in (q, k, v))
        beta, g = F.pad(beta, (0, padding)), F.pad(g, (0, padding))
    out, ht = _ChunkFinalStateFunction.apply(
        q, k, v, beta, g, BT, initial_state, output_final_state
    )
    return out[:, :, :length], ht
