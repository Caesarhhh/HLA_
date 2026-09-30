"""Decomposed FP32 GDN backward, avoiding the monolithic dqkw kernel.

Every dot has an explicit precision mode independent of autocast and global
cuBLAS settings. TF32x3 uses three tensor-core products and FP32 accumulation.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _exp2_kernel(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    ids = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + ids, ids < N, 0)
    tl.store(Y + ids, tl.math.exp2(x), ids < N)


def _exp2(x):
    # torch.exp2 uses NVRTC Jiterator in torch 2.8; its bundled CUDA 12.8
    # compiler does not recognize B300 sm_103. Match the original tl.exp2.
    x = x.contiguous()
    y = torch.empty_like(x)
    _exp2_kernel[(triton.cdiv(x.numel(), 256),)](x, y, x.numel(), 256)
    return y


@triton.jit
def _bmm_kernel(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                AZ: tl.constexpr, AM: tl.constexpr, AK: tl.constexpr,
                BZ: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr,
                BM: tl.constexpr, BN_TILE: tl.constexpr, BK_TILE: tl.constexpr,
                PRECISION: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN_TILE + tl.arange(0, BN_TILE)
    reduction = tl.arange(0, BK_TILE)
    z = tl.program_id(2)
    acc = tl.zeros((BM, BN_TILE), tl.float32)
    for kk in range(tl.cdiv(K, BK_TILE)):
        ks = kk * BK_TILE + reduction
        av = tl.load(A + z * AZ + rows[:, None] * AM + ks[None, :] * AK,
                     (rows[:, None] < M) & (ks[None, :] < K), 0)
        bv = tl.load(B + z * BZ + ks[:, None] * BK + cols[None, :] * BN,
                     (ks[:, None] < K) & (cols[None, :] < N), 0)
        acc = tl.dot(av, bv, acc, input_precision=PRECISION)
    tl.store(C + z * M * N + rows[:, None] * N + cols[None, :], acc,
             (rows[:, None] < M) & (cols[None, :] < N))


def bmm(a, b, precision="tf32x3"):
    if precision not in ("ieee", "tf32x3"):
        raise ValueError("Supported product precision modes are ieee and tf32x3")
    z, m, k = a.shape
    if b.shape[:2] != (z, k) or a.dtype != torch.float32 or b.dtype != torch.float32:
        raise ValueError("bmm requires compatible FP32 batches")
    n = b.shape[-1]
    result = torch.empty((z, m, n), device=a.device, dtype=torch.float32)
    _bmm_kernel[(triton.cdiv(m, 32), triton.cdiv(n, 64), z)](
        a, b, result, m, n, k, *a.stride(), *b.stride(),
        32, 64, 32, precision, num_warps=4, num_stages=2,
    )
    return result


def chunk_bwd_dqkw_batched(q, k, v_new, w, g, h, du, do, dh, BT,
                          precision="tf32x3"):
    """Same dq/dk/dw/dg algebra as chunk.chunk_bwd_dqkw_fn, FP32 path only."""
    if q.dtype != torch.float32 or k.dtype != torch.float32 or v_new.dtype != torch.float32:
        raise ValueError("The decomposed backward requires FP32 q/k/v")
    b, heads, length, dk = q.shape
    dv = v_new.shape[-1]
    if length % BT:
        raise ValueError("Input length must be padded to the GDN block size")
    count = b * heads * (length // BT)
    with torch.autocast("cuda", enabled=False):
        qt, kt, wt = [x.reshape(count, BT, dk) for x in (q, k, w)]
        vt, dot, dut = [x.reshape(count, BT, dv) for x in (v_new, do, du)]
        ht, dht = [x.reshape(count, dk, dv) for x in (h, dh)]
        gt = g.reshape(count, BT)
        expg = _exp2(gt)
        expglast = expg[:, -1]
        scale = dk ** -0.5

        dqt = bmm(dot, ht.transpose(1, 2), precision) * (expg * scale).unsqueeze(-1)
        dkt = bmm(vt, dht.transpose(1, 2), precision) * _exp2(gt[:, -1:] - gt).unsqueeze(-1)
        dwt = bmm(dut, ht.transpose(1, 2), precision) * expg.unsqueeze(-1)
        dgt = (dqt * qt - dwt * wt - dkt * kt).sum(-1)
        dglast = (ht * dht).sum((1, 2)) * expglast + (dkt * kt).sum((1, 2))

        ds = bmm(dot, vt.transpose(1, 2), precision)
        ds *= _exp2(gt.unsqueeze(-1) - gt.unsqueeze(-2)) * scale
        ds = ds.tril()
        dgmask = bmm(qt, kt.transpose(1, 2), precision) * ds
        dgt = dgt + dgmask.sum(-1) - dgmask.sum(-2)
        dgt[:, -1] += dglast
        dqt = dqt + bmm(ds, kt, precision)
        dkt = dkt + bmm(ds.transpose(1, 2), qt, precision)
    return dqt.reshape_as(q), dkt.reshape_as(k), -dwt.reshape_as(w), dgt.reshape_as(g)
