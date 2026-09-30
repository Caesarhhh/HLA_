"""Pointwise kernels for the isolated experimental history autograd."""
import triton
import triton.language as tl


@triton.jit
def history_forward_update(O, R, C, T, G, PREV, M: tl.constexpr,
                           FULL: tl.constexpr, SRC: tl.constexpr,
                           DK: tl.constexpr, DV: tl.constexpr,
                           CHUNKS: tl.constexpr, D: tl.constexpr):
    row = tl.program_id(0)
    bh = row // M
    m = row % M
    d = tl.arange(0, D)
    sequence = FULL - M + m
    gate = tl.load(G + (bh * FULL + sequence) * CHUNKS + SRC)
    r = tl.load(R + (bh * FULL + sequence) * DK + d, d < DK, 0).to(tl.float32)
    o = tl.load(O + (bh * FULL + sequence) * DV + d, d < DV, 0).to(tl.float32)
    contribution = tl.load(C + row * DV + d, d < DV, 0).to(tl.float32)
    transformed = tl.load(T + row * DK + d, d < DK, 0).to(tl.float32)
    tl.store(PREV + row * DK + d, r, d < DK)
    tl.store(O + (bh * FULL + sequence) * DV + d, o + gate * contribution, d < DV)
    tl.store(R + (bh * FULL + sequence) * DK + d,
             (1.0 - gate) * r + gate * transformed, d < DK)


@triton.jit
def history_backward_prepare(DO, DR, PREV, C, T, G, GC, GT, DG,
                             M: tl.constexpr, FULL: tl.constexpr,
                             SRC: tl.constexpr, DK: tl.constexpr,
                             DV: tl.constexpr, CHUNKS: tl.constexpr,
                             D: tl.constexpr):
    row = tl.program_id(0)
    bh = row // M
    m = row % M
    d = tl.arange(0, D)
    sequence = FULL - M + m
    gate_offset = (bh * FULL + sequence) * CHUNKS + SRC
    gate = tl.load(G + gate_offset)
    dr = tl.load(DR + (bh * FULL + sequence) * DK + d, d < DK, 0).to(tl.float32)
    do = tl.load(DO + (bh * FULL + sequence) * DV + d, d < DV, 0).to(tl.float32)
    r = tl.load(PREV + row * DK + d, d < DK, 0).to(tl.float32)
    contribution = tl.load(C + row * DV + d, d < DV, 0).to(tl.float32)
    transformed = tl.load(T + row * DK + d, d < DK, 0).to(tl.float32)
    dg = tl.sum(do * contribution, 0) + tl.sum(dr * transformed, 0) - tl.sum(dr * r, 0)
    tl.store(DG + gate_offset, dg)
    tl.store(GC + row * DV + d, gate * do, d < DV)
    tl.store(GT + row * DK + d, gate * dr, d < DK)


@triton.jit
def history_backward_combine(DR, BRANCH_A, BRANCH_B, G,
                             M: tl.constexpr, FULL: tl.constexpr,
                             SRC: tl.constexpr, DK: tl.constexpr,
                             CHUNKS: tl.constexpr, D: tl.constexpr):
    row = tl.program_id(0)
    bh = row // M
    m = row % M
    d = tl.arange(0, D)
    sequence = FULL - M + m
    gate = tl.load(G + (bh * FULL + sequence) * CHUNKS + SRC)
    offset = (bh * FULL + sequence) * DK + d
    dr = tl.load(DR + offset, d < DK, 0).to(tl.float32)
    a = tl.load(BRANCH_A + row * DK + d, d < DK, 0).to(tl.float32)
    b = tl.load(BRANCH_B + row * DK + d, d < DK, 0).to(tl.float32)
    tl.store(DR + offset, (1.0 - gate) * dr + a + b, d < DK)
