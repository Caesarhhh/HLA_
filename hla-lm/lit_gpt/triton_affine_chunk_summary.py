from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _affine_chunk_a_fwd_kernel(
    key,
    beta,
    gk,
    out,
    seq_len: tl.constexpr,
    key_dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    basis_idx = tl.program_id(1)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < key_dim

    r = tl.where(offs == basis_idx, 1.0, 0.0)
    # The returned matrix is the state transition A_total such that
    # S_out = A_total @ S_in + B.  Rows are computed as row-vector products,
    # so the per-token affine factors must be applied in reverse order:
    # A_total = A_{L-1} ... A_0.
    t = seq_len - 1
    while t >= 0:
        base = (row * seq_len + t) * key_dim
        k = tl.load(key + base + offs, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(beta + row * seq_len + t).to(tl.float32)
        decay = tl.exp(tl.load(gk + row * seq_len + t).to(tl.float32))
        coeff = b * tl.sum(r * k, axis=0)
        r = decay * (r - coeff * k)
        t -= 1

    out_base = (row * key_dim + basis_idx) * key_dim
    tl.store(out + out_base + offs, r, mask=mask)


def can_use_fused_affine_chunk_a(
    key: torch.Tensor,
    beta: torch.Tensor,
    gk: torch.Tensor,
) -> bool:
    if not (key.is_cuda and beta.is_cuda and gk.is_cuda):
        return False
    if key.dtype != torch.float32 or beta.dtype != torch.float32 or gk.dtype != torch.float32:
        return False
    if key.dim() != 3 or beta.shape != key.shape[:2] or gk.shape != key.shape[:2]:
        return False
    return key.shape[-1] <= 256 and key.shape[1] <= 2048


def fused_affine_chunk_a(
    key: torch.Tensor,
    beta: torch.Tensor,
    gk: torch.Tensor,
) -> torch.Tensor:
    if not can_use_fused_affine_chunk_a(key, beta, gk):
        raise ValueError("unsupported affine chunk summary inputs")
    key = key.contiguous()
    beta = beta.contiguous()
    gk = gk.contiguous()
    rows, seq_len, key_dim = key.shape
    block_d = triton.next_power_of_2(key_dim)
    out = torch.empty(rows, key_dim, key_dim, device=key.device, dtype=torch.float32)
    _affine_chunk_a_fwd_kernel[(rows, key_dim)](
        key,
        beta,
        gk,
        out,
        seq_len,
        key_dim,
        BLOCK_D=block_d,
        num_warps=8,
    )
    return out
