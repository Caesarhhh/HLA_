from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _affine_prefix_readout_fwd_kernel(
    q,
    gate,
    key,
    value,
    beta,
    gk,
    out_unit,
    r_unit,
    out,
    base_r,
    seq_len: tl.constexpr,
    key_dim: tl.constexpr,
    value_dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // seq_len
    token_idx = pid - row * seq_len
    offs_d = tl.arange(0, BLOCK_D)
    offs_v = tl.arange(0, BLOCK_V)

    q_base = (row * seq_len + token_idx) * key_dim
    qv = tl.load(q + q_base + offs_d, mask=offs_d < key_dim, other=0.0).to(tl.float32)
    r = qv
    acc = tl.zeros((BLOCK_V,), dtype=tl.float32)

    u = token_idx
    while u >= 0:
        k_base = (row * seq_len + u) * key_dim
        v_base = (row * seq_len + u) * value_dim
        kv = tl.load(key + k_base + offs_d, mask=offs_d < key_dim, other=0.0).to(tl.float32)
        vv = tl.load(value + v_base + offs_v, mask=offs_v < value_dim, other=0.0).to(tl.float32)
        bt = tl.load(beta + row * seq_len + u).to(tl.float32)
        decay = tl.exp(tl.load(gk + row * seq_len + u).to(tl.float32))
        dot = tl.sum(r * kv, axis=0)
        coeff = bt * dot
        acc += coeff * vv
        r = decay * (r - coeff * kv)
        u -= 1

    s = tl.load(gate + row * seq_len + token_idx).to(tl.float32)
    out_base = (row * seq_len + token_idx) * value_dim
    r_base = (row * seq_len + token_idx) * key_dim
    tl.store(out_unit + out_base + offs_v, acc, mask=offs_v < value_dim)
    tl.store(r_unit + r_base + offs_d, r, mask=offs_d < key_dim)
    tl.store(out + out_base + offs_v, s * acc, mask=offs_v < value_dim)
    tl.store(base_r + r_base + offs_d, (1.0 - s) * qv + s * r, mask=offs_d < key_dim)


class _FusedAffinePrefixReadout(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        gate: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        beta: torch.Tensor,
        gk: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not can_use_fused_affine_prefix_readout(q, gate, key, value, beta, gk):
            raise ValueError("unsupported affine prefix readout inputs")
        q = q.contiguous()
        gate = gate.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        beta = beta.contiguous()
        gk = gk.contiguous()

        rows, seq_len, key_dim = q.shape
        value_dim = value.shape[-1]
        block_d = triton.next_power_of_2(key_dim)
        block_v = triton.next_power_of_2(value_dim)
        out_unit = torch.empty(rows, seq_len, value_dim, device=q.device, dtype=torch.float32)
        r_unit = torch.empty(rows, seq_len, key_dim, device=q.device, dtype=torch.float32)
        out = torch.empty_like(out_unit)
        base_r = torch.empty_like(r_unit)
        _affine_prefix_readout_fwd_kernel[(rows * seq_len,)](
            q,
            gate,
            key,
            value,
            beta,
            gk,
            out_unit,
            r_unit,
            out,
            base_r,
            seq_len,
            key_dim,
            value_dim,
            BLOCK_D=block_d,
            BLOCK_V=block_v,
            num_warps=8,
        )
        ctx.save_for_backward(q, out_unit, r_unit)
        return out, base_r

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor, grad_base_r: torch.Tensor) -> tuple[None, torch.Tensor, None, None, None, None]:
        q, out_unit, r_unit = ctx.saved_tensors
        grad_gate = (grad_out.float() * out_unit).sum(dim=-1)
        grad_gate = grad_gate + (grad_base_r.float() * (r_unit - q.float())).sum(dim=-1)
        return None, grad_gate, None, None, None, None


def can_use_fused_affine_prefix_readout(
    q: torch.Tensor,
    gate: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    gk: torch.Tensor,
) -> bool:
    if not (q.is_cuda and gate.is_cuda and key.is_cuda and value.is_cuda and beta.is_cuda and gk.is_cuda):
        return False
    if q.dtype != torch.float32 or gate.dtype != torch.float32:
        return False
    if key.dtype != torch.float32 or value.dtype != torch.float32 or beta.dtype != torch.float32 or gk.dtype != torch.float32:
        return False
    if q.dim() != 3 or key.shape != q.shape:
        return False
    if value.dim() != 3 or value.shape[:2] != q.shape[:2]:
        return False
    if gate.shape != q.shape[:2] or beta.shape != q.shape[:2] or gk.shape != q.shape[:2]:
        return False
    return q.shape[1] <= 2048 and q.shape[2] <= 256 and value.shape[2] <= 256


def fused_affine_prefix_readout(
    q: torch.Tensor,
    gate: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    gk: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _FusedAffinePrefixReadout.apply(q, gate, key, value, beta, gk)


def reference_affine_prefix_readout(
    q: torch.Tensor,
    gate: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    gk: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, seq_len, _, = q.shape
    value_dim = value.shape[-1]
    out_chunks = []
    r_chunks = []
    for l in range(seq_len):
        r = q[:, l].float()
        out = torch.zeros(rows, value_dim, device=q.device, dtype=torch.float32)
        for u in range(l, -1, -1):
            dot = (r * key[:, u].float()).sum(dim=-1)
            coeff = beta[:, u].float() * dot
            out = out + coeff[:, None] * value[:, u].float()
            r = gk[:, u].float().exp()[:, None] * (r - coeff[:, None] * key[:, u].float())
        s = gate[:, l].float()
        out_chunks.append(s[:, None] * out)
        r_chunks.append((1.0 - s)[:, None] * q[:, l].float() + s[:, None] * r)
    return torch.stack(out_chunks, dim=1), torch.stack(r_chunks, dim=1)
