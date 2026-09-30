"""Opt-in FP32 WY backward using standard tiled GEMMs and native recurrence."""
import torch
import triton
from .chunk_dqkw_batched import bmm, _exp2
from .wy_fast import bwd_prepare_wy_repr_kernel_dA_recurrence


def bwd_prepare_wy_batched(k, v, beta, g, A_w, A_u, A_w_original,
                          A_u_original, dw, du, BT, precision="tf32x3"):
    if k.dtype != torch.float32 or v.dtype != torch.float32:
        raise ValueError("Decomposed WY backward requires FP32 k/v")
    b, heads, length, dk = k.shape
    dv = v.shape[-1]
    nt = triton.cdiv(length, BT)
    count = b * heads * nt
    with torch.autocast("cuda", enabled=False):
        kt, vt = k.reshape(count, BT, dk), v.reshape(count, BT, dv)
        dwt, dut = dw.reshape(count, BT, dk), du.reshape(count, BT, dv)
        bt, gt = beta.reshape(count, BT), g.reshape(count, BT)
        aw, au = A_w.reshape(count, BT, BT), A_u.reshape(count, BT, BT)
        kb, vb = kt * bt.unsqueeze(-1), vt * bt.unsqueeze(-1)
        daw = bmm(dwt, kb.transpose(1, 2), precision).reshape_as(A_w)
        dau = bmm(dut, vb.transpose(1, 2), precision).reshape_as(A_u)
        dkb = bmm(aw.transpose(1, 2), dwt, precision)
        dvb = bmm(au.transpose(1, 2), dut, precision)
        dk1, dv1 = dkb * bt.unsqueeze(-1), dvb * bt.unsqueeze(-1)
        db1 = (dkb * kt).sum(-1) + (dvb * vt).sum(-1)
        daw_original, dau_original = torch.empty_like(daw), torch.empty_like(dau)
        bk, bv = min(triton.next_power_of_2(dk), 64), min(triton.next_power_of_2(dv), 64)
        for a, a_original, da, da_original in (
            (A_w, A_w_original, daw, daw_original),
            (A_u, A_u_original, dau, dau_original),
        ):
            bwd_prepare_wy_repr_kernel_dA_recurrence[(nt, b * heads)](
                a, a_original, da, da_original,
                k.stride(1), k.stride(2), k.stride(3),
                v.stride(1), v.stride(2), v.stride(3),
                length, dk, dv, BT, bk, bv,
            )
        da_u = dau_original.reshape(count, BT, BT) * _exp2(gt.unsqueeze(-1) - gt.unsqueeze(-2))
        da = -(daw_original.reshape(count, BT, BT) + da_u).tril()
        dg_mask = (da_u * A_w_original.reshape(count, BT, BT)).tril()
        dg = dg_mask.sum(-1) - dg_mask.sum(-2)
        dk_beta2 = bmm(da, kt, precision)
        dk2 = bmm(da.transpose(1, 2), kb, precision) + dk_beta2 * bt.unsqueeze(-1)
        db2 = (dk_beta2 * kt).sum(-1)
    return (dk1 + dk2).reshape_as(k), dv1.reshape_as(v), (db1 + db2).reshape_as(beta), dg.reshape_as(g)
