"""Experimental memory-conscious affine-history autograd.

This module is deliberately not wired into the training path. It retains the
descending source recurrence and all five input gradients. The custom backward
recomputes the two affine products instead of retaining their forward graphs.
"""
from __future__ import annotations

import torch


class _AffineHistory(torch.autograd.Function):
    @staticmethod
    def forward(ctx, local_output, local_readout, chunk_a, chunk_b, weights, cache_products=False, fuse_pointwise=False):
        if local_output.dtype != torch.float32 or local_readout.dtype != torch.float32:
            raise TypeError("This experimental training path requires FP32 local output/readout, matching _fast_affine_chunk_data.")
        b, h, c, length, dk = local_readout.shape
        dv = local_output.shape[-1]
        ctx.shape = (b, h, c, length, dk, dv)
        ctx.cache_products = cache_products
        ctx.fused = fuse_pointwise and local_output.is_cuda
        if ctx.fused:
            from .affine_history_pointwise import history_forward_update
        ctx.input_dtypes = tuple(x.dtype for x in (local_output, local_readout, chunk_a, chunk_b, weights))
        ctx.autocast = torch.is_autocast_enabled(local_output.device.type)
        ctx.autocast_dtype = torch.get_autocast_dtype(local_output.device.type)
        compute_dtype = ctx.autocast_dtype if ctx.autocast else local_readout.dtype
        # Cast once, matching the dtype of the autocast einsum operands. These
        # casts are constant with respect to the history recurrence.
        a = chunk_a.to(compute_dtype)
        bb = chunk_b.to(compute_dtype)
        gates = weights.float().contiguous()
        output = local_output.clone(memory_format=torch.contiguous_format)
        readout = local_readout.clone(memory_format=torch.contiguous_format)
        trajectory = []
        products = []
        for source in range(c - 2, -1, -1):
            m = (c - source - 1) * length
            r_view = readout[:, :, source + 1:].reshape(b * h, m, dk)
            r = torch.empty((b * h, m, dk), device=readout.device, dtype=readout.dtype) if ctx.fused else r_view.clone()
            trajectory.append(r)
            rr = r_view.to(compute_dtype) if ctx.fused else r.to(compute_dtype)
            aa = a[:, :, source].reshape(b * h, dk, dk)
            bv = bb[:, :, source].reshape(b * h, dk, dv)
            contribution = torch.bmm(rr, bv).reshape(b, h, c - source - 1, length, dv)
            transformed = torch.bmm(rr, aa).reshape(b, h, c - source - 1, length, dk)
            if cache_products:
                products.extend((contribution, transformed))
            if ctx.fused:
                history_forward_update[(b * h * m,)](
                    output, readout, contribution, transformed, gates, r,
                    M=m, FULL=c * length, SRC=source, DK=dk, DV=dv,
                    CHUNKS=c, D=2 ** (max(dk, dv) - 1).bit_length(),
                    num_warps=4, enable_fp_fusion=False)
            else:
                gate = gates[:, :, source + 1:, :, source, None]
                output[:, :, source + 1:].add_(gate * contribution)
                readout[:, :, source + 1:] = (1.0 - gate) * r.reshape(b, h, c - source - 1, length, dk) + gate * transformed
        ctx.save_for_backward(a, bb, gates, *trajectory, *products)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        a, bb, gates, *saved = ctx.saved_tensors
        b, h, c, length, dk, dv = ctx.shape
        trajectory = saved[:c - 1]
        products = saved[c - 1:]
        output_dtype, readout_dtype, a_dtype, b_dtype, gate_dtype = ctx.input_dtypes
        if ctx.fused:
            from .affine_history_pointwise import history_backward_prepare, history_backward_combine
        grad_output = grad_output.contiguous()
        dr = torch.zeros((b, h, c, length, dk), dtype=readout_dtype, device=grad_output.device)
        da = torch.zeros(a.shape, dtype=a_dtype, device=a.device)
        db = torch.zeros(bb.shape, dtype=b_dtype, device=bb.device)
        dw = torch.zeros_like(gates)
        # Reverse-mode walks source indices in ascending order. grad_output is
        # invariant across source updates because output is an additive scan.
        with torch.autocast(grad_output.device.type, enabled=ctx.autocast, dtype=ctx.autocast_dtype):
            for source in range(c - 1):
                m = (c - source - 1) * length
                r = trajectory[c - source - 2]
                rr = r.to(a.dtype)
                aa = a[:, :, source].reshape(b * h, dk, dk)
                bv = bb[:, :, source].reshape(b * h, dk, dv)
                gate = gates[:, :, source + 1:, :, source].reshape(b * h, m, 1)
                dout = grad_output[:, :, source + 1:].reshape(b * h, m, dv)
                dr_next = dr[:, :, source + 1:].reshape(b * h, m, dk)
                if ctx.cache_products:
                    index = 2 * (c - source - 2)
                    contribution = products[index].reshape(b * h, m, dv)
                    transformed = products[index + 1].reshape(b * h, m, dk)
                else:
                    contribution = torch.bmm(rr, bv)
                    transformed = torch.bmm(rr, aa)
                if ctx.fused:
                    gc = torch.empty((b * h, m, dv), device=a.device, dtype=a.dtype)
                    gt = torch.empty((b * h, m, dk), device=a.device, dtype=a.dtype)
                    history_backward_prepare[(b * h * m,)](
                        grad_output, dr, r, contribution, transformed, gates, gc, gt, dw,
                        M=m, FULL=c * length, SRC=source, DK=dk, DV=dv,
                        CHUNKS=c, D=2 ** (max(dk, dv) - 1).bit_length(),
                        num_warps=4, enable_fp_fusion=False)
                else:
                    dg = (dout * contribution).sum(-1)
                    dg = dg + (dr_next * transformed).sum(-1) - (dr_next * r).sum(-1)
                    dw[:, :, source + 1:, :, source] = dg.reshape(b, h, c - source - 1, length)
                    gc = (gate * dout).to(a.dtype)
                    gt = (gate * dr_next).to(a.dtype)
                db[:, :, source] = torch.bmm(rr.transpose(1, 2), gc).reshape(b, h, dk, dv)
                da[:, :, source] = torch.bmm(rr.transpose(1, 2), gt).reshape(b, h, dk, dk)
                branch_c = torch.bmm(gc, bv.transpose(1, 2))
                branch_a = torch.bmm(gt, aa.transpose(1, 2))
                if ctx.fused:
                    history_backward_combine[(b * h * m,)](
                        dr, branch_a, branch_c, gates,
                        M=m, FULL=c * length, SRC=source, DK=dk,
                        CHUNKS=c, D=2 ** (dk - 1).bit_length(),
                        num_warps=4, enable_fp_fusion=False)
                else:
                    dr[:, :, source + 1:] = ((1.0 - gate) * dr_next + branch_a.to(readout_dtype) + branch_c.to(readout_dtype)).reshape(b, h, c - source - 1, length, dk)
        return grad_output.to(output_dtype), dr, da, db, dw.to(gate_dtype), None, None


def memory_efficient_affine_history_mix(local_output, local_readout, chunk_a, chunk_b, weights):
    """Drop-in experimental history operation; does not mutate its inputs."""
    return _AffineHistory.apply(local_output, local_readout, chunk_a, chunk_b, weights, False, False)


def cached_affine_history_mix(local_output, local_readout, chunk_a, chunk_b, weights):
    """Save BF16 affine products to avoid backward recomputation."""
    return _AffineHistory.apply(local_output, local_readout, chunk_a, chunk_b, weights, True, False)


def fused_affine_history_mix_training(local_output, local_readout, chunk_a, chunk_b, weights):
    """Experimental cached-product autograd with fused pointwise CUDA kernels."""
    return _AffineHistory.apply(local_output, local_readout, chunk_a, chunk_b, weights, True, True)
