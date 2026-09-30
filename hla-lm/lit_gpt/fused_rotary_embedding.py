# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Copyright (c) 2023, Tri Dao.

import math
from typing import Optional, Tuple

try:
    import rotary_emb
except ImportError:
    rotary_emb = None
import torch
from einops import rearrange, repeat

class ApplyRotaryEmb(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cos, sin, interleaved=False, inplace=False):
        """
            x: (batch_size, seqlen, nheads, headdim)
            cos, sin: (seqlen, rotary_dim / 2)
            interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
                of 1st half and 2nd half (GPT-NeoX style).
        rotary_dim must be <= headdim
        Apply rotary embedding to the first rotary_dim of x.
        """
        batch, seqlen, nheads, headdim = x.shape
        rotary_seqlen, rotary_dim = cos.shape
        rotary_dim *= 2
        assert rotary_dim <= headdim
        assert seqlen <= rotary_seqlen
        assert sin.shape == (rotary_seqlen, rotary_dim // 2)
        x_ro = x[..., :rotary_dim]
        x1, x2 = x_ro.chunk(2, dim=-1) if not interleaved else (x_ro[..., ::2], x_ro[..., 1::2])
        out = torch.empty_like(x) if not inplace else x
        out_ro = out[..., :rotary_dim]
        if inplace:
            o1, o2 = x1, x2
        else:
            o1, o2 = (
                out_ro.chunk(2, dim=-1)
                if not interleaved
                else (out_ro[..., ::2], out_ro[..., 1::2])
            )
        rotary_emb.apply_rotary(
            x1,
            x2,
            rearrange(cos[:seqlen], "s d -> s 1 d"),
            rearrange(sin[:seqlen], "s d -> s 1 d"),
            o1,
            o2,
            False,
        )
        if not inplace and rotary_dim < headdim:
            out[..., rotary_dim:].copy_(x[..., rotary_dim:])
        ctx.save_for_backward(cos, sin)
        ctx.interleaved = interleaved
        ctx.inplace = inplace
        return out if not inplace else x

    @staticmethod
    def backward(ctx, do):
        cos, sin = ctx.saved_tensors
        _, seqlen, _, headdim = do.shape
        rotary_dim = cos.shape[-1]
        rotary_dim *= 2
        inplace = ctx.inplace
        do_ro = do[..., :rotary_dim]
        do1, do2 = (
            do_ro.chunk(2, dim=-1) if not ctx.interleaved else (do_ro[..., ::2], do_ro[..., 1::2])
        )
        dx = torch.empty_like(do) if not inplace else do
        if inplace:
            dx1, dx2 = do1, do2
        else:
            dx_ro = dx[..., :rotary_dim]
            dx1, dx2 = (
                dx_ro.chunk(2, dim=-1)
                if not ctx.interleaved
                else (dx_ro[..., ::2], dx_ro[..., 1::2])
            )
        rotary_emb.apply_rotary(
            do1,
            do2,
            rearrange(cos[:seqlen], "s d -> s 1 d"),
            rearrange(sin[:seqlen], "s d -> s 1 d"),
            dx1,
            dx2,
            True,
        )
        if not inplace and rotary_dim < headdim:
            dx[..., rotary_dim:].copy_(do[..., rotary_dim:])
        return dx, None, None, None, None


def _torch_apply_rotary_emb(x, cos, sin, interleaved=False, inplace=False):
    seqlen = x.shape[1]
    rotary_dim = cos.shape[-1] * 2
    x_ro = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]
    cos = rearrange(cos[:seqlen], "s d -> 1 s 1 d")
    sin = rearrange(sin[:seqlen], "s d -> 1 s 1 d")
    if interleaved:
        x1, x2 = x_ro[..., ::2], x_ro[..., 1::2]
        out_ro = torch.empty_like(x_ro)
        out_ro[..., ::2] = x1 * cos - x2 * sin
        out_ro[..., 1::2] = x1 * sin + x2 * cos
    else:
        x1, x2 = x_ro.chunk(2, dim=-1)
        out_ro = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    if inplace:
        x[..., :rotary_dim] = out_ro
        return x
    if rotary_dim < x.shape[-1]:
        return torch.cat([out_ro, x_pass], dim=-1)
    return out_ro


apply_rotary_emb_func = _torch_apply_rotary_emb if rotary_emb is None else ApplyRotaryEmb.apply
