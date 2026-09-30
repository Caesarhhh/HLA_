from __future__ import annotations

import torch


def _source_major_history(local_output, local_readout, chunk_a, chunk_b, weight_chunk):
    output = local_output
    readout = local_readout
    num_chunks = output.shape[2]
    for source_idx in range(num_chunks - 2, -1, -1):
        query_slice = slice(source_idx + 1, None)
        gate = weight_chunk[:, :, query_slice, :, source_idx].float()
        readout_tail = readout[:, :, query_slice]
        contribution = torch.einsum(
            "bhcld,bhde->bhcle", readout_tail, chunk_b[:, :, source_idx]
        )
        transformed = torch.einsum(
            "bhde,bhcld->bhcle", chunk_a[:, :, source_idx], readout_tail
        )
        output_tail = output[:, :, query_slice] + gate[..., None] * contribution
        readout_tail = (1.0 - gate)[..., None] * readout_tail + gate[..., None] * transformed
        output = torch.cat((output[:, :, : source_idx + 1], output_tail), dim=2)
        readout = torch.cat((readout[:, :, : source_idx + 1], readout_tail), dim=2)
    return output


# Compilation is lazy on first invocation and shared by all 24 identical HLA
# layers. Default mode keeps restart latency reasonable; cuBLAS remains the
# selected exact-FP32 backend for the D=256 batched matrix products.
compiled_affine_history_mix = torch.compile(
    _source_major_history,
    fullgraph=True,
    dynamic=False,
    mode="default",
)
