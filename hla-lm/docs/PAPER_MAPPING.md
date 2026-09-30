# Paper correspondence

Source: project experimental results. No newly measured benchmark scores are substituted.

- Table 1: the `Avg.` column is reproduced in `paper-results.json` and the project page.
- Tables 2–3: Qwen3.5 adaptation results are described in the paper.
- Training: 1.3B-scale GDN, 24 layers, 4096-token sequences, 100B tokens, matched global batch 512; HLA C=256 / P=16.
- `GLA_GDN` / `gla_gdn` are legacy internal names for the GDN-based HLA path.
- `lit_gpt/gated_delta_net.py`: router, affine gating and cached autoregressive state.
- `lit_gpt/triton_affine_*`: chunk summaries, prefix/history composition and decode kernels.
- `lit_gpt/triton_hla_router.py` / `triton_pool_prefix_attention.py`: pooled routing kernels.

GPU smoke checks establish execution, not reproduction of the paper's benchmark scores. Evaluation suites and benchmark datasets are intentionally omitted.
