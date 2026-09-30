# Source provenance

Source revision: `2c4d6d7d458b7d3a21041bbbb6954e204a43b9f8`.

The model layers, router, kernels, and training and inference entry points are extracted from the GDN training branch. Original GatedDeltaNet NVIDIA Source Code License-NC and embedded Lit-GPT / FLA notices are preserved.

Release changes: portable training launcher, explicit paper HLA configuration, strict checkpoint generation entrypoint, environment setup, GPU integration check and documentation.

2026-09-29: specialized the public LLM release to the C256/P16 raw-QK/logmean, self-attention-pooling, affine-sigmoid recipe and matched GDN baseline. Removed alternate routing/state composition, MHLA models, adaptation-only training paths and experimental losses. Local experiment repositories were not modified.
