# Source provenance

Source revision: `8fc2cab808431a29e98ce554d8e7a8a88821251f`.

Copied from the working SANA-WM inference tree (which contains local HLA-WM modifications), not merely from the upstream commit. Core model, sampler, GDN summary/recomposition and refiner code is retained. Training/distillation pipelines and benchmark suites are excluded. Required upstream model-loading/data utilities remain to satisfy inference imports.

Release changes: public Top1 entrypoint with turn-aware recent context, input-intrinsics conversion, environment setup and documentation.

The public WM release retains the main Top1 chunk-retrieval method, turn-aware recent context, GDN summary recomposition and aligned Softmax K/V, plus native SANA-WM comparison. Alternative retrieval policies, partial-component and spatial-selection experiments, camera-state recomposition, and refiner history-selection variants are removed.
