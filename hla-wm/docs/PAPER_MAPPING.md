# Paper correspondence

Source: project experimental results. All reported scores follow the paper, without replacement by other profiling runs.

- Table 1: both SANA-WM-Bench splits, all three pipelines, all seven metrics are reproduced in `paper-results.json` and the project page.
- Table 2: all MBench-A aggregate rows are reproduced.
- Main setting: Top1 geometric retrieval, sink=1, recent=1, recent=3 at a within-chunk translation-direction turn of at least 25 degrees.
- AR and bidirectional refinement both load the streaming bundle's AR refiner weights; only the attention mode differs.
- `diffusion/scheduler/self_forcing_flow_euler_sampler.py`: geometry retrieval, selected softmax KV, summary storage and recomposition.
- `diffusion/model/ops/fused_streaming.py`: frame-wise Phase-A summary retention and Phase-B scan.
- `diffusion/model/ops/fused_gdn_chunkwise.py`: Triton GDN kernels.
- The t=0 cache-writing pass is inherited from the baseline. HLA retains sufficient statistics in that existing pass. Statistics are state-independent conditional on the layer coefficients recorded in that pass.

GPU smoke checks establish execution, not reproduction of the paper's benchmark scores. Evaluation suites and benchmark datasets are intentionally omitted.
