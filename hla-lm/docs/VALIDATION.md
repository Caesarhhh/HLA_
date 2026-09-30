# Main-setting release validation — 2026-09-29

Compared against the pre-cleanup release at `c497b26`. The release now includes only the C256/P16 HLA recipe and the matched 1.3B GDN baseline. Model dimensions can be reduced by the smoke checks.

- Python compilation and undefined-name checks of changed Python files: PASS.
- Tiny GPT construction: PASS, 1,773,068 parameters.
- Strict HLA layer state-dictionary compatibility: PASS.
- 18 router forward/backward comparisons across train/eval modes and pooling/chunk boundaries: exact agreement, including input and parameter gradients.
- Affine A/B chunk summaries and history composition: exact agreement; training gate gradients agree.
- 8 cached prefill and 24 decode comparisons, covering chunk boundaries and batch sizes 1 and 2: exact agreement.
- Budget, effective-support and binary auxiliary losses and input gradients: exact agreement.
- HLA and GDN training launcher dry-runs: PASS; all emitted flags exist in the trainer.
- Trainer and generation CLI help: PASS.

The numerical comparisons ran on CPU. Cached inference used a PyTorch recurrent reference in place of the CUDA GDN kernel and disabled the fused CUDA router; it exercised the released pooling, affine composition and cache logic. This is not validation of GPU kernel execution. GPU training/generation was not rerun for this cleanup.

Current logs: `main-setting-cpu.log`, `train-launcher-dry-run.log`, `train-help.log`. Historical GPU logs below describe the earlier release, not a rerun of the current code.

## Historical GPU validation — 2026-09-26

Executed from this release directory on an allocated NVIDIA H200 NVL (m3u012), using PyTorch 2.9.1+cu128, Triton 3.5.1 and Python 3.13 in the existing GDN runtime.

| Check | Result |
|---|---|
| Tiny HLA forward / backward / AdamW step | PASS; finite loss and gradients |
| Cached prefill versus uncached last-token logits | PASS within BF16 tolerance |
| Cached one-token decode | PASS; finite output |
| Real training entrypoint | PASS; synthetic 4096-token batch, 2-layer / 1.77M-parameter model |
| Training auxiliary losses | Budget, effective-support and binary gate losses executed |
| Optimizer and checkpoints | Step 1 completed; latest, step000001 and final checkpoints written |
| Portable training launcher | DRY_RUN passed with the paper C256/P16 recipe |
| Trainer CLI | `pretrain.py --help` passed |
| Environment dependency resolution | PASS; `environment-resolution.log` |
| Website | Desktop and mobile browser checks passed |

The real trainer check uses `scripts/train_smoke.py`, with `GDN_COMPILE_AFFINE_HISTORY_TRAINING=1`. It runs the same pretrain.py loop and loss hooks with a reduced model and synthetic packed data; it does not reproduce the 1.3B / 100B-token experiment or any benchmark scores. The first optimizer step included cold compiler work.

Logs: `smoke-gpu.log`, `train-smoke.log`, `train-launcher-dry-run.log`, `train-help.log`.

Checkpoint: `outputs/train_smoke_e5grepmf/outputs/512x4k_100B_tiny_execution_check/final-model-ckpt.pth` (synthetic smoke model, not a usable pretrained language model).

Installation requirements were dependency-resolved. GPU execution reused the existing runtime; a separate fresh full virtual environment was not installed.
