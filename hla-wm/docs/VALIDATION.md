# Main-setting cleanup — 2026-09-29

- Python syntax/import-reference audit and undefined-name checks passed for changed code.
- 5,936 Top1 selections match the pre-cleanup implementation, covering recent=1/3, ties, and startup chunks.
- Frustum overlap matrices match exactly on synthetic translated-camera trajectories; the turn-aware recent override is unchanged.
- 16 CPU comparisons of the main/camera kernel wrappers match, using mocked GPU kernels: initial/cached state, denoise/cache-save passes, and padded/unpadded dimensions.
- Full GPU video generation was not rerun for this cleanup. The runs below describe the earlier release, not fresh validation of this revision.

# Validation — 2026-09-26

Executed from this release directory on an allocated NVIDIA H200 NVL (m3u012). No original research repository was on PYTHONPATH. Third-party dependencies came from the existing SANA environment and its package overlay.

| Check | Result |
|---|---|
| Stage 1, Top1, 97 frames at 1280×704 | PASS; `outputs/smoke/top1_generated.mp4` |
| Stage 1 + AR refinement, Top1, 145 input frames | PASS; `outputs/smoke/top1_ar_generated.mp4` |
| Stage 1 + bidirectional refinement, Top1, 145 input frames | PASS; `outputs/smoke/top1_bi_generated.mp4` |
| Nontrivial Top1 selection | Chunk 4 selected [0,2,3]; chunk 5 selected [0,3,4] |
| Historical Transformer replay | 0 forwards in all three runs |
| Environment dependency resolution | PASS, Python 3.10; `environment-resolution.log` |
| Static website | Desktop 1440px / mobile 390px; images loaded; no horizontal page overflow; split tabs work |

Both refiners load `SANA-WM_streaming/refiner_diffusers` and `gemma3_12b`; AR uses block size 3, bidirectional uses no block restriction. These are execution checks, not benchmark score reproductions. Model loading and cold compiler time appear in the logs and are not reported as paper FPS.

Environment installation was dependency-resolved; a new full virtual environment was not installed. The actual GPU runs used the existing pinned-compatible runtime. MMCV build constraints retain setuptools' legacy pkg_resources support; WM Python is limited to 3.10–3.12.

Logs: `smoke-gpu.log`, `smoke-ar.log`, `smoke-bi.log`.
