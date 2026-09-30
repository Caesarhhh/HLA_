"""HLA-WM Top1 inference: Stage 1, AR refinement, or bidirectional refinement."""

import json
import os
import numpy as np

POLICY = {
    "REPLAY_IMAGE_WIDTH": "1280",
    "REPLAY_IMAGE_HEIGHT": "704",
    "REPLAY_LOG_SELECTION": "1",
    "STORE_CHUNK_SUMMARY": "1",
}


def turn_overrides(c2w):
    result = {}
    for chunk in range(1, (len(c2w) - 1) // 24 + 1):
        xyz = c2w[chunk * 24 : (chunk + 1) * 24, :3, 3]
        if len(xyz) < 8:
            continue
        first = xyz[7] - xyz[0]
        last = xyz[-1] - xyz[-8]
        denom = np.linalg.norm(first) * np.linalg.norm(last)
        angle = (
            0
            if denom <= 1e-10
            else np.degrees(np.arccos(np.clip(np.dot(first, last) / denom, -1, 1)))
        )
        if angle >= 25:
            result[str(chunk)] = 3
    return result


def main():
    from inference_video_scripts.wm import inference_sana_wm_streaming as streaming
    from inference_video_scripts.wm.inference_sana_wm import (
        InferenceConfig,
        GenerationParams,
        RefinerSettings,
        SanaWMPipeline,
        resize_and_center_crop,
        load_intrinsics,
        transform_intrinsics_for_crop,
        write_video,
    )
    from diffusion.utils.logger import get_root_logger
    import pyrallis
    import torch
    from PIL import Image

    p = streaming._build_parser()
    p.description = __doc__
    p.add_argument("--mode", choices=["stage1", "ar", "bi"], default="stage1")
    p.add_argument("--baseline", action="store_true")
    p.add_argument(
        "--median-depth",
        type=float,
        default=3.0,
        help="Scene depth in camera translation units",
    )
    args = p.parse_args()
    own = args
    if own.median_depth <= 0:
        p.error("--median-depth must be positive")
    if args.camera is None or args.intrinsics is None:
        p.error("Supply --camera and --intrinsics for geometric retrieval")
    if args.num_frame_per_block != 3:
        p.error("Paper policy requires --num_frame_per_block 3")
    c2w = np.load(args.camera)
    n = min(args.num_frames, len(c2w))
    n = ((n - 1) // 24) * 24 + 1
    if n < 25:
        p.error("At least 25 frames are required")
    c2w = c2w[:n]
    for key in list(os.environ):
        if key.startswith("SANA_WM_GDN_"):
            del os.environ[key]
    image = Image.open(args.image).convert("RGB")
    cropped, src, resized, offset = resize_and_center_crop(image)
    intr = transform_intrinsics_for_crop(
        load_intrinsics(args.intrinsics, n), src, resized, offset
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    camera_file = args.output_dir / (args.name + "_camera.npy")
    intr_file = args.output_dir / (args.name + "_intrinsics.npy")
    np.save(camera_file, c2w)
    intr_matrix = np.zeros((n, 3, 3), dtype=np.float32)
    intr_matrix[:, 0, 0] = intr[:, 0]
    intr_matrix[:, 1, 1] = intr[:, 1]
    intr_matrix[:, 0, 2] = intr[:, 2]
    intr_matrix[:, 1, 2] = intr[:, 3]
    intr_matrix[:, 2, 2] = 1
    np.save(intr_file, intr_matrix)
    if not own.baseline:
        values = {
            **POLICY,
            "REPLAY_OVERLAP_CAMERA": str(camera_file.resolve()),
            "REPLAY_OVERLAP_INTRINSICS": str(intr_file.resolve()),
            "REPLAY_MEDIAN_DEPTH": str(own.median_depth),
            "REPLAY_KEEP_LAST_CHUNKS_OVERRIDES_JSON": json.dumps(turn_overrides(c2w)),
        }
        os.environ.update({"SANA_WM_GDN_" + k: v for k, v in values.items()})
    logger = get_root_logger()
    streaming._apply_fast_defaults()
    streaming._apply_precision_args(args, logger)
    config_path, model_path, vae_path, refiner_root, gemma_root = (
        streaming._resolve_streaming_paths(args)
    )
    config = pyrallis.parse(
        config_class=InferenceConfig, config_path=str(config_path), args=[]
    )
    config.vae.vae_type = "LTX2VAE_diffusers_causal"
    config.vae.vae_pretrained = str(vae_path)
    refiner = (
        None
        if own.mode == "stage1"
        else RefinerSettings(
            root=str(refiner_root),
            gemma_root=str(gemma_root),
            sink_size=args.sink_size,
            seed=args.refiner_seed,
            block_size=3 if own.mode == "ar" else None,
            kv_max_frames=args.refiner_kv_max_frames,
        )
    )
    pipe = SanaWMPipeline(
        config=config,
        model_path=str(model_path),
        device=torch.device("cuda"),
        refiner=refiner,
        offload_vae=args.offload_vae,
        offload_refiner=args.offload_refiner,
        offload_text_encoder=args.offload_text_encoder,
        logger=logger,
    )
    params = GenerationParams(
        num_frames=n,
        fps=args.fps,
        cfg_scale=args.cfg_scale,
        flow_shift=args.flow_shift,
        seed=args.seed,
        negative_prompt=args.negative_prompt,
        sampling_algo="self_forcing",
        num_cached_blocks=3,
        sink_token=True,
        num_frame_per_block=3,
        denoising_step_list=[int(x) for x in args.denoising_step_list.split(",")],
    )
    result = pipe.generate(cropped, args.prompt.read_text().strip(), c2w, intr, params)
    path = write_video(args.output_dir, args.name, result["video"], args.fps, logger)
    (args.output_dir / (args.name + "_policy.json")).write_text(
        json.dumps(
            {
                "mode": own.mode,
                "baseline": own.baseline,
                "frames": n,
                "policy": {
                    k: v for k, v in os.environ.items() if k.startswith("SANA_WM_GDN_")
                },
            },
            indent=2,
        )
    )
    print(path)


if __name__ == "__main__":
    main()
