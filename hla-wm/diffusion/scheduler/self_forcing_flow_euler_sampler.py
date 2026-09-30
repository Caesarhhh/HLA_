# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Self-forcing flow Euler samplers for chunk-causal autoregressive video.

This module provides the streaming Sana-WM inference samplers:

* ``SelfForcingFlowEuler`` is the base chunk-causal autoregressive sampler.
  It walks ``base_chunk_frames``-sized chunks left-to-right, denoising each
  chunk against a KV cache accumulated from previously generated chunks.

* ``SelfForcingFlowEulerCamCtrl`` extends the base sampler with the camera
  conditioning extras (``camera_conditions``, ``chunk_plucker``, etc.),
  first-frame conditioning, and the 10-slot dual-mode (state / concat) KV
  cache layout used by the camctrl ``forward_long`` path. This is the
  sampler used by the end-to-end streaming Sana-WM + LTX-2 refiner.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import time
from pathlib import Path

import torch


def _gdn_replay_profile_enabled() -> bool:
    return os.environ.get("SANA_WM_GDN_REPLAY_PROFILE", "").strip() == "1"


def _profile_sync(device: torch.device | None = None) -> None:
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)


# Diffusers ships with a hard ``import flash_attn`` in some attention backends
# that raises before ``flash_attn_interface`` (FA4) is considered. We
# temporarily hide the installed ``flash_attn`` module so diffusers takes the
# FA-not-installed branch, then restore it so downstream code can still use FA.
_fa_spec = importlib.util.find_spec("flash_attn")
_has_fa = _fa_spec is not None

_real_fa_module = None

if _has_fa:
    _real_fa_module = sys.modules.get("flash_attn")
    sys.modules["flash_attn"] = None

try:
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.models.modeling_outputs import Transformer2DModelOutput
    from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
finally:
    if _has_fa:
        if _real_fa_module is not None:
            sys.modules["flash_attn"] = _real_fa_module
        else:
            del sys.modules["flash_attn"]

from tqdm import tqdm

from diffusion.model.nets.basic_modules import CachedGLUMBConvTemp
from diffusion.model.nets.sana_blocks import CachedCausalAttention

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _inject_sliced_extras(
    extra: dict[str, object],
    kwargs: dict,
    num_chunk_frames: int,
    end_f: int,
) -> None:
    """Inject ``extra`` kwargs into ``kwargs``, slicing temporal dims.

    Tensors whose temporal axis is longer than ``num_chunk_frames`` are sliced
    to ``[end_f - num_chunk_frames, end_f)``. Layouts handled:

    * ``(B, C, T, H, W)`` — e.g. ``chunk_plucker``; sliced on dim 2.
    * ``(B, T, ...)`` — e.g. ``camera_conditions``; sliced on dim 1.

    Any key already present in ``kwargs`` is left untouched.
    """
    begin_f = end_f - num_chunk_frames
    for k, v in extra.items():
        if k in kwargs:
            continue
        if isinstance(v, torch.Tensor):
            if v.ndim == 5:
                kwargs[k] = v[:, :, begin_f:end_f] if v.shape[2] > num_chunk_frames else v
            elif v.ndim >= 3 and v.shape[1] > num_chunk_frames:
                kwargs[k] = v[:, begin_f:end_f]
            else:
                kwargs[k] = v
        else:
            kwargs[k] = v


def _pop_extra_model_kwargs(model_kwargs: dict) -> dict:
    """Pop all keys from ``model_kwargs`` except ``mask`` and ``data_info``.

    The popped entries are the "extras" (camera tensors, RoPE caches, etc.)
    that need per-chunk temporal slicing before being forwarded to the model.
    """
    extra: dict = {}
    for key in list(model_kwargs):
        if key not in ("mask", "data_info"):
            extra[key] = model_kwargs.pop(key)
    return extra


# ---------------------------------------------------------------------------
# Cache-slot index constants
# ---------------------------------------------------------------------------
#
# The camctrl ``forward_long`` path uses a 10-slot KV cache per block. Two
# layouts share the slot table, distinguished by slot 6 (``_SLOT_TYPE_FLAG``):
#
# Old layout (ChunkCausalGDN / ChunkCausalSoftmaxAttn):
#   0: k, 1: v, 2: beta, 3: decay, 4: shortconv, 5: tconv, 6-9: None
#
# New layout (CachedChunkCausalGDN / CachedChunkCausalSoftmaxAttn):
#   GDN:     0: S_kv state, 1: S_z state, 2: cam_S_kv state, 3: cam K-conv state
#   Softmax: 0: k post-RoPE, 1: v, 2: cam_k post-UCPE, 3: cam_v post-UCPE
#   Both:    4: shortconv, 5: tconv, 6: type flag (1.0=state, 0.0=concat),
#            9: FFN tconv state written by ``CachedGLUMBConvTemp``.

_NUM_CACHE_SLOTS = 10  # 7 active + 3 reserved

_SLOT_K = 0
_SLOT_V = 1
_SLOT_BETA = 2
_SLOT_DECAY = 3
_SLOT_SHORTCONV = 4
_SLOT_TCONV = 5
_SLOT_TYPE_FLAG = 6
_SLOT_GDN_MAIN_SUMMARY = 7
_SLOT_GDN_CAM_SUMMARY = 8

# Old-layout concat slots (when type flag is absent).
_CONCAT_SLOTS = (_SLOT_K, _SLOT_V, _SLOT_BETA, _SLOT_DECAY)
# New-layout softmax concat slots (when type flag == 0.0).
_SOFTMAX_CONCAT_SLOTS = (0, 1, 2, 3)
_LAST_CHUNK_SLOTS = (_SLOT_SHORTCONV, _SLOT_TCONV)


# ---------------------------------------------------------------------------
# Base self-forcing sampler
# ---------------------------------------------------------------------------


class SelfForcingFlowEuler:
    """Chunk-causal autoregressive flow-Euler sampler with KV cache support.

    Walks ``base_chunk_frames``-sized chunks left-to-right. For each chunk the
    sampler runs the diffusion schedule, then performs one extra ``t = 0``
    forward with ``save_kv_cache=True`` to write the KV cache that subsequent
    chunks consume. This implements the "self-forcing" recipe where every
    chunk is conditioned on the model's own previously-generated context.
    """

    def __init__(
        self,
        model_fn: object,
        condition: torch.Tensor,
        uncondition: torch.Tensor,
        cfg_scale: float,
        flow_shift: float = 3.0,
        model_kwargs: dict | None = None,
        base_chunk_frames: int = 10,
        num_cached_blocks: int = -1,
        **kwargs: object,
    ) -> None:
        self.model = model_fn
        self.condition = condition
        self.uncondition = uncondition
        self.cfg_scale = cfg_scale
        self.model_kwargs = model_kwargs or {}
        self.mask = self.model_kwargs.pop("mask", None)
        self.flow_shift = flow_shift
        self.base_chunk_frames = base_chunk_frames
        self.rank = os.environ.get("RANK", 0)
        self.cached_modules = None
        # Populate ``self.cached_modules`` and ``self.num_model_blocks``.
        self.get_cached_modules_by_block()
        self.num_cached_blocks = num_cached_blocks
        self.use_softmax_attention = kwargs.get("use_softmax_attention", False)
        self.sink_token = kwargs.get("sink_token", False)

    def create_autoregressive_segments(self, total_frames: int) -> list[int]:
        """Build chunk boundaries for an autoregressive sweep.

        Returns a list of frame indices ``[0, c1, c2, ..., total_frames]`` of
        length ``num_chunks + 1`` such that chunk ``i`` covers frames
        ``[chunk_indices[i], chunk_indices[i + 1])``. The first chunk absorbs
        any remainder so subsequent chunks all have exactly
        ``base_chunk_frames`` frames.
        """
        remained_frames = total_frames % self.base_chunk_frames
        num_chunks = total_frames // self.base_chunk_frames
        chunk_indices = [0]
        for i in range(num_chunks):
            cur_idx = chunk_indices[-1] + self.base_chunk_frames
            if i == 0:
                cur_idx += remained_frames
            chunk_indices.append(cur_idx)
        return chunk_indices

    def get_cached_modules_by_block(self) -> list[list[torch.nn.Module]]:
        """Locate ``CachedCausalAttention`` and ``CachedGLUMBConvTemp`` modules.

        The result is a list (one entry per transformer block) of the cached
        modules inside that block. ``self.num_model_blocks`` is set as a side
        effect.
        """
        if self.cached_modules is not None:
            return self.cached_modules

        # Unwrap DDP if present.
        model = self.model.module if hasattr(self.model, "module") else self.model

        cached_modules: list[list[torch.nn.Module]] = []

        def collect_from_block(block: torch.nn.Module, block_idx: int) -> list[torch.nn.Module]:
            attention_modules: list[torch.nn.Module] = []
            conv_modules: list[torch.nn.Module] = []

            def collect_recursive(module: torch.nn.Module) -> None:
                if isinstance(module, CachedCausalAttention):
                    attention_modules.append(module)
                elif isinstance(module, CachedGLUMBConvTemp):
                    conv_modules.append(module)
                for child in module.children():
                    collect_recursive(child)

            collect_recursive(block)
            return attention_modules + conv_modules

        if hasattr(model, "blocks"):
            blocks = model.blocks
        elif hasattr(model, "transformer_blocks"):
            blocks = model.transformer_blocks
        elif hasattr(model, "layers"):
            blocks = model.layers
        else:
            raise ValueError("Model does not have any blocks")

        self.num_model_blocks = len(blocks)
        for block_idx, block in enumerate(blocks):
            block_modules = collect_from_block(block, block_idx)
            cached_modules.append(block_modules)

        self.cached_modules = cached_modules
        return cached_modules

    # NOTE: SelfForcingFlowEulerCamCtrl overrides ``_initialize_kv_cache``,
    # ``_accumulate_softmax_kv_cache``, ``accumulate_kv_cache`` and ``sample``
    # with the 10-slot dual-mode (state / concat) cache layout used by the
    # camctrl ``forward_long`` path.  The base class keeps ``__init__``,
    # ``create_autoregressive_segments`` and ``get_cached_modules_by_block``
    # only — the inherited entry points for CamCtrl.  The non-camctrl
    # streaming path (6-slot softmax + 3-slot linear caches) is intentionally
    # not shipped in this repo.


# ---------------------------------------------------------------------------
# CamCtrl self-forcing sampler
# ---------------------------------------------------------------------------


class SelfForcingFlowEulerCamCtrl(SelfForcingFlowEuler):
    """SelfForcingFlowEuler with camera conditioning and first-frame anchoring.

    Wraps ``SelfForcingFlowEuler`` to support the camctrl ``forward_long`` API
    used by the streaming Sana-WM pipeline:

    * Camera tensors (``camera_conditions``, ``chunk_plucker``, etc.) are
      popped from ``model_kwargs`` at init and injected into each model call
      sliced to the current temporal window.
    * The KV cache uses the 10-slot layout with a dual-mode (state / concat)
      type flag at slot 6, and supports a "sink" chunk anchored at the start
      of the sequence when the sliding window has scrolled past chunk 0.
    * ``condition_frame_info`` in ``data_info`` (e.g. ``{0: 0.0}``) marks
      frames that should be treated as fully clean and restored after every
      denoising step so the per-token scheduler cannot corrupt them.
    """

    def __init__(
        self,
        model_fn: object,
        condition: torch.Tensor,
        uncondition: torch.Tensor,
        cfg_scale: float,
        model_kwargs: dict | None = None,
        **kw: object,
    ) -> None:
        model_kwargs = model_kwargs or {}
        self._extra_model_kwargs = _pop_extra_model_kwargs(model_kwargs)
        super().__init__(
            model_fn,
            condition,
            uncondition,
            cfg_scale,
            model_kwargs=model_kwargs,
            **kw,
        )
        self._patch_model()

    # ------------------------------------------------------------------
    # Camera tensor slicing
    # ------------------------------------------------------------------

    def _patch_model(self) -> None:
        """Monkey-patch ``model.forward_long`` to inject sliced camera tensors."""
        extra = self._extra_model_kwargs

        orig_forward_long = self.model.forward_long

        # Keys that ``forward_long`` recomputes or doesn't need:
        # - ``pos_embeds``: RoPE is recomputed from (start_f, end_f) or frame_index.
        # - ``cam_pos_embeds``: dict of full-sequence tensors; forward_long
        #   recomputes from sliced camera_conditions instead.
        # - ``chunk_index``: not used in KV-cache mode (blocks check kv_cache).
        # - ``frame_index``: forwarded explicitly (not via _inject_sliced_extras).
        _SKIP_FOR_FORWARD_LONG = frozenset({"pos_embeds", "chunk_index", "cam_pos_embeds", "frame_index"})

        def _forward_long_with_extras(
            x: torch.Tensor,
            timestep: torch.Tensor,
            y: torch.Tensor,
            mask: torch.Tensor | None = None,
            **kwargs: object,
        ) -> object:
            end_f = kwargs.get("end_f", x.shape[2])
            num_chunk_frames = x.shape[2]
            filtered_extra = {k: v for k, v in extra.items() if k not in _SKIP_FOR_FORWARD_LONG}
            _inject_sliced_extras(filtered_extra, kwargs, num_chunk_frames, end_f)
            return orig_forward_long(x, timestep, y, mask=mask, **kwargs)

        self.model.forward_long = _forward_long_with_extras

    def _select_replay_chunks(self, chunk_idx: int, keep_last: int, sink_chunks: int) -> list[int]:
        """Select sink, recent context and the best historical frustum match."""
        sink_end = min(sink_chunks, chunk_idx)
        recent_start = max(sink_end, chunk_idx - keep_last)
        selected = list(range(sink_end)) + list(range(recent_start, chunk_idx))
        candidates = range(sink_end, recent_start)
        scores = getattr(self, "_replay_chunk_frustum_overlap", None)
        if candidates:
            if scores is None:
                raise ValueError("HLA-WM requires camera poses and intrinsics for history retrieval")
            # Preserve the original ranking's newer-chunk tie break.
            selected.append(max(candidates, key=lambda idx: (float(scores[idx][chunk_idx]), idx)))
        return sorted(set(selected))

    def _prepare_replay_overlap_from_camera(self, num_chunks: int) -> None:
        """Prepare the main setting's symmetric frustum overlap scores."""
        import numpy as np

        self._replay_log_selection = os.environ.get("SANA_WM_GDN_REPLAY_LOG_SELECTION", "").strip() == "1"
        camera_path = os.environ.get("SANA_WM_GDN_REPLAY_OVERLAP_CAMERA", "").strip()
        if not camera_path:
            return
        poses = np.load(Path(camera_path).expanduser())
        if poses.ndim != 3 or poses.shape[-2:] != (4, 4):
            raise ValueError(f"Camera poses must be (F,4,4), got {poses.shape}")
        num_video_frames = int(poses.shape[0] - 1)
        frame_samples = []
        for chunk_idx in range(num_chunks):
            start = round(chunk_idx * num_video_frames / num_chunks)
            end = max(start + 1, round((chunk_idx + 1) * num_video_frames / num_chunks))
            frame_samples.append(np.arange(start, min(end, num_video_frames), dtype=np.int64))
        intrinsics_path = os.environ.get("SANA_WM_GDN_REPLAY_OVERLAP_INTRINSICS", "").strip()
        if not intrinsics_path:
            raise ValueError("HLA-WM requires intrinsics for history retrieval")
        intrinsics = np.load(Path(intrinsics_path).expanduser())
        if intrinsics.ndim == 2:
            intrinsics = np.broadcast_to(intrinsics[None], (poses.shape[0], 3, 3))
        if intrinsics.ndim != 3 or intrinsics.shape[-2:] != (3, 3):
            raise ValueError(f"Intrinsics must be (F,3,3) or (3,3), got {intrinsics.shape}")
        median_depth = float(os.environ.get("SANA_WM_GDN_REPLAY_MEDIAN_DEPTH", "3.0"))
        self._replay_chunk_frustum_overlap = self._compute_chunk_frustum_overlap(
            poses=poses, intrinsics=intrinsics, frame_samples=frame_samples,
            depths=median_depth * np.array([0.5, 1.0, 1.5], dtype=np.float32), grid_size=7,
        )

    def _compute_chunk_frustum_overlap(
        self,
        *,
        poses,
        intrinsics,
        frame_samples,
        depths,
        grid_size: int,
    ):
        """Compute symmetric chunk visibility from camera frustums."""
        import numpy as np

        first_k = intrinsics[0]
        width = float(os.environ.get("SANA_WM_GDN_REPLAY_IMAGE_WIDTH", "") or (2.0 * first_k[0, 2]))
        height = float(os.environ.get("SANA_WM_GDN_REPLAY_IMAGE_HEIGHT", "") or (2.0 * first_k[1, 2]))
        xs = np.linspace(0.05 * width, 0.95 * width, grid_size, dtype=np.float32)
        ys = np.linspace(0.05 * height, 0.95 * height, grid_size, dtype=np.float32)
        pix = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)

        chunk_points = []
        for frame_ids in frame_samples:
            pts_world = []
            for fid in frame_ids:
                k = intrinsics[min(int(fid), intrinsics.shape[0] - 1)]
                fx, fy = float(k[0, 0]), float(k[1, 1])
                cx, cy = float(k[0, 2]), float(k[1, 2])
                dirs = np.stack([(pix[:, 0] - cx) / fx, (pix[:, 1] - cy) / fy, np.ones(len(pix))], axis=1)
                for depth in depths:
                    pts_cam = dirs * float(depth)
                    pts_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1), dtype=np.float32)], axis=1)
                    pts_world.append((poses[fid] @ pts_h.T).T[:, :3])
            chunk_points.append(np.concatenate(pts_world, axis=0).astype(np.float32))

        def visibility_mask(points, frame_ids):
            if points.size == 0:
                return np.zeros((0,), dtype=bool)
            pts_h = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
            visible = np.zeros(points.shape[0], dtype=bool)
            for fid in frame_ids:
                w2c = np.linalg.inv(poses[int(fid)])
                pts_cam = (w2c @ pts_h.T).T[:, :3]
                z = pts_cam[:, 2]
                valid_z = z > 1e-4
                if not np.any(valid_z):
                    continue
                k = intrinsics[min(int(fid), intrinsics.shape[0] - 1)]
                u = k[0, 0] * (pts_cam[:, 0] / np.maximum(z, 1e-6)) + k[0, 2]
                v = k[1, 1] * (pts_cam[:, 1] / np.maximum(z, 1e-6)) + k[1, 2]
                visible |= valid_z & (u >= 0.0) & (u < width) & (v >= 0.0) & (v < height)
            return visible

        num_chunks = len(frame_samples)
        directed = np.zeros((num_chunks, num_chunks), dtype=np.float32)
        for i in range(num_chunks):
            for j in range(num_chunks):
                mask = visibility_mask(chunk_points[i], frame_samples[j])
                directed[i, j] = float(np.mean(mask)) if mask.size else 0.0

        symmetric = 0.5 * (directed + directed.T)
        return symmetric.astype(np.float32)

    def _local_data_info_for_chunk(
        self,
        data_info: dict,
        image_vae_embeds: torch.Tensor | None,
        start_f: int,
        end_f: int,
    ) -> dict:
        local_data_info = dict(data_info)
        if image_vae_embeds is not None:
            local_data_info["image_vae_embeds"] = image_vae_embeds[:, :, start_f:end_f]
        return local_data_info

    def _prompt_embeds_for_chunk(self, chunk_idx: int, do_classifier_free_guidance: bool) -> torch.Tensor:
        prompt_embeds = self.condition[chunk_idx].unsqueeze(0)
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([self.uncondition, prompt_embeds], dim=0)
        return prompt_embeds


    def _recompose_main_gdn_cache_from_summaries_batched(
        self,
        replay_chunks: list[int],
        baseline_chunk_kv_cache: list,
    ) -> tuple[list, bool]:
        """Recompose all shape-compatible GDN blocks with one Phase-B call.

        Phase B is independent along its leading ``B*H`` dimension.  The old
        implementation launched one scan per model block; stacking blocks on
        that dimension preserves the recurrence while substantially reducing
        Python, concatenation, and Triton launch overhead.
        """
        from diffusion.model.ops.fused_gdn_chunkwise import phase_b_triton

        profile = _gdn_replay_profile_enabled()
        profile_t0 = time.perf_counter() if profile else 0.0
        summary_store = getattr(self, "_gdn_chunk_summary_cache", None)
        recomposed_cache = [[slot for slot in block_cache] for block_cache in baseline_chunk_kv_cache]
        if summary_store is None or not replay_chunks:
            return recomposed_cache, False

        # A non-GDN block has no summary in every chunk.  Discover GDN blocks
        # from the first selected chunk without synchronizing GPU type flags
        # back to Python via ``Tensor.item()``.
        first_chunk = replay_chunks[0]
        if first_chunk >= len(summary_store):
            return recomposed_cache, False
        block_ids = [block_id for block_id, summary in enumerate(summary_store[first_chunk]) if summary is not None]
        if not block_ids:
            return recomposed_cache, False

        groups: dict[tuple, list[tuple[int, list[tuple]]]] = {}
        num_summaries = 0
        for block_id in block_ids:
            summaries = []
            for chunk_idx in replay_chunks:
                if chunk_idx >= len(summary_store):
                    return recomposed_cache, False
                summary = summary_store[chunk_idx][block_id]
                if summary is None:
                    return recomposed_cache, False
                summaries.append(summary)
            first = summaries[0]
            B, H, D, dot_prec = first[5], first[6], first[7], first[8]
            total_f = sum(int(summary[0].shape[1]) for summary in summaries)
            key = (
                first[0].device,
                tuple(tensor.dtype for tensor in first[:5]),
                tuple(tuple(tensor.shape[2:]) for tensor in first[:4]),
                B,
                H,
                D,
                dot_prec,
                total_f,
            )
            groups.setdefault(key, []).append((block_id, summaries))
            num_summaries += len(summaries)

        collect_t0 = time.perf_counter() if profile else 0.0
        packed_groups = []
        for key, blocks in groups.items():
            device, _, _, B, H, D, dot_prec, total_f = key
            per_block = []
            for block_id, summaries in blocks:
                fields = [torch.cat([summary[i].to(device, non_blocking=True) for summary in summaries], dim=1) for i in range(4)]
                decay = torch.cat(
                    [summary[4].to(device, non_blocking=True).reshape(summary[0].shape[0], summary[0].shape[1]) for summary in summaries],
                    dim=1,
                )
                per_block.append((block_id, fields, decay))
            packed = [torch.cat([entry[1][i] for entry in per_block], dim=0) for i in range(4)]
            packed_decay = torch.cat([entry[2] for entry in per_block], dim=0)
            packed_groups.append((B, H, D, dot_prec, total_f, per_block, packed, packed_decay))
        if profile:
            profile_device = packed_groups[0][6][0].device
            _profile_sync(profile_device)
            collect_s = time.perf_counter() - collect_t0
        else:
            profile_device = None
            collect_s = 0.0

        phase_t0 = time.perf_counter() if profile else 0.0
        results = []
        for B, H, D, dot_prec, total_f, per_block, packed, packed_decay in packed_groups:
            _, _, _, _, final_kv, final_z = phase_b_triton(
                packed[0],
                packed[1],
                packed[2],
                packed[3],
                packed_decay,
                init_state_kv=None,
                init_state_z=None,
                F=total_f,
                dot_precision=dot_prec,
                direction=1,
                return_final_state=True,
            )
            results.append((B, H, D, per_block, final_kv, final_z))
        if profile:
            _profile_sync(profile_device)
            phase_s = time.perf_counter() - phase_t0
        else:
            phase_s = 0.0

        assign_t0 = time.perf_counter() if profile else 0.0
        for B, H, D, per_block, final_kv, final_z in results:
            block_batch = B * H
            for packed_index, (block_id, _, _) in enumerate(per_block):
                start = packed_index * block_batch
                end = start + block_batch
                block_kv = final_kv[start:end]
                block_z = final_z[start:end]
                block_d = block_kv.shape[-1]
                recomposed_cache[block_id][_SLOT_K] = (
                    block_kv.view(B, H, block_d, block_d)[:, :, :D, :D].transpose(-1, -2).contiguous()
                )
                recomposed_cache[block_id][_SLOT_V] = block_z.view(B, H, block_d)[:, :, :D].unsqueeze(-1).contiguous()
        if profile:
            _profile_sync(profile_device)
            assign_s = time.perf_counter() - assign_t0
            total_s = time.perf_counter() - profile_t0
            tqdm.write(
                "[Sana][recompose-profile] "
                f"backend=triton_batched chunks={len(replay_chunks)} blocks={len(block_ids)} "
                f"groups={len(groups)} summaries={num_summaries} total={total_s:.4f}s "
                f"collect={collect_s:.4f}s phase_b={phase_s:.4f}s assign={assign_s:.4f}s ok=1"
            )
        return recomposed_cache, True


    def _pack_selected_softmax_kv(
        self,
        replay_chunks: list[int],
        recomposed_cache: list,
    ) -> tuple[int, bool]:
        """Gather permanently cached per-chunk Softmax K/V for selection."""
        softmax_store = getattr(self, "_softmax_chunk_kv_cache", None)
        if softmax_store is None or not replay_chunks:
            return 0, False

        copied = 0
        for block_id in range(self.num_model_blocks):
            selected = []
            for chunk_idx in replay_chunks:
                if chunk_idx >= len(softmax_store):
                    return copied, False
                chunk_values = softmax_store[chunk_idx][block_id]
                if chunk_values is None:
                    selected = []
                    break
                selected.append(chunk_values)
            if not selected:
                continue
            for slot_id in _SOFTMAX_CONCAT_SLOTS:
                recomposed_cache[block_id][slot_id] = torch.cat(
                    [chunk_values[slot_id] for chunk_values in selected], dim=2
                ).contiguous()
            copied += 1
        return copied, copied > 0

    def _stash_gdn_chunk_summaries(self, kv_cache: list, chunk_idx: int) -> None:
        """Permanently keep all selection data outside the runtime window."""
        if os.environ.get("SANA_WM_GDN_STORE_CHUNK_SUMMARY", "").strip() != "1":
            return
        summary_store = getattr(self, "_gdn_chunk_summary_cache", None)
        if summary_store is None or chunk_idx >= len(summary_store):
            return
        for block_id in range(self.num_model_blocks):
            block_cache = kv_cache[chunk_idx][block_id]
            summary = block_cache[_SLOT_GDN_MAIN_SUMMARY]
            if summary is not None:
                summary_store[chunk_idx][block_id] = summary
            elif all(block_cache[slot_id] is not None for slot_id in _SOFTMAX_CONCAT_SLOTS):
                # Softmax save passes write current-chunk-only post-transform
                # K/V into slots 0..3.  Retain references permanently; later
                # eviction only clears the sliding-window list entries.
                self._softmax_chunk_kv_cache[chunk_idx][block_id] = tuple(
                    block_cache[slot_id].detach() for slot_id in _SOFTMAX_CONCAT_SLOTS
                )


    # ------------------------------------------------------------------
    # Sample (override for first-frame conditioning + distilled schedules)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        latents: torch.Tensor,
        steps: int = 50,
        generator: torch.Generator | None = None,
        *,
        denoising_step_list: list[int] | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        """Sample with first-frame conditioning support.

        Reads ``condition_frame_info`` from ``data_info`` (e.g. ``{0: 0.0}``
        means frame 0 is fully clean). For each conditioned frame that falls
        inside the current chunk, its latent is restored after every denoising
        step so the scheduler cannot corrupt it.

        Args:
            latents: Initial latent tensor of shape ``(B, C, T, H, W)``.
            steps: Number of denoising steps per chunk. Ignored when
                ``denoising_step_list`` is supplied.
            generator: Optional torch generator (currently unused; the
                scheduler is deterministic given the noise input).
            denoising_step_list: Optional explicit student timestep schedule
                (e.g. ``[1000, 967, 908, 764, 0]``). MUST end with 0. When
                provided, ``steps`` is ignored and the
                ``FlowMatchEulerDiscreteScheduler`` is set up with these
                exact sigmas (no shift re-applied — the schedule is taken
                verbatim, so it should already incorporate the teacher's
                ``flow_shift``). Use this for distilled students that were
                trained on a fixed subsampled subset of teacher timesteps.

        Returns:
            Denoised latent tensor with the same shape as ``latents``.
        """
        for _ in self.sample_chunks(
            latents,
            steps=steps,
            generator=generator,
            denoising_step_list=denoising_step_list,
            **kwargs,
        ):
            pass
        return latents

    @torch.no_grad()
    def sample_chunks(
        self,
        latents: torch.Tensor,
        steps: int = 50,
        generator: torch.Generator | None = None,
        *,
        denoising_step_list: list[int] | None = None,
        **kwargs: object,
    ):
        """Streaming variant of :meth:`sample` — yields one chunk at a time.

        After each AR chunk completes (denoising + KV-cache save pass), yields
        a tuple ``(chunk_idx, latent_chunk_view, start_f, end_f)`` where
        ``latent_chunk_view`` is a *view* into the in-place-mutated ``latents``
        tensor for the just-finished chunk. The view stays valid for the
        remainder of inference (subsequent chunks never overwrite earlier
        frames), so the orchestrator may launch downstream work on a separate
        CUDA stream and continue pulling chunks without copying.

        ``sample(latents, ...)`` is implemented as ``for _ in sample_chunks(...)``
        and returns ``latents`` after exhaustion, so the legacy whole-volume
        API is preserved.
        """
        # Resolve scheduler factory once (a fresh instance is built per chunk).
        if denoising_step_list is not None:
            if len(denoising_step_list) < 2 or denoising_step_list[-1] != 0:
                raise ValueError(
                    "denoising_step_list must have >=2 entries and end with 0; " f"got {denoising_step_list}"
                )
            # Drop trailing 0; FlowMatchEulerDiscreteScheduler auto-appends sigma=0.
            # ``shift=1.0`` keeps our explicit sigmas verbatim (no second shift).
            _explicit_sigmas = [float(t) / 1000.0 for t in denoising_step_list[:-1]]
        else:
            _explicit_sigmas = None

        device = self.condition.device
        do_classifier_free_guidance = self.cfg_scale > 1
        batch_size, num_latent_channels, total_frames, height, width = latents.shape
        self.total_frames = total_frames

        if total_frames <= self.base_chunk_frames:
            raise ValueError("Please use FlowEuler for short videos")

        chunk_indices = self.create_autoregressive_segments(total_frames)
        self._chunk_indices = chunk_indices
        num_chunks = len(chunk_indices) - 1
        self._prepare_replay_overlap_from_camera(num_chunks)
        kv_cache = self._initialize_kv_cache(num_chunks)
        self._gdn_chunk_summary_cache = [[None] * self.num_model_blocks for _ in range(num_chunks)]
        self._softmax_chunk_kv_cache = [[None] * self.num_model_blocks for _ in range(num_chunks)]
        # A model block's attention implementation is fixed for the whole
        # sample. Resolve its tensor type flag once instead of calling
        # ``Tensor.item()`` (a device synchronization) on every chunk.
        self._cache_block_is_gdn = [None] * self.num_model_blocks
        self._historical_replay_forward_count = 0
        replay_keep_last = int(os.environ.get("SANA_WM_GDN_STORE_CHUNK_SUMMARY", "") == "1")
        replay_sink_chunks = 1
        replay_keep_last_overrides_raw = os.environ.get(
            "SANA_WM_GDN_REPLAY_KEEP_LAST_CHUNKS_OVERRIDES_JSON", ""
        ).strip()
        replay_keep_last_overrides: dict[int, int] = {}
        if replay_keep_last_overrides_raw:
            replay_keep_last_overrides = {
                int(chunk_idx): int(value)
                for chunk_idx, value in json.loads(replay_keep_last_overrides_raw).items()
            }
            invalid_overrides = {
                chunk_idx: value
                for chunk_idx, value in replay_keep_last_overrides.items()
                if chunk_idx < 0 or chunk_idx >= num_chunks or value != 3
            }
            if invalid_overrides:
                raise ValueError(
                    "SANA_WM_GDN_REPLAY_KEEP_LAST_CHUNKS_OVERRIDES_JSON contains "
                    f"invalid chunk/value pairs: {invalid_overrides}"
                )
        replay_logged = False

        assert self.condition.shape[0] == batch_size or self.condition.shape[0] == num_chunks
        if self.condition.shape[0] == batch_size:
            self.condition = self.condition.repeat_interleave(num_chunks, dim=0)
            self.mask = self.mask[None].repeat_interleave(num_chunks, dim=0) if self.mask is not None else None

        # -- First-frame conditioning --
        data_info = self.model_kwargs.pop("data_info", {})
        condition_frame_info = data_info.pop("condition_frame_info", {})
        # Save a clean copy of conditioned frame latents.
        init_latents = latents.clone()
        image_vae_embeds = data_info.get("image_vae_embeds", None)

        # Build the scheduler once (sigmas / shift don't change per chunk).
        if _explicit_sigmas is not None:
            _shared_scheduler = FlowMatchEulerDiscreteScheduler(shift=1.0)
            _shared_scheduler.set_timesteps(sigmas=_explicit_sigmas, device=device)
            _shared_timesteps = _shared_scheduler.timesteps
            _shared_num_steps = len(_shared_timesteps)
        else:
            _shared_scheduler = FlowMatchEulerDiscreteScheduler(shift=self.flow_shift)
            _shared_timesteps, _shared_num_steps = retrieve_timesteps(_shared_scheduler, steps, device, None)

        for chunk_idx in range(num_chunks):
            chunk_replay_keep_last = replay_keep_last_overrides.get(chunk_idx, replay_keep_last)
            if chunk_idx in replay_keep_last_overrides:
                tqdm.write(
                    "[Sana] per-chunk recent override: "
                    f"chunk={chunk_idx} keep_last={chunk_replay_keep_last} "
                    f"(default={replay_keep_last})"
                )
            # Selected K/V replaces the runtime window after retrieval.
            selected_softmax_will_replace_baseline = (
                chunk_replay_keep_last > 0
                and chunk_idx > replay_sink_chunks + chunk_replay_keep_last
            )
            (
                chunk_kv_cache,
                num_chunks_to_accumulate,
                sink_num,
                num_cached_frames,
            ) = self.accumulate_kv_cache(
                kv_cache,
                chunk_idx,
                skip_softmax_concat=selected_softmax_will_replace_baseline,
            )
            prompt_embeds = self.condition[chunk_idx].unsqueeze(0)
            if do_classifier_free_guidance:
                prompt_embeds = torch.cat([self.uncondition, prompt_embeds], dim=0)

            mask = self.mask[chunk_idx] if self.mask is not None else None

            # Reuse the scheduler built outside the chunk loop; re-set
            # timesteps to reset its internal ``_step_index`` to zero.
            self.scheduler = _shared_scheduler
            if _explicit_sigmas is not None:
                self.scheduler.set_timesteps(sigmas=_explicit_sigmas, device=device)
                timesteps = self.scheduler.timesteps
                num_inference_steps = _shared_num_steps
            else:
                timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, steps, device, None)

            start_f = chunk_indices[chunk_idx]
            end_f = chunk_indices[chunk_idx + 1]
            end_f - start_f
            max(chunk_idx - self.num_cached_blocks, 0) if self.num_cached_blocks > 0 else 0

            # Build frame_index for the current chunk's RoPE positions.
            #
            # In camctrl, ``CachedChunkCausalSoftmaxAttn`` caches POST-RoPE
            # K/V (each chunk keeps its own absolute positions). When attending
            # over [sink + window + current], cached tokens already have the
            # correct rope baked in — only the CURRENT chunk's Q/K need
            # rotation, which uses positions ``[start_f, end_f)``. So
            # frame_index here describes only the current chunk; sink_token
            # affects cache contents but not the per-chunk rope shape.
            #
            frame_index: torch.Tensor | None = None
            rope_start_f = start_f
            rope_end_f = end_f
            if sink_num > 0:
                frame_index = torch.arange(start_f, end_f, device=device, dtype=torch.long)

            # Shallow copy — we only need to override image_vae_embeds per chunk.
            local_data_info = self._local_data_info_for_chunk(data_info, image_vae_embeds, start_f, end_f)

            if chunk_replay_keep_last > 0 and chunk_idx > replay_sink_chunks + chunk_replay_keep_last:
                replay_chunks = self._select_replay_chunks(
                    chunk_idx, chunk_replay_keep_last, replay_sink_chunks
                )
                if not replay_logged:
                    tqdm.write(
                        "[Sana] recent_chunks="
                        f"{chunk_replay_keep_last}, sink_chunks={replay_sink_chunks}: "
                        "building cache from sink + recent + retrieved chunk"
                    )
                    replay_logged = True
                if getattr(self, "_replay_log_selection", False):
                    tqdm.write(f"[Sana] replay chunk {chunk_idx}: selected past chunks {replay_chunks}")
                replayed_chunk_kv_cache, summary_ok = self._recompose_main_gdn_cache_from_summaries_batched(
                    replay_chunks, chunk_kv_cache,
                )
                if not summary_ok:
                    raise RuntimeError(f"Missing GDN summaries for selected chunks {replay_chunks}")
                _, softmax_ok = self._pack_selected_softmax_kv(replay_chunks, replayed_chunk_kv_cache)
                if not softmax_ok:
                    raise RuntimeError(f"Missing Softmax K/V for selected chunks {replay_chunks}")
                chunk_kv_cache = replayed_chunk_kv_cache
            # Identify conditioned frames inside this chunk (local indices).
            chunk_frames = end_f - start_f
            cond_local_indices: list[int] = []
            for frame_idx in condition_frame_info:
                if start_f <= frame_idx < end_f:
                    cond_local_indices.append(frame_idx - start_f)

            # Build a per-frame mask instead of a full (B,C,F,H,W) tensor.
            # The model consumes frame-level timesteps and the scheduler uses
            # the same frame value broadcast over spatial tokens.
            condition_frame_mask = None
            if cond_local_indices:
                condition_frame_mask = torch.zeros(
                    batch_size,
                    chunk_frames,
                    device=device,
                    dtype=torch.float32,
                )
                for loc in cond_local_indices:
                    condition_frame_mask[:, loc] = 1.0
            spatial_tokens = height * width

            for i, t in tqdm(
                list(enumerate(timesteps)),
                disable=os.getenv("DPM_TQDM", "False") == "True",
                desc=f"Processing chunk {chunk_idx}",
            ):
                latent_model_input = (
                    torch.cat([latents[:, :, start_f:end_f]] * 2)
                    if do_classifier_free_guidance
                    else latents[:, :, start_f:end_f]
                )

                # Keep the timestep on device without `.item()` syncs, but
                # avoid materialising a full channel x spatial mask.
                t_dev = t.to(device=device, dtype=torch.float32).reshape(1)
                if condition_frame_mask is None:
                    timestep_frames = t_dev.expand(batch_size, chunk_frames)
                    per_token_timesteps = t_dev.expand(batch_size, chunk_frames * spatial_tokens)
                else:
                    timestep_frames = (1.0 - condition_frame_mask) * t_dev
                    per_token_timesteps = (
                        timestep_frames[:, :, None]
                        .expand(
                            batch_size,
                            chunk_frames,
                            spatial_tokens,
                        )
                        .reshape(batch_size, -1)
                    )

                timestep_tensor_model = timestep_frames[:, None, :]
                if do_classifier_free_guidance:
                    timestep_tensor_model = torch.cat([timestep_tensor_model, timestep_tensor_model], dim=0)

                noise_pred, _ = self.model(
                    latent_model_input,
                    timestep_tensor_model,
                    prompt_embeds,
                    start_f=rope_start_f,
                    end_f=rope_end_f,
                    frame_index=frame_index,
                    save_kv_cache=False,
                    kv_cache=chunk_kv_cache,
                    mask=mask,
                    data_info=local_data_info,
                    **self.model_kwargs,
                )
                if isinstance(noise_pred, Transformer2DModelOutput):
                    noise_pred = noise_pred[0]
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.cfg_scale * (noise_pred_text - noise_pred_uncond)

                # Per-token scheduler step with ``per_token_timesteps`` (same
                # reshape convention as ChunkFlowEuler).
                latents_dtype = latents.dtype
                chunk_latents_cur = latents[:, :, start_f:end_f]
                chunk_shape = chunk_latents_cur.shape

                denoised = self.scheduler.step(
                    -noise_pred.reshape(
                        batch_size,
                        num_latent_channels,
                        -1,
                    ).transpose(1, 2),
                    t,
                    chunk_latents_cur.reshape(
                        batch_size,
                        num_latent_channels,
                        -1,
                    ).transpose(1, 2),
                    per_token_timesteps=per_token_timesteps,
                    return_dict=False,
                )[0]
                denoised = denoised.transpose(1, 2).reshape(chunk_shape)
                latents[:, :, start_f:end_f] = denoised

                # Safety: explicitly restore conditioned frames in case of
                # numerical drift from the per-token scheduler step.
                for loc in cond_local_indices:
                    latents[:, :, start_f + loc] = init_latents[:, :, start_f + loc]

                if latents.dtype != latents_dtype:
                    latents = latents.to(latents_dtype)

            # Save clean-latent cache and Phase-A summaries once per chunk.
            latent_model_input = (
                torch.cat([latents[:, :, start_f:end_f]] * 2)
                if do_classifier_free_guidance
                else latents[:, :, start_f:end_f]
            )
            timestep = torch.zeros(latent_model_input.shape[0], device=device)

            noise_pred, updated_kv_cache = self.model(
                latent_model_input,
                timestep,
                prompt_embeds,
                start_f=rope_start_f,
                end_f=rope_end_f,
                frame_index=frame_index,
                save_kv_cache=True,
                kv_cache=chunk_kv_cache,
                mask=mask,
                data_info=local_data_info,
                **self.model_kwargs,
            )
            kv_cache[chunk_idx] = updated_kv_cache
            self._stash_gdn_chunk_summaries(kv_cache, chunk_idx)

            yield chunk_idx, latents[:, :, start_f:end_f], start_f, end_f

        assert self._historical_replay_forward_count == 0, (
            "historical_replay_forward_count must remain zero; historical model forwards are forbidden"
        )
        if replay_logged:
            tqdm.write("[Sana][historical-replay-audit] historical_replay_forward_count=0")

    # ------------------------------------------------------------------
    # KV cache management
    # ------------------------------------------------------------------

    def _initialize_kv_cache(self, num_chunks: int) -> list[list[list[torch.Tensor | None]]]:
        """Create empty 10-slot cache: ``kv_cache[chunk][block] = [None]*10``."""
        return [[[None] * _NUM_CACHE_SLOTS for _ in range(self.num_model_blocks)] for _ in range(num_chunks)]

    def accumulate_kv_cache(
        self,
        kv_cache: list,
        chunk_idx: int,
        *,
        skip_softmax_concat: bool = False,
    ):
        """Override parent dispatcher to always use 10-slot cache logic."""
        return self._accumulate_softmax_kv_cache(
            kv_cache,
            chunk_idx,
            skip_softmax_concat=skip_softmax_concat,
        )

    def _accumulate_softmax_kv_cache(
        self,
        kv_cache: list,
        chunk_idx: int,
        *,
        skip_softmax_concat: bool = False,
    ) -> tuple[list, int, int, int]:
        """Accumulate KV cache for chunk ``chunk_idx``.

        Two cache layouts are supported, distinguished by slot 6 (type flag):

        * **State-based** (type flag == 1.0, GDN blocks): slots 0-3 hold
          recurrent states from the last chunk — no concatenation needed.
          ``sink_token`` has no effect here: state already encodes full
          history.
        * **Concat-based** (type flag == 0.0, Softmax blocks): slots 0-3 hold
          K/V tensors concatenated across cached chunks. When
          ``self.sink_token`` is set and the sliding window has scrolled past
          chunk 0, chunk 0 is always retained at the front (sink anchor).
          If ``skip_softmax_concat`` is true, slots 0-3 are left empty for a
          selected-history pack later in the same chunk iteration.
        * **Legacy** (type flag absent): falls back to the old concat logic
          using beta/decay detection. Sink behavior mirrors the softmax path.

        Slot 4 (shortconv) always comes from the preceding chunk.
        Slot 9 (``kv_cache[-1]``) holds the FFN tconv state written by
        ``CachedGLUMBConvTemp`` and comes from the preceding chunk.

        Returns:
            ``(cur_kv_cache, num_chunks_accumulated, sink_num,
            num_cached_frames)``.
        """
        if chunk_idx == 0:
            return kv_cache[0], 0, 0, 0

        cur_kv_cache = kv_cache[chunk_idx]
        block_is_gdn = getattr(self, "_cache_block_is_gdn", None)
        if block_is_gdn is None or len(block_is_gdn) != self.num_model_blocks:
            block_is_gdn = [None] * self.num_model_blocks
            self._cache_block_is_gdn = block_is_gdn
        # Clamp to >= 0: when ``chunk_idx < num_cached_blocks`` the window has
        # not yet slid past chunk 0, so the effective start is 0.
        start_chunk_idx = max(chunk_idx - self.num_cached_blocks, 0) if self.num_cached_blocks > 0 else 0

        # Sink-aware iteration order: when ``num_cached_blocks`` slid past
        # chunk 0, prepend chunk 0 as a permanent anchor.
        sink_num = 0
        valid_cached_chunks = list(range(start_chunk_idx, chunk_idx))
        if self.sink_token and self.num_cached_blocks > 0:
            s = max(chunk_idx - self.num_cached_blocks + 1, 0)
            if s > 0:
                valid_cached_chunks = [0] + list(range(s, chunk_idx))
                sink_num = self._chunk_indices[1] - self._chunk_indices[0]

        valid_cached_chunks = [
            i
            for i in valid_cached_chunks
            if kv_cache[i][0][_SLOT_K] is not None or kv_cache[i][0][_SLOT_TYPE_FLAG] is not None
        ]

        # Count cached frames in latent units (independent of patch_size). The
        # sampler builds frame_index in latent units so this stays consistent.
        num_cached_frames = sum(self._chunk_indices[i + 1] - self._chunk_indices[i] for i in valid_cached_chunks)
        prev_cache_idx = valid_cached_chunks[-1] if valid_cached_chunks else chunk_idx

        for block_id in range(self.num_model_blocks):
            prev_last = kv_cache[prev_cache_idx][block_id]

            # Detect cache layout from type flag (slot 6).
            type_flag = prev_last[_SLOT_TYPE_FLAG] if prev_last[_SLOT_TYPE_FLAG] is not None else None
            type_flag_value = None
            if type_flag is not None:
                cached_is_gdn = block_is_gdn[block_id]
                if cached_is_gdn is None:
                    type_flag_value = (
                        float(type_flag.item()) if isinstance(type_flag, torch.Tensor) else float(type_flag)
                    )
                    block_is_gdn[block_id] = type_flag_value > 0.5
                else:
                    type_flag_value = 1.0 if cached_is_gdn else 0.0

            if type_flag_value is not None and type_flag_value > 0.5:
                # --- State-based (GDN): last chunk's state is the full history ---
                # NOTE: ``CachedGLUMBConvTemp`` writes tconv state to
                # ``kv_cache[-1]`` (slot 9), not ``_SLOT_TCONV`` (slot 5). We
                # must read from [-1] and place into [-1] so the MLP finds it
                # on the next chunk.
                cur_kv_cache[block_id] = [
                    prev_last[0],  # S_kv state (or accumulated softmax k)
                    prev_last[1],  # S_z state (or accumulated softmax v)
                    prev_last[2],  # cam_S_kv state
                    prev_last[3],  # camera K ShortConv state
                    prev_last[_SLOT_SHORTCONV],  # ShortConv state
                    None,  # (slot 5 unused)
                    prev_last[_SLOT_TYPE_FLAG],  # type flag
                    None,
                    None,
                    prev_last[-1],  # FFN tconv state (slot 9)
                ]

            elif type_flag_value is not None:
                # --- Concat-based (Softmax): concatenate K/V across chunks ---
                acc: list[torch.Tensor | None] = [None] * _NUM_CACHE_SLOTS

                if not skip_softmax_concat:
                    # Collect first, then concatenate each slot once.  The
                    # previous incremental cat copied the growing prefix for
                    # every cached chunk (quadratic bytes moved).
                    values_by_slot: dict[int, list[torch.Tensor]] = {
                        slot_id: [] for slot_id in _SOFTMAX_CONCAT_SLOTS
                    }
                    for i in valid_cached_chunks:
                        prev = kv_cache[i][block_id]
                        if prev[0] is None:
                            continue
                        for slot_id in _SOFTMAX_CONCAT_SLOTS:
                            if prev[slot_id] is not None:
                                values_by_slot[slot_id].append(prev[slot_id])

                    for slot_id, values in values_by_slot.items():
                        if len(values) == 1:
                            acc[slot_id] = values[0]
                        elif values:
                            # Softmax K/V are (B, H, N, D) — concat on N.
                            acc[slot_id] = torch.cat(values, dim=2)

                cur_kv_cache[block_id] = [
                    acc[0],  # accumulated k
                    acc[1],  # accumulated v
                    acc[2],  # accumulated cam_k
                    acc[3],  # accumulated cam_v
                    prev_last[_SLOT_SHORTCONV],  # ShortConv state (last chunk)
                    None,  # (slot 5 unused)
                    prev_last[_SLOT_TYPE_FLAG],  # type flag
                    None,
                    None,
                    prev_last[-1],  # FFN tconv state (slot 9)
                ]

            else:
                # --- Legacy layout (no type flag): old concat logic ---
                acc_legacy: list[torch.Tensor | None] = [None] * _NUM_CACHE_SLOTS

                for i in valid_cached_chunks:
                    prev = kv_cache[i][block_id]
                    if prev[_SLOT_K] is None:
                        continue

                    is_gdn = prev[_SLOT_BETA] is not None
                    kv_cat_dim = -1 if is_gdn else 2

                    for s in _CONCAT_SLOTS:
                        if prev[s] is None:
                            continue
                        if s in (_SLOT_K, _SLOT_V):
                            cat_dim = kv_cat_dim
                        elif s == _SLOT_BETA:
                            cat_dim = 2
                        else:
                            cat_dim = -1

                        if acc_legacy[s] is None:
                            acc_legacy[s] = prev[s]
                        else:
                            acc_legacy[s] = torch.cat([acc_legacy[s], prev[s]], dim=cat_dim)

                cur_kv_cache[block_id] = [
                    acc_legacy[_SLOT_K],
                    acc_legacy[_SLOT_V],
                    acc_legacy[_SLOT_BETA],
                    acc_legacy[_SLOT_DECAY],
                    prev_last[_SLOT_SHORTCONV],
                    None,  # (slot 5 unused)
                    None,
                    None,
                    None,
                    prev_last[-1],  # FFN tconv state (slot 9)
                ]

            # Evict cached chunks outside the (possibly sink-augmented) window.
            if self.num_cached_blocks > 0:
                kept = set(valid_cached_chunks)
                for i in range(chunk_idx):
                    if i not in kept:
                        kv_cache[i][block_id] = [None] * _NUM_CACHE_SLOTS

        return (
            cur_kv_cache,
            chunk_idx - start_chunk_idx,
            sink_num,
            num_cached_frames,
        )
