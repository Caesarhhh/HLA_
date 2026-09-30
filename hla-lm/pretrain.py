# Modified by Songlin Yang & Ali Hatamizadeh
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# Copyright Lightning AI. Licensed under the Apache License 2.0,
# see LICENSE file at https://github.com/Lightning-AI/litgpt/blob/main/LICENSE
import glob
import math
import sys
import time
from pathlib import Path
from typing import Optional, Tuple
import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy
from torch.utils.data import DataLoader
from functools import partial

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
from lit_gpt.model import GPT, Block, Config
from lit_gpt.packed_dataset import CombinedDataset, PackedDataset
from lit_gpt.speed_monitor import SpeedMonitorFabric as Monitor
from lit_gpt.utils import chunked_cross_entropy, num_parameters
from pytorch_lightning.loggers import WandbLogger
from lit_gpt import FusedCrossEntropyLoss
import random
import os
import argparse
import torch.multiprocessing as mp

os.environ["TRITON_CACHE_MANAGER"] = "cache:ParallelFileCacheManager"


def set_trainable_parameters(model: torch.nn.Module, mode: str) -> dict:
    for param in model.parameters():
        param.requires_grad_(True)
    return {
        "mode": mode,
        "trainable_tensors": sum((1 for _ in model.parameters())),
        "trainable_params": sum((p.numel() for p in model.parameters())),
        "sample": [],
    }


def set_gla_router_logit_scale(model: torch.nn.Module, scale: float) -> int:
    changed = 0
    for module in model.modules():
        if getattr(module, "route_q_proj", None) is not None and hasattr(
            module, "_mix_weights"
        ):
            module.router_logit_scale = float(scale)
            changed += 1
    return changed


def set_gla_router_current_always_on(model: torch.nn.Module, enabled: bool) -> int:
    changed = 0
    for module in model.modules():
        if getattr(module, "route_q_proj", None) is not None and hasattr(
            module, "_mix_weights"
        ):
            module.router_current_always_on = bool(enabled)
            changed += 1
    return changed


def set_gla_router_sigmoid_temperature(
    model: torch.nn.Module, temperature: float
) -> int:
    changed = 0
    for module in model.modules():
        if getattr(module, "route_q_proj", None) is not None and hasattr(
            module, "_mix_weights"
        ):
            module.router_sigmoid_temperature = float(temperature)
            changed += 1
    return changed


@torch.compiler.disable
def _collect_gla_router_stats_eager(
    self, q, k, num_tokens, num_chunks, weights
) -> None:
    """Keep scalar logging values outside Dynamo's tensor graphs and guards.

    Only called on statistics steps. The routing computation stays compilable;
    all Tensor.item() conversions and dictionary/list mutations happen eagerly.
    """
    with torch.no_grad():
        q_route, k_route = self._route_qk(q[:, :, :num_tokens], k[:, :, :num_tokens])
        chunk_size = 256
        pool_size = 16
        buckets = chunk_size // pool_size
        pad_len = num_chunks * chunk_size - num_tokens
        k_padded = self._pad_tokens(k_route, pad_len)
        k_chunk = k_padded.reshape(
            k_padded.shape[0],
            k_padded.shape[1],
            num_chunks,
            chunk_size,
            k_padded.shape[-1],
        )
        pooled = k_chunk.reshape(
            k_chunk.shape[0],
            k_chunk.shape[1],
            num_chunks,
            buckets,
            pool_size,
            k_chunk.shape[-1],
        ).mean(dim=4)
        router_scale = q_route.shape[-1] ** (-0.5) * 1.0
        bucket_logits = (
            torch.einsum(
                "b h l d, b h c p d -> b h l c p", q_route.float(), pooled.float()
            )
            * router_scale
        )
        scores = torch.logsumexp(bucket_logits, dim=-1) - math.log(buckets)
        token_chunk_idx = torch.div(
            torch.arange(num_tokens, device=q.device), chunk_size, rounding_mode="floor"
        )
        chunk_idx = torch.arange(num_chunks, device=q.device)
        causal_chunk_mask = chunk_idx.view(1, num_chunks) <= token_chunk_idx.view(
            num_tokens, 1
        )
        current_idx = token_chunk_idx.view(1, 1, num_tokens, 1).expand(
            scores.shape[0], scores.shape[1], num_tokens, 1
        )
        current_scores = self._current_chunk_scores(
            q_route, k_padded, pooled, num_tokens, num_chunks
        )
        scores = scores.masked_fill(
            ~causal_chunk_mask.view(1, 1, num_tokens, num_chunks), -1e309
        )
        scores = scores.scatter(
            3, current_idx, current_scores.unsqueeze(-1).to(scores.dtype)
        )
        visible_scores = scores.masked_select(
            causal_chunk_mask.view(1, 1, num_tokens, num_chunks).expand_as(scores)
        )
        visible = (token_chunk_idx + 1).float().view(1, 1, num_tokens)
        visible_mask = causal_chunk_mask.view(1, 1, num_tokens, num_chunks).expand_as(
            weights
        )
        raw_weights = weights.detach().float().masked_fill(~visible_mask, 0.0)
        gate_sum = raw_weights.sum(dim=-1)
        gate_mean = gate_sum / visible.clamp_min(1.0)
        current_gate = raw_weights.gather(
            -1,
            token_chunk_idx.view(1, 1, raw_weights.shape[2], 1).expand(
                raw_weights.shape[0], raw_weights.shape[1], raw_weights.shape[2], 1
            ),
        ).squeeze(-1)
        max_gate = raw_weights.masked_fill(~visible_mask, -1e309).max(dim=-1).values
        probs = raw_weights / gate_sum.unsqueeze(-1).clamp_min(1e-30)
        uniform_probs = torch.zeros_like(probs)
        uniform_probs = uniform_probs.masked_fill(visible_mask, 1.0)
        uniform_probs = uniform_probs / visible.unsqueeze(-1)
        max_prob = probs.max(dim=-1).values
        entropy = -(probs.clamp_min(1e-30) * probs.clamp_min(1e-30).log()).sum(dim=-1)
        norm_entropy = entropy / visible.log().clamp_min(1e-06)
        norm_entropy = norm_entropy.masked_fill(visible <= 1, 1e309 - 1e309)
        max_prob_dev = max_prob - visible.reciprocal()
        max_prob_dev = max_prob_dev.masked_fill(visible <= 1, -1e309)
        token_uniform_dev = (probs - uniform_probs).abs().amax(dim=-1)
        current_mass = probs.gather(
            -1,
            token_chunk_idx.view(1, 1, probs.shape[2], 1).expand(
                probs.shape[0], probs.shape[1], probs.shape[2], 1
            ),
        ).squeeze(-1)
        last_probs = probs[:, :, -1, :]
        last_uniform = 1.0 / last_probs.shape[-1]
        self._router_stats_buffer.append(
            {
                "module": getattr(self, "_router_stats_name", ""),
                "router_gate": "affine_sigmoid",
                "avg_gate": float(gate_mean.mean().item()),
                "avg_gate_sum": float(gate_sum.mean().item()),
                "avg_current_gate": float(current_gate.mean().item()),
                "avg_max_gate": float(max_gate.mean().item()),
                "last_token_gate_sum": float(gate_sum[:, :, -1].mean().item()),
                "last_token_current_gate": float(current_gate[:, :, -1].mean().item()),
                "avg_max_prob": float(max_prob.mean().item()),
                "avg_current_chunk_mass": float(current_mass.mean().item()),
                "avg_norm_entropy_visible_gt1": float(
                    torch.nanmean(norm_entropy).item()
                ),
                "min_norm_entropy_visible_gt1": float(
                    torch.nan_to_num(norm_entropy, nan=1e309).min().item()
                ),
                "max_token_max_prob_dev": float(max_prob_dev.max().item()),
                "max_token_uniform_dev": float(token_uniform_dev.max().item()),
                "last_token_max_uniform_dev": float(
                    (last_probs - last_uniform).abs().max().item()
                ),
                "q_route_std": float(q_route.float().std().item()),
                "k_route_std": float(k_route.float().std().item()),
                "score_std": float(visible_scores.float().std().item()),
                "score_range": float(
                    (visible_scores.float().max() - visible_scores.float().min()).item()
                ),
            }
        )


def attach_gla_router_stats(model: torch.nn.Module) -> int:
    attached = 0
    for name, module in model.named_modules():
        if getattr(module, "route_q_proj", None) is None or not hasattr(
            module, "_mix_weights"
        ):
            continue
        if hasattr(module, "_router_stats_original_mix_weights"):
            continue
        module._router_stats_name = name
        module._router_stats_active = False
        module._router_stats_buffer = []
        module._router_stats_original_mix_weights = module._mix_weights
        original = module._mix_weights

        def wrapped_mix_weights(
            self, q, k, num_tokens, num_chunks, *, original=original
        ):
            weights = original(q, k, num_tokens, num_chunks)
            if getattr(self, "_router_stats_active", False):
                _collect_gla_router_stats_eager(
                    self, q, k, num_tokens, num_chunks, weights
                )
            return weights

        module._mix_weights = wrapped_mix_weights.__get__(module, module.__class__)
        attached += 1
    return attached


def set_gla_router_stats_active(model: torch.nn.Module, active: bool) -> None:
    for module in model.modules():
        if hasattr(module, "_router_stats_buffer"):
            module._router_stats_active = active
            if active:
                module._router_stats_buffer.clear()


def collect_gla_router_stats(model: torch.nn.Module) -> list[dict]:
    rows = []
    for module in model.modules():
        if hasattr(module, "_router_stats_buffer"):
            rows.extend(module._router_stats_buffer)
            module._router_stats_buffer.clear()
    return rows


def attach_gla_router_gate_budget_loss(
    model: torch.nn.Module, target: float, exclude_current: bool = False
) -> int:
    attached = 0
    for module in model.modules():
        if getattr(module, "route_q_proj", None) is None or not hasattr(
            module, "_mix_weights"
        ):
            continue
        if hasattr(module, "_router_gate_budget_original_mix_weights"):
            continue
        module._router_gate_budget_active = False
        module._router_gate_budget_buffer = []
        module._router_gate_budget_target = float(target)
        module._router_gate_budget_exclude_current = bool(exclude_current)
        module._router_gate_budget_original_mix_weights = module._mix_weights
        original = module._mix_weights

        def wrapped_mix_weights(
            self, q, k, num_tokens, num_chunks, *, original=original
        ):
            weights = original(q, k, num_tokens, num_chunks)
            if getattr(self, "_router_gate_budget_active", False):
                chunk_size = 256
                token_chunk_idx = torch.div(
                    torch.arange(num_tokens, device=weights.device),
                    chunk_size,
                    rounding_mode="floor",
                )
                chunk_idx = torch.arange(num_chunks, device=weights.device)
                visible_mask = chunk_idx.view(1, num_chunks) <= token_chunk_idx.view(
                    num_tokens, 1
                )
                gate_sum = (
                    weights.float()
                    .masked_fill(~visible_mask.view(1, 1, num_tokens, num_chunks), 0.0)
                    .sum(dim=-1)
                )
                current_idx = token_chunk_idx.view(1, 1, num_tokens, 1).expand(
                    weights.shape[0], weights.shape[1], num_tokens, 1
                )
                current_gate = weights.float().gather(-1, current_idx).squeeze(-1)
                gate_sum = (gate_sum - current_gate).clamp_min(0.0)
                target = float(getattr(self, "_router_gate_budget_target", 0.0))
                penalty = (torch.relu(gate_sum - target) / max(target, 1.0)).square()
                self._router_gate_budget_buffer.append(penalty)
            return weights

        module._mix_weights = wrapped_mix_weights.__get__(module, module.__class__)
        attached += 1
    return attached


def set_gla_router_gate_budget_active(model: torch.nn.Module, active: bool) -> None:
    for module in model.modules():
        if hasattr(module, "_router_gate_budget_buffer"):
            module._router_gate_budget_active = active
            if active:
                module._router_gate_budget_buffer.clear()


def collect_gla_router_gate_budget_loss(
    model: torch.nn.Module,
) -> Optional[torch.Tensor]:
    losses = []
    for module in model.modules():
        if hasattr(module, "_router_gate_budget_buffer"):
            for penalty in module._router_gate_budget_buffer:
                losses.append(penalty.mean())
            module._router_gate_budget_buffer.clear()
    if not losses:
        return None
    return torch.stack(losses).mean()


def attach_gla_router_gate_sparse_loss(model: torch.nn.Module, target: float) -> int:
    """Capture the HLA effective-support and binary gate objectives."""
    attached = 0
    for module in model.modules():
        if getattr(module, "route_q_proj", None) is None or not hasattr(
            module, "_mix_weights"
        ):
            continue
        if hasattr(module, "_router_gate_sparse_original_mix_weights"):
            continue
        module._router_gate_sparse_active = False
        module._router_gate_sparse_buffer = []
        module._router_gate_binary_buffer = []
        module._router_gate_sparse_target = float(target)
        module._router_gate_sparse_original_mix_weights = module._mix_weights
        original = module._mix_weights

        def wrapped_mix_weights(
            self, q, k, num_tokens, num_chunks, *, original=original
        ):
            weights = original(q, k, num_tokens, num_chunks)
            if getattr(self, "_router_gate_sparse_active", False):
                chunk_size = 256
                token_chunk_idx = torch.div(
                    torch.arange(num_tokens, device=weights.device),
                    chunk_size,
                    rounding_mode="floor",
                )
                chunk_idx = torch.arange(num_chunks, device=weights.device)
                historical = chunk_idx.view(1, num_chunks) < token_chunk_idx.view(
                    num_tokens, 1
                )
                historical_4d = historical.view(1, 1, num_tokens, num_chunks)
                weights_float = weights.float()
                raw_historical_gates = weights_float.masked_fill(~historical_4d, 0.0)
                historical_gates = torch.where(
                    historical_4d,
                    weights_float.clamp_min(1e-08),
                    torch.zeros((), device=weights.device, dtype=torch.float32),
                )
                historical_mass = historical_gates.sum(dim=-1)
                historical_probability = historical_gates / historical_mass.clamp_min(
                    1e-08
                ).unsqueeze(-1)
                effective_support = (
                    historical_probability.square()
                    .sum(dim=-1)
                    .clamp_min(1e-08)
                    .reciprocal()
                )
                target = float(getattr(self, "_router_gate_sparse_target", 2.0))
                valid_sparse = token_chunk_idx.view(1, 1, num_tokens) > target
                sparse_penalty = torch.relu(effective_support - target).square()
                sparse_loss = sparse_penalty.masked_select(
                    valid_sparse.expand_as(sparse_penalty)
                ).mean()
                self._router_gate_sparse_buffer.append(sparse_loss)
                historical_entries = historical_4d.expand_as(weights)
                binary_loss = (
                    (raw_historical_gates * (1.0 - raw_historical_gates))
                    .masked_select(historical_entries)
                    .mean()
                )
                self._router_gate_binary_buffer.append(binary_loss)
            return weights

        module._mix_weights = wrapped_mix_weights.__get__(module, module.__class__)
        attached += 1
    return attached


def set_gla_router_gate_sparse_active(model: torch.nn.Module, active: bool) -> None:
    for module in model.modules():
        if hasattr(module, "_router_gate_sparse_buffer"):
            module._router_gate_sparse_active = active
            if active:
                module._router_gate_sparse_buffer.clear()
                module._router_gate_binary_buffer.clear()


def collect_gla_router_gate_sparse_losses(
    model: torch.nn.Module,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    sparse_losses = []
    binary_losses = []
    for module in model.modules():
        if hasattr(module, "_router_gate_sparse_buffer"):
            sparse_losses.extend(module._router_gate_sparse_buffer)
            binary_losses.extend(module._router_gate_binary_buffer)
            module._router_gate_sparse_buffer.clear()
            module._router_gate_binary_buffer.clear()
    sparse = torch.stack(sparse_losses).mean() if sparse_losses else None
    binary = torch.stack(binary_losses).mean() if binary_losses else None
    return (sparse, binary)


def format_gla_router_stats(rows: list[dict]) -> str:
    if not rows:
        return "router_stats: no rows captured"
    gate_kinds = sorted({str(row.get("router_gate", "affine_sigmoid")) for row in rows})
    gate_kind = gate_kinds[0] if len(gate_kinds) == 1 else ",".join(gate_kinds)
    avg_entropy = sum((row["avg_norm_entropy_visible_gt1"] for row in rows)) / len(rows)
    avg_max = sum((row["avg_max_prob"] for row in rows)) / len(rows)
    avg_current = sum((row["avg_current_chunk_mass"] for row in rows)) / len(rows)
    avg_gate = sum((row.get("avg_gate", 0.0) for row in rows)) / len(rows)
    avg_gate_sum = sum((row.get("avg_gate_sum", 0.0) for row in rows)) / len(rows)
    avg_current_gate = sum((row.get("avg_current_gate", 0.0) for row in rows)) / len(
        rows
    )
    avg_max_gate = sum((row.get("avg_max_gate", 0.0) for row in rows)) / len(rows)
    avg_last_gate_sum = sum(
        (row.get("last_token_gate_sum", 0.0) for row in rows)
    ) / len(rows)
    avg_last_current_gate = sum(
        (row.get("last_token_current_gate", 0.0) for row in rows)
    ) / len(rows)
    min_entropy = min((row["min_norm_entropy_visible_gt1"] for row in rows))
    max_token_prob_dev = max((row["max_token_max_prob_dev"] for row in rows))
    max_token_uniform_dev = max((row["max_token_uniform_dev"] for row in rows))
    max_last_dev = max((row["last_token_max_uniform_dev"] for row in rows))
    avg_q_std = sum((row["q_route_std"] for row in rows)) / len(rows)
    avg_k_std = sum((row["k_route_std"] for row in rows)) / len(rows)
    avg_score_std = sum((row["score_std"] for row in rows)) / len(rows)
    max_score_range = max((row["score_range"] for row in rows))
    worst = min(rows, key=lambda row: row["avg_norm_entropy_visible_gt1"])
    if gate_kind in {"sigmoid", "affine_sigmoid"}:
        return f"router_stats: layers {len(rows)}, gate {gate_kind}, avg_gate {avg_gate:.6e}, avg_gate_sum {avg_gate_sum:.6e}, avg_current_gate {avg_current_gate:.6e}, avg_max_gate {avg_max_gate:.6e}, avg_last_gate_sum {avg_last_gate_sum:.6e}, avg_last_current_gate {avg_last_current_gate:.6e}, avg_normed_gate_entropy {avg_entropy:.6f}, avg_normed_max_share {avg_max:.6f}, avg_normed_current_share {avg_current:.6f}, min_token_normed_gate_entropy {min_entropy:.6f}, max_token_normed_uniform_dev {max_token_uniform_dev:.6f}, max_last_token_normed_uniform_dev {max_last_dev:.6f}, avg_q_route_std {avg_q_std:.6e}, avg_k_route_std {avg_k_std:.6e}, avg_score_std {avg_score_std:.6e}, max_score_range {max_score_range:.6e}, lowest_entropy_module {worst['module']}={worst['avg_norm_entropy_visible_gt1']:.6f}"
    return f"router_stats: layers {len(rows)}, avg_norm_entropy {avg_entropy:.6f}, avg_max_prob {avg_max:.6f}, avg_current_chunk_mass {avg_current:.6f}, min_token_norm_entropy {min_entropy:.6f}, max_token_max_prob_dev {max_token_prob_dev:.6f}, max_token_uniform_dev {max_token_uniform_dev:.6f}, max_last_token_uniform_dev {max_last_dev:.6f}, avg_q_route_std {avg_q_std:.6e}, avg_k_route_std {avg_k_std:.6e}, avg_score_std {avg_score_std:.6e}, max_score_range {max_score_range:.6e}, lowest_entropy_module {worst['module']}={worst['avg_norm_entropy_visible_gt1']:.6f}"


def _block_size_from_train_config(train_config: str) -> Optional[int]:
    train_config = train_config.lower()
    for marker, block_size in (
        ("x32k", 32768),
        ("x16k", 16384),
        ("x8k", 8192),
        ("x4k", 4096),
        ("x2k", 2048),
    ):
        if marker in train_config:
            return block_size
    return None


def compile_transformer_blocks_in_place(model: torch.nn.Module) -> int:
    """Compile existing blocks after loss hooks, preserving parameter/state names.

    Module.compile() changes only the call implementation. Replacing a block
    with torch.compile(block) would introduce an OptimizedModule and potentially
    an `_orig_mod` checkpoint prefix, and interfere with FSDP's class wrapping.
    """
    blocks = list(model.transformer.h)
    if not blocks or any((not isinstance(block, Block) for block in blocks)):
        raise RuntimeError(
            "GDN_COMPILE_BLOCKS expects transformer.h to contain Block modules"
        )
    state_keys = tuple(model.state_dict())
    parameters = tuple(((name, id(param)) for name, param in model.named_parameters()))
    import torch._dynamo.config as dynamo_config

    dynamo_config.recompile_limit = max(
        dynamo_config.recompile_limit, 4 * len(blocks) + 8
    )
    dynamo_config.accumulated_recompile_limit = max(
        dynamo_config.accumulated_recompile_limit, 16 * len(blocks)
    )
    for block in blocks:
        block.compile(fullgraph=False, dynamic=False)
    if state_keys != tuple(model.state_dict()):
        raise RuntimeError(
            "In-place block compilation unexpectedly changed checkpoint keys"
        )
    if parameters != tuple(
        ((name, id(param)) for name, param in model.named_parameters())
    ):
        raise RuntimeError(
            "In-place block compilation unexpectedly replaced parameters"
        )
    return len(blocks)


def validate_compiled_router_loss_buffers(
    model: torch.nn.Module, budget_modules: int, sparse_modules: int
) -> None:
    """Fail promptly if compilation loses hook side effects or detaches losses.

    Checks Python metadata only, without GPU synchronization or extra backward.
    Called before the collectors clear the buffers for this forward pass.
    """
    expected = {
        "_router_gate_budget_buffer": budget_modules,
        "_router_gate_sparse_buffer": sparse_modules,
        "_router_gate_binary_buffer": sparse_modules,
    }
    expected = {name: count for name, count in expected.items() if count}
    if not expected:
        return
    counts = dict.fromkeys(expected, 0)
    for module in model.modules():
        for name in expected:
            if not hasattr(module, name):
                continue
            values = getattr(module, name)
            if len(values) != 1:
                raise RuntimeError(
                    f"Compiled block {name} captured {len(values)} values; expected exactly one"
                )
            value = values[0]
            if (
                not isinstance(value, torch.Tensor)
                or not value.requires_grad
                or value.grad_fn is None
            ):
                raise RuntimeError(
                    f"Compiled block {name} lost the auxiliary-loss gradient graph"
                )
            counts[name] += 1
    if counts != expected:
        raise RuntimeError(
            f"Compiled router loss buffers missing: found {counts}, expected {expected}"
        )


def main(args):
    compile_blocks = os.environ.get("GDN_COMPILE_BLOCKS", "0") == "1"
    args.hparams["activation_checkpointing"] = False
    activation_kwargs = {}
    args.gdn_compile_blocks = compile_blocks
    args.hparams["compile_blocks"] = compile_blocks
    args.hparams["affine_kernel_flags"] = {
        name: os.environ.get(name, default)
        for name, default in (
            ("GDN_AFFINE_FINAL_STATE_BACKWARD", "0"),
            ("GDN_AFFINE_NATIVE_BF16", "0"),
            ("GDN_AFFINE_BATCHED_DQKW", "0"),
            ("GDN_AFFINE_BATCHED_WY", "0"),
            ("GDN_AFFINE_BATCHED_PRECISION", "tf32x3"),
            ("GDN_AFFINE_TF32X3", "0"),
            ("GDN_COMPILE_AFFINE_HISTORY_TRAINING", "0"),
            ("GDN_SKIP_DIRECT_POOL_CURRENT_SCORES", "0"),
            ("GDN_SKIP_SELFATTN_POOL_CURRENT_SCORES", "0"),
            ("DISABLE_MMA_V5", "0"),
        )
    }
    if args.debug:
        wandb_logger = WandbLogger(
            project="llm_next_gen",
            mode="disabled",
            name=args.exp_name,
            id=args.exp_name,
            save_dir=args.wandb_dir,
            dir=args.wandb_dir,
            version=args.exp_name,
            group="debug",
        )
    else:
        wandb_logger = WandbLogger(
            project="llm_next_gen",
            name=args.exp_name,
            id=args.exp_name,
            save_dir=args.wandb_dir,
            dir=args.wandb_dir,
            version=args.exp_name,
            group=args.exp_group,
        )
    if devices == 1:
        strategy = "auto"
    elif args.distributed_strategy == "ddp":
        strategy = "ddp"
    elif args.interactive_job:
        strategy = FSDPStrategy(
            auto_wrap_policy={Block},
            state_dict_type="full",
            **activation_kwargs,
            **{"use_orig_params": True} if compile_blocks or False else {},
        )
    else:
        strategy = FSDPStrategy(
            auto_wrap_policy={Block},
            state_dict_type="full",
            sharding_strategy="HYBRID_SHARD",
            **activation_kwargs,
            **{"use_orig_params": True} if compile_blocks or False else {},
        )
    fabric = L.Fabric(
        devices=devices,
        strategy=strategy,
        precision="bf16-mixed",
        loggers=[wandb_logger],
    )
    fabric.launch()
    fabric.seed_everything(args.seed)
    fabric.print("##### Infra Details #####")
    fabric.print(f"Number of Nodes: {args.nodes}")
    fabric.print(f"Number of GPUs: {fabric.world_size}")
    fabric.print("##### Training Details #####")
    fabric.print(f"Maximum number of training tokens: {args.max_tokens}")
    fabric.print(f"Micro batch size: {args.micro_batch_size}")
    fabric.print(f"Batch size: {args.batch_size}")
    if fabric.global_rank == 0:
        fabric.print(args)
    fabric.logger.log_hyperparams(args)
    monitor = Monitor(
        fabric,
        window_size=2,
        time_unit="seconds",
        log_iter_interval=args.log_iter_interval,
    )
    auto_resume = None
    if fabric.global_rank == 0:
        auto_resume = os.path.exists(args.out_dir) and (not False)
    auto_resume = fabric.broadcast(auto_resume, src=0)
    if auto_resume:
        args.resume = True
        print("Resuming from {}".format(args.out_dir))
    elif fabric.global_rank == 0:
        os.makedirs(args.out_dir, exist_ok=False)
        target_litgpt_save_dir = os.path.join(args.out_dir, "lit_gpt")
        target_bash_scripts_save_dir = os.path.join(args.out_dir, "bash_scripts")
        os.makedirs(target_litgpt_save_dir, exist_ok=True)
        os.makedirs(target_bash_scripts_save_dir, exist_ok=True)
    fabric.barrier()
    config = Config.from_name(args.model_name)
    requested_block_size = args.train_block_size
    if requested_block_size is not None and requested_block_size != config.block_size:
        fabric.print(
            f"Overriding config block_size from {config.block_size} to {requested_block_size} based on train_config={args.train_config}"
        )
        config.block_size = requested_block_size
    args.block_size = config.block_size
    fabric.print(f"Training block_size: {config.block_size}")
    train_dataloader, val_dataloader = create_dataloaders(
        batch_size=args.micro_batch_size,
        block_size=config.block_size,
        fabric=fabric,
        train_data_dir=args.train_data_dir,
        val_data_dir=args.val_data_dir,
        seed=args.seed,
    )
    if val_dataloader is None:
        train_dataloader = fabric.setup_dataloaders(train_dataloader)
    else:
        train_dataloader, val_dataloader = fabric.setup_dataloaders(
            train_dataloader, val_dataloader
        )
    if getattr(config, "gated_delta_variant", None) == "gla_gdn":
        args.hparams["hla_routing"] = {
            "mix_chunk_size": config.mix_chunk_size,
            "mix_pool_size": config.mix_pool_size,
            "mix_apply_gdn_decay": config.mix_apply_gdn_decay,
            "gla_router_route_source": config.gla_router_route_source,
            "gla_router_use_logmean": config.gla_router_use_logmean,
            "gla_router_use_rope": config.gla_router_use_rope,
            "gla_router_pool_self_attention": config.gla_router_pool_self_attention,
            "gla_router_gate": config.gla_router_gate,
            "gla_router_sigmoid_bias": config.gla_router_sigmoid_bias,
            "current_always_on": True,
        }
    if fabric.global_rank == 0:
        fabric.print(f"Loading model with {config.__dict__}")
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=False):
        model = GPT(config)
        model.apply(partial(model._init_weights, n_layer=config.n_layer))
    if fabric.global_rank == 0:
        fabric.print(
            f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds."
        )
        fabric.print(f"Total parameters {num_parameters(model.transformer.h):,}")
        fabric.print(model)
    if getattr(config, "gated_delta_variant", None) == "gla_gdn":
        router_scale_modules = set_gla_router_logit_scale(model, 1.0)
        if fabric.global_rank == 0:
            fabric.print(
                f"Set GLA router_logit_scale={1.0} on {router_scale_modules} modules"
            )
        current_always_on_modules = set_gla_router_current_always_on(model, True)
        if fabric.global_rank == 0:
            fabric.print(
                f"Set GLA router_current_always_on=True on {current_always_on_modules} modules"
            )
        sigmoid_temperature_modules = set_gla_router_sigmoid_temperature(model, 1.0)
        if fabric.global_rank == 0:
            fabric.print(
                f"Set GLA router_sigmoid_temperature={1.0} on {sigmoid_temperature_modules} modules"
            )
    trainable_summary = set_trainable_parameters(model, "all")
    if fabric.global_rank == 0:
        fabric.print(
            f"Trainable parameter mode={trainable_summary['mode']}: {trainable_summary['trainable_params']:,} params across {trainable_summary['trainable_tensors']} tensors"
        )
        if trainable_summary["sample"]:
            fabric.print(f"Trainable parameter sample: {trainable_summary['sample']}")
    router_stats_modules = 0
    if (
        args.router_stats_step_interval > 0
        and getattr(config, "gated_delta_variant", None) == "gla_gdn"
    ):
        router_stats_modules = attach_gla_router_stats(model)
        if fabric.global_rank == 0:
            fabric.print(
                f"Attached router stats hooks to {router_stats_modules} GLA modules"
            )
    router_gate_budget_modules = 0
    if getattr(config, "gated_delta_variant", None) == "gla_gdn":
        router_gate_budget_modules = attach_gla_router_gate_budget_loss(
            model, 3.0, exclude_current=True
        )
        if fabric.global_rank == 0:
            fabric.print(
                f"Attached router gate budget loss hooks to {router_gate_budget_modules} GLA modules with coef={0.005} target={3.0} exclude_current={True}"
            )
    router_gate_sparse_modules = 0
    if getattr(config, "gated_delta_variant", None) == "gla_gdn":
        router_gate_sparse_modules = attach_gla_router_gate_sparse_loss(model, 2.0)
        if fabric.global_rank == 0:
            fabric.print(
                f"Attached router sparse/binary loss hooks to {router_gate_sparse_modules} GLA modules with sparse_coef={0.001} target={2.0} binary_coef={0.001}"
            )
    if compile_blocks:
        compiled_blocks = compile_transformer_blocks_in_place(model)
        fabric.print(
            f"Compiled {compiled_blocks} transformer blocks in place (fullgraph=False, dynamic=False; checkpoint keys and parameters unchanged; FSDP use_orig_params=True)"
        )
    model = fabric.setup(model)
    trainable_model_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_model_params:
        raise RuntimeError(f"No trainable parameters for trainable_params={'all'!r}")
    optimizer = torch.optim.AdamW(
        trainable_model_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        fused=True,
    )
    optimizer = fabric.setup_optimizers(optimizer)
    state = {
        "model": model,
        "optimizer": optimizer,
        "hparams": args.hparams,
        "iter_num": 0,
        "step_count": 0,
    }
    resume_dataloader = args.resume and (not args.resume_from_checkpoint)
    if args.resume_dataloader_from_counters:
        resume_dataloader = True
    if args.resume or args.resume_from_checkpoint:
        try:
            resume = args.resume_from_checkpoint or os.path.join(
                args.out_dir, "latest-model-ckpt.pth"
            )
            if fabric.global_rank == 0:
                fabric.print(f"Resuming training from {resume}")
            fabric.load(resume, state)
            fabric.print(f"Successfully resumed from {resume}")
            if args.resume_micro_batch_size == -1 and int(state["step_count"]) > 0:
                old_iter_num = int(state["iter_num"])
                step_count = int(state["step_count"])
                if old_iter_num % step_count != 0:
                    raise ValueError(
                        f"Cannot infer the resumed accumulation exactly: iter_num={old_iter_num}, step_count={step_count}"
                    )
                old_accumulation = old_iter_num // step_count
                state["iter_num"] = step_count * args.gradient_accumulation_steps
                if fabric.global_rank == 0:
                    fabric.print(
                        f"Rescaled resumed iter_num from optimizer-step counters: {old_iter_num} (accumulation={old_accumulation}) -> {state['iter_num']} (accumulation={args.gradient_accumulation_steps}); step_count={step_count}"
                    )
            elif (
                args.resume_micro_batch_size > 0
                and args.resume_micro_batch_size != args.micro_batch_size
            ):
                old_iter_num = int(state["iter_num"])
                consumed_sequences_per_rank = (
                    old_iter_num * args.resume_micro_batch_size
                )
                if consumed_sequences_per_rank % args.micro_batch_size != 0:
                    raise ValueError(
                        f"Cannot preserve the resumed dataloader position exactly: iter_num={old_iter_num}, old micro batch={args.resume_micro_batch_size}, new micro batch={args.micro_batch_size}"
                    )
                state["iter_num"] = consumed_sequences_per_rank // args.micro_batch_size
                if state["iter_num"] % args.gradient_accumulation_steps != 0:
                    raise ValueError(
                        f"Rescaled iter_num is not on an optimizer-step boundary: iter_num={state['iter_num']}, accumulation={args.gradient_accumulation_steps}"
                    )
                if fabric.global_rank == 0:
                    fabric.print(
                        f"Rescaled resumed iter_num to preserve consumed sequences: {old_iter_num} (micro={args.resume_micro_batch_size}) -> {state['iter_num']} (micro={args.micro_batch_size}); step_count={state['step_count']}"
                    )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to resume from {resume}; refusing to silently restart training"
            ) from exc
        args.hparams["resumed_from_checkpoint"] = str(resume)
    state["hparams"] = args.hparams
    train_time = time.perf_counter()
    train(
        args,
        fabric,
        state,
        train_dataloader,
        val_dataloader,
        monitor,
        resume_dataloader,
        router_stats_modules,
        0,
        router_gate_budget_modules,
        router_gate_sparse_modules,
    )
    if fabric.global_rank == 0:
        fabric.print(f"Training time: {time.perf_counter() - train_time:.2f}s")
    if fabric.device.type == "cuda":
        if fabric.global_rank == 0:
            fabric.print(
                f"Memory used: {torch.cuda.max_memory_allocated() / 1000000000.0:.02f} GB"
            )


def train(
    args,
    fabric,
    state,
    train_dataloader,
    val_dataloader,
    monitor,
    resume,
    router_stats_modules=0,
    router_gate_budget_modules=0,
    router_gate_sparse_modules=0,
):
    model = state["model"]
    optimizer = state["optimizer"]
    total_lengths = 0
    total_t0 = time.perf_counter()
    max_tokens_per_device = args.max_tokens // fabric.world_size
    tokens_per_iter = args.micro_batch_size * model.config.block_size
    max_iters = max_tokens_per_device // tokens_per_iter
    warmup_iters = args.warmup_tokens // fabric.world_size // tokens_per_iter
    initial_iter = state["iter_num"]
    curr_iter = 0
    loss_func = FusedCrossEntropyLoss()
    tokens = 0
    train_t0 = time.perf_counter()
    if args.eval_before_training:
        fabric.print("Do validation before training:")
        val_loss = validate(args, fabric, model, val_dataloader, None)
        for i in range(args.num_extrapol):
            if fabric.global_rank == 0:
                fabric.print(
                    f"step {state['iter_num']} {i + 1} x: val loss {val_loss[i]:.4f}"
                )

    def save_checkpoint(final=False):
        name = "latest" if not final else "final"
        checkpoint_path = os.path.join(args.out_dir, f"{name}-model-ckpt.pth")
        fabric.print(f"Saving checkpoint to {str(checkpoint_path)!r}")

        def atomic_save(path):
            tmp_path = f"{path}.tmp"
            fabric.save(tmp_path, state)
            if fabric.global_rank == 0:
                os.replace(tmp_path, path)
            fabric.barrier()

        if not final:
            atomic_save(checkpoint_path)
            archive_interval = (
                args.save_step_archive_interval or args.save_step_interval
            )
            if (
                args.save_step_checkpoints
                and state["step_count"] % archive_interval == 0
            ):
                step_checkpoint_path = os.path.join(
                    args.out_dir, f"step{state['step_count']:06d}-model-ckpt.pth"
                )
                fabric.print(f"Saving checkpoint to {str(step_checkpoint_path)!r}")
                atomic_save(step_checkpoint_path)
        else:
            state["optimizer"] = None
            atomic_save(checkpoint_path)

    for train_data in train_dataloader:
        if resume:
            if curr_iter < initial_iter:
                curr_iter += 1
                continue
            else:
                resume = False
                curr_iter = -1
                fabric.barrier()
                if fabric.global_rank == 0:
                    fabric.print(
                        "resume finished, taken {} seconds".format(
                            time.perf_counter() - total_t0
                        )
                    )
        if state["iter_num"] >= max_iters or (
            args.stop_after_step > 0 and state["step_count"] >= args.stop_after_step
        ):
            break
        tokens += model.config.block_size * args.micro_batch_size
        iter_t0 = time.perf_counter()
        input_ids = train_data[:, 0 : model.config.block_size].contiguous()
        targets = train_data[:, 1 : model.config.block_size + 1].contiguous()
        lr = get_lr(args, state["iter_num"], warmup_iters, max_iters)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        is_accumulating = (
            state["iter_num"] + 1
        ) % args.gradient_accumulation_steps != 0
        next_step = state["step_count"] + (0 if is_accumulating else 1)
        collect_router_stats = (
            router_stats_modules > 0
            and (not is_accumulating)
            and (
                next_step % args.router_stats_step_interval == 0
                or (args.router_stats_first_step and next_step == 1)
            )
        )
        if router_stats_modules > 0:
            set_gla_router_stats_active(model, collect_router_stats)
        if router_gate_budget_modules > 0:
            set_gla_router_gate_budget_active(model, True)
        if router_gate_sparse_modules > 0:
            set_gla_router_gate_sparse_active(model, True)
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(input_ids)
            if getattr(args, "gdn_compile_blocks", False):
                validate_compiled_router_loss_buffers(
                    model, router_gate_budget_modules, router_gate_sparse_modules
                )
            if collect_router_stats and fabric.global_rank == 0:
                fabric.print(
                    f"step {next_step} {format_gla_router_stats(collect_gla_router_stats(model))}"
                )
            elif router_stats_modules > 0:
                collect_gla_router_stats(model)
            loss = loss_func(logits, targets)
            if router_gate_budget_modules > 0:
                router_gate_budget_loss = collect_gla_router_gate_budget_loss(model)
                if router_gate_budget_loss is not None:
                    loss = loss + 0.005 * router_gate_budget_loss
                    if not is_accumulating and fabric.global_rank == 0:
                        msg = f"step {next_step} router_gate_budget_loss {router_gate_budget_loss.item():.6f} coef {0.005} target {3.0}"
                        fabric.print(msg)
            if router_gate_sparse_modules > 0:
                router_sparse_loss, router_binary_loss = (
                    collect_gla_router_gate_sparse_losses(model)
                )
                if router_sparse_loss is None or router_binary_loss is None:
                    raise RuntimeError("router sparse/binary losses were not captured")
                loss = loss + 0.001 * router_sparse_loss + 0.001 * router_binary_loss
                if not is_accumulating and fabric.global_rank == 0:
                    fabric.print(
                        f"step {next_step} router_sparse_loss {router_sparse_loss.item():.6f} coef {0.001} target {2.0} router_binary_loss {router_binary_loss.item():.6f} coef {0.001}"
                    )
            fabric.backward(loss / args.gradient_accumulation_steps)
        if router_stats_modules > 0:
            set_gla_router_stats_active(model, False)
        if router_gate_budget_modules > 0:
            set_gla_router_gate_budget_active(model, False)
        if router_gate_sparse_modules > 0:
            set_gla_router_gate_sparse_active(model, False)
        if not is_accumulating:
            fabric.clip_gradients(model, optimizer, max_norm=args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            state["step_count"] += 1
        state["iter_num"] += 1
        total_lengths += input_ids.size(1)
        t1 = time.perf_counter()
        if (
            fabric.global_rank == 0
            and state["iter_num"] % max(1, args.log_step_interval) == 0
        ):
            total_tokens = (
                model.config.block_size
                * state["iter_num"]
                * args.micro_batch_size
                * fabric.world_size
                / 1000000000.0
            )
            fabric.print(
                f"iter {state['iter_num']} step {state['step_count']}: loss {loss.item():.4f}, iter time: {(t1 - iter_t0) * 1000:.2f}ms{(' (optimizer.step)' if not is_accumulating else '')} remaining time: {(t1 - total_t0) / (state['iter_num'] - initial_iter) * (max_iters - state['iter_num']) / 3600:.2f} hours.  or {(t1 - total_t0) / (state['iter_num'] - initial_iter) * (max_iters - state['iter_num']) / 3600 / 24:.2f} days.  total training throughput {tokens / (t1 - train_t0) / 1000.0:.2f}K tokens/s per GPU. total trained tokens: {total_tokens} B tokens peak memory allocate {torch.cuda.memory_stats(0)['allocated_bytes.all.peak'] / 1000000000.0} GB"
            )
        estimated_flops = 1
        monitor.on_train_batch_end(
            state["iter_num"] * args.micro_batch_size,
            t1 - total_t0,
            fabric.world_size,
            state["step_count"],
            flops_per_batch=estimated_flops,
            lengths=total_lengths,
            train_loss=loss.item(),
        )
        if not is_accumulating and state["step_count"] % args.save_step_interval == 0:
            save_checkpoint()
        if (
            val_dataloader is not None
            and (not is_accumulating)
            and (state["step_count"] % args.eval_step_interval == 0)
        ):
            t0 = time.perf_counter()
            val_loss = validate(args, fabric, model, val_dataloader, args.eval_iters)
            t1 = time.perf_counter() - t0
            monitor.eval_end(t1)
            for i in range(args.num_extrapol):
                if fabric.global_rank == 0:
                    fabric.print(
                        f"step {state['iter_num']} {i + 1} x: val loss {val_loss[i]:.4f}, val time: {t1 * 1000:.2f}ms"
                    )
                    fabric.log_dict(
                        {"metric/val_loss@" + str(i + 1) + "x": val_loss[i].item()},
                        state["step_count"],
                    )
                    fabric.log_dict(
                        {
                            "metric/val_ppl@"
                            + str(i + 1)
                            + "x": math.exp(val_loss[i].item())
                        },
                        state["step_count"],
                    )
            fabric.barrier()
    save_checkpoint(final=state["iter_num"] >= max_iters)


@torch.no_grad()
def validate(
    args,
    fabric: L.Fabric,
    model: torch.nn.Module,
    val_dataloader: DataLoader,
    eval_iters=10,
) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()
    losses = torch.zeros(eval_iters, args.num_extrapol, device=fabric.device)
    for k, val_data in enumerate(val_dataloader):
        if k >= eval_iters:
            break
        for i, length in enumerate([2048, 4096]):
            input_ids = val_data[:, 0:length].contiguous()
            targets = val_data[:, 1 : length + 1].contiguous()
            logits = model(input_ids)
            loss = chunked_cross_entropy(logits, targets, chunk_size=0)
            losses[k, i] = loss.item()
    out = losses.mean(0)
    model.train()
    return out


def create_dataloader(
    batch_size: int,
    block_size: int,
    data_dir: Path,
    fabric,
    shuffle: bool = True,
    seed: int = 12345,
    split="train",
) -> DataLoader:
    datasets = []
    data_config = train_data_config if split == "train" else val_data_config
    data_specs = []
    for prefix, weight in data_config:
        data_specs.append((data_dir, prefix, weight))
    for source_dir, prefix, _ in data_specs:
        filenames = sorted(glob.glob(os.path.join(source_dir, f"{prefix}*")))
        if not filenames:
            raise RuntimeError(
                f"No data files found at {source_dir} with prefix {prefix!r}."
            )
        random.seed(seed)
        random.shuffle(filenames)
        if split != "train":
            n_chunks = -(8 // -nodes)
        else:
            n_chunks = 8
        dataset = PackedDataset(
            filenames,
            n_chunks=n_chunks,
            block_size=block_size,
            shuffle=shuffle,
            seed=seed + fabric.global_rank,
            num_processes=fabric.world_size,
            process_rank=fabric.global_rank,
        )
        datasets.append(dataset)
    if not datasets:
        raise RuntimeError(
            f"No data found at {data_dir}. Make sure you ran prepare_redpajama.py to create the dataset."
        )
    weights = [weight for _, _, weight in data_specs]
    sum_weights = sum(weights)
    weights = [el / sum_weights for el in weights]
    combined_dataset = CombinedDataset(datasets=datasets, seed=seed, weights=weights)
    return DataLoader(
        combined_dataset, batch_size=batch_size, shuffle=False, pin_memory=True
    )


def create_dataloaders(
    batch_size: int,
    block_size: int,
    fabric,
    train_data_dir: Path = Path("data/redpajama_sample"),
    val_data_dir: Optional[Path] = None,
    seed: int = 12345,
) -> Tuple[DataLoader, DataLoader]:
    effective_block_size = block_size + 1
    train_dataloader = create_dataloader(
        batch_size=batch_size,
        block_size=effective_block_size,
        fabric=fabric,
        data_dir=train_data_dir,
        shuffle=True,
        seed=seed,
        split="train",
    )
    val_dataloader = (
        create_dataloader(
            batch_size=-(batch_size // -2),
            block_size=16384 + 1,
            fabric=fabric,
            data_dir=val_data_dir,
            shuffle=False,
            seed=seed,
            split="validation",
        )
        if val_data_dir
        else None
    )
    return (train_dataloader, val_dataloader)


def get_lr(args, it: int, warmup_iters: int, max_iters: int) -> float:
    if it < warmup_iters:
        return args.learning_rate * it / warmup_iters
    if it > max_iters:
        return args.min_lr
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return args.min_lr + coeff * (args.learning_rate - args.min_lr)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    devices = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(description="LLM Training")
    group = parser.add_argument_group("hyperparameters")
    group.add_argument(
        "--output_root", default="", type=str, help="output root directory"
    )
    group.add_argument("--wandb_dir", default="", type=str, help="wandb directory")
    group.add_argument(
        "--train_data_dir", default="", type=str, help="training data directory"
    )
    group.add_argument(
        "--train_data_dir_raw",
        default="",
        type=str,
        help="training data directory (raw file for stream tok)",
    )
    group.add_argument(
        "--val_data_dir", default="", type=str, help="validation data directory"
    )
    group.add_argument(
        "--val_data_dir_raw",
        default="",
        type=str,
        help="validation data directory (raw file for stream tok)",
    )
    group.add_argument(
        "--model_name",
        default="GatedDeltaNet_GLA_GDN_Release_1.3B",
        choices=("GatedDeltaNet_GLA_GDN_Release_1.3B", "GatedDeltaNet_Release_1.3B"),
        type=str,
        help="model name",
    )
    group.add_argument("--exp_name", default="", type=str, help="experiment name")
    group.add_argument(
        "--exp_group", default="", type=str, help="experiment group name"
    )
    group.add_argument(
        "--train_config", default="512x4k_100B", type=str, help="training config"
    )
    group.add_argument(
        "--resume", action="store_true", default=False, help="resume flag"
    )
    group.add_argument(
        "--resume_from_checkpoint",
        default="",
        type=str,
        help="load training state from this checkpoint without skipping the new dataloader",
    )
    group.add_argument(
        "--resume_micro_batch_size",
        type=int,
        default=0,
        help="micro batch size used to create a resumed checkpoint; rescales iter_num to preserve its dataloader/token position. Use -1 to infer the old accumulation from checkpoint counters.",
    )
    group.add_argument(
        "--resume_dataloader_from_counters",
        action="store_true",
        default=False,
        help="skip the dataloader up to the initialized iter_num without loading optimizer state",
    )
    group.add_argument("--debug", action="store_true", default=False, help="debug flag")
    group.add_argument(
        "--interactive_job", action="store_true", default=False, help="debug flag"
    )
    group.add_argument(
        "--distributed_strategy",
        choices=("fsdp", "ddp"),
        default="fsdp",
        help="multi-GPU strategy",
    )
    group.add_argument("--tokenizer_name", type=str, default="TinyLlama/TinyLlama_v1.1")
    group.add_argument(
        "--learning_rate", type=float, default=0.0001, help="learning rate"
    )
    group.add_argument(
        "--total_evals", type=int, default=400, help="total number of evals"
    )
    group.add_argument(
        "--eval_iters", type=int, default=10, help="number of evaluation iterations"
    )
    group.add_argument(
        "--log_step_interval", type=int, default=10, help="log_step_interval"
    )
    group.add_argument(
        "--save_step_interval", type=int, default=1000, help="save_step_interval"
    )
    group.add_argument(
        "--save_step_archive_interval",
        type=int,
        default=0,
        help="interval for non-overwriting step checkpoints; 0 uses save_step_interval",
    )
    group.add_argument(
        "--eval_step_interval", type=int, default=1000, help="eval_step_interval"
    )
    group.add_argument("--seed", type=int, default=3407, help="seed")
    group.add_argument("--num_extrapol", type=int, default=2, help="num_extrapol")
    group.add_argument("--weight_decay", type=float, default=0.1, help="weight decay")
    group.add_argument("--beta1", type=float, default=0.9, help="beta1")
    group.add_argument("--beta2", type=float, default=0.95, help="beta2")
    group.add_argument("--grad_clip", type=float, default=1.0, help="gradient clip")
    group.add_argument(
        "--eval_before_training",
        action="store_true",
        default=False,
        help="do validation before the training starts",
    )
    group.add_argument("--nnodes", type=int, default=None, help="number of nodes")
    group.add_argument("--train_num_workers", type=int, default=8)
    group.add_argument("--val_num_workers", type=int, default=1)
    group.add_argument(
        "--micro_batch_size", type=int, default=8, help="micro batch size"
    )
    group.add_argument(
        "--max_tokens_override",
        type=int,
        default=0,
        help="override max tokens for smoke/debug runs",
    )
    group.add_argument(
        "--stop_after_step",
        type=int,
        default=0,
        help="stop after this optimizer step while retaining the max-token LR schedule; 0 disables",
    )
    group.add_argument(
        "--warmup_tokens_override",
        type=int,
        default=0,
        help="override warmup tokens; 0 keeps one percent of max tokens",
    )
    group.add_argument(
        "--gradient_accumulation_steps_override",
        type=int,
        default=0,
        help="override grad accumulation for smoke/debug runs",
    )
    group.add_argument(
        "--save_step_checkpoints",
        action="store_true",
        default=False,
        help="also save non-overwriting stepNNNNNN checkpoints at save_step_interval",
    )
    group.add_argument(
        "--router_stats_step_interval",
        type=int,
        default=0,
        help="log GLA/HLA router distribution stats every N optimizer steps; 0 disables",
    )
    group.add_argument(
        "--router_stats_first_step",
        action="store_true",
        default=False,
        help="also log router distribution stats on optimizer step 1",
    )
    args = parser.parse_args()
    args.train_block_size = _block_size_from_train_config(args.train_config)
    name = args.train_config + "_" + args.exp_name
    args.out_dir = args.output_root + "/outputs/" + name
    args.wandb_dir = args.output_root + "/wandb/" + name
    train_data_config = [("train_slim", 1.0)]
    val_data_config = [("validation", 1.0)]
    nodes = 1
    args.nodes = nodes
    micro_batch_size = 8
    if "20B" in name:
        max_tokens = 100000000000 // 5
    elif "100B" in name:
        max_tokens = 100000000000
    elif "50B" in name:
        max_tokens = 100000000000 // 2
    elif "30B" in name:
        max_tokens = 30000000000
    elif "15B" in name:
        max_tokens = 30000000000 // 2
    else:
        raise ValueError("Unknown training token config")
    if args.max_tokens_override > 0:
        max_tokens = args.max_tokens_override
    if "512x4k" in name:
        micro_batch_size = 8
        global_batch_size = 512 // nodes
    elif "1024x4k" in name:
        micro_batch_size = 8
        global_batch_size = 1024 // nodes
    elif "256x8k" in name:
        global_batch_size = 256 // nodes
        micro_batch_size = 8
    elif "128x16k" in name:
        global_batch_size = 128 // nodes
        micro_batch_size = 4
    elif "64x32k" in name:
        global_batch_size = 64 // nodes
        micro_batch_size = 2
    elif "1024x2k" in name:
        global_batch_size = 1024 // nodes
        micro_batch_size = 32
    if "1.3B" in name:
        micro_batch_size = 4
    micro_batch_size = max(1, micro_batch_size)
    args.min_lr = args.learning_rate / 10
    args.batch_size = global_batch_size // devices
    gradient_accumulation_steps = args.batch_size // micro_batch_size
    if args.gradient_accumulation_steps_override > 0:
        gradient_accumulation_steps = args.gradient_accumulation_steps_override
    assert gradient_accumulation_steps > 0
    log_iter_interval = args.log_step_interval * gradient_accumulation_steps
    args.gradient_accumulation_steps = gradient_accumulation_steps
    args.warmup_tokens = int(max_tokens * 0.01)
    if args.warmup_tokens_override > 0:
        args.warmup_tokens = args.warmup_tokens_override
    args.max_tokens = max_tokens
    if args.micro_batch_size == 0:
        args.micro_batch_size = micro_batch_size
    args.log_iter_interval = log_iter_interval
    hparams = {
        k: v
        for k, v in locals().items()
        if isinstance(v, (int, float, str)) and (not k.startswith("_"))
    }
    hparams["micro_batch_size"] = args.micro_batch_size
    hparams["global_batch_size"] = (
        args.micro_batch_size * args.gradient_accumulation_steps * devices
    )
    args.hparams = hparams
    main(args)
