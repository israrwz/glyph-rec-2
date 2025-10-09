#!/usr/bin/env python3
"""
train.py
========

Minimal training loop for the raster LeViT glyph embedding project.

Design Goals:
-------------
- Keep arguments minimal; rely on sensible defaults.
- Train a LeViT_128S (img_size=128) variant with:
    * CrossEntropy classification head
    * Parallel embedding head (128-D L2-normalized)
- Provide basic metrics per epoch:
    * Train loss
    * Validation accuracy (Top-1)
    * Optional retrieval-style Top-K (on a capped subset)
    * Simple effect size proxy (difference of intra vs inter cosine means / pooled std)
- Save:
    * checkpoints/best.pt (best val acc)
    * checkpoints/last.pt (latest epoch)
    * artifacts/train_log.jsonl (per-epoch JSON lines)
- Keep everything CPU-friendly and deterministic.

Enhancements (A–E):
-------------------
A. Persist label_to_index mapping for inference decoding.
B. Cosine LR schedule now supports linear warmup (configurable fraction).
C. Gradient norm (global) logged each epoch.
D. Optional font-disjoint splitting (--font-disjoint) to avoid font leakage.
E. Retrieval cadence control (--retrieval-every N) to reduce validation cost.

Author: Raster Phase 1
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", message=".*_register_pytree_node.*")
warnings.filterwarnings("ignore", message=".*timm.models.registry.*")

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Local imports
from .model import (
    build_glyph_levit_128s,
    GlyphLeViTConfig,
    save_checkpoint,
    load_checkpoint,
)
from .dataset import (
    DatasetConfig,
    GlyphRasterDataset,
    make_train_val_split,
    simple_collate,
)


# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Train LeViT_128S raster glyph classification + embedding model."
    )
    ap.add_argument("--db", required=True, help="Path to glyphs.db SQLite")
    ap.add_argument("--limit", type=int, default=30000, help="Max glyphs to load")
    ap.add_argument("--epochs", type=int, default=30, help="Training epochs")
    ap.add_argument("--batch-size", type=int, default=128, help="Batch size")
    ap.add_argument(
        "--img-size",
        type=int,
        default=224,
        help="Input image size (use 224 for original LeViT token grid; previous runs used 128).",
    )
    ap.add_argument(
        "--val-frac", type=float, default=0.1, help="Validation fraction (0,1)"
    )
    ap.add_argument("--device", type=str, default="cpu", help="cpu or cuda:0")
    ap.add_argument(
        "--out-dir", type=str, default="raster", help="Base output directory"
    )
    ap.add_argument(
        "--log-file",
        type=str,
        default="artifacts/train_log.jsonl",
        help="Relative path under out-dir for JSONL logs",
    )
    ap.add_argument(
        "--no-retrieval",
        action="store_true",
        help="Disable retrieval (Top-K & effect size) metrics (faster).",
    )
    ap.add_argument(
        "--retrieval-cap",
        type=int,
        default=3000,
        help="Max eval samples for retrieval metrics (memory/time cap).",
    )
    ap.add_argument(
        "--retrieval-every",
        type=int,
        default=1,
        help="Run retrieval metrics every N epochs (default: every epoch).",
    )
    ap.add_argument("--seed", type=int, default=42, help="Global RNG seed")
    ap.add_argument(
        "--exclude-shaped",
        action="store_true",
        help="Exclude glyph rows whose labels contain the suffix '_shaped' (e.g. U+062B_isol_shaped).",
    )
    # NOTE: --pre-raster-workers defined later (single canonical definition)
    # (Removed duplicate --suppress-warnings definition; single canonical definition kept later)
    ap.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from (backbone + embed head).",
    )
    ap.add_argument(
        "--grad-clip",
        type=float,
        default=0.0,
        help="If >0, apply global gradient clipping to this norm value.",
    )
    ap.add_argument(
        "--arcface-scale",
        type=float,
        default=30.0,
        help="ArcFace-like scale parameter (s). Only used if --arcface-margin>0.",
    )
    ap.add_argument(
        "--arcface-margin",
        type=float,
        default=0.0,
        help="If >0, enable ArcFace-like angular margin loss with given margin (m, radians).",
    )
    ap.add_argument(
        "--emb-loss-weight",
        type=float,
        default=0.3,
        help="Weight for embedding contrastive loss (0.0 disables). Recommended: 0.1-0.5. Lower for high class counts.",
    )
    ap.add_argument(
        "--pre-raster-workers",
        type=int,
        default=0,
        help="Parallel workers (threads) for pre-raster build phase (0=sequential).",
    )
    ap.add_argument(
        "--suppress-warnings",
        action="store_true",
        help="Suppress common deprecation warnings (timm registry, pytree).",
    )
    ap.add_argument(
        "--lr-backbone", type=float, default=1e-3, help="Backbone base learning rate"
    )
    ap.add_argument(
        "--lr-head",
        type=float,
        default=2e-3,
        help="Head (embedding/class) learning rate",
    )
    ap.add_argument(
        "--weight-decay", type=float, default=0.05, help="AdamW weight decay"
    )
    ap.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="If >0, also save checkpoint every N epochs (besides best/last).",
    )
    ap.add_argument(
        "--font-disjoint",
        action="store_true",
        help="Make validation split font-disjoint (no overlapping font_hash).",
    )
    ap.add_argument(
        "--warmup-frac",
        type=float,
        default=0.05,
        help="Fraction of total epochs used for LR linear warmup (0 disables).",
    )
    ap.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers for parallel batch preparation.",
    )
    ap.add_argument(
        "--pre-rasterize",
        action="store_true",
        help="Pre-render all glyphs (unaugmented) into an in-memory tensor cache.",
    )
    ap.add_argument(
        "--pre-raster-dtype",
        type=str,
        default="uint8",
        choices=["uint8", "float32"],
        help="Storage dtype for pre-raster cache (uint8 saves memory).",
    )
    ap.add_argument(
        "--pre-raster-mmap",
        action="store_true",
        help="Write pre-raster tensor to a memory-mapped file for lower RAM pressure.",
    )
    ap.add_argument(
        "--pre-raster-mmap-path",
        type=str,
        default=None,
        help="Path for pre-raster memmap file (auto-generated if omitted).",
    )
    ap.add_argument(
        "--min-label-count",
        type=int,
        default=1,
        help="Filter out labels with fewer than this many samples (after limit).",
    )
    ap.add_argument(
        "--drop-singletons",
        action="store_true",
        help="Convenience: if set and min-label-count=1, treat threshold as 2.",
    )
    ap.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable augmentations (useful for deterministic eval or caching).",
    )
    ap.add_argument(
        "--augment-mode",
        type=str,
        default="dataset",
        choices=["dataset", "gpu", "none"],
        help="Augmentation location: 'dataset' (existing per-sample), 'gpu' (batched on GPU), 'none' (disable).",
    )
    ap.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable mixed precision (AMP). Enabled by default on CUDA.",
    )
    ap.add_argument(
        "--log-interval",
        type=int,
        default=200,
        help="Batches between intra-epoch progress prints (0 disables).",
    )
    ap.add_argument(
        "--lr-schedule",
        type=str,
        default="cosine",
        choices=["cosine", "constant"],
        help="LR schedule: 'cosine' (warmup + cosine decay) or 'constant' (flat LR after warmup).",
    )
    return ap.parse_args(argv)


# ---------------------------------------------------------------------------
# Supervised Contrastive Loss
# ---------------------------------------------------------------------------


def batched_gpu_augment(
    imgs: torch.Tensor,
    translate_px: int = 2,
    scale_jitter: float = 0.05,
    contrast_jitter: float = 0.10,
    gamma_jitter: float = 0.10,
    blur_prob: float = 0.0,
    blur_kernel: int = 3,
) -> torch.Tensor:
    """
    Batched GPU augmentation for (B,1,H,W) images in [0,1].
    Applied after DataLoader collation to eliminate per-sample CPU overhead.
    All ops keep tensor on the same device/dtype (supports AMP).
    """
    if imgs.ndim != 4 or imgs.size(1) != 1:
        return imgs  # safeguard
    B, C, H, W = imgs.shape
    device = imgs.device
    dtype = imgs.dtype

    # Geometric (scale + translation)
    if (scale_jitter > 0) or (translate_px > 0):
        # Random scales
        scales = (
            1.0 + (torch.rand(B, device=device, dtype=dtype) * 2.0 - 1.0) * scale_jitter
        )
        # Integer pixel shifts
        if translate_px > 0:
            tx = torch.randint(-translate_px, translate_px + 1, (B,), device=device)
            ty = torch.randint(-translate_px, translate_px + 1, (B,), device=device)
        else:
            tx = torch.zeros(B, device=device, dtype=torch.long)
            ty = torch.zeros(B, device=device, dtype=torch.long)
        # Normalize shifts to [-1,1] grid space (pixel -> normalized)
        tx_n = tx.to(dtype) * (2.0 / W)
        ty_n = ty.to(dtype) * (2.0 / H)
        theta = torch.zeros(B, 2, 3, device=device, dtype=dtype)
        theta[:, 0, 0] = scales
        theta[:, 1, 1] = scales
        theta[:, 0, 2] = tx_n
        theta[:, 1, 2] = ty_n
        grid = torch.nn.functional.affine_grid(
            theta, size=imgs.shape, align_corners=False
        )
        imgs = torch.nn.functional.grid_sample(
            imgs, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )

    # Contrast jitter
    if contrast_jitter > 0:
        factors = (
            1.0
            + (torch.rand(B, 1, 1, 1, device=device, dtype=dtype) * 2.0 - 1.0)
            * contrast_jitter
        )
        imgs = (imgs - 0.5) * factors + 0.5
        imgs = imgs.clamp(0.0, 1.0)

    # Gamma jitter
    if gamma_jitter > 0:
        gammas = (
            1.0
            + (torch.rand(B, 1, 1, 1, device=device, dtype=dtype) * 2.0 - 1.0)
            * gamma_jitter
        )
        imgs = torch.clamp(imgs, 1e-5, 1.0) ** gammas
        imgs = imgs.clamp(0.0, 1.0)

    # Optional blur (simple mean blur, applied selectively)
    if blur_prob > 0 and blur_kernel >= 3 and (blur_kernel % 2 == 1):
        mask = torch.rand(B, device=device) < blur_prob
        if mask.any():
            k = blur_kernel
            pad = k // 2
            weight = torch.ones(1, 1, k, k, device=device, dtype=dtype) / (k * k)
            idx = mask.nonzero(as_tuple=False).flatten()
            blurred = torch.nn.functional.conv2d(imgs[idx], weight, padding=pad)
            imgs[idx] = blurred

    return imgs


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Supervised contrastive loss for L2-normalized embeddings.

    For each sample i:
    - Positives: all j with labels[j] == labels[i], j != i
    - Negatives: all j with labels[j] != labels[i]

    Loss: -log(sum(exp(sim_pos)) / sum(exp(sim_all_except_self)))

    embeddings: (B, D) L2-normalized vectors
    labels: (B,) integer class labels
    temperature: scaling factor for similarities
    """
    device = embeddings.device
    B = embeddings.size(0)

    # Ensure labels are on the same device as embeddings
    labels = labels.to(device)

    # Compute pairwise cosine similarities (already L2-normalized)
    sim_matrix = embeddings @ embeddings.t()  # (B, B)
    sim_matrix = sim_matrix / temperature

    # Create label equality mask directly on device (avoid intermediate tensors)
    labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B) boolean

    # Create self-mask
    mask = torch.eye(B, device=device, dtype=torch.bool)

    # Exclude self-similarity from label mask
    labels_eq = labels_eq & ~mask  # (B, B) boolean - positives excluding self

    # For numerical stability, subtract max
    sim_max, _ = sim_matrix.max(dim=1, keepdim=True)
    sim_matrix = sim_matrix - sim_max.detach()

    # Compute exp(sim)
    exp_sim = torch.exp(sim_matrix)

    # Sum over positives (convert boolean mask to float only when needed)
    pos_sum = (exp_sim * labels_eq.float()).sum(dim=1)  # (B,)

    # Sum over all except self
    exp_sim_masked = exp_sim.masked_fill(mask, 0)
    all_sum = exp_sim_masked.sum(dim=1)  # (B,)

    # Loss for samples that have at least one positive
    has_pos = labels_eq.any(dim=1)  # (B,) boolean - more efficient than sum > 0

    if not has_pos.any():
        return torch.tensor(0.0, device=device)

    # -log(pos / all)
    log_prob = torch.log(pos_sum[has_pos] / (all_sum[has_pos] + 1e-8) + 1e-8)
    loss = -log_prob.mean()

    return loss


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def set_seed(seed: int):
    import random
    import numpy as np

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _make_font_disjoint_split(
    dataset: GlyphRasterDataset, val_fraction: float, seed: int
) -> Tuple[GlyphRasterDataset, GlyphRasterDataset]:
    """
    Split dataset by font_hash groups so that fonts do not overlap between
    train and validation sets. Label vocab is still shared.
    """
    if not (0.0 < val_fraction < 1.0):
        raise ValueError("val_fraction must be in (0,1).")

    import random as _r

    fonts: Dict[str, List[Any]] = {}
    for r in dataset._rows:
        fonts.setdefault(r.font_hash, []).append(r)
    font_ids = list(fonts.keys())
    _r.Random(seed).shuffle(font_ids)
    val_font_count = max(1, int(len(font_ids) * val_fraction))
    val_set = set(font_ids[:val_font_count])
    train_rows: List[Any] = []
    val_rows: List[Any] = []
    for fh, rows in fonts.items():
        (val_rows if fh in val_set else train_rows).extend(rows)

    def _clone(rows: List[Any]) -> GlyphRasterDataset:
        clone = GlyphRasterDataset(dataset.cfg, rasterizer=dataset.rasterizer)
        clone._rows = rows
        clone.label_to_index = dataset.label_to_index
        clone.index_to_label = dataset.index_to_label
        clone._cache.clear()
        clone._cache_order.clear()
        return clone

    return _clone(train_rows), _clone(val_rows)


def build_loaders(
    args,
    *,
    stratified_min1: bool = False,
    coverage_report: bool = False,
) -> Tuple[GlyphRasterDataset, GlyphRasterDataset, DataLoader, DataLoader]:
    ds_cfg = DatasetConfig(
        db_path=args.db,
        limit=args.limit,
        randomize_query=True,
        image_size=args.img_size,
        supersample=2,
        augment=not args.no_augment,
        seed=args.seed,
        pre_rasterize=args.pre_rasterize,
        pre_raster_dtype=args.pre_raster_dtype,
        pre_raster_mmap=args.pre_raster_mmap,
        pre_raster_mmap_path=args.pre_raster_mmap_path,
        min_label_count=args.min_label_count,
        drop_singletons=args.drop_singletons,
        verbose_stats=True,
        pre_raster_workers=args.pre_raster_workers,
    )
    # Instantiate full dataset first
    full_ds = GlyphRasterDataset(ds_cfg)
    # Optional exclusion of labels containing '_shaped'
    if getattr(args, "exclude_shaped", False):
        original_count = len(full_ds._rows)
        filtered_rows = [r for r in full_ds._rows if "_shaped" not in r.label]
        removed = original_count - len(filtered_rows)
        if removed > 0:
            print(
                f"[FILTER] Excluded {removed} shaped labels ('_shaped' suffix). Remaining rows={len(filtered_rows)}"
            )
        # Apply filtering
        full_ds._rows = filtered_rows
        # Rebuild vocab & ordering after filtering
        from collections import Counter

        counts = Counter(r.label for r in full_ds._rows)
        labels = sorted(counts.keys())
        full_ds.label_to_index = {lab: i for i, lab in enumerate(labels)}
        full_ds.index_to_label = labels
        full_ds.glyph_id_order = [r.glyph_id for r in full_ds._rows]
    if args.font_disjoint:
        train_ds, val_ds = _make_font_disjoint_split(full_ds, args.val_frac, args.seed)
    else:
        train_ds, val_ds = make_train_val_split(
            full_ds,
            val_fraction=args.val_frac,
            seed=args.seed,
            stratified_min1=stratified_min1,
            coverage_report=coverage_report,
        )

    # Enable pin_memory for faster CPU->GPU transfers when using CUDA
    use_pin_memory = args.device.startswith("cuda")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=use_pin_memory,
        collate_fn=simple_collate,
        persistent_workers=True if args.num_workers > 0 else False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_pin_memory,
        collate_fn=simple_collate,
        persistent_workers=True if args.num_workers > 0 else False,
    )
    return train_ds, val_ds, train_loader, val_loader


def build_model(num_classes: int, device: str, args) -> torch.nn.Module:
    cfg = GlyphLeViTConfig(
        img_size=args.img_size,
        num_classes=num_classes,
        embedding_out_dim=128,
        hidden_dim=256,
        activation="gelu",
        distillation=False,
        replicate_gray=True,
    )
    model = build_glyph_levit_128s(cfg)
    model.to(device)
    return model


def create_optimizer(model: torch.nn.Module, args) -> torch.optim.Optimizer:
    backbone_params = []
    head_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("backbone.") and "head" not in name:
            backbone_params.append(p)
        else:
            head_params.append(p)
    param_groups = [
        {"params": backbone_params, "lr": args.lr_backbone},
        {"params": head_params, "lr": args.lr_head},
    ]
    opt = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    return opt


def adjust_lrs(
    optimizer,
    epoch: int,
    max_epochs: int,
    base_lrs: List[float],
    warmup_epochs: int,
    schedule: str = "cosine",
):
    """
    LR schedule with optional linear warmup for first `warmup_epochs`.
    epoch here is zero-based.
    schedule: 'cosine' (decay after warmup) or 'constant' (flat after warmup).
    """
    for pg, base_lr in zip(optimizer.param_groups, base_lrs):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            lr = base_lr * float(epoch + 1) / float(warmup_epochs)
        elif schedule == "constant":
            lr = base_lr
        else:
            effective_epoch = epoch - warmup_epochs
            effective_total = max(1, max_epochs - warmup_epochs)
            cos = 0.5 * (1 + math.cos(math.pi * effective_epoch / effective_total))
            lr = base_lr * cos
        pg["lr"] = lr


@torch.no_grad()
def extract_embeddings(
    model: torch.nn.Module, loader: DataLoader, device: str, cap: int | None = None
) -> Tuple[torch.Tensor, List[int], List[str]]:
    model.eval()
    embs: List[torch.Tensor] = []
    glyph_ids: List[int] = []
    labels: List[str] = []
    total = 0
    for batch in loader:
        imgs = batch["images"].to(device)
        out = model(imgs)
        emb = out["embedding"].detach().cpu()
        embs.append(emb)
        glyph_ids.extend(batch["glyph_ids"])
        labels.extend(batch["raw_labels"])
        total += imgs.size(0)
        if cap and total >= cap:
            break
    return torch.cat(embs, dim=0), glyph_ids, labels


@torch.no_grad()
def compute_retrieval_metrics(
    embeddings: torch.Tensor,
    labels: List[str],
    k: int = 10,
) -> Dict[str, Any]:
    """
    Simple brute-force cosine Top-K + effect size.
    embeddings: (N,D) assumed L2 normalized.
    labels: list of length N
    """
    N, D = embeddings.shape
    if N < 2:
        return {"topk_accuracy": 0.0, "mrr": 0.0, "effect_size": 0.0}

    sims = embeddings @ embeddings.t()
    sims.fill_diagonal_(-2.0)

    topk_vals, topk_idx = sims.topk(k, dim=1)

    hits = 0
    rr_sum = 0.0
    for i in range(N):
        li = labels[i]
        row = topk_idx[i]
        found_rank = None
        for r, j in enumerate(row):
            if labels[j] == li:
                hits += 1
                found_rank = r + 1
                break
        if found_rank:
            rr_sum += 1.0 / found_rank
    topk_acc = hits / N
    mrr = rr_sum / N

    import random

    label_to_indices: Dict[str, List[int]] = {}
    for i, lab in enumerate(labels):
        label_to_indices.setdefault(lab, []).append(i)

    intra_samples = []
    inter_samples = []
    rng = random.Random(123)
    for lab, idxs in label_to_indices.items():
        if len(idxs) < 2:
            continue
        for _ in range(min(50, len(idxs))):
            a, b = rng.sample(idxs, 2)
            intra_samples.append(float(embeddings[a] @ embeddings[b]))
            if len(intra_samples) >= 10000:
                break
        if len(intra_samples) >= 10000:
            break

    all_indices = list(range(N))
    while len(inter_samples) < min(10000, len(intra_samples)):
        a, b = rng.sample(all_indices, 2)
        if labels[a] != labels[b]:
            inter_samples.append(float(embeddings[a] @ embeddings[b]))

    if intra_samples and inter_samples:
        import statistics as stats

        mu_in = stats.mean(intra_samples)
        mu_out = stats.mean(inter_samples)
        all_s = intra_samples + inter_samples
        sigma = stats.pstdev(all_s) if len(all_s) > 1 else 1.0
        effect = (mu_in - mu_out) / sigma if sigma > 1e-9 else 0.0
    else:
        effect = 0.0
        mu_in = mu_out = 0.0

    return {
        "topk_accuracy": topk_acc,
        "mrr": mrr,
        "effect_size": effect,
        "intra_mean": mu_in,
        "inter_mean": mu_out,
        "sim_diff": (mu_in - mu_out),
        "sim_gap_ratio": ((mu_in - mu_out) / (abs(mu_out) + 1e-9))
        if mu_out != 0
        else 0.0,
        "n_embeddings": N,
    }


def _global_grad_norm(model: torch.nn.Module) -> float:
    """Compute global gradient norm efficiently on GPU, sync once at end."""
    # Collect all grad norms on GPU
    norms = []
    for p in model.parameters():
        if p.grad is not None:
            norms.append(p.grad.data.norm(2))

    if not norms:
        return 0.0

    # Stack and compute total norm on GPU
    total_norm = torch.stack(norms).norm(2)
    # Only sync to CPU once
    return total_norm.item()


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    *,
    arcface_margin: float = 0.0,
    arcface_scale: float = 30.0,
    grad_clip: float = 0.0,
    log_interval: int = 200,
    emb_loss_weight: float = 0.3,
    use_amp: bool = False,
    scaler: "torch.cuda.amp.GradScaler | None" = None,
    gpu_augment: bool = False,
    gpu_aug_cfg: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    model.train()
    # Accumulate losses on GPU to avoid synchronization
    total_loss_tensor = torch.tensor(0.0, device=device)
    total_ce_loss_tensor = torch.tensor(0.0, device=device)
    total_emb_loss_tensor = torch.tensor(0.0, device=device)
    total_samples = 0
    ce = nn.CrossEntropyLoss()
    last_grad_norm = 0.0
    last_unclipped_grad_norm = 0.0
    first_batch_reported = False

    def _arcface_loss(
        logits: torch.Tensor, labels: torch.Tensor, feats: torch.Tensor
    ) -> torch.Tensor:
        """
        True-ish ArcFace:
          - Normalize feature vectors and classification weight vectors
          - Compute cosine
          - Add angular margin to target classes: cos(theta+m)
          - Scale by arcface_scale then cross entropy
        """
        if arcface_margin <= 0:
            return ce(logits, labels)

        # Extract the linear head weight/bias (assumes backbone.head exists)
        head = getattr(model.backbone, "head", None)
        if head is None or not hasattr(head, "l"):
            # Fallback to legacy shift if structure unexpected
            with torch.no_grad():
                one_hot = torch.zeros_like(logits)
                one_hot.scatter_(1, labels.view(-1, 1), 1.0)
            adjusted = logits - one_hot * arcface_margin * arcface_scale
            return ce(adjusted / arcface_scale, labels)

        linear = head.l  # BN_Linear.l (Linear)
        W = linear.weight  # (C, D)
        # Normalize features & weights
        # Normalize weights & features with gradients enabled (ArcFace requires grads through both)
        W_norm = torch.nn.functional.normalize(W, dim=1)
        feat_norm = torch.nn.functional.normalize(feats, dim=1)
        if feat_norm.shape[1] != W_norm.shape[1]:
            # Safeguard: feature dim (e.g. 128) mismatches classifier weight dim (e.g. 384)
            # Fallback to standard CE on original logits (avoid silent failure).
            return ce(logits, labels)
        # Cosine logits (dimensions now aligned)
        cos = torch.matmul(feat_norm, W_norm.t()).clamp(-1.0, 1.0)  # (B,C)
        # Angular margin application (no in-place ops to preserve autograd history)
        theta = torch.acos(cos.clamp(-1 + 1e-7, 1 - 1e-7))
        target_theta = theta[torch.arange(theta.size(0)), labels] + arcface_margin
        cos_target = torch.cos(target_theta)
        cos_margin = cos.clone()
        cos_margin[torch.arange(cos_margin.size(0)), labels] = cos_target
        scaled = cos_margin * arcface_scale
        return ce(scaled, labels)

    for batch_idx, batch in enumerate(loader, start=1):
        imgs = batch["images"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        joining_groups = batch.get("joining_groups", None)
        if joining_groups is not None:
            joining_groups = joining_groups.to(device, non_blocking=True)
        # Batched GPU augmentation (after move) if enabled
        if gpu_augment:
            imgs = batched_gpu_augment(
                imgs,
                translate_px=int(gpu_aug_cfg.get("translate_px", 2)),
                scale_jitter=float(gpu_aug_cfg.get("scale_jitter", 0.05)),
                contrast_jitter=float(gpu_aug_cfg.get("contrast_jitter", 0.10)),
                gamma_jitter=float(gpu_aug_cfg.get("gamma_jitter", 0.10)),
                blur_prob=float(gpu_aug_cfg.get("blur_prob", 0.0)),
                blur_kernel=int(gpu_aug_cfg.get("blur_kernel", 3)),
            )

        optimizer.zero_grad()
        # Forward (AMP autocast)
        autocast_ctx = (
            torch.cuda.amp.autocast(enabled=use_amp)
            if use_amp
            else torch.autocast("cuda", enabled=False)
        )
        with autocast_ctx:
            out = model(imgs)
            logits = out["logits"]
            embeddings = out["embedding"]

        # Classification loss (uses fine-grained labels)
        feats_for_margin = getattr(model, "_last_backbone_features", None)
        if feats_for_margin is None:
            # Fallback to embedding head output (may trigger dim guard in _arcface_loss)
            feats_for_margin = embeddings
        if arcface_margin > 0.0:
            ce_loss = _arcface_loss(logits, labels, feats_for_margin)
        else:
            ce_loss = ce(logits, labels)

        # Embedding contrastive loss (uses coarse joining_groups for better clustering)
        if emb_loss_weight <= 0:
            # Completely skip contrastive computation when weight is zero
            emb_loss = torch.tensor(0.0, device=embeddings.device)
        else:
            if joining_groups is not None:
                emb_loss = supervised_contrastive_loss(
                    embeddings, joining_groups, temperature=0.07
                )
            else:
                # Fallback to fine labels if no coarse grouping provided
                emb_loss = supervised_contrastive_loss(
                    embeddings, labels, temperature=0.07
                )
            # One-time debug (first batch) – show positive pair statistics
            if batch_idx == 1:
                with torch.no_grad():
                    tgt = joining_groups if joining_groups is not None else labels
                    lbl_eq = tgt.unsqueeze(0) == tgt.unsqueeze(1)
                    # Exclude self
                    lbl_eq.fill_diagonal_(False)
                    pos_counts = lbl_eq.sum(dim=1)
                    if pos_counts.numel() > 0:
                        print(
                            f"[DEBUG] contrastive positives: min={int(pos_counts.min().item())} "
                            f"mean={pos_counts.float().mean().item():.2f}"
                        )

        # Combined loss
        loss = ce_loss + emb_loss_weight * emb_loss

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                last_unclipped_grad_norm = _global_grad_norm(model)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                last_grad_norm = _global_grad_norm(model)
            else:
                last_unclipped_grad_norm = _global_grad_norm(model)
                last_grad_norm = last_unclipped_grad_norm
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip and grad_clip > 0:
                last_unclipped_grad_norm = _global_grad_norm(model)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                last_grad_norm = _global_grad_norm(model)
            else:
                last_unclipped_grad_norm = _global_grad_norm(model)
                last_grad_norm = last_unclipped_grad_norm
            optimizer.step()
        # Intra-epoch progress (prints every log_interval batches, if enabled)
        # Only sync to CPU for logging, not every batch
        if log_interval > 0 and (
            batch_idx % log_interval == 0 or batch_idx == len(loader)
        ):
            try:
                total_batches = len(loader)
            except Exception:
                total_batches = -1
            # Only sync loss to CPU when logging
            print(
                f"[BATCH] prog={batch_idx}/{total_batches if total_batches > 0 else '?'} "
                f"loss={loss.item():.4f}"
            )
        # Debug stats only for first batch (moved outside hot loop)
        if not first_batch_reported:
            # Defer actual printing to avoid sync in tight loop
            first_batch_stats = {
                "shape": tuple(imgs.shape),
                "min": imgs.min(),
                "max": imgs.max(),
                "mean": imgs.mean(),
                "nonzero": (imgs > 0.01).sum(),
                "numel": imgs.numel(),
            }
            first_batch_reported = True
        bs = imgs.size(0)
        # Accumulate on GPU to avoid sync bottleneck
        total_loss_tensor += loss.detach() * bs
        total_ce_loss_tensor += ce_loss.detach() * bs
        total_emb_loss_tensor += emb_loss.detach() * bs
        total_samples += bs

    # Print first batch stats after loop (sync once)
    if "first_batch_stats" in locals():
        s = first_batch_stats
        print(
            f"[DEBUG] first batch shape={s['shape']}, "
            f"min={s['min'].item():.4f}, max={s['max'].item():.4f}, mean={s['mean'].item():.4f}, "
            f"nonzero_px={s['nonzero'].item()}/{s['numel']}"
        )

    # Only sync to CPU at epoch end
    return {
        "train_loss": total_loss_tensor.item() / max(1, total_samples),
        "ce_loss": total_ce_loss_tensor.item() / max(1, total_samples),
        "emb_loss": total_emb_loss_tensor.item() / max(1, total_samples),
        "grad_norm": last_grad_norm,
        "grad_norm_unclipped": last_unclipped_grad_norm,
    }


@torch.no_grad()
def validate(model: torch.nn.Module, loader: DataLoader, device: str) -> Dict[str, Any]:
    model.eval()
    ce = nn.CrossEntropyLoss()
    # Accumulate on GPU to avoid sync bottleneck
    total_loss_tensor = torch.tensor(0.0, device=device)
    total_correct_tensor = torch.tensor(0, device=device, dtype=torch.long)
    total_samples = 0
    for batch in loader:
        imgs = batch["images"].to(device)
        labels = batch["labels"].to(device)
        out = model(imgs)
        logits = out["logits"]
        loss = ce(logits, labels)
        preds = logits.argmax(dim=1)
        # Accumulate on GPU
        total_correct_tensor += (preds == labels).sum()
        bs = imgs.size(0)
        total_samples += bs
        total_loss_tensor += loss.detach() * bs
    # Only sync to CPU at end
    return {
        "val_loss": total_loss_tensor.item() / max(1, total_samples),
        "val_accuracy": total_correct_tensor.item() / max(1, total_samples),
        "val_samples": total_samples,
    }


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Main Training Logic
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)

    out_base = Path(args.out_dir)
    checkpoints_dir = out_base / "checkpoints"
    artifacts_dir = out_base / "artifacts"
    ensure_dir(checkpoints_dir)
    ensure_dir(artifacts_dir)

    log_path = out_base / args.log_file
    ensure_dir(log_path.parent)

    print(f"[INFO] Building datasets...")
    if args.exclude_shaped:
        print(
            "[INFO] Shaped label exclusion enabled: rows with '_shaped' in label will be dropped."
        )
    # Adjust dataset augment flag based on augment-mode
    if getattr(args, "augment_mode", "dataset") == "gpu":
        if not args.no_augment:
            # Force dataset (per-sample) aug off; we'll do batched GPU aug
            args.no_augment = True
    if getattr(args, "augment_mode", "dataset") == "none":
        args.no_augment = True
    # Use stratified split ensuring every label with >=2 samples has at least 1 in training
    # and report coverage statistics.
    train_ds, val_ds, train_loader, val_loader = build_loaders(
        args, stratified_min1=True, coverage_report=True
    )
    print(
        f"[INFO] Train size={len(train_ds)} | Val size={len(val_ds)} | Num classes={len(train_ds.label_to_index)}"
    )

    # Enhanced rasterization integrity check with visualization
    print("[INFO] Performing rasterization integrity check...")
    try:
        # Sample a few items from the dataset and check their statistics
        sample_indices = [
            0,
            len(train_ds) // 4,
            len(train_ds) // 2,
            (3 * len(train_ds)) // 4,
            len(train_ds) - 1,
        ]
        samples_valid = 0
        samples_blank = 0

        for idx in sample_indices:
            if idx >= len(train_ds):
                continue
            try:
                sample = train_ds[idx]
                img = sample["image"]  # Should be (1, H, W) tensor
                img_min = float(img.min())
                img_max = float(img.max())
                img_mean = float(img.mean())
                img_std = float(img.std())
                nonzero_count = int((img > 0.01).sum())
                total_pixels = img.numel()

                if img_max > 0.01 and nonzero_count > 10:
                    samples_valid += 1
                    if samples_valid == 1:
                        print(
                            f"[INTEGRITY] Sample {idx} (glyph_id={sample['glyph_id']}, label='{sample['raw_label']}'): "
                            f"shape={tuple(img.shape)}, min={img_min:.4f}, max={img_max:.4f}, "
                            f"mean={img_mean:.4f}, std={img_std:.4f}, nonzero={nonzero_count}/{total_pixels}"
                        )
                else:
                    samples_blank += 1
                    print(
                        f"[WARN] Sample {idx} appears blank: min={img_min:.4f}, max={img_max:.4f}, nonzero={nonzero_count}"
                    )
            except Exception as e:
                print(f"[ERROR] Failed to load sample {idx}: {e}")

        if samples_blank >= len(sample_indices) - 1:
            print(
                "[FATAL] Most sampled images are blank! Rasterization pipeline may be broken."
            )
            print(
                "[FATAL] Check preraster memmap file or disable --pre-raster-mmap to force re-rasterization."
            )
            raise RuntimeError(
                "Rasterization integrity check failed: most samples are blank"
            )
        elif samples_valid > 0:
            print(
                f"[INTEGRITY] Check passed: {samples_valid}/{len(sample_indices)} samples valid, {samples_blank} blank"
            )
        else:
            print(
                "[WARN] Could not verify rasterization integrity (no valid samples found)"
            )

        # Additional check for val dataset
        if len(val_ds) > 0:
            val_sample = val_ds[0]
            val_img = val_sample["image"]
            val_mean = float(val_img.mean())
            val_nonzero = int((val_img > 0.01).sum())
            print(
                f"[INTEGRITY] Val sample 0: mean={val_mean:.4f}, nonzero={val_nonzero}/{val_img.numel()}, augment={val_ds.cfg.augment}"
            )

    except Exception as _e:
        print(f"[ERROR] Rasterization integrity check failed: {_e}")
        import traceback

        traceback.print_exc()
        raise

    # A. Save label mapping
    label_map_path = artifacts_dir / "label_to_index.json"
    with label_map_path.open("w", encoding="utf-8") as f:
        json.dump(train_ds.label_to_index, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved label mapping: {label_map_path}")

    print("[INFO] Building model...")
    device = args.device
    model = build_model(
        num_classes=len(train_ds.label_to_index), device=device, args=args
    )
    n_params = count_parameters(model)
    print(f"[INFO] Model params: {n_params / 1e6:.3f}M")

    # Print training configuration
    print(f"[INFO] Training config:")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Classes (classification): {len(train_ds.label_to_index)}")
    print(
        f"  - Samples per class (avg): {len(train_ds) / len(train_ds.label_to_index):.1f}"
    )
    print(f"  - Embedding loss weight: {args.emb_loss_weight}")
    if args.emb_loss_weight > 0.0:
        print(f"  - Contrastive grouping: HYBRID (joining_group + char_class)")
        print(f"    • Arabic letters: grouped by joining_group (BEH, HAH, etc.)")
        print(f"    • NO_JOIN glyphs: grouped by char_class (latin, diacritic, etc.)")
        print(f"    • Prevents mixing Latin/diacritic/punctuation in embeddings")
    if args.emb_loss_weight == 0.0:
        print(f"    WARNING: Embedding loss disabled (weight=0.0)")

    # Resume logic (F): load checkpoint weights & report prior val accuracy if available
    resume_epoch_offset = 0
    if args.resume:
        if os.path.isfile(args.resume):
            try:
                ckpt = torch.load(args.resume, map_location="cpu")
                load_checkpoint(model, args.resume, strict=False)
                prev_val = ckpt.get("extra", {}).get("val_accuracy", None)
                opt_state_path = Path(args.resume).parent / "optimizer_state.pt"
                if opt_state_path.is_file():
                    try:
                        opt_state = torch.load(opt_state_path, map_location="cpu")
                        prev_epoch = opt_state.get("epoch", 0)
                        resume_epoch_offset = prev_epoch
                        print(f"[INFO] Loaded optimizer state epoch={prev_epoch}")
                    except Exception as e:
                        print(f"[WARN] Could not load optimizer state: {e}")
                if prev_val is not None:
                    print(
                        f"[INFO] Resumed from {args.resume} (prev best val_acc={prev_val:.4f})"
                    )
                else:
                    print(f"[INFO] Resumed from {args.resume} (no stored val_accuracy)")
            except Exception as e:
                print(f"[WARN] Failed to resume from {args.resume}: {e}")
        else:
            print(f"[WARN] --resume path not found: {args.resume}")

    # AMP setup (must be defined before entering the epoch loop so scaler persists)
    use_amp = args.device.startswith("cuda") and (not args.no_amp)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None

    optimizer = create_optimizer(model, args)
    base_lrs = [args.lr_backbone, args.lr_head]
    warmup_epochs = int(args.epochs * args.warmup_frac) if args.warmup_frac > 0 else 0
    # Attempt optimizer state resume
    if args.resume:
        state_path = Path(args.resume).parent / "optimizer_state.pt"
        if state_path.is_file():
            try:
                opt_state = torch.load(state_path, map_location="cpu")
                if "optimizer" in opt_state:
                    optimizer.load_state_dict(opt_state["optimizer"])
                    if "best_val_acc" in opt_state:
                        best_val_acc = opt_state["best_val_acc"]
                        prev_epoch = opt_state.get("epoch", None)
                        if prev_epoch:
                            print(
                                f"[INFO] Loaded optimizer state (epoch={prev_epoch}, best_val_acc={best_val_acc:.4f})"
                            )
                    else:
                        print(
                            "[INFO] Loaded optimizer state (no best_val_acc in state)."
                        )
                else:
                    print("[WARN] optimizer_state.pt missing 'optimizer' key.")
            except Exception as e:
                print(f"[WARN] Failed to load optimizer state: {e}")

    best_val_acc = -1.0
    best_epoch = -1

    train_start = time.time()
    for epoch in range(resume_epoch_offset + 1, resume_epoch_offset + args.epochs + 1):
        adjust_lrs(
            optimizer,
            epoch - 1,
            args.epochs,
            base_lrs,
            warmup_epochs=warmup_epochs,
            schedule=args.lr_schedule,
        )

        # Update dataset epoch for augmentation variety
        if hasattr(train_ds, "set_epoch"):
            train_ds.set_epoch(epoch)

        t0 = time.time()
        # Train
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device=args.device,
            arcface_margin=args.arcface_margin,
            arcface_scale=args.arcface_scale,
            grad_clip=args.grad_clip,
            log_interval=args.log_interval,
            emb_loss_weight=args.emb_loss_weight,
            use_amp=use_amp,
            scaler=scaler,
            gpu_augment=(getattr(args, "augment_mode", "dataset") == "gpu"),
            gpu_aug_cfg=dict(
                translate_px=2,
                scale_jitter=0.05,
                contrast_jitter=0.10,
                gamma_jitter=0.10,
                blur_prob=0.0,  # disable blur by default in batched path (cost/benefit)
                blur_kernel=3,
            ),
        )
        val_stats = validate(model, val_loader, device)

        retrieval_stats = {}
        run_retrieval_this_epoch = (
            (not args.no_retrieval)
            and args.retrieval_every > 0
            and (epoch % args.retrieval_every == 0)
        )
        if run_retrieval_this_epoch:
            emb, gids, labs = extract_embeddings(
                model, val_loader, device, cap=args.retrieval_cap
            )
            emb = F.normalize(emb, dim=1)
            retrieval_stats = compute_retrieval_metrics(emb, labs, k=10)

        epoch_time = time.time() - t0

        log_record = {
            "epoch": epoch,
            "lr_backbone": optimizer.param_groups[0]["lr"],
            "lr_head": optimizer.param_groups[1]["lr"],
            **train_metrics,
            **val_stats,
            **retrieval_stats,
            "epoch_time_sec": epoch_time,
            "font_disjoint": args.font_disjoint,
            "warmup_epochs": warmup_epochs,
        }

        save_checkpoint(
            model,
            str(checkpoints_dir / "last.pt"),
            step=epoch,
            extra={"val_accuracy": val_stats["val_accuracy"]},
        )

        if val_stats["val_accuracy"] > best_val_acc:
            best_val_acc = val_stats["val_accuracy"]
            best_epoch = epoch
            save_checkpoint(
                model,
                str(checkpoints_dir / "best.pt"),
                step=epoch,
                extra={"val_accuracy": best_val_acc},
            )

        # Only save per-epoch checkpoint every save_every epochs (reduce I/O overhead)
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(
                model,
                str(checkpoints_dir / f"epoch_{epoch}.pt"),
                step=epoch,
                extra={"val_accuracy": val_stats["val_accuracy"]},
            )
        # Save/update optimizer state (full resume support)
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_val_acc": best_val_acc,
            },
            checkpoints_dir / "optimizer_state.pt",
        )

        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_record) + "\n")

        summary = (
            f"[EPOCH {epoch - resume_epoch_offset}/{args.epochs}] "
            f"(abs_epoch={epoch}) "
            f"train_loss={train_metrics['train_loss']:.4f} "
            f"ce_loss={train_metrics.get('ce_loss', 0.0):.4f} "
            f"emb_loss={train_metrics.get('emb_loss', 0.0):.4f} "
            f"val_loss={val_stats['val_loss']:.4f} "
            f"val_acc={val_stats['val_accuracy']:.4f} "
            f"grad_norm={train_metrics['grad_norm']:.2f} "
            f"grad_unclipped={train_metrics.get('grad_norm_unclipped', 0.0):.2f} "
        )
        if retrieval_stats:
            summary += (
                f"top10_acc={retrieval_stats['topk_accuracy']:.4f} "
                f"mrr={retrieval_stats['mrr']:.4f} "
                f"effect={retrieval_stats['effect_size']:.4f} "
                f"diff={retrieval_stats.get('sim_diff', 0.0):.4f} "
                f"gap={retrieval_stats.get('sim_gap_ratio', 0.0):.4f} "
            )
        summary += f"time={epoch_time:.1f}s"
        print(summary)

    total_time = time.time() - train_start
    print(
        f"[INFO] Training complete in {total_time / 60.0:.2f} min. "
        f"Best val acc={best_val_acc:.4f} @ epoch {best_epoch}"
    )
    print(f"[INFO] Logs written to {log_path}")
    print(f"[INFO] Best checkpoint: {checkpoints_dir / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
