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
        "--log-interval",
        type=int,
        default=200,
        help="Batches between intra-epoch progress prints (0 disables).",
    )
    return ap.parse_args(argv)


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
) -> Tuple[GlyphRasterDataset, GlyphRasterDataset, DataLoader, DataLoader]:
    ds_cfg = DatasetConfig(
        db_path=args.db,
        limit=args.limit,
        randomize_query=True,
        image_size=128,
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
    full_ds = GlyphRasterDataset(ds_cfg)
    if args.font_disjoint:
        train_ds, val_ds = _make_font_disjoint_split(
            full_ds, val_fraction=args.val_frac, seed=args.seed
        )
    else:
        train_ds, val_ds = make_train_val_split(
            full_ds, val_fraction=args.val_frac, seed=args.seed
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
        collate_fn=simple_collate,
        persistent_workers=True if args.num_workers > 0 else False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        collate_fn=simple_collate,
        persistent_workers=True if args.num_workers > 0 else False,
    )
    return train_ds, val_ds, train_loader, val_loader


def build_model(num_classes: int, device: str, args) -> torch.nn.Module:
    cfg = GlyphLeViTConfig(
        img_size=128,
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
    optimizer, epoch: int, max_epochs: int, base_lrs: List[float], warmup_epochs: int
):
    """
    Cosine schedule with optional linear warmup for first `warmup_epochs`.
    epoch here is zero-based.
    """
    for pg, base_lr in zip(optimizer.param_groups, base_lrs):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            lr = base_lr * float(epoch + 1) / float(warmup_epochs)
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
    import math

    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total += param_norm.item() ** 2
    return math.sqrt(total)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    *,
    arcface_margin: float = 0.0,
    arcface_scale: float = 30.0,
    grad_clip: float = 0.0,
    log_interval: int = 200,  # batches between progress prints (0 disables)
) -> Dict[str, Any]:
    model.train()
    total_loss = 0.0
    total_samples = 0
    ce = nn.CrossEntropyLoss()
    last_grad_norm = 0.0

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
        imgs = batch["images"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad()
        out = model(imgs)
        logits = out["logits"]
        # Prefer backbone raw features for ArcFace so dimensionality matches classifier weight matrix
        feats_for_margin = getattr(model, "_last_backbone_features", None)
        if feats_for_margin is None:
            # Fallback to embedding head output (may trigger dim guard in _arcface_loss)
            feats_for_margin = out["embedding"]
        if arcface_margin > 0.0:
            loss = _arcface_loss(logits, labels, feats_for_margin)
        else:
            loss = ce(logits, labels)
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        last_grad_norm = _global_grad_norm(model)
        optimizer.step()
        # Intra-epoch progress (prints every log_interval batches, if enabled)
        if log_interval > 0 and (
            batch_idx % log_interval == 0 or batch_idx == len(loader)
        ):
            try:
                total_batches = len(loader)
            except Exception:
                total_batches = -1
            print(
                f"[BATCH] prog={batch_idx}/{total_batches if total_batches > 0 else '?'} "
                f"loss={loss.item():.4f}"
            )
        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

    return {
        "train_loss": total_loss / max(1, total_samples),
        "grad_norm": last_grad_norm,
    }


@torch.no_grad()
def validate(model: torch.nn.Module, loader: DataLoader, device: str) -> Dict[str, Any]:
    model.eval()
    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for batch in loader:
        imgs = batch["images"].to(device)
        labels = batch["labels"].to(device)
        out = model(imgs)
        logits = out["logits"]
        loss = ce(logits, labels)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        bs = imgs.size(0)
        total_samples += bs
        total_loss += loss.item() * bs
    return {
        "val_loss": total_loss / max(1, total_samples),
        "val_accuracy": total_correct / max(1, total_samples),
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

    print("[INFO] Building datasets...")
    train_ds, val_ds, train_loader, val_loader = build_loaders(args)
    print(
        f"[INFO] Train size={len(train_ds)} | Val size={len(val_ds)} | Num classes={len(train_ds.label_to_index)}"
    )
    # Memmap integrity quick check: if reused memmap appears entirely blank at a few sampled indices,
    # rebuild an in-memory tensor fallback (prevents silent all-zero inputs).
    try:
        if (
            hasattr(train_ds, "_preraster_memmap")
            and train_ds._preraster_memmap is not None
        ):
            mm = train_ds._preraster_memmap
            H = train_ds.cfg.image_size
            W = train_ds.cfg.image_size
            sample_indices = [0, len(mm) // 4, len(mm) // 2, (3 * len(mm)) // 4]
            zero_like = 0
            for si in sample_indices:
                if si < len(mm):
                    if mm[si].sum() == 0:
                        zero_like += 1
            if zero_like == len(sample_indices):
                print(
                    "[WARN] Detected all sampled preraster memmap blocks are zero; rebuilding in-memory preraster tensor."
                )
                import torch as _torch

                rebuilt = []
                for row in train_ds._rows:
                    t = train_ds._rasterize(row)
                    rebuilt.append((t.clamp(0, 1) * 255).to(_torch.uint8))
                train_ds._preraster_tensor = _torch.stack(rebuilt, dim=0)
                train_ds._preraster_memmap = None
                # Mirror to val split if it shares adoption
                if (
                    hasattr(val_ds, "_preraster_memmap")
                    and val_ds._preraster_memmap is not None
                ):
                    val_rebuilt = []
                    for row in val_ds._rows:
                        t = val_ds._rasterize(row)
                        val_rebuilt.append((t.clamp(0, 1) * 255).to(_torch.uint8))
                    val_ds._preraster_tensor = _torch.stack(val_rebuilt, dim=0)
                    val_ds._preraster_memmap = None
                print("[INFO] In-memory preraster rebuild complete (memmap replaced).")
    except Exception as _e:
        print(f"[WARN] Memmap integrity check skipped due to error: {_e}")

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
            optimizer, epoch - 1, args.epochs, base_lrs, warmup_epochs=warmup_epochs
        )

        t0 = time.time()
        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            arcface_margin=args.arcface_margin,
            arcface_scale=args.arcface_scale,
            grad_clip=args.grad_clip,
            log_interval=args.log_interval,
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
            **train_stats,
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
            f"train_loss={train_stats['train_loss']:.4f} "
            f"val_loss={val_stats['val_loss']:.4f} "
            f"val_acc={val_stats['val_accuracy']:.4f} "
            f"grad_norm={train_stats['grad_norm']:.2f} "
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
