#!/usr/bin/env python3
"""
embed.py
========

Extract L2-normalized glyph embeddings from a trained raster LeViT model checkpoint.

Overview
--------
1. Loads glyph rows from the SQLite `glyphs.db`.
2. Builds a non-augmented raster dataset (on-the-fly rendering).
3. Reconstructs the LeViT_128S wrapper model (img_size=128) and loads checkpoint
   (backbone + embedding head). Classification head shape mismatches are ignored (strict=False).
4. Runs batched inference to produce a contiguous embeddings tensor (N, D).
5. Writes:
     * Embeddings tensor (.pt) -> torch.FloatTensor (N, D)
     * Metadata JSONL (glyph_id, label, font_hash, embedding_index)
     * Optional label vocabulary JSON.

Defaults keep CLI surface minimal. All heavy parameters are fixed per the Phase 1 plan.

Example
-------
    python raster/embed.py \
        --db dataset/glyphs.db \
        --checkpoint raster/checkpoints/best.pt \
        --out-embeds raster/artifacts/raster_embeddings.pt \
        --out-meta raster/artifacts/raster_meta.jsonl

Optional:
    --limit 10000
    --batch-size 256
    --device cuda:0

Notes
-----
- Embedding head already applies L2 normalization; use --no-renorm if you want
  to trust that (default) or --force-renorm to enforce normalization again.
- If you trained with augmentations, extraction always disables them.
- If the dataset label set differs from training time, classification weights may
  not load (we ignore head weight mismatch). Embeddings remain valid.

Checkpoint Format (from train.py / model.py):
{
  "backbone": ...state_dict...
  "embed_head": ...state_dict...
  "num_classes": int,
  "embedding_dim": int,
  "step": int,
  "extra": {...}
}

Author: Raster Phase 1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Local project imports
try:
    from .model import (
        build_glyph_levit_128s,
        GlyphLeViTConfig,
        load_checkpoint,
    )
    from .dataset import (
        DatasetConfig,
        GlyphRasterDataset,
        simple_collate,
    )
except ImportError:
    # Allow running as module: python -m raster.embed ...
    from raster.model import (
        build_glyph_levit_128s,
        GlyphLeViTConfig,
        load_checkpoint,
    )
    from raster.dataset import (
        DatasetConfig,
        GlyphRasterDataset,
        simple_collate,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Extract glyph raster embeddings with a trained LeViT_128S model."
    )
    ap.add_argument("--db", required=True, help="Path to glyphs.db SQLite")
    ap.add_argument(
        "--checkpoint",
        required=True,
        help="Path to trained checkpoint (best.pt / last.pt)",
    )
    ap.add_argument(
        "--out-embeds",
        type=str,
        default="raster/artifacts/raster_embeddings.pt",
        help="Output .pt for embeddings (FloatTensor N,D)",
    )
    ap.add_argument(
        "--out-meta",
        type=str,
        default="raster/artifacts/raster_meta.jsonl",
        help="Output metadata JSONL",
    )
    ap.add_argument(
        "--out-labels",
        type=str,
        default=None,
        help="Optional label vocab JSON (maps index->label)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, cap number of glyphs loaded (subset extraction).",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Inference batch size (tune for memory / CPU speed).",
    )
    ap.add_argument("--device", type=str, default="cpu", help="cpu or cuda:0")
    ap.add_argument(
        "--force-renorm",
        action="store_true",
        help="Re-apply L2 normalization even if embeddings are already normalized.",
    )
    ap.add_argument(
        "--no-progress",
        action="store_true",
        help="Suppress per-batch progress logging.",
    )
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for any ordering")
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


def ensure_parent(path: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


@torch.no_grad()
def extract_embeddings(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    force_renorm: bool = False,
    progress: bool = True,
) -> torch.Tensor:
    model.eval()
    all_chunks: List[torch.Tensor] = []
    total = 0
    t0 = time.time()
    for i, batch in enumerate(loader):
        imgs = batch["images"].to(device)
        out = model(imgs)
        emb = out["embedding"]
        if force_renorm:
            emb = F.normalize(emb, dim=1)
        all_chunks.append(emb.cpu())
        total += imgs.size(0)
        if progress:
            print(
                f"[BATCH {i}] processed={total} chunk_shape={tuple(emb.shape)} elapsed={time.time() - t0:.1f}s"
            )
    return torch.cat(all_chunks, dim=0) if all_chunks else torch.empty(0, 128)


def write_metadata_jsonl(
    path: str,
    dataset: GlyphRasterDataset,
    embedding_count: int,
):
    """
    Write one line per item in the order the DataLoader produced them.
    Assumes DataLoader iteration order matches dataset order (shuffle disabled).

    Extended:
        If the dataset exposes get_glyph_meta(glyph_id) and the rasterizer
        captured normalization metadata, augment each record with:
            bbox_orig: [min_x, min_y, max_x, max_y] in original (pre-scale/center) coords
            scale_factor: float scale applied (1.0 if preserve mode or degenerate)
            fit_mode: scaling policy ("tight" or "preserve")
    """
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        for i, row in enumerate(dataset._rows[:embedding_count]):
            obj = {
                "glyph_id": row.glyph_id,
                "font_hash": row.font_hash,
                "label": row.label,
                "embedding_index": i,
            }
            meta = None
            if hasattr(dataset, "get_glyph_meta"):
                try:
                    meta = dataset.get_glyph_meta(row.glyph_id)
                except Exception:
                    meta = None
            if meta:
                # Only add expected keys to avoid accidental large blobs
                for k in (
                    "bbox_orig",
                    "scale_factor",
                    "fit_mode",
                    "overflow_downscale_applied",
                    "upem",
                    "font_global_scale",
                    "baseline_shift",
                    "overflow_downscale",
                    "used_ascent",
                    "used_descent",
                    "final_ascent",
                    "final_descent",
                    "ratio_compensation",
                    "comp_scale",
                    "center_shift",
                ):
                    if k in meta:
                        obj[k] = meta[k]
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)

    device = args.device
    print("[INFO] Building dataset (augment disabled for extraction)...")
    ds_cfg = DatasetConfig(
        db_path=args.db,
        limit=args.limit if args.limit > 0 else 300000000,  # effectively unlimited
        randomize_query=False,
        image_size=128,
        supersample=2,
        augment=False,
        seed=args.seed,
    )
    dataset = GlyphRasterDataset(ds_cfg)
    print(f"[INFO] Loaded rows={len(dataset)} num_labels={len(dataset.label_to_index)}")

    # DataLoader (no shuffle!)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=simple_collate,
    )

    # Rebuild model. Use dataset label size for classification dimension (will ignore mismatch).
    model_cfg = GlyphLeViTConfig(
        img_size=128,
        num_classes=len(dataset.label_to_index),
        embedding_out_dim=128,
        hidden_dim=256,
        activation="gelu",
        distillation=False,
        replicate_gray=True,
    )
    model = build_glyph_levit_128s(model_cfg).to(device)
    print("[INFO] Loading checkpoint:", args.checkpoint)
    ckpt = load_checkpoint(model, args.checkpoint, strict=False)
    emb_dim = ckpt.get("embedding_dim", model_cfg.embedding_out_dim)
    print(
        f"[INFO] Checkpoint loaded. Embedding dim={emb_dim} step={ckpt.get('step', '?')}"
    )

    print("[INFO] Extracting embeddings...")
    embeds = extract_embeddings(
        model,
        loader,
        device=device,
        force_renorm=args.force_renorm,
        progress=not args.no_progress,
    )
    print(f"[INFO] Final embeddings tensor shape: {tuple(embeds.shape)}")

    # Save embeddings
    ensure_parent(args.out_embeds)
    torch.save(embeds, args.out_embeds)
    print(f"[INFO] Wrote embeddings: {args.out_embeds}")

    # Save metadata (ordered)
    write_metadata_jsonl(args.out_meta, dataset, embedding_count=embeds.shape[0])
    print(f"[INFO] Wrote metadata: {args.out_meta}")

    # Optional labels
    if args.out_labels:
        ensure_parent(args.out_labels)
        with open(args.out_labels, "w", encoding="utf-8") as f:
            json.dump(
                {i: lab for i, lab in enumerate(dataset.index_to_label)},
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"[INFO] Wrote label vocab: {args.out_labels}")

    # Basic diagnostics
    if embeds.numel() > 0:
        norms = embeds.norm(dim=1)
        print(
            f"[INFO] Norm stats: mean={norms.mean():.4f} std={norms.std():.4f} min={norms.min():.4f} max={norms.max():.4f}"
        )

    print("[INFO] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
