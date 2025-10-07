"""
dataset.py
==========

On-the-fly raster glyph dataset for training the LeViT-based raster embedding model.

Goals (Phase 1):
- Minimal, dependency-light integration with existing SQLite glyph DB.
- Deterministic / reproducible splits (seeded).
- On-demand rasterization (no requirement to persist intermediate PNG files).
- Lightweight, domain-safe augmentations (translation, scale jitter, slight contrast / gamma, optional blur).
- Simple label indexing (string/integer labels -> contiguous [0..N-1]).

Design Choices:
---------------
1. We fetch glyph rows once at Dataset initialization (up to a configurable limit).
2. Rasterization uses Rasterizer from `rasterize.py` (parse -> normalize -> render).
3. Augmentations are implemented in tensor space (after rasterization) to avoid repeated
   contour parsing overhead and to keep them simple for small grayscale images.
4. For horizontal flips / rotations we deliberately DO NOT apply them (glyph semantics often change).
5. A tiny LRU cache (optional) can store the last K un-augmented rasters to reduce repeated parsing
   cost if multiple epochs revisit the same glyph (useful for small training sets).

Returned Sample:
----------------
{
  "image": FloatTensor (1, H, W) in [0,1],
  "label": int (index into label_vocab),
  "glyph_id": int,
  "font_hash": str,
  "raw_label": original label string,
}

Usage:
------
    from raster.dataset import GlyphRasterDataset, make_train_val_split
    ds = GlyphRasterDataset(db_path="dataset/glyphs.db", limit=30000, augment=True)
    train_ds, val_ds = make_train_val_split(ds, val_fraction=0.1, seed=42)

    sample = train_ds[0]
    img, label = sample["image"], sample["label"]

Notes / Future:
---------------
- Could add multi-channel rendering (distance transform) by extending Rasterizer config
  and stacking channels prior to normalization.
- May add caching of parsed contour structures if DB size >> RAM is not a concern.

"""

from __future__ import annotations

import math
import random
import sqlite3
import numpy as np
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from .rasterize import (
    Rasterizer,
    RasterizerConfig,
    fetch_glyph_rows,
    GlyphRow,
    rasterize_glyphs_to_tensor,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DatasetConfig:
    db_path: str
    limit: int = 30000
    randomize_query: bool = True
    # Rasterization
    image_size: int = 128
    supersample: int = 2
    cubic_subdiv: int = 8
    stroke_px: int = 0
    fill_closed: bool = True
    # New: normalization / hole handling policies
    fit_mode: str = "tight"  # "tight" or "preserve"
    hole_strategy: str = "orientation"  # "orientation" | "even-odd" | "none"
    clip_out_of_bounds: bool = True
    # Augmentations
    augment: bool = True
    translate_px: int = 2
    scale_jitter: float = 0.05  # ±5%
    contrast_jitter: float = 0.10  # ±10% linear contrast
    gamma_jitter: float = 0.10  # ±10% gamma exponent deviation
    blur_prob: float = 0.10
    blur_kernel: int = 3  # must be odd
    # Caching
    cache_size: int = 512  # number of *unaugmented* rasters to cache
    # Pre-rasterization (optional full tensor build to accelerate very large CPU farms)
    pre_rasterize: bool = (
        False  # If True, pre-render all glyphs (unaugmented) into a tensor
    )
    pre_raster_dtype: str = "uint8"  # "uint8" or "float32" storage for pre-raster
    # Parallel pre-raster workers (threaded) – 0 = sequential
    pre_raster_workers: int = 0
    # Memory-mapped pre-raster (optional: backed by np.memmap on disk)
    pre_raster_mmap: bool = False
    pre_raster_mmap_path: Optional[str] = (
        None  # If None and mmap enabled, auto-generate
    )
    # Label filtering / frequency audit
    min_label_count: int = (
        1  # Drop labels with fewer than this many train samples (1 = keep all)
    )
    drop_singletons: bool = (
        False  # Convenience: if True and min_label_count==1, treat as min_label_count=2
    )
    verbose_stats: bool = True  # Print frequency audit
    # Margin loss scaffolding (phase 2)
    enable_margin_targets: bool = (
        False  # If True, downstream training can use alternative margin losses
    )
    # Reproducibility
    seed: int = 42
    # Optional: restrict by label list
    label_filter: Optional[Sequence[str]] = None


# ---------------------------------------------------------------------------
# Utility: Convolutional Blur (Naive)
# ---------------------------------------------------------------------------


def _maybe_blur(
    img: torch.Tensor, kernel: int, p: float, rng: random.Random
) -> torch.Tensor:
    """
    Simple mean blur. img shape: (1, H, W)
    """
    if p <= 0 or rng.random() >= p:
        return img
    if kernel <= 1:
        return img
    if kernel % 2 == 0:
        kernel += 1
    k = torch.ones(1, 1, kernel, kernel, device=img.device, dtype=img.dtype) / (
        kernel * kernel
    )
    pad = kernel // 2
    with torch.no_grad():
        out = torch.nn.functional.conv2d(img.unsqueeze(0), k, padding=pad)
    return out.squeeze(0)


# ---------------------------------------------------------------------------
# Augmentations
# ---------------------------------------------------------------------------


def _augment_tensor(
    t: torch.Tensor,
    *,
    rng: random.Random,
    translate_px: int,
    scale_jitter: float,
    contrast_jitter: float,
    gamma_jitter: float,
    blur_prob: float,
    blur_kernel: int,
) -> torch.Tensor:
    """
    Apply light geometric + photometric augmentations.
    t: (1,H,W) in [0,1]
    """
    c, h, w = t.shape
    assert c == 1, "Augment assumes single-channel input."

    # Geometry: translation + scale via affine_grid + grid_sample
    # Compose scale + translation into 2x3
    if scale_jitter > 0 or translate_px > 0:
        scale = 1.0 + (rng.random() * 2 - 1) * scale_jitter
        # Convert pixel shift to normalized grid coords: shift_px / (size/2)
        shift_x_px = rng.randint(-translate_px, translate_px) if translate_px > 0 else 0
        shift_y_px = rng.randint(-translate_px, translate_px) if translate_px > 0 else 0
        shift_x = shift_x_px * 2.0 / w
        shift_y = shift_y_px * 2.0 / h
        theta = torch.tensor(
            [[scale, 0.0, shift_x], [0.0, scale, shift_y]],
            dtype=t.dtype,
            device=t.device,
        ).unsqueeze(0)  # (1,2,3)
        grid = torch.nn.functional.affine_grid(
            theta, size=(1, 1, h, w), align_corners=False
        )
        t = torch.nn.functional.grid_sample(
            t.unsqueeze(0),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).squeeze(0)

    # Photometric: contrast
    if contrast_jitter > 0:
        delta = (rng.random() * 2 - 1) * contrast_jitter
        # center at 0.5
        t = (t - 0.5) * (1.0 + delta) + 0.5
        t = t.clamp(0.0, 1.0)

    # Photometric: gamma
    if gamma_jitter > 0:
        g_delta = (rng.random() * 2 - 1) * gamma_jitter
        gamma = 1.0 + g_delta
        # Avoid zero -> add small epsilon
        t = torch.clamp(t, 1e-5, 1.0) ** gamma
        t = t.clamp(0.0, 1.0)

    # Blur
    t = _maybe_blur(t, blur_kernel, blur_prob, rng)

    return t


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class GlyphRasterDataset(Dataset):
    """
    On-the-fly rasterization dataset.

    - Fetches glyph rows from DB (optionally random order).
    - Builds label vocabulary (string -> int).
    - Rasterizes contours on each __getitem__ call (optionally cached).
    - Applies light augmentation if enabled.

    Args:
        config: DatasetConfig
        rasterizer: Optional externally provided Rasterizer (for custom params)
    """

    def __init__(
        self,
        config: DatasetConfig,
        rasterizer: Optional[Rasterizer] = None,
    ):
        super().__init__()
        self.cfg = config
        self._rng = random.Random(config.seed)
        self._rows: List[GlyphRow] = fetch_glyph_rows(
            db_path=config.db_path,
            limit=config.limit,
            randomize=config.randomize_query,
            label_filter=config.label_filter,
        )
        if not self._rows:
            raise RuntimeError("No glyph rows fetched from database.")
        # Initialize cache bookkeeping early (needed because pre-raster calls _rasterize)
        self.cache_enabled = self.cfg.cache_size > 0
        self._cache: Dict[int, torch.Tensor] = {}
        self._cache_order: List[int] = []

        # Label frequency audit + optional filtering
        from collections import Counter

        counts = Counter(r.label for r in self._rows)
        # Determine effective min count
        eff_min = self.cfg.min_label_count
        if self.cfg.drop_singletons and eff_min <= 1:
            eff_min = 2
        if eff_min > 1:
            before = len(counts)
            kept_labels = {lab for lab, c in counts.items() if c >= eff_min}
            self._rows = [r for r in self._rows if r.label in kept_labels]
            counts = Counter(r.label for r in self._rows)
            after = len(counts)
            if self.cfg.verbose_stats:
                print(
                    f"[DATASET] Dropped labels with freq < {eff_min}: {before - after} removed (remaining={after})"
                )
        else:
            if self.cfg.verbose_stats:
                print(
                    f"[DATASET] Label frequency audit (no filtering). Unique labels={len(counts)}"
                )
        if self.cfg.verbose_stats:
            rare = sum(1 for c in counts.values() if c == 1)
            print(f"[DATASET] Singleton labels (freq=1) count={rare}")
        # Build label vocab post-filter
        labels = sorted(counts.keys())
        self.label_to_index: Dict[str, int] = {lab: i for i, lab in enumerate(labels)}
        self.index_to_label: List[str] = labels
        # Glyph ordering for resume / reproducibility
        self.glyph_id_order: List[int] = [r.glyph_id for r in self._rows]
        # If adopting existing rows (clone path), skip pre-raster & early exit
        if getattr(self.cfg, "_adopt_rows", False):
            # Cloned dataset shares pre-raster buffers & vocab; ensure placeholders present
            self._preraster_tensor = getattr(self.cfg, "_parent_preraster_tensor", None)
            self._preraster_memmap = getattr(self.cfg, "_parent_preraster_memmap", None)
            self.rasterizer = rasterizer or Rasterizer(
                RasterizerConfig(
                    size=config.image_size,
                    supersample=config.supersample,
                    cubic_subdiv=config.cubic_subdiv,
                    stroke_px=config.stroke_px,
                    fill_closed=config.fill_closed,
                    fit_mode=config.fit_mode,
                    hole_strategy=config.hole_strategy,
                    clip_out_of_bounds=config.clip_out_of_bounds,
                )
            )
            return

        # Initialize rasterizer BEFORE optional pre-rasterization so _rasterize() works
        r_cfg = RasterizerConfig(
            size=config.image_size,
            supersample=config.supersample,
            cubic_subdiv=config.cubic_subdiv,
            stroke_px=config.stroke_px,
            fill_closed=config.fill_closed,
            fit_mode=config.fit_mode,
            hole_strategy=config.hole_strategy,
            clip_out_of_bounds=config.clip_out_of_bounds,
        )
        self.rasterizer = rasterizer or Rasterizer(r_cfg)

        # Optional pre-rasterization (unaugmented)
        self._preraster_tensor: Optional[torch.Tensor] = None
        self._preraster_memmap: Optional[np.memmap] = None
        if self.cfg.pre_rasterize:
            total = len(self._rows)
            H = self.cfg.image_size
            W = self.cfg.image_size
            use_uint8 = self.cfg.pre_raster_dtype == "uint8"
            if self.cfg.pre_raster_mmap:
                # Reuse existing memmap if path exists and shape/dtype match
                mmap_path = self.cfg.pre_raster_mmap_path
                total = len(self._rows)
                H = self.cfg.image_size
                W = self.cfg.image_size
                use_uint8 = self.cfg.pre_raster_dtype == "uint8"
                dtype_np = np.uint8 if use_uint8 else np.float32
                expected_shape = (total, 1, H, W)
                can_reuse = False
                if mmap_path is None:
                    mmap_path = (
                        f"preraster_{total}_{H}x{W}_{'u8' if use_uint8 else 'f32'}.dat"
                    )
                else:
                    # If user supplied path and it exists, probe it
                    if os.path.isfile(mmap_path):
                        try:
                            existing = np.memmap(
                                mmap_path,
                                mode="r",
                                dtype=dtype_np,
                                shape=expected_shape,
                            )
                            # Simple content sanity: check a few bytes not all zero
                            sample_bytes = existing[0, 0, :2, :2].sum()
                            can_reuse = True
                            if self.cfg.verbose_stats:
                                print(
                                    f"[DATASET] Reusing existing preraster memmap '{mmap_path}' sample_sum={float(sample_bytes):.1f}"
                                )
                            self._preraster_memmap = existing
                            # Initialize rasterizer (later in normal path) then return early
                            r_cfg = RasterizerConfig(
                                size=self.cfg.image_size,
                                supersample=self.cfg.supersample,
                                cubic_subdiv=self.cfg.cubic_subdiv,
                                stroke_px=self.cfg.stroke_px,
                                fill_closed=self.cfg.fill_closed,
                                fit_mode=self.cfg.fit_mode,
                                hole_strategy=self.cfg.hole_strategy,
                                clip_out_of_bounds=self.cfg.clip_out_of_bounds,
                            )
                            self.rasterizer = rasterizer or Rasterizer(r_cfg)
                            return
                        except Exception:
                            can_reuse = False
                self.cfg.pre_raster_mmap_path = mmap_path
                # Set up memory-mapped file
                mmap_path = self.cfg.pre_raster_mmap_path
                if mmap_path is None:
                    mmap_path = (
                        f"preraster_{total}_{H}x{W}_{'u8' if use_uint8 else 'f32'}.dat"
                    )
                dtype_np = np.uint8 if use_uint8 else np.float32
                if self.cfg.verbose_stats:
                    print(
                        f"[DATASET] Pre-raster (memmap) path={mmap_path} dtype={dtype_np.__name__}"
                    )
                mm = np.memmap(
                    mmap_path, mode="w+", dtype=dtype_np, shape=(total, 1, H, W)
                )
            else:
                mm = None
                if self.cfg.verbose_stats:
                    print(
                        f"[DATASET] Pre-rasterizing {total} glyphs (in-memory dtype={self.cfg.pre_raster_dtype})..."
                    )
            imgs = [] if not self.cfg.pre_raster_mmap else None
            # Parallel (thread) pre-rasterization if requested
            if self.cfg.pre_raster_workers and self.cfg.pre_raster_workers > 0:
                from concurrent.futures import ThreadPoolExecutor, as_completed

                def _render(row):
                    t_local = self._rasterize(row)
                    if use_uint8:
                        return (t_local.clamp(0, 1) * 255).to(torch.uint8)
                    return t_local.to(torch.float32)

                with ThreadPoolExecutor(max_workers=self.cfg.pre_raster_workers) as ex:
                    futures = {
                        ex.submit(_render, r): i for i, r in enumerate(self._rows)
                    }
                    for fut in as_completed(futures):
                        i = futures[fut]
                        try:
                            arr = fut.result()
                        except Exception:
                            # Fallback blank tensor on failure
                            arr = torch.zeros(
                                1,
                                H,
                                W,
                                dtype=torch.uint8 if use_uint8 else torch.float32,
                            )
                        if mm is not None:
                            mm[i] = arr.cpu().numpy()
                        else:
                            imgs.append(arr)
            else:
                for i, r in enumerate(self._rows):
                    t = self._rasterize(r)
                    if use_uint8:
                        arr = (t.clamp(0, 1) * 255).to(torch.uint8)
                    else:
                        arr = t.to(torch.float32)
                    if mm is not None:
                        mm[i] = arr.cpu().numpy()
                    else:
                        imgs.append(arr)
            if mm is not None:
                mm.flush()
                self._preraster_memmap = mm
            else:
                self._preraster_tensor = torch.stack(imgs, dim=0)
            if self.cfg.verbose_stats:
                if mm is not None:
                    est_mb = (total * H * W * (1 if use_uint8 else 4)) / 1e6
                    print(
                        f"[DATASET] Pre-raster memmap ready ~{est_mb:.1f} MB (lazy OS paging)."
                    )
                else:
                    mb = (
                        self._preraster_tensor.numel()
                        * self._preraster_tensor.element_size()
                        / 1e6
                    )
                    print(
                        f"[DATASET] Pre-raster tensor built: shape={tuple(self._preraster_tensor.shape)} ~{mb:.1f} MB"
                    )

        # Rasterizer already initialized earlier (before pre-rasterization).
        # (No action needed here now.)

        # Optional LRU cache (already initialized earlier; keep for clarity / future re-init)
        # (No change needed here; attributes exist to avoid AttributeError during pre-raster phase)

    def __len__(self) -> int:
        return len(self._rows)

    def get_glyph_meta(self, glyph_id: int) -> Optional[Dict[str, Any]]:
        """
        Return normalization metadata captured by the Rasterizer for a glyph:
        {
          "bbox_orig": [min_x, min_y, max_x, max_y],
          "scale_factor": float,
          "fit_mode": str
        }
        May return None if the glyph has not yet been rasterized in this session.
        """
        return self.rasterizer.meta_by_glyph.get(glyph_id)

    def _cache_put(self, glyph_id: int, tensor: torch.Tensor):
        if not self.cache_enabled:
            return
        if glyph_id in self._cache:
            return
        self._cache[glyph_id] = tensor
        self._cache_order.append(glyph_id)
        # Evict oldest
        if len(self._cache_order) > self.cfg.cache_size:
            old = self._cache_order.pop(0)
            self._cache.pop(old, None)

    def _cache_get(self, glyph_id: int) -> Optional[torch.Tensor]:
        if not self.cache_enabled:
            return None
        return self._cache.get(glyph_id)

    def _rasterize(self, row: GlyphRow) -> torch.Tensor:
        cached = self._cache_get(row.glyph_id)
        if cached is not None:
            return cached.clone()

        t = self.rasterizer.render_glyph(row)
        if t is None:
            # Return blank tensor to avoid crashing training; log once optionally.
            t = torch.zeros(
                1, self.cfg.image_size, self.cfg.image_size, dtype=torch.float32
            )
        else:
            self._cache_put(row.glyph_id, t)
        return t.clone()

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self._rows[idx]
        img = self._rasterize(row)

        if self.cfg.augment:
            # Use a per-sample RNG derived from global seed + glyph_id to keep determinism
            local_rng = random.Random(self.cfg.seed + row.glyph_id)
            img = _augment_tensor(
                img,
                rng=local_rng,
                translate_px=self.cfg.translate_px,
                scale_jitter=self.cfg.scale_jitter,
                contrast_jitter=self.cfg.contrast_jitter,
                gamma_jitter=self.cfg.gamma_jitter,
                blur_prob=self.cfg.blur_prob,
                blur_kernel=self.cfg.blur_kernel,
            )

        label_idx = self.label_to_index[row.label]
        # If pre-rasterized, pull from tensor or memmap (avoids re-render cost)
        if self._preraster_tensor is not None or self._preraster_memmap is not None:
            if not hasattr(self, "_gid_to_pr_idx"):
                self._gid_to_pr_idx = {
                    gid: i for i, gid in enumerate(self.glyph_id_order)
                }
            pr_idx = self._gid_to_pr_idx.get(row.glyph_id, None)
            if pr_idx is not None:
                if self._preraster_tensor is not None:
                    cached_img = self._preraster_tensor[pr_idx]
                    if cached_img.dtype == torch.uint8:
                        img = cached_img.to(torch.float32) / 255.0
                    else:
                        img = cached_img
                else:
                    # Memmap path
                    raw = self._preraster_memmap[pr_idx]  # numpy array
                    if raw.dtype == np.uint8:
                        img = torch.from_numpy(raw).to(torch.float32) / 255.0
                    else:
                        img = torch.from_numpy(raw).to(torch.float32)
                if self.cfg.augment:
                    local_rng = random.Random(self.cfg.seed + row.glyph_id)
                    img = _augment_tensor(
                        img,
                        rng=local_rng,
                        translate_px=self.cfg.translate_px,
                        scale_jitter=self.cfg.scale_jitter,
                        contrast_jitter=self.cfg.contrast_jitter,
                        gamma_jitter=self.cfg.gamma_jitter,
                        blur_prob=self.cfg.blur_prob,
                        blur_kernel=self.cfg.blur_kernel,
                    )

        return {
            "image": img,  # (1,H,W)
            "label": label_idx,
            "glyph_id": row.glyph_id,
            "font_hash": row.font_hash,
            "raw_label": row.label,
        }


# ---------------------------------------------------------------------------
# Splitting Utilities
# ---------------------------------------------------------------------------


def make_train_val_split(
    dataset: GlyphRasterDataset,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> Tuple[GlyphRasterDataset, GlyphRasterDataset]:
    """
    Produce two view-like dataset objects sharing underlying rows but with
    disjoint index lists.

    For simplicity we implement this by shallow copying and replacing _rows.

    NOTE: Label vocab is the same across splits for consistent classification head.
    """
    if not (0.0 < val_fraction < 1.0):
        raise ValueError("val_fraction must be in (0,1).")

    n = len(dataset._rows)
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_size = int(n * val_fraction)
    val_idx = set(indices[:val_size])
    train_rows = [dataset._rows[i] for i in indices[val_size:]]
    val_rows = [dataset._rows[i] for i in indices[:val_size]]

    def _clone_with_rows(rows: List[GlyphRow]) -> GlyphRasterDataset:
        # Build a lightweight cfg clone that signals row adoption
        cfg = dataset.cfg
        adopt_cfg = DatasetConfig(
            db_path=cfg.db_path,
            limit=len(rows),
            randomize_query=False,
            image_size=cfg.image_size,
            supersample=cfg.supersample,
            cubic_subdiv=cfg.cubic_subdiv,
            stroke_px=cfg.stroke_px,
            fill_closed=cfg.fill_closed,
            fit_mode=cfg.fit_mode,
            hole_strategy=cfg.hole_strategy,
            clip_out_of_bounds=cfg.clip_out_of_bounds,
            augment=cfg.augment,
            translate_px=cfg.translate_px,
            scale_jitter=cfg.scale_jitter,
            contrast_jitter=cfg.contrast_jitter,
            gamma_jitter=cfg.gamma_jitter,
            blur_prob=cfg.blur_prob,
            blur_kernel=cfg.blur_kernel,
            cache_size=cfg.cache_size,
            pre_rasterize=False,  # prevent rebuild
            pre_raster_dtype=cfg.pre_raster_dtype,
            pre_raster_mmap=cfg.pre_raster_mmap,
            pre_raster_mmap_path=cfg.pre_raster_mmap_path,
            min_label_count=cfg.min_label_count,
            drop_singletons=False,  # already filtered
            verbose_stats=False,
            enable_margin_targets=cfg.enable_margin_targets,
            seed=cfg.seed,
            label_filter=None,
        )
        # Internal adoption markers
        setattr(adopt_cfg, "_adopt_rows", True)
        setattr(
            adopt_cfg,
            "_parent_preraster_tensor",
            getattr(dataset, "_preraster_tensor", None),
        )
        setattr(
            adopt_cfg,
            "_parent_preraster_memmap",
            getattr(dataset, "_preraster_memmap", None),
        )
        clone = GlyphRasterDataset(adopt_cfg, rasterizer=dataset.rasterizer)
        clone._rows = rows
        clone.label_to_index = dataset.label_to_index
        clone.index_to_label = dataset.index_to_label
        clone.glyph_id_order = [r.glyph_id for r in rows]
        # Share preraster buffers directly (already assigned in constructor via adoption flags)
        # Reset per-clone caches
        clone._cache.clear()
        clone._cache_order.clear()
        return clone

    return _clone_with_rows(train_rows), _clone_with_rows(val_rows)


# ---------------------------------------------------------------------------
# Simple Collate
# ---------------------------------------------------------------------------


def simple_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for DataLoader:
    - Stack images
    - Stack labels
    - Keep metadata lists
    """
    imgs = torch.stack([b["image"] for b in batch], dim=0)  # (B,1,H,W)
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    glyph_ids = [b["glyph_id"] for b in batch]
    font_hashes = [b["font_hash"] for b in batch]
    raw_labels = [b["raw_label"] for b in batch]
    return {
        "images": imgs,
        "labels": labels,
        "glyph_ids": glyph_ids,
        "font_hashes": font_hashes,
        "raw_labels": raw_labels,
    }


# ---------------------------------------------------------------------------
# Self-Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Lightweight sanity (requires existing dataset/glyphs.db)
    import os

    test_db = "dataset/glyphs.db"
    if not os.path.exists(test_db):
        print("[WARN] No dataset/glyphs.db found; self-test skipped.")
    else:
        cfg = DatasetConfig(db_path=test_db, limit=64, augment=True)
        ds = GlyphRasterDataset(cfg)
        item = ds[0]
        print(
            "Sample image shape:",
            item["image"].shape,
            "label:",
            item["label"],
            "glyph_id:",
            item["glyph_id"],
        )
        train_ds, val_ds = make_train_val_split(ds, val_fraction=0.2)
        print("Train/Val sizes:", len(train_ds), len(val_ds))
        batch = simple_collate([train_ds[i] for i in range(4)])
        print("Batch images:", batch["images"].shape, "labels:", batch["labels"])
