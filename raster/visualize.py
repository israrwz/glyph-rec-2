#!/usr/bin/env python3
"""
visualize.py
============

Utilities to visually inspect raster glyph renderings produced by the raster
pipeline (rasterize.py / dataset.py). Generates composite grids, optional
vector polyline overlays, and simple diagnostics to validate normalization,
centering, and stroke fidelity.

Primary Use Cases
-----------------
1. Quick sanity check after implementing rasterization.
2. Comparing a random sample vs specific glyph IDs.
3. Overlaying extracted polylines (from contours) to ensure curve sampling
   aligns with filled raster result.
4. Saving a grid PNG or opening an interactive matplotlib window (if available).

Example
-------
Generate a 6×8 grid (48 random glyphs) and save as PNG:

    python raster/visualize.py \
        --db dataset/glyphs.db \
        --limit 48 \
        --rows 6 --cols 8 \
        --out raster/artifacts/vis_grid.png

Show interactively (if matplotlib installed) with polyline overlays:

    python raster/visualize.py \
        --db dataset/glyphs.db \
        --limit 32 \
        --show --overlay

Specify explicit glyph IDs:

    python raster/visualize.py \
        --db dataset/glyphs.db \
        --glyph-ids 101,202,303,404 \
        --out raster/artifacts/specific.png

Notes
-----
- Overlays draw sampled polylines (cubic -> polyline expansion) AFTER normalization.
- Rendering here reuses Rasterizer logic; results should match training input.
- This script avoids heavy dependencies; matplotlib is optional.

Author: Raster Phase 1
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont

# Local imports (Phase 1 modules)
try:
    from .rasterize import (
        Rasterizer,
        RasterizerConfig,
        fetch_glyph_rows,
        GlyphRow,
        parse_contours,
        normalize_contours,
        contours_to_polylines,
    )
except ImportError:
    # Fallback allowing: python -m raster.visualize
    from raster.rasterize import (
        Rasterizer,
        RasterizerConfig,
        fetch_glyph_rows,
        GlyphRow,
        parse_contours,
        normalize_contours,
        contours_to_polylines,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_glyph_id_list(arg: Optional[str]) -> Optional[List[int]]:
    if not arg:
        return None
    out: List[int] = []
    for token in arg.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            out.append(int(token))
        except ValueError:
            raise ValueError(f"Invalid glyph id '{token}' in --glyph-ids")
    return out


def select_rows(
    rows: List[GlyphRow],
    limit: int,
    randomize: bool,
    glyph_ids: Optional[List[int]],
    seed: int,
) -> List[GlyphRow]:
    if glyph_ids:
        idset = set(glyph_ids)
        subset = [r for r in rows if r.glyph_id in idset]
        return subset[:limit] if limit > 0 else subset
    if limit <= 0 or limit >= len(rows):
        if randomize:
            rng = random.Random(seed)
            rng.shuffle(rows)
        return rows
    rng = random.Random(seed)
    if randomize:
        rng.shuffle(rows)
    return rows[:limit]


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """
    t: (1,H,W) float32 in [0,1]
    """
    if t.ndim != 3 or t.size(0) != 1:
        raise ValueError("Expected tensor shape (1,H,W)")
    arr = (t[0].clamp(0, 1) * 255).byte().cpu().numpy()
    return Image.fromarray(arr, mode="L")


def draw_overlay_polylines(
    img: Image.Image,
    parsed_contours,
    size: int,
    supersample: int,
    cubic_subdiv: int,
    line_color: Optional[int] = None,
    line_width: int = 1,
):
    """
    Draw polyline overlays (post-normalization). If line_color is None, overlays are disabled (no outlines).
    """
    if line_color is None:
        return
    draw = ImageDraw.Draw(img)
    polylines = contours_to_polylines(parsed_contours, cubic_subdiv=cubic_subdiv)
    render_size = size  # image is already final size
    for poly in polylines:
        if len(poly) < 2:
            continue
        scaled = [
            (
                (x * 0.5 + 0.5) * (render_size - 1),
                (y * -0.5 + 0.5) * (render_size - 1),
            )
            for (x, y) in poly
        ]
        draw.line(scaled, fill=line_color, width=line_width)


def annotate(img: Image.Image, text: str, font: Optional[ImageFont.ImageFont] = None):
    """
    Add a small label in the upper-left corner (no background bar).
    """
    draw = ImageDraw.Draw(img)
    pad = 2
    draw.text((pad, 0), text, fill=255, font=font)


def annotate_bottom_left(
    img: Image.Image, text: str, font: Optional[ImageFont.ImageFont] = None
):
    """
    Add a small label in the bottom-left corner (no background bar).
    Truncates long IDs to keep visuals uncluttered.
    """
    if not text:
        return
    draw = ImageDraw.Draw(img)
    pad = 2
    w, h = img.size
    # Estimate text height
    if font:
        try:
            ascent, descent = font.getmetrics()
            text_h = ascent + descent
        except Exception:
            text_h = 10
    else:
        text_h = 10
    y = h - text_h
    if y < 0:
        y = 0
    draw.text((pad, y), text, fill=255, font=font)


def build_grid(
    images: List[Image.Image],
    rows: int,
    cols: int,
    cell_pad: int = 4,
    background: int = 0,
) -> Image.Image:
    if not images:
        raise ValueError("No images to grid")
    cell_w, cell_h = images[0].size
    grid_w = cols * cell_w + (cols + 1) * cell_pad
    grid_h = rows * cell_h + (rows + 1) * cell_pad
    canvas = Image.new("L", (grid_w, grid_h), color=background)
    i = 0
    for r in range(rows):
        for c in range(cols):
            if i >= len(images):
                break
            x = cell_pad + c * (cell_w + cell_pad)
            y = cell_pad + r * (cell_h + cell_pad)
            canvas.paste(images[i], (x, y))
            i += 1
    return canvas


def try_load_font(size: int = 10) -> Optional[ImageFont.ImageFont]:
    try:
        return ImageFont.load_default()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Visualization Pipeline
# ---------------------------------------------------------------------------


def visualize(
    db_path: str,
    limit: int,
    rows: int,
    cols: int,
    out_path: Optional[str],
    show: bool,
    overlay: bool,
    cubic_subdiv: int,
    supersample: int,
    randomize: bool,
    glyph_ids: Optional[List[int]],
    seed: int,
    size: int,
) -> int:
    glyph_rows = fetch_glyph_rows(
        db_path=db_path,
        limit=max(limit, rows * cols) if limit > 0 else (rows * cols),
        randomize=randomize and (glyph_ids is None),
    )
    if not glyph_rows:
        print("[WARN] No glyph rows fetched.")
        return 0

    selected = select_rows(
        glyph_rows,
        limit=rows * cols if limit <= 0 else min(limit, rows * cols),
        randomize=randomize and (glyph_ids is None),
        glyph_ids=glyph_ids,
        seed=seed,
    )

    r_cfg = RasterizerConfig(size=size, supersample=supersample)
    rasterizer = Rasterizer(r_cfg)
    font = try_load_font()

    images: List[Image.Image] = []
    failures = 0
    for gr in selected:
        tensor = rasterizer.render_glyph(gr)
        if tensor is None:
            failures += 1
            continue
        img = tensor_to_pil(tensor)
        glyph_id_txt = f"{gr.glyph_id}"
        annotate(img, glyph_id_txt, font=font)
        fid_txt = (gr.font_hash or "")[:6]
        if fid_txt:
            annotate_bottom_left(img, fid_txt, font=font)

        if overlay:
            try:
                parsed = parse_contours(gr.contours_json, qcurve_mode="midpoint")
                normed = normalize_contours(
                    parsed,
                    upem=None,
                    em_normalize=True,
                    center_origin=True,
                    scale_to_unit=True,
                    target_range=1.0,
                    flip_y=False,
                )
                # Overlay outlines disabled (removed gray wireframe).
                # draw_overlay_polylines call intentionally removed.
                # If needed for debugging, re-enable with a non-None line_color.
                # draw_overlay_polylines(
                #     img,
                #     normed,
                #     size=size,
                #     supersample=supersample,
                #     cubic_subdiv=cubic_subdiv,
                #     line_color=128,
                #     line_width=1,
                # )
            except Exception as e:
                # Non-fatal
                pass
        images.append(img)

    if not images:
        print("[ERROR] All glyphs failed to rasterize.")
        return 1

    # Ensure we have enough to fill grid; if not, adjust row count
    total = len(images)
    max_cells = rows * cols
    if total < max_cells:
        # Recompute rows to fit available cells
        cols_eff = cols
        rows_eff = math.ceil(total / cols_eff)
        rows = rows_eff

    grid = build_grid(images, rows=rows, cols=cols, cell_pad=4)

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        grid.save(out_path)
        print(f"[INFO] Saved grid image: {out_path}")

    print(
        f"[INFO] Visualized {len(images)} glyphs "
        f"(failures={failures}) size={size} overlay={overlay}"
    )

    if show:
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except ImportError:
            print("[WARN] matplotlib not installed; cannot --show.")
            return 0
        plt.figure(figsize=(cols * 1.5, rows * 1.5))
        plt.axis("off")
        plt.imshow(grid, cmap="gray")
        plt.tight_layout()
        plt.show()

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_argparser():
    ap = argparse.ArgumentParser(
        description="Visualize raster glyph renderings as a grid (optional overlays)."
    )
    ap.add_argument("--db", required=True, help="Path to glyphs.db")
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max glyphs to sample (0 = rows*cols). Ignored when --glyph-ids is used.",
    )
    ap.add_argument("--rows", type=int, default=6, help="Grid rows")
    ap.add_argument("--cols", type=int, default=8, help="Grid columns")
    ap.add_argument("--size", type=int, default=128, help="Raster size (pixels)")
    ap.add_argument(
        "--supersample",
        type=int,
        default=2,
        help="Supersample factor for anti-alias (render at size*factor)",
    )
    ap.add_argument(
        "--overlay",
        action="store_true",
        help="Draw polyline overlays reconstructed from contours.",
    )
    ap.add_argument(
        "--cubic-subdiv",
        type=int,
        default=8,
        help="Cubic Bézier subdivision steps for overlay polylines.",
    )
    ap.add_argument(
        "--glyph-ids",
        type=str,
        default=None,
        help="Comma-separated list of explicit glyph IDs to visualize.",
    )
    ap.add_argument(
        "--no-random",
        action="store_true",
        help="Disable randomization (deterministic ordering by DB).",
    )
    ap.add_argument(
        "--seed", type=int, default=42, help="Seed for random selection (if enabled)."
    )
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output PNG path (if omitted and --show not set, still warns).",
    )
    ap.add_argument("--show", action="store_true", help="Show interactive window.")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = build_argparser()
    args = ap.parse_args(argv)

    glyph_ids = parse_glyph_id_list(args.glyph_ids)
    if glyph_ids and args.limit and args.limit < len(glyph_ids):
        print("[WARN] --limit ignored when --glyph-ids provided (using all IDs).")

    return visualize(
        db_path=args.db,
        limit=args.limit,
        rows=args.rows,
        cols=args.cols,
        out_path=args.out,
        show=args.show,
        overlay=args.overlay,
        cubic_subdiv=args.cubic_subdiv,
        supersample=args.supersample,
        randomize=not args.no_random,
        glyph_ids=glyph_ids,
        seed=args.seed,
        size=args.size,
    )


if __name__ == "__main__":
    raise SystemExit(main())
