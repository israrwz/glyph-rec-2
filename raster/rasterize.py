"""
rasterize.py
============

Glyph contour → raster (grayscale tensor) rendering utilities.

Design Principles
-----------------
- Minimal external configuration (hard-coded sensible defaults).
- Reuse existing contour parsing from `src.data.contour_parser`.
- Support *on-the-fly* rasterization for training without persisting every PNG.
- Optional supersampling + downsampling for anti-aliased output.
- Clean separation:
    * Database access (SQLite fetch)
    * Contour parsing / normalization
    * Curve sampling (cubic to polyline)
    * Polygon & stroke rasterization
    * Batch utilities

Intended Usage (Phase 1)
------------------------
    from raster.rasterize import Rasterizer, rasterize_glyphs_to_tensor

    rst = Rasterizer()
    img = rst.render_glyph(glyph_row)           # single (H, W) float32 tensor
    batch = rasterize_glyphs_to_tensor(rows)    # (B, 1, H, W)

CLI (Light)
-----------
    python -m raster.rasterize --db dataset/glyphs.db --out-dir raster/renders --limit 100

This will dump PNGs (optional) AND a single `raster_tensors.pt` file if --save-tensor is passed.

Database Schema Assumptions
---------------------------
We assume a `glyphs` table with AT LEAST:
    glyph_id (INTEGER / PRIMARY KEY)
    font_hash (TEXT)
    label (TEXT or INTEGER)
    contours (TEXT JSON - list-of-subpaths)

We do not fail hard if extra columns exist. If schema differs, adapt `_DEFAULT_GLYPH_QUERY`.

NOTE: We intentionally avoid importing heavy ML libs at module import time except torch & PIL.

Author: Raster Phase 1
"""

from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image, ImageDraw

# Import parser utilities (existing vector pipeline)
try:
    from src.data.contour_parser import (
        parse_contours,
        normalize_contours,
        ContourCommand,
        ParsedGlyphContours,
    )
except ImportError as e:  # Fallback helpful message
    raise RuntimeError(
        "Failed to import contour parser. Ensure project root is on PYTHONPATH."
    ) from e


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass
class GlyphRow:
    glyph_id: int
    font_hash: str
    label: str
    contours_json: str
    joining_group: Optional[str] = None
    char_class: Optional[str] = None
    upem: Optional[int] = None
    ascent: Optional[int] = None
    descent: Optional[int] = None
    typo_ascent: Optional[int] = None
    typo_descent: Optional[int] = None
    x_height: Optional[int] = None
    cap_height: Optional[int] = None
    line_gap: Optional[int] = None


# ---------------------------------------------------------------------------
# SQLite Access
# ---------------------------------------------------------------------------

_DEFAULT_GLYPH_QUERY = """
SELECT glyph_id, f_id AS font_hash, label, contours, joining_group, char_class
FROM glyphs
WHERE contours IS NOT NULL
LIMIT ?
"""


def fetch_glyph_rows(
    db_path: str,
    limit: int,
    randomize: bool = True,
    label_filter: Optional[Sequence[str]] = None,
) -> List[GlyphRow]:
    """
    Fetch glyph metadata + raw contours JSON.

    Parameters
    ----------
    db_path : str
        Path to SQLite database.
    limit : int
        Max rows to fetch.
    randomize : bool
        If True, ORDER BY RANDOM().
    label_filter : list of labels
        If provided, restrict to these labels.

    Returns
    -------
    List[GlyphRow]
    """
    if limit <= 0:
        return []
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    clauses = ["contours IS NOT NULL"]
    params: List[Any] = []
    if label_filter:
        placeholders = ",".join("?" for _ in label_filter)
        clauses.append(f"label IN ({placeholders})")
        params.extend(label_filter)

    where = " AND ".join(clauses)
    order = "ORDER BY RANDOM()" if randomize else ""
    q = f"""SELECT g.glyph_id,
                   g.f_id AS font_hash,
                   g.label,
                   g.contours,
                   g.joining_group,
                   g.char_class,
                   f.upem,
                   f.ascent,
                   f.descent,
                   f.typo_ascent,
                   f.typo_descent,
                   f.x_height,
                   f.cap_height,
                   f.line_gap
            FROM glyphs g
            LEFT JOIN fonts f ON f.file_hash = g.f_id
            WHERE {where} {order} LIMIT ?"""
    params.append(limit)
    cur.execute(q, params)

    rows: List[GlyphRow] = []
    for r in cur.fetchall():
        rows.append(
            GlyphRow(
                glyph_id=int(r[0]),
                font_hash=str(r[1]),
                label=str(r[2]),
                contours_json=r[3],
                joining_group=str(r[4]) if r[4] is not None else None,
                char_class=str(r[5]) if r[5] is not None else None,
                upem=int(r[6]) if r[6] is not None else None,
                ascent=int(r[7]) if r[7] is not None else None,
                descent=int(r[8]) if r[8] is not None else None,
                typo_ascent=int(r[9]) if r[9] is not None else None,
                typo_descent=int(r[10]) if r[10] is not None else None,
                x_height=int(r[11]) if r[11] is not None else None,
                cap_height=int(r[12]) if r[12] is not None else None,
                line_gap=int(r[13]) if r[13] is not None else None,
            )
        )
    conn.close()
    return rows


# ---------------------------------------------------------------------------
# Curve Sampling / Geometry Helpers
# ---------------------------------------------------------------------------


def _sample_cubic(p0, p1, p2, p3, n: int = 8) -> List[Tuple[float, float]]:
    """
    Uniformly sample a cubic Bézier into n segments (n+1 points including endpoints).
    """
    pts = []
    for i in range(n + 1):
        t = i / n
        mt = 1 - t
        x = (
            mt * mt * mt * p0[0]
            + 3 * mt * mt * t * p1[0]
            + 3 * mt * t * t * p2[0]
            + t * t * t * p3[0]
        )
        y = (
            mt * mt * mt * p0[1]
            + 3 * mt * mt * t * p1[1]
            + 3 * mt * t * t * p2[1]
            + t * t * t * p3[1]
        )
        pts.append((x, y))
    return pts


def contours_to_polylines(
    parsed: ParsedGlyphContours,
    cubic_subdiv: int = 8,
    force_close_threshold: float = 0.0,
) -> List[List[Tuple[float, float]]]:
    """
    Convert canonical contour commands to polylines (list of list of (x,y)).
    Each subpath becomes a polyline; closed paths will repeat the start point at end.

    We treat consecutive 'c' commands as connected segments.
    """
    polylines: List[List[Tuple[float, float]]] = []
    for sub in parsed:
        current: List[Tuple[float, float]] = []
        pen: Optional[Tuple[float, float]] = None
        for cmd in sub:
            if cmd.cmd == "m":
                if current:
                    polylines.append(current)
                    current = []
                pen = cmd.points[0]
                current.append(pen)
            elif cmd.cmd == "l":
                pt = cmd.points[0]
                if pen is None:
                    pen = pt
                    current.append(pt)
                else:
                    current.append(pt)
                pen = pt
            elif cmd.cmd == "c":
                if pen is None:
                    # Skip malformed
                    continue
                c1, c2, p3 = cmd.points
                samples = _sample_cubic(pen, c1, c2, p3, n=cubic_subdiv)
                # Avoid duplicating first sample (already pen)
                current.extend(samples[1:])
                pen = p3
            elif cmd.cmd == "z":
                # Close path: add first point again if not present
                if current and current[0] != current[-1]:
                    current.append(current[0])
            else:
                continue
        if current:
            # If path not explicitly closed but endpoints are within threshold, close it
            if (
                force_close_threshold > 0.0
                and len(current) > 2
                and current[0] != current[-1]
            ):
                dx = current[0][0] - current[-1][0]
                dy = current[0][1] - current[-1][1]
                if (dx * dx + dy * dy) <= force_close_threshold * force_close_threshold:
                    current.append(current[0])
            polylines.append(current)
    return polylines


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_polylines_to_image(
    polylines: List[List[Tuple[float, float]]],
    size: int = 128,
    supersample: int = 2,
    stroke: int = 0,
    fill_closed: bool = True,
    background: int = 0,
    hole_strategy: str = "orientation",
    clip_out_of_bounds: bool = True,
) -> Image.Image:
    """
    Render polylines into a grayscale Pillow Image with optional hole preservation.

    Hole Strategies:
        orientation (default):
            - Determine dominant winding orientation by cumulative signed area.
            - Fill polygons with dominant orientation; treat opposite orientation as holes.
        even-odd:
            - XOR parity fill (classic even-odd rule). Each closed ring toggles filled state,
              handling arbitrary nesting depth (holes within holes within solids).
        none:
            - Legacy: fill all closed polygons (no hole carving).

    Coordinate Assumptions:
        Polylines expected roughly in [-1,1] after normalization if tight scaling.
        In preserve mode some coordinates may exceed this range; we optionally
        clip them to image bounds.
    """
    from PIL import ImageChops

    render_size = size * supersample
    img = Image.new("L", (render_size, render_size), color=background)
    draw = ImageDraw.Draw(img)

    def _signed_area(poly: List[Tuple[float, float]]) -> float:
        if len(poly) < 3:
            return 0.0
        s = 0.0
        for i in range(len(poly) - 1):
            x1, y1 = poly[i]
            x2, y2 = poly[i + 1]
            s += x1 * y2 - x2 * y1
        return 0.5 * s

    # Preprocess: classify closed vs open (with closure epsilon)
    closed_polys: List[Tuple[List[Tuple[float, float]], float]] = []
    open_polys: List[List[Tuple[float, float]]] = []
    close_eps_sq = 1e-6  # (~1e-3 distance tolerance before scaling to pixels)
    for poly in polylines:
        if len(poly) < 2:
            continue
        x0, y0 = poly[0]
        xn, yn = poly[-1]
        if (x0 - xn) * (x0 - xn) + (y0 - yn) * (y0 - yn) <= close_eps_sq:
            # Treat as closed; ensure explicit closure by duplicating start if needed
            if poly[-1] != poly[0]:
                poly = poly + [poly[0]]
            closed_polys.append((poly, _signed_area(poly)))
        else:
            open_polys.append(poly)

    # Determine dominant orientation (sum absolute area grouped by sign)
    solid_orientation = None
    if hole_strategy == "orientation" and closed_polys:
        pos_sum = sum(abs(a) for (_, a) in closed_polys if a > 0)
        neg_sum = sum(abs(a) for (_, a) in closed_polys if a < 0)
        solid_orientation = 1 if pos_sum >= neg_sum else -1

    def _scale_and_clip(seq: List[Tuple[float, float]]):
        out = []
        for x, y in seq:
            sx = (x * 0.5 + 0.5) * (render_size - 1)
            sy = (y * -0.5 + 0.5) * (render_size - 1)
            if clip_out_of_bounds:
                sx = max(0, min(render_size - 1, sx))
                sy = max(0, min(render_size - 1, sy))
            out.append((sx, sy))
        return out

    # Filling logic
    if fill_closed and closed_polys:
        if hole_strategy == "even-odd":
            # Parity (XOR) fill (use 1-bit mask so logical_xor is valid)
            mask = Image.new("1", (render_size, render_size), color=0)
            for poly, _ in closed_polys:
                scaled = _scale_and_clip(poly)
                tmp = Image.new("1", (render_size, render_size), color=0)
                ImageDraw.Draw(tmp).polygon(scaled, fill=1)
                mask = ImageChops.logical_xor(mask, tmp)
            # Apply mask (white where parity == 1)
            img.paste(255, mask=mask)
        elif hole_strategy == "orientation":
            # Fill solids
            for poly, area in closed_polys:
                if solid_orientation is None:
                    scaled = _scale_and_clip(poly)
                    draw.polygon(scaled, fill=255)
                else:
                    is_solid = (area >= 0 and solid_orientation == 1) or (
                        area < 0 and solid_orientation == -1
                    )
                    if is_solid:
                        scaled = _scale_and_clip(poly)
                        draw.polygon(scaled, fill=255)
            # Carve holes
            if solid_orientation is not None:
                for poly, area in closed_polys:
                    is_hole = (area >= 0 and solid_orientation == -1) or (
                        area < 0 and solid_orientation == 1
                    )
                    if is_hole:
                        scaled = _scale_and_clip(poly)
                        draw.polygon(scaled, fill=background)
        else:  # "none" or unrecognized -> legacy fill all
            for poly, _ in closed_polys:
                scaled = _scale_and_clip(poly)
                draw.polygon(scaled, fill=255)

    # Strokes
    if stroke > 0:
        for poly in open_polys:
            scaled = _scale_and_clip(poly)
            draw.line(scaled, fill=255, width=stroke, joint="curve")
        if fill_closed is False:
            for poly, _ in closed_polys:
                scaled = _scale_and_clip(poly)
                draw.line(scaled, fill=255, width=stroke, joint="curve")

    if supersample > 1:
        # Pillow >=9 prefers Image.Resampling; fallback for older versions
        try:
            resample_filter = Image.Resampling.BICUBIC  # type: ignore[attr-defined]
        except AttributeError:
            resample_filter = Image.BICUBIC  # type: ignore[attr-defined]
        img = img.resize((size, size), resample=resample_filter)
    return img


def render_contours_to_tensor(
    parsed: ParsedGlyphContours,
    size: int = 128,
    supersample: int = 2,
    cubic_subdiv: int = 8,
    stroke_px: int = 0,
    fill_closed: bool = True,
    hole_strategy: str = "orientation",
    clip_out_of_bounds: bool = True,
    fit_mode: str = "tight",
    force_close_threshold: float = 0.0,
) -> torch.Tensor:
    """
    High-level: parsed contours -> raster tensor (1, H, W), float32 in [0,1].
    """
    polylines = contours_to_polylines(
        parsed,
        cubic_subdiv=cubic_subdiv,
        force_close_threshold=force_close_threshold,
    )
    img = _render_polylines_to_image(
        polylines,
        size=size,
        supersample=supersample,
        stroke=stroke_px,
        fill_closed=fill_closed,
        hole_strategy=hole_strategy,
        clip_out_of_bounds=clip_out_of_bounds,
    )
    import numpy as _np

    t = torch.from_numpy(_np.array(img, dtype=_np.uint8))
    t = t.unsqueeze(0).float() / 255.0  # (1,H,W)
    return t


# ---------------------------------------------------------------------------
# Rasterizer Class
# ---------------------------------------------------------------------------


@dataclass
class RasterizerConfig:
    # Core raster dimensions
    size: int = 128
    supersample: int = 2
    cubic_subdiv: int = 8
    stroke_px: int = 0
    fill_closed: bool = True
    # Normalization flags
    em_normalize: bool = True
    # Backward-compat flag (used only if fit_mode == "tight")
    scale_to_unit: bool = True
    center_origin: bool = True
    flip_y: bool = False  # We'll flip in rendering mapping anyway
    target_range: float = 1.0
    # Scaling policy:
    #   "tight"    -> scale glyph to fill target_range (legacy)
    #   "preserve" -> do NOT scale to unit; only center (retain relative EM size)
    fit_mode: str = "preserve"
    # Hole preservation strategy:
    #   "orientation" -> dominant winding = solid, opposite = hole
    #   "even-odd"    -> parity (XOR) fill for arbitrary nesting depth
    #   "none"        -> fill all (no holes)
    hole_strategy: str = "orientation"
    # Safety: clip coordinates after mapping to pixel grid
    clip_out_of_bounds: bool = True
    # New: tolerance for auto-closing nearly closed paths (in normalized coord units)
    force_close_threshold: float = 0.0001


class Rasterizer:
    """
    Stateful helper (could later hold caches, metrics).
    Tracks per-glyph normalization metadata (original bbox + applied scale factor).
    """

    def __init__(self, cfg: Optional[RasterizerConfig] = None):
        self.cfg = cfg or RasterizerConfig()
        # glyph_id -> {"bbox_orig":[min_x,min_y,max_x,max_y], "scale_factor":float, "fit_mode":str}
        self.meta_by_glyph: Dict[int, Dict[str, Any]] = {}
        # Per-font running stats (lightweight) for preserve mode scaling.
        # font_hash -> {"avg_h": float, "count": int}
        self.font_stats: Dict[str, Dict[str, float]] = {}

    def parse_and_normalize(self, glyph_row: GlyphRow) -> Optional[ParsedGlyphContours]:
        """
        Parse raw JSON contours and apply normalization while capturing
        original (pre-center/scale) bounding box and scale factor used.
        """
        try:
            parsed = parse_contours(glyph_row.contours_json, qcurve_mode="midpoint")
        except Exception:
            return None
        if not parsed:
            return None

        # Gather original scaled (EM + flip) coordinates to compute bbox & scale factor
        xs: List[float] = []
        ys: List[float] = []
        for sub in parsed:
            for cmd in sub:
                for x, y in cmd.points:
                    # Apply optional EM normalization here so bbox reflects final relative scale
                    if self.cfg.em_normalize and glyph_row.upem and glyph_row.upem > 0:
                        x_norm = x / glyph_row.upem
                        y_norm = y / glyph_row.upem
                    else:
                        x_norm, y_norm = x, y
                    xf = x_norm
                    yf = -y_norm if self.cfg.flip_y else y_norm
                    xs.append(xf)
                    ys.append(yf)
        if not xs or not ys:
            return None
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        width = max_x - min_x
        height = max_y - min_y
        max_dim = max(width, height) if max(width, height) > 1e-6 else 1.0
        # Estimate provisional ascent/descent for font stats (after EM normalization but before any scaling).
        # We approximate ascent as max_y if baseline is at 0 (some glyphs may not cross baseline).
        # If glyph has negative y (descender), descent is -min_y else a small epsilon to keep ratio stable.
        est_ascent = max_y
        est_descent = -min_y if min_y < 0 else 0.0
        if est_ascent < 1e-6 and max_dim > 0:
            est_ascent = max_dim  # fallback
        # Avoid zero descent (keeps baseline placement numerically stable)
        if est_descent < 1e-6:
            est_descent = max_dim * 0.2  # heuristic small descent if absent

        # New unified scaling logic:
        # tight:
        #   - Keep legacy behavior (independent per-glyph unit scaling to fill target range).
        # preserve (default intent):
        #   - EM normalize (already applied via upem in bbox calc).
        #   - Collect a running per-font average height in EM space.
        #   - Compute a prospective global font scale so the running average occupies
        #     ~fill_fraction (0.85) of vertical span (2*target_range).
        #   - Never up-scale a glyph above 1.0 global font scale (retain relative size across fonts).
        #   - Down-scale only if glyph (after global scaling) still overflows.
        #   - Do NOT center-origin; instead baseline-align (baseline at y=0) with a fixed bottom margin.
        fill_fraction = 0.85
        bottom_margin_frac = 0.05  # of full vertical span (2*target_range)
        full_span = 2.0 * self.cfg.target_range
        scale_factor = 1.0
        baseline_shift = 0.0
        font_global_scale = 1.0
        overflow_downscale = 1.0
        metric_source = "unknown"
        metric_fallback_used = False
        if self.cfg.fit_mode == "tight":
            # Legacy path
            scale_factor = (full_span / max_dim) if max_dim > 0 else 1.0
            normed = normalize_contours(
                parsed,
                upem=glyph_row.upem,
                em_normalize=self.cfg.em_normalize,
                center_origin=self.cfg.center_origin,
                scale_to_unit=True,
                target_range=self.cfg.target_range,
                flip_y=self.cfg.flip_y,
            )
            font_global_scale = scale_factor
            metric_source = "tight"
        else:
            # Preserve mode path using DB font metrics + glyph ratio compensation.
            def _n(val):
                if val is None:
                    return None
                if glyph_row.upem and glyph_row.upem > 0:
                    return float(val) / glyph_row.upem
                return float(val)

            # Prefer typo metrics, then hhea
            m_ascent = _n(glyph_row.typo_ascent) or _n(glyph_row.ascent) or est_ascent
            raw_descent = (
                glyph_row.typo_descent
                if glyph_row.typo_descent is not None
                else glyph_row.descent
            )
            m_descent = abs(_n(raw_descent)) if raw_descent is not None else est_descent
            metric_span = m_ascent + m_descent if (m_ascent and m_descent) else height
            metric_source = "typo_or_hhea"
            raw_bbox_h = height
            # Heuristic fallback: if typo ascent is implausibly small compared to glyph bbox
            # and a much larger ascent is available, fallback to hhea ascent/descent.
            if (
                glyph_row.typo_ascent is not None
                and glyph_row.ascent is not None
                and glyph_row.ascent > glyph_row.typo_ascent * 2
                and raw_bbox_h > (m_ascent or 0) * 2
            ):
                alt_ascent = _n(glyph_row.ascent)
                alt_descent = (
                    abs(_n(glyph_row.descent))
                    if glyph_row.descent is not None
                    else m_descent
                )
                if (
                    alt_ascent
                    and alt_descent
                    and (alt_ascent + alt_descent) > (m_ascent + m_descent)
                ):
                    m_ascent = alt_ascent
                    m_descent = alt_descent
                    metric_span = m_ascent + m_descent
                    metric_fallback_used = True
                    metric_source = "hhea_fallback"
            # Secondary guard: if metric_span much smaller than bbox height, clamp using bbox
            if metric_span < raw_bbox_h * 0.5:
                # Inflate metric span minimally to avoid extreme global scale
                metric_span = max(metric_span, raw_bbox_h * 0.8)
                metric_fallback_used = True
                if metric_source == "typo_or_hhea":
                    metric_source = "bbox_clamp"
            desired_span = fill_fraction * full_span
            font_global_scale = (
                desired_span / metric_span if metric_span > 1e-6 else 1.0
            )
            # Normalize WITHOUT centering/scaling
            normed = normalize_contours(
                parsed,
                upem=glyph_row.upem,
                em_normalize=self.cfg.em_normalize,
                center_origin=False,
                scale_to_unit=False,
                target_range=self.cfg.target_range,
                flip_y=self.cfg.flip_y,
            )
            # Apply global scale
            if abs(font_global_scale - 1.0) > 1e-9:
                scaled_normed = []
                for sub in normed:
                    new_sub = []
                    for cmd in sub:
                        pts = tuple(
                            (x * font_global_scale, y * font_global_scale)
                            for x, y in cmd.points
                        )
                        new_sub.append(ContourCommand(cmd.cmd, pts))
                    scaled_normed.append(new_sub)
                normed = scaled_normed
            # Baseline shift so descent sits above bottom margin
            baseline_shift = (
                -self.cfg.target_range
                + bottom_margin_frac * full_span
                + (m_descent * font_global_scale)
            )
            if abs(baseline_shift) > 1e-9:
                shifted = []
                for sub in normed:
                    new_sub = []
                    for cmd in sub:
                        pts = tuple((x, y + baseline_shift) for x, y in cmd.points)
                        new_sub.append(ContourCommand(cmd.cmd, pts))
                    shifted.append(new_sub)
                normed = shifted
            # -------------------------------
            # Ratio-based compensation (Option A)
            # -------------------------------
            # Compute glyph bbox after global scale + baseline shift (pre-overflow clamp)
            gxs, gys = [], []
            for sub in normed:
                for cmd in sub:
                    for x, y in cmd.points:
                        gxs.append(x)
                        gys.append(y)
            if gxs and gys and metric_span > 1e-6:
                g_min_y, g_max_y = min(gys), max(gys)
                glyph_h = g_max_y - g_min_y
                # Effective pre-scale metric span in pixel-normalized units ~ font_global_scale * metric_span
                scaled_metric_span = font_global_scale * metric_span
                ratio = (
                    glyph_h / scaled_metric_span if scaled_metric_span > 1e-9 else 1.0
                )
                small_mark_thresh = 0.20
                base_mid_thresh = 0.60
                comp_k = 0.5
                comp_scale_max = 1.35
                comp_scale = 1.0
                if ratio >= small_mark_thresh and ratio < base_mid_thresh:
                    # Linear ramp
                    comp_scale = 1.0 + comp_k * (
                        (base_mid_thresh - ratio)
                        / (base_mid_thresh - small_mark_thresh)
                    )
                    if comp_scale > comp_scale_max:
                        comp_scale = comp_scale_max
                # Apply compensation scaling (maintain baseline)
                if abs(comp_scale - 1.0) > 1e-9:
                    normed_comp = []
                    for sub in normed:
                        new_sub = []
                        for cmd in sub:
                            pts = tuple(
                                (x * comp_scale, y * comp_scale) for x, y in cmd.points
                            )
                            new_sub.append(ContourCommand(cmd.cmd, pts))
                        normed_comp.append(new_sub)
                    normed = normed_comp
                    font_global_scale *= comp_scale  # reflect in effective scale
                    # Recompute glyph bbox for centering compensation
                    gxs2, gys2 = [], []
                    for sub in normed:
                        for cmd in sub:
                            for x, y in cmd.points:
                                gxs2.append(x)
                                gys2.append(y)
                    if gys2:
                        g_min_y2, g_max_y2 = min(gys2), max(gys2)
                        glyph_h2 = g_max_y2 - g_min_y2
                        # Vertical centering for short baseline glyphs (Option C)
                        center_thresh = 0.45
                        if ratio < center_thresh and ratio >= small_mark_thresh:
                            remaining = scaled_metric_span * comp_scale - glyph_h2
                            if remaining > 1e-9:
                                center_shift_fraction = 0.40
                                delta = remaining * center_shift_fraction * 0.5
                                # Shift downward (negative y) by half the chosen delta to visually center
                                centered = []
                                for sub in normed:
                                    new_sub = []
                                    for cmd in sub:
                                        pts = tuple(
                                            (x, y - delta) for x, y in cmd.points
                                        )
                                        new_sub.append(ContourCommand(cmd.cmd, pts))
                                    centered.append(new_sub)
                                normed = centered
                # End ratio compensation
            # Overflow clamp (vertical/horizontal)
            post_xs, post_ys = [], []
            for sub in normed:
                for cmd in sub:
                    for x, y in cmd.points:
                        post_xs.append(x)
                        post_ys.append(y)
            if post_xs and post_ys:
                p_min_y, p_max_y = min(post_ys), max(post_ys)
                p_min_x, p_max_x = min(post_xs), max(post_xs)
                p_vert_extent = max(p_max_y, -p_min_y)
                p_horiz_extent = max(p_max_x, -p_min_x)
                v_down = (
                    self.cfg.target_range / p_vert_extent
                    if p_vert_extent > self.cfg.target_range
                    else 1.0
                )
                h_down = (
                    self.cfg.target_range / p_horiz_extent
                    if p_horiz_extent > self.cfg.target_range
                    else 1.0
                )
                overflow_downscale = min(v_down, h_down)
                if overflow_downscale < 1.0:
                    clamped = []
                    for sub in normed:
                        new_sub = []
                        for cmd in sub:
                            pts = tuple(
                                (x * overflow_downscale, y * overflow_downscale)
                                for x, y in cmd.points
                            )
                            new_sub.append(ContourCommand(cmd.cmd, pts))
                        clamped.append(new_sub)
                    normed = clamped
            scale_factor = font_global_scale * overflow_downscale
            ascent = m_ascent * font_global_scale
            descent = m_descent * font_global_scale

        # Store metadata
        # Ensure compensation metadata locals exist (safe defaults if block not triggered)
        if "ratio" not in locals():
            ratio = None
        if "comp_scale" not in locals():
            comp_scale = 1.0
        if "delta" not in locals():
            delta = 0.0
        self.meta_by_glyph[glyph_row.glyph_id] = {
            "bbox_orig": [min_x, min_y, max_x, max_y],
            "scale_factor": scale_factor,
            "fit_mode": self.cfg.fit_mode,
            "overflow_downscale_applied": (
                self.cfg.fit_mode == "preserve" and scale_factor != 1.0
            ),
            "upem": glyph_row.upem,
            "font_global_scale": font_global_scale,
            "baseline_shift": baseline_shift,
            "overflow_downscale": overflow_downscale,
            "used_ascent": glyph_row.typo_ascent or glyph_row.ascent,
            "used_descent": glyph_row.typo_descent or glyph_row.descent,
            "final_ascent": ascent if self.cfg.fit_mode != "tight" else None,
            "final_descent": descent if self.cfg.fit_mode != "tight" else None,
            # Compensation diagnostics
            "ratio_compensation": ratio,
            "comp_scale": comp_scale,
            "center_shift": delta,
            # Metric source / fallback diagnostics
            "metric_source": metric_source,
            "metric_fallback": metric_fallback_used,
        }
        return normed

    def render_glyph(self, glyph_row: GlyphRow) -> Optional[torch.Tensor]:
        """
        Full pipeline: parse -> normalize -> raster -> tensor (1,H,W).
        Returns None on failure.
        """
        parsed = self.parse_and_normalize(glyph_row)
        if parsed is None:
            return None
        return render_contours_to_tensor(
            parsed,
            size=self.cfg.size,
            supersample=self.cfg.supersample,
            cubic_subdiv=self.cfg.cubic_subdiv,
            stroke_px=self.cfg.stroke_px,
            fill_closed=self.cfg.fill_closed,
            hole_strategy=self.cfg.hole_strategy,
            clip_out_of_bounds=self.cfg.clip_out_of_bounds,
            fit_mode=self.cfg.fit_mode,
        )


# ---------------------------------------------------------------------------
# Batch Utilities
# ---------------------------------------------------------------------------


def rasterize_glyphs_to_tensor(
    rows: Sequence[GlyphRow],
    rasterizer: Optional[Rasterizer] = None,
    drop_failures: bool = True,
) -> Tuple[torch.Tensor, List[int]]:
    """
    Rasterize a batch of glyph rows into a single tensor.

    Parameters
    ----------
    rows : list[GlyphRow]
    rasterizer : Rasterizer
        If None, a default rasterizer config is used.
    drop_failures : bool
        If True, silently skip failed glyphs; else raise RuntimeError.

    Returns
    -------
    tensor : torch.Tensor
        Shape (B, 1, H, W)
    kept_ids : List[int]
        glyph_ids for the rows successfully rasterized.
    """
    r = rasterizer or Rasterizer()
    tensors: List[torch.Tensor] = []
    ids: List[int] = []
    for gr in rows:
        t = r.render_glyph(gr)
        if t is None:
            if drop_failures:
                continue
            raise RuntimeError(f"Failed to rasterize glyph_id={gr.glyph_id}")
        tensors.append(t)
        ids.append(gr.glyph_id)
    if not tensors:
        return torch.empty(0, 1, r.cfg.size, r.cfg.size), []
    batch = torch.stack(tensors, dim=0)
    return batch, ids


# ---------------------------------------------------------------------------
# Optional Saving
# ---------------------------------------------------------------------------


def save_pngs(
    tensor: torch.Tensor,
    glyph_ids: Sequence[int],
    out_dir: str,
    prefix: str = "glyph",
):
    """
    Save each raster (assumes shape (B,1,H,W)) as an 8-bit PNG.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for i, gid in enumerate(glyph_ids):
        arr = (tensor[i, 0].clamp(0, 1) * 255).byte().cpu().numpy()
        img = Image.fromarray(arr, mode="L")
        img.save(Path(out_dir) / f"{prefix}_{gid}.png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser():
    ap = argparse.ArgumentParser(
        description="Rasterize glyph contours from glyphs.db into grayscale tensors / PNGs."
    )
    ap.add_argument("--db", required=True, help="Path to glyphs.db SQLite")
    ap.add_argument("--limit", type=int, default=256, help="Number of glyphs to sample")
    ap.add_argument("--no-random", action="store_true", help="Deterministic order")
    ap.add_argument(
        "--out-dir", type=str, default=None, help="If set, write PNG images here"
    )
    ap.add_argument(
        "--save-tensor",
        type=str,
        default=None,
        help="If set, path to save stacked tensor (.pt)",
    )
    ap.add_argument("--size", type=int, default=128, help="Output raster size")
    ap.add_argument(
        "--supersample",
        type=int,
        default=2,
        help="Supersample factor for anti-alias (render at size*factor)",
    )
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = _build_argparser()
    args = ap.parse_args(argv)

    rows = fetch_glyph_rows(
        db_path=args.db,
        limit=args.limit,
        randomize=not args.no_random,
    )
    if not rows:
        print("[WARN] No glyph rows fetched.")
        return 0

    cfg = RasterizerConfig(size=args.size, supersample=args.supersample)
    rasterizer = Rasterizer(cfg)
    batch, ids = rasterize_glyphs_to_tensor(rows, rasterizer=rasterizer)
    print(
        f"[INFO] Rasterized {len(ids)}/{len(rows)} glyphs -> tensor {tuple(batch.shape)}"
    )

    if args.out_dir:
        save_pngs(batch, ids, args.out_dir)
        print(f"[INFO] Saved PNGs to {args.out_dir}")

    if args.save_tensor:
        outp = Path(args.save_tensor)
        outp.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"tensor": batch, "glyph_ids": ids}, outp)
        print(f"[INFO] Saved tensor to {args.save_tensor}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
