#!/usr/bin/env python3
"""
Visualize a sample of glyph contours (normalized) as standalone SVG files.

Purpose
-------
Before integrating the DeepSVG model encoder, we want to confirm:
1. We can successfully load glyph contour data from the SQLite database.
2. Quadratic (qCurveTo) chains are converted appropriately to cubic curves.
3. Normalization (EM-relative, centering, scaling) produces coordinates in a
   consistent range that map cleanly into a visualization canvas.
4. Shapes from multiple fonts (varying UPEM values) appear comparable in size.
5. Orientation is correct (no upside-down rendering after applying flip logic).
6. Include specific categories (diacritics & long ligatures) for relative size inspection.

Key Features
------------
- Random or stratified sampling with guaranteed inclusion of diacritics and ligatures.
- Robust qCurveTo midpoint inference (re-uses parse_contours from contour_parser).
- Normalization using font units-per-em (UPEM) for cross-font consistency.
- Option to disable per-glyph scale-to-unit so relative natural EM sizes are preserved.
- Orientation handling: avoid double vertical inversion when flip_y normalization is used.
- Exports each glyph as an SVG file (one file per glyph) with optional JSON summary.
- Aggregates simple statistics: command counts, cubic conversions, payload length distribution.

Output
------
- SVG files named: glyph_{glyph_id}.svg in --outdir (default: artifacts/vis)
- Optional JSON stats file if --stats-json is provided.

Example
-------
    # Show relative sizes (do NOT scale each glyph to unit box) and include diacritics/ligatures
    python -m src.scripts.visualize_glyphs \
        --db dataset/glyphs.db \
        --limit 30 \
        --outdir artifacts/vis \
        --flip-y \
        --min-diacritics 4 \
        --min-ligatures 6 \
        --no-scale-unit \
        --stats-json artifacts/vis/vis_stats.json

Assumptions
-----------
- The database contains tables: glyphs, fonts.
- glyphs.contours is a JSON string (validated previously).
- fonts table provides `upem` keyed by glyphs.f_id (which matches fonts.file_hash).

Coordinate Mapping
------------------
Normalized contours (after parse_contours + normalize_contours) are in roughly [-1, 1].
We map:
    X_svg = (x + 1)/2 * size
    Y_svg = (1 - (y + 1)/2) * size   (ONLY if we did NOT already flip in normalization)
If flip_y=True during normalization, we skip the additional inversion here to prevent upside-down rendering.

Caveats
-------
This visualization is for qualitative validation, not production artifact quality.
We do not truncate long shapes. Extremely complex glyphs may increase rendering time.

Author
------
Phase 1 scaffolding (enhanced sampling & orientation handling).
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

# Import project contour utilities
try:
    from src.data.contour_parser import (
        parse_contours,
        normalize_contours,
        contour_stats,
        ContourCommand,
        ParsedGlyphContours,
    )
except ImportError as e:  # pragma: no cover
    print(
        "[ERROR] Failed to import contour_parser. Ensure PYTHONPATH includes project root.",
        file=sys.stderr,
    )
    raise


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass
class GlyphRecord:
    glyph_db_id: int
    font_hash: str
    label: str
    contours_raw: str
    upem: Optional[int]
    category: str  # 'generic' | 'diacritic' | 'ligature'


@dataclass
class GlyphVisualizationMeta:
    glyph_db_id: int
    font_hash: str
    label: str
    upem: Optional[int]
    command_total: int
    subpaths: int
    move: int
    line: int
    cubic: int
    close: int
    max_subpath_length: int
    svg_filename: str
    qcurve_payload_hist: Dict[str, int]
    cubic_from_quadratic: int
    normalization_mode: str
    qcurve_mode: str
    category: str
    scaled: bool
    flipped_in_normalization: bool


# ---------------------------------------------------------------------------
# SQLite Helpers
# ---------------------------------------------------------------------------


def open_db(db_path: str) -> sqlite3.Connection:
    dbp = Path(db_path).expanduser().resolve()
    if not dbp.exists():
        raise FileNotFoundError(f"Database not found: {dbp}")
    uri = f"file:{dbp}?mode=ro"
    try:
        return sqlite3.connect(uri, uri=True)
    except sqlite3.OperationalError:
        return sqlite3.connect(str(dbp))


def _fetch_category_samples(
    conn: sqlite3.Connection,
    where_fragment: str,
    limit: int,
    order_by_complexity: bool = False,
) -> List[GlyphRecord]:
    """
    Helper to fetch category-specific samples.
    order_by_complexity: if True, sorts by LENGTH(g.contours) DESC to bias longer shapes.
    """
    if limit <= 0:
        return []
    order_clause = (
        "ORDER BY LENGTH(g.contours) DESC"
        if order_by_complexity
        else "ORDER BY RANDOM()"
    )
    sql = f"""
        SELECT g.id, g.f_id, g.label, g.contours, f.upem
        FROM glyphs g
        JOIN fonts f ON f.file_hash = g.f_id
        WHERE g.contours IS NOT NULL
          AND length(g.contours) > 0
          AND ({where_fragment})
        {order_clause}
        LIMIT ?
    """
    cur = conn.execute(sql, (limit,))
    records: List[GlyphRecord] = []
    for row in cur.fetchall():
        cat = "generic"
        if "_diacritic" in row[2]:
            cat = "diacritic"
        elif "_liga" in row[2]:
            cat = "ligature"
        records.append(
            GlyphRecord(
                glyph_db_id=row[0],
                font_hash=row[1],
                label=row[2],
                contours_raw=row[3],
                upem=row[4],
                category=cat,
            )
        )
    return records


def sample_glyphs_stratified(
    conn: sqlite3.Connection,
    limit: int,
    min_diacritics: int,
    min_ligatures: int,
    randomize: bool,
) -> List[GlyphRecord]:
    """
    Stratified sampling:
    - Fetch diacritics (random).
    - Fetch ligatures (biased toward longer ones for 'long arabic ligatures').
    - Fill remainder with generic glyphs.
    """
    diacritics = _fetch_category_samples(
        conn,
        "g.label LIKE '%_diacritic'",
        min_diacritics,
        order_by_complexity=False,
    )
    ligatures = _fetch_category_samples(
        conn,
        "g.label LIKE '%_liga'",
        min_ligatures,
        order_by_complexity=True,  # prefer longer ligatures
    )

    used_ids: Set[int] = {r.glyph_db_id for r in diacritics + ligatures}
    remaining = max(0, limit - len(used_ids))
    if remaining > 0:
        order_clause = "ORDER BY RANDOM()" if randomize else ""
        sql = (
            f"""
            SELECT g.id, g.f_id, g.label, g.contours, f.upem
            FROM glyphs g
            JOIN fonts f ON f.file_hash = g.f_id
            WHERE g.contours IS NOT NULL
              AND length(g.contours) > 0
              AND g.id NOT IN ({",".join("?" for _ in used_ids)})
            {order_clause}
            LIMIT ?
        """
            if used_ids
            else f"""
            SELECT g.id, g.f_id, g.label, g.contours, f.upem
            FROM glyphs g
            JOIN fonts f ON f.file_hash = g.f_id
            WHERE g.contours IS NOT NULL
              AND length(g.contours) > 0
            {"ORDER BY RANDOM()" if randomize else ""}
            LIMIT ?
        """
        )
        params: List = list(used_ids) + [remaining] if used_ids else [remaining]
        cur = conn.execute(sql, params)
        generics: List[GlyphRecord] = []
        for row in cur.fetchall():
            cat = "generic"
            if "_diacritic" in row[2]:
                cat = "diacritic"
            elif "_liga" in row[2]:
                cat = "ligature"
            generics.append(
                GlyphRecord(
                    glyph_db_id=row[0],
                    font_hash=row[1],
                    label=row[2],
                    contours_raw=row[3],
                    upem=row[4],
                    category=cat,
                )
            )
    else:
        generics = []

    combined = diacritics + ligatures + generics
    # If we somehow overshoot (rare), truncate deterministically
    return combined[:limit]


# ---------------------------------------------------------------------------
# SVG Construction
# ---------------------------------------------------------------------------


def _map_coord(x: float, y: float, size: int, invert_y: bool) -> Tuple[float, float]:
    """
    Map normalized coordinates in [-1, 1] to SVG pixel space (0..size).

    invert_y:
        True  -> apply conventional SVG inversion (top-left origin).
        False -> preserve y as-is (already flipped earlier).
    """
    sx = (x + 1.0) * 0.5 * size
    if invert_y:
        sy = (1.0 - (y + 1.0) * 0.5) * size
    else:
        sy = (y + 1.0) * 0.5 * size
    return sx, sy


def contours_to_svg_path_d(
    parsed: ParsedGlyphContours,
    size: int,
    invert_y: bool,
    fit_mode: str = "unit",
    margin: float = 0.05,
) -> str:
    """
    Convert canonical parsed contours into a single SVG path 'd' attribute.

    Parameters
    ----------
    parsed : ParsedGlyphContours
        Normalized contour commands.
    size : int
        Canvas size (width=height=size).
    invert_y : bool
        Whether to apply SVG Y inversion (only if not already flipped in normalization).
    fit_mode : {"unit","tight"}
        unit  -> assume coordinates already roughly in [-1,1] range (legacy behavior).
        tight -> compute glyph bounding box and scale to fill canvas with given margin.
    margin : float
        Relative margin (0..0.49) applied on each side when fit_mode="tight".
    """
    segments: List[str] = []
    current_point: Optional[Tuple[float, float]] = None

    # Pre-compute tight bbox if needed
    if fit_mode == "tight":
        all_x: List[float] = []
        all_y: List[float] = []
        for sub in parsed:
            for cmd in sub:
                for x, y in cmd.points:
                    all_x.append(x)
                    all_y.append(y)
        if not all_x or not all_y:
            fit_mode = "unit"  # Fallback
        else:
            min_x = min(all_x)
            max_x = max(all_x)
            min_y = min(all_y)
            max_y = max(all_y)
            width = max_x - min_x
            height = max_y - min_y
            if width <= 1e-9 or height <= 1e-9:
                fit_mode = "unit"
            else:
                # Compute scale to fit largest dimension into (1 - 2*margin)*size pixels
                margin = max(0.0, min(0.49, margin))
                usable = (1.0 - 2 * margin) * size
                scale = usable / max(width, height)
                # Offsets so that glyph is centered inside margin
                # After scaling, glyph size in pixels: width*scale or height*scale
                scaled_w = width * scale
                scaled_h = height * scale
                offset_x = (size - scaled_w) / 2.0 - min_x * scale
                offset_y_base = (size - scaled_h) / 2.0 - min_y * scale

                def tight_map(x_val: float, y_val: float) -> Tuple[float, float]:
                    px = x_val * scale + offset_x
                    if invert_y:
                        # invert relative to canvas
                        py = size - (y_val * scale + offset_y_base)
                    else:
                        py = y_val * scale + offset_y_base
                    return px, py

            # Provide closure mapping
            if fit_mode == "tight":

                def mapping(ptx: float, pty: float) -> Tuple[float, float]:
                    return tight_map(ptx, pty)
            else:
                mapping = None
    else:
        mapping = None  # will use default unit mapping below

    def unit_map(x: float, y: float) -> Tuple[float, float]:
        sx = (x + 1.0) * 0.5 * size
        if invert_y:
            sy = (1.0 - (y + 1.0) * 0.5) * size
        else:
            sy = (y + 1.0) * 0.5 * size
        return sx, sy

    coord_map = mapping if mapping is not None else unit_map

    for sub in parsed:
        sub_started = False
        for cmd in sub:
            if cmd.cmd == "m":
                pt = cmd.points[0]
                x, y = coord_map(pt[0], pt[1])
                segments.append(f"M {x:.2f} {y:.2f}")
                current_point = (x, y)
                sub_started = True
            elif cmd.cmd == "l":
                if not sub_started:
                    pt0 = cmd.points[0]
                    mx, my = coord_map(pt0[0], pt0[1])
                    segments.append(f"M {mx:.2f} {my:.2f}")
                    sub_started = True
                    current_point = (mx, my)
                    continue
                pt = cmd.points[0]
                lx, ly = coord_map(pt[0], pt[1])
                segments.append(f"L {lx:.2f} {ly:.2f}")
                current_point = (lx, ly)
            elif cmd.cmd == "c":
                if not sub_started:
                    end_pt = cmd.points[-1]
                    mx, my = coord_map(end_pt[0], end_pt[1])
                    segments.append(f"M {mx:.2f} {my:.2f}")
                    current_point = (mx, my)
                    sub_started = True
                    continue
                c1, c2, p3 = cmd.points
                c1x, c1y = coord_map(c1[0], c1[1])
                c2x, c2y = coord_map(c2[0], c2[1])
                p3x, p3y = coord_map(p3[0], p3[1])
                segments.append(
                    f"C {c1x:.2f} {c1y:.2f} {c2x:.2f} {c2y:.2f} {p3x:.2f} {p3y:.2f}"
                )
                current_point = (p3x, p3y)
            elif cmd.cmd == "z":
                segments.append("Z")
            else:
                continue

    return " ".join(segments)


def build_svg_content(
    path_d: str,
    size: int,
    stroke: str = "none",
    fill: str = "#000000",
    background: Optional[str] = None,
) -> str:
    """
    Produce minimal SVG content with optional background rectangle.
    """
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}">',
    ]
    if background:
        parts.append(
            f'<rect x="0" y="0" width="{size}" height="{size}" fill="{background}" />'
        )
    parts.append(f'<path d="{path_d}" fill="{fill}" stroke="{stroke}" />')
    parts.append("</svg>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Visualization Pipeline
# ---------------------------------------------------------------------------


def visualize_glyphs(
    db_path: str,
    outdir: str,
    limit: int = 20,
    randomize: bool = True,
    size: int = 256,
    qcurve_mode: str = "midpoint",
    flip_y: bool = False,
    em_normalize: bool = True,
    scale_to_unit: bool = True,
    target_range: float = 1.0,
    stats_json: Optional[str] = None,
    min_diacritics: int = 3,
    min_ligatures: int = 3,
    fit_mode: str = "unit",
    margin: float = 0.05,
) -> Dict[str, any]:
    """
    Main execution function returning aggregated statistics.
    """
    Path(outdir).mkdir(parents=True, exist_ok=True)

    conn = open_db(db_path)
    glyphs = sample_glyphs_stratified(
        conn,
        limit=limit,
        min_diacritics=min_diacritics,
        min_ligatures=min_ligatures,
        randomize=randomize,
    )
    if not glyphs:
        raise RuntimeError("No glyph rows fetched; adjust sampling criteria.")

    overall_stats: Dict[str, int] = {}
    metas: List[GlyphVisualizationMeta] = []

    for rec in glyphs:
        qcurve_stats: Dict[str, int] = {}
        try:
            parsed = parse_contours(
                rec.contours_raw,
                qcurve_mode=qcurve_mode,
                qcurve_stats=qcurve_stats,
            )
        except Exception as e:
            print(
                f"[WARN] Failed to parse glyph {rec.glyph_db_id}: {e}", file=sys.stderr
            )
            continue

        norm = normalize_contours(
            parsed,
            upem=rec.upem,
            em_normalize=em_normalize,
            center_origin=True,
            scale_to_unit=scale_to_unit,
            target_range=target_range,
            flip_y=flip_y,
        )

        stats = contour_stats(norm)
        invert_y_for_display = not flip_y  # avoid double inversion if already flipped
        path_d = contours_to_svg_path_d(
            norm,
            size=size,
            invert_y=invert_y_for_display,
            fit_mode=fit_mode,
            margin=margin,
        )
        svg_content = build_svg_content(path_d, size=size)

        filename = f"glyph_{rec.glyph_db_id}.svg"
        filepath = Path(outdir) / filename
        filepath.write_text(svg_content, encoding="utf-8")

        cubic_from_quadratic = qcurve_stats.get("qcurve_segments_to_cubic", 0)
        meta = GlyphVisualizationMeta(
            glyph_db_id=rec.glyph_db_id,
            font_hash=rec.font_hash,
            label=rec.label,
            upem=rec.upem,
            command_total=stats["commands"],
            subpaths=stats["subpaths"],
            move=stats["move"],
            line=stats["line"],
            cubic=stats["cubic"],
            close=stats["close"],
            max_subpath_length=stats["max_subpath_length"],
            svg_filename=filename,
            qcurve_payload_hist={
                k: v
                for k, v in qcurve_stats.items()
                if k.startswith("qcurve_payload_len_")
            },
            cubic_from_quadratic=cubic_from_quadratic,
            normalization_mode=f"em={em_normalize},scale={scale_to_unit},target={target_range}",
            qcurve_mode=qcurve_mode,
            category=rec.category,
            scaled=scale_to_unit,
            flipped_in_normalization=flip_y,
        )
        metas.append(meta)

        for k, v in qcurve_stats.items():
            overall_stats[k] = overall_stats.get(k, 0) + v

    category_counts = {
        "diacritic": sum(1 for m in metas if m.category == "diacritic"),
        "ligature": sum(1 for m in metas if m.category == "ligature"),
        "generic": sum(1 for m in metas if m.category == "generic"),
    }

    aggregate = {
        "sampled_glyphs": len(metas),
        "qcurve_mode": qcurve_mode,
        "visualization_size": size,
        "flip_y_normalization": flip_y,
        "invert_y_in_display": not flip_y,
        "overall_qcurve_stats": overall_stats,
        "category_counts": category_counts,
        "glyphs": [asdict(m) for m in metas],
        "fit_mode": fit_mode,
        "margin": margin,
    }

    if stats_json:
        out_stats_path = Path(stats_json)
        out_stats_path.parent.mkdir(parents=True, exist_ok=True)
        out_stats_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
        print(f"[INFO] Wrote stats JSON: {out_stats_path}")

    return aggregate


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize sample glyph contours as normalized SVGs (with stratified category sampling)."
    )
    p.add_argument(
        "--db", required=True, help="Path to glyphs SQLite DB (e.g., dataset/glyphs.db)"
    )
    p.add_argument(
        "--outdir", default="artifacts/vis", help="Output directory for SVGs."
    )
    p.add_argument(
        "--limit", type=int, default=20, help="Total number of glyphs to visualize."
    )
    p.add_argument(
        "--min-diacritics",
        type=int,
        default=3,
        help="Minimum diacritic glyphs (label contains '_diacritic').",
    )
    p.add_argument(
        "--min-ligatures",
        type=int,
        default=3,
        help="Minimum ligature glyphs (label contains '_liga', biased to longer ones).",
    )
    p.add_argument(
        "--no-random",
        action="store_true",
        help="Disable random sampling for generic pool (still random for diacritics if unspecified).",
    )
    p.add_argument(
        "--size",
        type=int,
        default=256,
        help="Canvas size (square) for output SVG.",
    )
    p.add_argument(
        "--qcurve-mode",
        choices=("midpoint", "naive"),
        default="midpoint",
        help="Quadratic chain handling strategy.",
    )
    p.add_argument(
        "--flip-y",
        action="store_true",
        help="Flip Y axis during normalization (font coordinates are typically y-up).",
    )
    p.add_argument(
        "--no-em-normalize",
        action="store_true",
        help="Disable division by upem before bounding box normalization.",
    )
    p.add_argument(
        "--no-scale-unit",
        action="store_true",
        help="Disable uniform scaling to target range (preserve relative EM size differences).",
    )
    p.add_argument(
        "--target-range",
        type=float,
        default=1.0,
        help="Half-extent after scaling (coords in approx [-R, R]) when fit-mode=unit.",
    )
    p.add_argument(
        "--fit-mode",
        choices=("unit", "tight"),
        default="unit",
        help="unit=legacy normalized mapping; tight=fit glyph bbox to canvas with margin.",
    )
    p.add_argument(
        "--margin",
        type=float,
        default=0.05,
        help="Relative margin (each side) when --fit-mode tight (0..0.49).",
    )
    p.add_argument(
        "--stats-json",
        default=None,
        help="Optional path to write JSON summary stats.",
    )
    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    try:
        visualize_glyphs(
            db_path=args.db,
            outdir=args.outdir,
            limit=args.limit,
            randomize=not args.no_random,
            size=args.size,
            qcurve_mode=args.qcurve_mode,
            flip_y=args.flip_y,
            em_normalize=not args.no_em_normalize,
            scale_to_unit=not args.no_scale_unit,
            target_range=args.target_range,
            stats_json=args.stats_json,
            min_diacritics=args.min_diacritics,
            min_ligatures=args.min_ligatures,
            fit_mode=args.fit_mode,
            margin=args.margin,
        )
    except Exception as e:
        print(f"[ERROR] Visualization failed: {e}", file=sys.stderr)
        return 2
    print("[INFO] Visualization complete.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
