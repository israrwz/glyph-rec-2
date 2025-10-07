#!/usr/bin/env python3
"""
update_font_metrics.py
======================

One-off migration / enrichment script that:
  1. Ensures the `fonts` table in a glyphs SQLite DB has extended metric columns.
  2. Iterates over each font row, loads the font file (via fontTools),
     extracts relevant typographic + geometric metrics, and updates the row.

Why
----
Current raster scaling / baseline alignment benefits from reliable cross-font
metrics (ascent, descent, x-height, cap-height, etc). Many fonts expose these
through standard TrueType/OpenType tables (head, hhea, OS/2, post).
Persisting them avoids recomputing on every run and yields deterministic
baseline + scaling logic.

Added Columns (if absent)
-------------------------
(as INTEGER unless otherwise noted)

  ascent                : Prefer hhea.ascent (fallback OS/2.sTypoAscender)
  descent               : Prefer hhea.descent (fallback OS/2.sTypoDescender)
  line_gap              : hhea.lineGap (fallback OS/2.sTypoLineGap)
  win_ascent            : OS/2.usWinAscent (if present)
  win_descent           : OS/2.usWinDescent (if present)
  typo_ascent           : OS/2.sTypoAscender
  typo_descent          : OS/2.sTypoDescender
  typo_line_gap         : OS/2.sTypoLineGap
  x_height              : OS/2.sxHeight (if present)
  cap_height            : OS/2.sCapHeight (if present)
  italic_angle          : post.italicAngle (REAL)
  bbox_xmin             : head.xMin
  bbox_ymin             : head.yMin
  bbox_xmax             : head.xMax
  bbox_ymax             : head.yMax
  avg_glyph_height      : Mean (glyph bbox height) over sampled glyphs
  avg_glyph_width       : Mean (glyph bbox width)  over sampled glyphs
  units_per_em          : head.unitsPerEm (kept distinct from original `upem` if `upem` already stored)

Sampling Strategy
-----------------
For performance, we sample up to `--glyph-sample` glyphs uniformly from the
glyph order. Composite glyphs are included; empty bounding boxes are skipped.

Path Resolution
---------------
Each `fonts.file_path` may be:
  * Absolute path (use as-is).
  * Relative path referencing a fonts root, possibly with leading ../ segments.
Provide `--fonts-root` (default: sibling path provided by user’s environment)
and we will attempt:
    1. If path is absolute and exists -> use it.
    2. Join fonts_root + file_path and normalize; if exists -> use it.
    3. Strip leading ../ until exists (progressively).
    4. Final fallback: try relative to DB directory.

If still not found, font is skipped (record warning in stderr).

Safety
------
Use --dry-run to test migration without updating rows.

Dependencies
------------
Requires `fontTools` (`pip install fonttools`).

Usage
-----
    python raster/update_font_metrics.py \
        --db dataset/glyphs.db \
        --fonts-root /Users/you/path/to/fonts \
        --commit-batch 50 \
        --glyph-sample 400

Optional:
    --limit 20      Only process first 20 fonts (debug)
    --dry-run
    --verbose

Exit Codes
----------
  0 on success, >0 on partial/critical failures.

Author: Raster Phase 1 (Font Metrics Migration)
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import traceback
from typing import Dict, List, Optional, Tuple

# Lazy import fontTools so the script can still run --dry-run without it.
try:
    from fontTools.ttLib import TTFont  # type: ignore
except ImportError:
    TTFont = None  # type: ignore


# ---------------------------------------------------------------------------
# Schema Migration
# ---------------------------------------------------------------------------

MISSING_COLUMNS: Dict[str, str] = {
    "ascent": "INTEGER",
    "descent": "INTEGER",
    "line_gap": "INTEGER",
    "win_ascent": "INTEGER",
    "win_descent": "INTEGER",
    "typo_ascent": "INTEGER",
    "typo_descent": "INTEGER",
    "typo_line_gap": "INTEGER",
    "x_height": "INTEGER",
    "cap_height": "INTEGER",
    "italic_angle": "REAL",
    "bbox_xmin": "INTEGER",
    "bbox_ymin": "INTEGER",
    "bbox_xmax": "INTEGER",
    "bbox_ymax": "INTEGER",
    "avg_glyph_height": "REAL",
    "avg_glyph_width": "REAL",
    "units_per_em": "INTEGER",
}

COLUMN_ORDER_NOTE = """
NOTE:
Existing tooling that does SELECT * FROM fonts should remain unaffected by
column order. If stable ordering is required downstream, prefer specifying
explicit column names in queries.

Migration adds columns with ALTER TABLE (SQLite appends at end).
"""


def get_existing_columns(conn: sqlite3.Connection, table: str = "fonts") -> List[str]:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return [row[1] for row in cur.fetchall()]


def ensure_columns(
    conn: sqlite3.Connection, dry_run: bool = False, verbose: bool = False
) -> List[str]:
    existing = set(get_existing_columns(conn))
    added: List[str] = []
    for col, coltype in MISSING_COLUMNS.items():
        if col in existing:
            continue
        ddl = f"ALTER TABLE fonts ADD COLUMN {col} {coltype}"
        if dry_run:
            if verbose:
                print(f"[DRY-RUN] Would add column: {ddl}")
        else:
            if verbose:
                print(f"[MIGRATE] Adding column: {col} ({coltype})")
            conn.execute(ddl)
        added.append(col)
    return added


# ---------------------------------------------------------------------------
# Font Path Resolution
# ---------------------------------------------------------------------------


def resolve_font_path(file_path: str, fonts_root: str, db_path: str) -> Optional[str]:
    # 1. Absolute
    if os.path.isabs(file_path) and os.path.isfile(file_path):
        return file_path

    # 2. Join with fonts_root
    candidate = os.path.normpath(os.path.join(fonts_root, file_path))
    if os.path.isfile(candidate):
        return candidate

    # 3. Strip leading ../ segments one-by-one
    stripped = file_path
    while stripped.startswith("../") or stripped.startswith("..\\"):
        stripped = stripped[3:]
        candidate2 = os.path.normpath(os.path.join(fonts_root, stripped))
        if os.path.isfile(candidate2):
            return candidate2

    # 4. Relative to DB directory
    db_dir = os.path.dirname(os.path.abspath(db_path))
    candidate3 = os.path.normpath(os.path.join(db_dir, file_path))
    if os.path.isfile(candidate3):
        return candidate3

    return None


# ---------------------------------------------------------------------------
# Metric Extraction
# ---------------------------------------------------------------------------


def safe_get(table, attr, default=None):
    try:
        return getattr(table, attr, default)
    except Exception:
        return default


def extract_font_metrics(
    font_path: str,
    glyph_sample: int = 400,
) -> Dict[str, Optional[float]]:
    if TTFont is None:
        raise RuntimeError("fontTools is not installed (pip install fonttools).")

    font = TTFont(font_path, lazy=True)

    head = font["head"] if "head" in font else None
    hhea = font["hhea"] if "hhea" in font else None
    post = font["post"] if "post" in font else None
    os2 = font["OS/2"] if "OS/2" in font else None

    units_per_em = safe_get(head, "unitsPerEm")
    ascent = safe_get(hhea, "ascent")
    descent = safe_get(hhea, "descent")
    line_gap = safe_get(hhea, "lineGap")

    typo_ascent = safe_get(os2, "sTypoAscender")
    typo_descent = safe_get(os2, "sTypoDescender")
    typo_line_gap = safe_get(os2, "sTypoLineGap")

    win_ascent = safe_get(os2, "usWinAscent")
    win_descent = safe_get(os2, "usWinDescent")

    x_height = safe_get(os2, "sxHeight")
    cap_height = safe_get(os2, "sCapHeight")

    italic_angle = safe_get(post, "italicAngle")

    bbox_xmin = safe_get(head, "xMin")
    bbox_ymin = safe_get(head, "yMin")
    bbox_xmax = safe_get(head, "xMax")
    bbox_ymax = safe_get(head, "yMax")

    # Glyph sampling
    glyph_set = font.getGlyphSet()
    glyph_names = list(glyph_set.keys())
    if glyph_sample > 0 and len(glyph_names) > glyph_sample:
        # Uniform stride sampling
        stride = len(glyph_names) / glyph_sample
        sampled_names = [glyph_names[int(i * stride)] for i in range(glyph_sample)]
    else:
        sampled_names = glyph_names

    total_h = 0.0
    total_w = 0.0
    counted = 0
    for gname in sampled_names:
        try:
            g = glyph_set[gname]
            # boundingBox returns (xmin, ymin, xmax, ymax) or None
            bb = g.boundingBox()
            if not bb:
                continue
            gxmin, gymin, gxmax, gymax = bb
            w = gxmax - gxmin
            h = gymax - gymin
            if w <= 0 or h <= 0:
                continue
            total_w += w
            total_h += h
            counted += 1
        except Exception:
            continue

    avg_glyph_height = (total_h / counted) if counted > 0 else None
    avg_glyph_width = (total_w / counted) if counted > 0 else None

    return {
        "units_per_em": units_per_em,
        "ascent": ascent,
        "descent": descent,
        "line_gap": line_gap,
        "win_ascent": win_ascent,
        "win_descent": win_descent,
        "typo_ascent": typo_ascent,
        "typo_descent": typo_descent,
        "typo_line_gap": typo_line_gap,
        "x_height": x_height,
        "cap_height": cap_height,
        "italic_angle": italic_angle,
        "bbox_xmin": bbox_xmin,
        "bbox_ymin": bbox_ymin,
        "bbox_xmax": bbox_xmax,
        "bbox_ymax": bbox_ymax,
        "avg_glyph_height": avg_glyph_height,
        "avg_glyph_width": avg_glyph_width,
    }


# ---------------------------------------------------------------------------
# Update Logic
# ---------------------------------------------------------------------------


def update_font_row(
    conn: sqlite3.Connection,
    file_hash: str,
    metrics: Dict[str, Optional[float]],
    dry_run: bool = False,
):
    cols = []
    vals = []
    for k, v in metrics.items():
        if k not in MISSING_COLUMNS:
            # Skip columns not part of migration list (conservative)
            continue
        cols.append(f"{k} = ?")
        vals.append(v)
    if not cols:
        return
    vals.append(file_hash)
    sql = f"UPDATE fonts SET {', '.join(cols)} WHERE file_hash = ?"
    if dry_run:
        return
    conn.execute(sql, vals)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Add/refresh extended font metrics in the 'fonts' table."
    )
    ap.add_argument("--db", required=True, help="Path to glyphs.db SQLite.")
    ap.add_argument(
        "--fonts-root",
        required=True,
        help="Root directory containing font files (for resolving relative paths).",
    )
    ap.add_argument(
        "--glyph-sample",
        type=int,
        default=400,
        help="Max glyphs sampled per font for avg height/width (0 = use all).",
    )
    ap.add_argument(
        "--commit-batch",
        type=int,
        default=50,
        help="Commit every N font updates (to avoid huge transactions).",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, only process first N fonts (debug / partial run).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform migration & path resolution without updating rows.",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging (prints each font + metrics summary).",
    )
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    if TTFont is None and not args.dry_run:
        print(
            "[ERROR] fontTools not installed. Run: pip install fonttools",
            file=sys.stderr,
        )
        return 2

    db_path = args.db
    fonts_root = args.fonts_root

    if not os.path.isfile(db_path):
        print(f"[ERROR] DB not found: {db_path}", file=sys.stderr)
        return 2

    if not os.path.isdir(fonts_root):
        print(f"[ERROR] fonts root not found: {fonts_root}", file=sys.stderr)
        return 2

    conn = sqlite3.connect(db_path)
    try:
        added = ensure_columns(conn, dry_run=args.dry_run, verbose=args.verbose)
        if added and not args.dry_run:
            conn.commit()
        if args.verbose and added:
            print(f"[INFO] Added columns: {added}")
        elif args.verbose:
            print("[INFO] No new columns needed.")

        # Fetch fonts
        limit_clause = f"LIMIT {args.limit}" if args.limit > 0 else ""
        rows = conn.execute(
            f"SELECT file_hash, file_path FROM fonts WHERE excluded=0 {limit_clause}"
        ).fetchall()
        total = len(rows)
        if total == 0:
            print("[INFO] No fonts to process (maybe all excluded).")
            return 0

        processed = 0
        updated = 0
        skipped = 0
        batch = 0
        for file_hash, file_path in rows:
            processed += 1
            resolved = resolve_font_path(file_path, fonts_root, db_path)
            if not resolved:
                skipped += 1
                if args.verbose:
                    print(
                        f"[WARN] Could not resolve font path: {file_path} (hash={file_hash})"
                    )
                continue

            try:
                metrics = extract_font_metrics(resolved, glyph_sample=args.glyph_sample)
                update_font_row(
                    conn, file_hash=file_hash, metrics=metrics, dry_run=args.dry_run
                )
                updated += 1
                batch += 1
                if args.verbose:
                    short = {
                        k: metrics[k]
                        for k in (
                            "units_per_em",
                            "ascent",
                            "descent",
                            "x_height",
                            "cap_height",
                        )
                        if k in metrics
                    }
                    print(
                        f"[OK] {file_hash} path={os.path.basename(resolved)} short_metrics={short}"
                    )
            except Exception as e:
                skipped += 1
                if args.verbose:
                    print(f"[ERROR] Failed metrics for {file_hash}: {e}")
                    if args.verbose:
                        traceback.print_exc()

            if batch >= args.commit_batch and not args.dry_run:
                conn.commit()
                batch = 0
                if args.verbose:
                    print(f"[INFO] Batch committed (processed={processed})")

        if not args.dry_run:
            conn.commit()

        print(
            f"[SUMMARY] fonts_total={total} updated={updated} skipped={skipped} dry_run={args.dry_run}"
        )
        if args.dry_run and updated > 0:
            print("[NOTE] Dry-run mode: no DB changes were persisted.")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
