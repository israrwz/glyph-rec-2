"""
Font metrics loading utilities (SQLite-backed).

Purpose
-------
Provides convenient access to per-font metrics (currently Units Per EM / `upem`)
stored in the `fonts` table of the project's SQLite database (`dataset/glyphs.db`).

Schema Assumptions (Observed)
-----------------------------
fonts(
    file_hash TEXT PRIMARY KEY,
    file_name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    family_name TEXT,
    style_name TEXT,
    version TEXT,
    upem INTEGER,
    num_glyphs INTEGER,
    total_processed_glyphs INTEGER DEFAULT 0,
    warnings TEXT,
    excluded INTEGER DEFAULT 0
)

Key Points
----------
- `file_hash` is treated as the unique font identifier.
- `upem` varies across fonts (e.g., 500 .. 4096). Normalization of glyph shapes
  should take `upem` into account for cross-font geometric consistency.
- `excluded` can be used to skip fonts flagged during preprocessing.

Features
--------
1. Bulk loading functions:
   - `load_font_upem_map()`: Returns a dict {file_hash: upem}.
   - `iter_font_records()`: Yields full `FontRecord` instances (optionally filtered).
2. Helper `build_sqlite_upem_loader()` returning a callable compatible with
   the `FontMetricsCache` in `contour_parser.py`.
3. Simple statistics helper `compute_upem_stats()` to summarize distribution.
4. Lightweight optional logging (no external logging dependency required).

Usage Example
-------------
    from data.font_metrics import load_font_upem_map, compute_upem_stats

    upem_map = load_font_upem_map("dataset/glyphs.db")
    stats = compute_upem_stats(upem_map)
    print(stats)

Compatibility
-------------
- Designed for Python >= 3.9.
- No heavy dependencies; only standard library.

CLI
---
`python -m data.font_metrics --db dataset/glyphs.db --limit 5`
Shows a sample of fonts and prints UPEM statistics.

Author
------
Phase 1 scaffolding.
"""

from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

# Attempt to import shared FontMetricsCache (non-fatal if unavailable)
try:
    from .contour_parser import FontMetricsCache  # type: ignore
except Exception:  # pragma: no cover - fallback definition

    class FontMetricsCache:  # type: ignore
        """
        Fallback cache definition (duplicate minimal behavior).
        """

        def __init__(self, loader_fn: Callable[[str], Optional[int]]):
            self._loader_fn = loader_fn
            self._cache: Dict[str, int] = {}

        def get_upem(self, font_hash: str) -> Optional[int]:
            if font_hash in self._cache:
                return self._cache[font_hash]
            upem = self._loader_fn(font_hash)
            if upem is not None:
                self._cache[font_hash] = upem
            return upem


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FontRecord:
    file_hash: str
    file_name: str
    file_path: str
    family_name: Optional[str]
    style_name: Optional[str]
    version: Optional[str]
    upem: Optional[int]
    num_glyphs: Optional[int]
    total_processed_glyphs: Optional[int]
    warnings: Optional[str]
    excluded: int


# ---------------------------------------------------------------------------
# Core Query Helpers
# ---------------------------------------------------------------------------


def _connect(db_path: str) -> sqlite3.Connection:
    """
    Create a read-only SQLite connection (where supported) or standard connection as fallback.
    """
    dbp = Path(db_path).expanduser().resolve()
    if not dbp.exists():
        raise FileNotFoundError(f"SQLite database not found: {dbp}")

    # SQLite URI read-only mode
    uri = f"file:{dbp}?mode=ro"
    try:
        return sqlite3.connect(uri, uri=True)
    except sqlite3.OperationalError:
        # Fallback to regular connection
        return sqlite3.connect(str(dbp))


def load_font_upem_map(
    db_path: str,
    *,
    include_excluded: bool = False,
    subset_hashes: Optional[Sequence[str]] = None,
) -> Dict[str, int]:
    """
    Load a mapping of font file_hash → upem.

    Parameters
    ----------
    db_path : str
        Path to SQLite database.
    include_excluded : bool
        Include rows where excluded != 0 if True.
    subset_hashes : Sequence[str] | None
        If provided, restrict results to these file_hash values.

    Returns
    -------
    Dict[str, int]
        Mapping from file_hash to upem (filters out null / missing upem).
    """
    where_clauses: List[str] = []
    params: List = []

    if not include_excluded:
        where_clauses.append("excluded = 0")
    if subset_hashes:
        placeholders = ",".join("?" for _ in subset_hashes)
        where_clauses.append(f"file_hash IN ({placeholders})")
        params.extend(subset_hashes)

    where_sql = ""
    if where_clauses:
        where_sql = "WHERE " + " AND ".join(where_clauses)

    sql = f"""
        SELECT file_hash, upem
        FROM fonts
        {where_sql}
    """

    mapping: Dict[str, int] = {}
    with _connect(db_path) as conn:
        cursor = conn.execute(sql, params)
        for file_hash, upem in cursor.fetchall():
            if upem is None:
                continue
            mapping[file_hash] = int(upem)
    return mapping


def iter_font_records(
    db_path: str,
    *,
    include_excluded: bool = False,
    subset_hashes: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
) -> Iterator[FontRecord]:
    """
    Iterate over complete font records.

    Parameters
    ----------
    db_path : str
        SQLite database path.
    include_excluded : bool
        Include excluded fonts if True.
    subset_hashes : sequence[str] | None
        Restrict to given font hashes.
    limit : int | None
        Limit number of rows returned (for sampling).
    """
    where_clauses: List[str] = []
    params: List = []

    if not include_excluded:
        where_clauses.append("excluded = 0")
    if subset_hashes:
        placeholders = ",".join("?" for _ in subset_hashes)
        where_clauses.append(f"file_hash IN ({placeholders})")
        params.extend(subset_hashes)

    where_sql = ""
    if where_clauses:
        where_sql = "WHERE " + " AND ".join(where_clauses)

    limit_sql = f"LIMIT {int(limit)}" if limit and limit > 0 else ""

    sql = f"""
        SELECT
            file_hash,
            file_name,
            file_path,
            family_name,
            style_name,
            version,
            upem,
            num_glyphs,
            total_processed_glyphs,
            warnings,
            excluded
        FROM fonts
        {where_sql}
        {limit_sql}
    """

    with _connect(db_path) as conn:
        cursor = conn.execute(sql, params)
        for row in cursor.fetchall():
            yield FontRecord(
                file_hash=row[0],
                file_name=row[1],
                file_path=row[2],
                family_name=row[3],
                style_name=row[4],
                version=row[5],
                upem=row[6],
                num_glyphs=row[7],
                total_processed_glyphs=row[8],
                warnings=row[9],
                excluded=row[10],
            )


# ---------------------------------------------------------------------------
# Loader Closure / Cache Integration
# ---------------------------------------------------------------------------


def build_sqlite_upem_loader(db_path: str) -> Callable[[str], Optional[int]]:
    """
    Build a loader function suitable for `FontMetricsCache`, resolving a single font hash to upem.
    """

    def _loader(font_hash: str) -> Optional[int]:
        sql = "SELECT upem FROM fonts WHERE file_hash = ? LIMIT 1"
        with _connect(db_path) as conn:
            cur = conn.execute(sql, (font_hash,))
            row = cur.fetchone()
            if row is None:
                return None
            return int(row[0]) if row[0] is not None else None

    return _loader


def build_cached_upem_accessor(db_path: str) -> FontMetricsCache:
    """
    Convenience factory returning a `FontMetricsCache` bound to the given database.
    """
    return FontMetricsCache(build_sqlite_upem_loader(db_path))


# ---------------------------------------------------------------------------
# Statistics & Diagnostics
# ---------------------------------------------------------------------------


def compute_upem_stats(upem_map: Dict[str, int]) -> Dict[str, float]:
    """
    Compute basic statistics over a font upem mapping.

    Returns
    -------
    dict
        {
            'count': int,
            'distinct': int,
            'min': int,
            'max': int,
            'mean': float
        }
    """
    if not upem_map:
        return {
            "count": 0,
            "distinct": 0,
            "min": 0,
            "max": 0,
            "mean": 0.0,
        }
    values = list(upem_map.values())
    return {
        "count": len(values),
        "distinct": len(set(values)),
        "min": min(values),
        "max": max(values),
        "mean": float(mean(values)),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect font metrics (upem) from the SQLite database."
    )
    parser.add_argument(
        "--db",
        required=True,
        help="Path to glyphs SQLite database (e.g., dataset/glyphs.db)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Sample size of font records to display.",
    )
    parser.add_argument(
        "--include-excluded",
        action="store_true",
        help="Include fonts marked as excluded.",
    )
    args = parser.parse_args()

    print(f"[INFO] Loading font UPEM map from: {args.db}")
    upem_map = load_font_upem_map(
        args.db,
        include_excluded=args.include_excluded,
        subset_hashes=None,
    )
    stats = compute_upem_stats(upem_map)
    print("[INFO] UPEM Stats:", stats)

    print(f"[INFO] Sample font records (limit={args.limit}):")
    for rec in iter_font_records(
        args.db,
        include_excluded=args.include_excluded,
        limit=args.limit,
    ):
        print(
            f"  {rec.file_hash} | upem={rec.upem} | family={rec.family_name} | "
            f"style={rec.style_name} | excluded={rec.excluded}"
        )


if __name__ == "__main__":
    _cli()
