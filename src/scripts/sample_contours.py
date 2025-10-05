#!/usr/bin/env python3
"""
Sample glyph contours from the SQLite database to analyze:
1. Distribution of `qCurveTo` payload variants (number of point tuples per command).
2. Basic command count statistics (move/line/curve/qcurve/close) before conversion.
3. Sequence length (total commands per glyph) distribution.
4. Optional export of a small set of raw contour JSON strings for manual inspection.

This script is intended as an exploratory / diagnostic utility during Phase 1
to inform robust handling of quadratic curves (`qCurveTo`) when converting
to cubic Bézier segments for the DeepSVG encoder pipeline.

Usage (from project root):
    python -m src.scripts.sample_contours \
        --db dataset/glyphs.db \
        --limit 1000 \
        --output artifacts/reports/qcurve_payload_stats.json

If the module path invocation fails, you can also run directly:
    python src/scripts/sample_contours.py --db dataset/glyphs.db

Outputs
-------
- A JSON report (default: artifacts/reports/qcurve_payload_stats.json) containing:
    {
      "summary": { ... high-level counts ... },
      "qcurve_payload_lengths": {
          "2": {"count": ..., "example_glyph_ids": [...]},
          "3": {"count": ..., "example_glyph_ids": [...]},
          ...
      },
      "command_type_counts": {
          "moveTo": ..., "lineTo": ..., "curveTo": ..., "qCurveTo": ..., "closePath": ...
      },
      "sequence_length_distribution": [
          {"length": L, "count": C}, ...
      ],
      "errors": {
          "json_parse_fail": N,
          "non_list_root": N
      }
    }

Notes
-----
- The script does NOT convert quadratics; it only inspects raw serialized contours.
- The SQLite `glyphs` table is assumed to contain columns: id, contours.
- Large databases: consider lowering --limit or adding filtering options
  if performance becomes an issue.

Future Enhancements
-------------------
- Add font-based sampling (e.g., restrict to specific upem).
- Stratified sampling across font families.
- Integration with the contour parser to cross-check post-normalization stats.

Author: Phase 1 scaffolding.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Any, Optional


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class QCurveStats:
    payload_length: int
    count: int = 0
    example_ids: List[int] = None

    def add_example(self, glyph_id: int, max_examples: int = 5):
        if self.example_ids is None:
            self.example_ids = []
        if len(self.example_ids) < max_examples:
            self.example_ids.append(glyph_id)


# ---------------------------------------------------------------------------
# SQLite Helpers
# ---------------------------------------------------------------------------


def open_db(db_path: str) -> sqlite3.Connection:
    """
    Open a SQLite connection (read-only if possible).
    """
    dbp = Path(db_path).expanduser().resolve()
    if not dbp.exists():
        raise FileNotFoundError(f"SQLite database not found: {dbp}")
    uri = f"file:{dbp}?mode=ro"
    try:
        return sqlite3.connect(uri, uri=True)
    except sqlite3.OperationalError:
        # Fallback to regular mode (still fine for read).
        return sqlite3.connect(str(dbp))


def sample_glyph_rows(
    conn: sqlite3.Connection,
    limit: int,
    randomize: bool = True,
    where: Optional[str] = None,
) -> Iterable[Tuple[int, str]]:
    """
    Yield (id, contours_json) rows from the glyphs table.

    If randomize is True, selects a random subset using ORDER BY RANDOM().
    Otherwise, returns the first `limit` rows (optionally with a WHERE clause).

    WARNING: ORDER BY RANDOM() can be expensive on very large tables.
             For very large tables, consider a two-phase sampling approach.
    """
    base = "SELECT id, contours FROM glyphs WHERE contours IS NOT NULL AND length(contours) > 0"
    if where:
        base += f" AND ({where})"
    if randomize:
        base += " ORDER BY RANDOM()"
    base += " LIMIT ?"
    cursor = conn.execute(base, (limit,))
    for row in cursor:
        yield int(row[0]), row[1]


# ---------------------------------------------------------------------------
# Analysis Logic
# ---------------------------------------------------------------------------


def analyze_contours(
    rows: Iterable[Tuple[int, str]],
    max_qcurve_examples: int = 5,
    max_raw_examples: int = 10,
) -> Dict[str, Any]:
    """
    Analyze glyph contour JSON strings for qCurveTo payload patterns and command stats.

    Parameters
    ----------
    rows : iterable of (glyph_id, contours_json)
    max_qcurve_examples : int
        Max example glyph ids stored per qCurveTo payload length.
    max_raw_examples : int
        Number of raw contour JSON snippets to preserve for manual review.

    Returns
    -------
    dict
        Collected statistics and examples.
    """
    qcurve_length_map: Dict[int, QCurveStats] = {}
    command_counter = Counter()
    seq_length_counter = Counter()
    errors = Counter()
    raw_examples: List[Dict[str, Any]] = []

    processed = 0

    for glyph_id, raw in rows:
        processed += 1
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            errors["json_parse_fail"] += 1
            continue

        if not isinstance(data, list):
            errors["non_list_root"] += 1
            continue

        glyph_command_count = 0

        # Optionally store a few raw samples:
        if len(raw_examples) < max_raw_examples:
            raw_examples.append({"glyph_id": glyph_id, "raw_prefix": raw[:250]})

        for subpath in data:
            if not isinstance(subpath, list):
                continue
            for cmd_entry in subpath:
                if (
                    not isinstance(cmd_entry, list)
                    or len(cmd_entry) != 2
                    or not isinstance(cmd_entry[0], str)
                ):
                    continue
                op = cmd_entry[0]
                payload = cmd_entry[1]
                glyph_command_count += 1
                command_counter[op] += 1

                if op == "qCurveTo":
                    if isinstance(payload, list):
                        # Count valid coordinate pair style items
                        point_count = 0
                        valid = True
                        for p in payload:
                            if (
                                isinstance(p, list)
                                and len(p) == 2
                                and all(isinstance(v, (int, float)) for v in p)
                            ):
                                point_count += 1
                            else:
                                valid = False
                                break
                        if valid and point_count > 0:
                            stats_obj = qcurve_length_map.get(point_count)
                            if stats_obj is None:
                                stats_obj = QCurveStats(payload_length=point_count)
                                qcurve_length_map[point_count] = stats_obj
                            stats_obj.count += 1
                            stats_obj.add_example(
                                glyph_id, max_examples=max_qcurve_examples
                            )
                        else:
                            errors["qcurve_invalid_payload"] += 1
                    else:
                        errors["qcurve_non_list_payload"] += 1

        seq_length_counter[glyph_command_count] += 1

    # Build output structure
    qcurve_payload_lengths = {
        str(length): {
            "count": stats.count,
            "example_glyph_ids": stats.example_ids or [],
        }
        for length, stats in sorted(qcurve_length_map.items(), key=lambda x: x[0])
    }

    seq_length_distribution = [
        {"length": length, "count": count}
        for length, count in sorted(seq_length_counter.items(), key=lambda x: x[0])
    ]

    result = {
        "summary": {
            "processed_rows": processed,
            "unique_qcurve_payload_lengths": len(qcurve_payload_lengths),
            "total_qcurve_commands": sum(
                stats["count"] for stats in qcurve_payload_lengths.values()
            ),
        },
        "qcurve_payload_lengths": qcurve_payload_lengths,
        "command_type_counts": dict(command_counter),
        "sequence_length_distribution": seq_length_distribution,
        "errors": dict(errors),
        "raw_examples": raw_examples,
    }
    return result


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_report(report: Dict[str, Any], output_path: str) -> None:
    """
    Write JSON report to the specified output path, creating directories as needed.
    """
    outp = Path(output_path).expanduser()
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def pretty_print_report(report: Dict[str, Any]) -> None:
    """
    Print a concise summary to stdout.
    """
    summary = report.get("summary", {})
    command_counts = report.get("command_type_counts", {})
    qcurve_lengths = report.get("qcurve_payload_lengths", {})

    print("=== Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print("\n=== Command Type Counts ===")
    for k, v in sorted(command_counts.items(), key=lambda x: x[0].lower()):
        print(f"{k}: {v}")

    print("\n=== qCurveTo Payload Lengths ===")
    for length, entry in qcurve_lengths.items():
        print(
            f"length={length} -> count={entry['count']} examples={entry['example_glyph_ids']}"
        )

    errors = report.get("errors", {})
    if errors:
        print("\n=== Errors ===")
        for k, v in errors.items():
            print(f"{k}: {v}")

    print("\n=== Sample Raw Examples (truncated) ===")
    for ex in report.get("raw_examples", [])[:5]:
        print(f"id={ex['glyph_id']} raw_prefix={ex['raw_prefix']!r}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample glyph contours and analyze qCurveTo payload patterns."
    )
    parser.add_argument(
        "--db",
        required=True,
        help="Path to SQLite glyph database (e.g., dataset/glyphs.db)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Number of glyph rows to sample (random).",
    )
    parser.add_argument(
        "--no-random",
        action="store_true",
        help="Disable random sampling (take first N rows).",
    )
    parser.add_argument(
        "--output",
        default="artifacts/reports/qcurve_payload_stats.json",
        help="Path to write JSON report.",
    )
    parser.add_argument(
        "--max-qcurve-examples",
        type=int,
        default=5,
        help="Max example glyph ids per qCurveTo payload length.",
    )
    parser.add_argument(
        "--max-raw-examples",
        type=int,
        default=10,
        help="Max raw contour JSON snippet examples to store.",
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    print(f"[INFO] Opening database: {args.db}")
    try:
        conn = open_db(args.db)
    except Exception as e:
        print(f"[ERROR] Cannot open database: {e}", file=sys.stderr)
        return 2

    print(
        f"[INFO] Sampling glyph rows: limit={args.limit} randomize={not args.no_random}"
    )

    rows = sample_glyph_rows(
        conn,
        limit=args.limit,
        randomize=not args.no_random,
        where=None,
    )

    report = analyze_contours(
        rows,
        max_qcurve_examples=args.max_qcurve_examples,
        max_raw_examples=args.max_raw_examples,
    )

    write_report(report, args.output)
    print(f"[INFO] Report written to: {args.output}")
    pretty_print_report(report)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
