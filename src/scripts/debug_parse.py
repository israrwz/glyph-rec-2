#!/usr/bin/env python3
"""
debug_parse.py

Utility script to introspect a single glyph's contour JSON -> parsed commands ->
(normalized) contours -> SVGTensor builder tensors.

Goals:
1. Verify parse_contours succeeds for a given glyph_id.
2. Inspect quadratic qCurveTo expansion statistics.
3. Show normalization metadata (norm_v2 by default).
4. (Optional) Convert to grouped command/arg tensors and dump a compact view.
5. Help diagnose why embedding extraction script might be skipping glyphs.

Typical usage:
    python -m src.scripts.debug_parse \
        --db dataset/glyphs.db \
        --glyph-id 218897 \
        --show-raw \
        --show-parsed \
        --show-norm \
        --build-tensors \
        --qcurve-mode midpoint

Multiple glyphs:
    python -m src.scripts.debug_parse --db dataset/glyphs.db --glyph-ids 218897,455825

Random sample (limit 3):
    python -m src.scripts.debug_parse --db dataset/glyphs.db --random 3 --show-parsed

Outputs are textual; no files are written unless --out-json is provided.

NOTE:
This script is intentionally verbose; adapt as needed once the root cause of failures
is identified.

"""

from __future__ import annotations

# --- Dynamic path injection for local deepsvg repository (added) ---
import sys as _sys_dbg_inject
from pathlib import Path as _Path_dbg_inject

_pr_dbg = _Path_dbg_inject(__file__).resolve().parents[2]
_dsvg_dbg = _pr_dbg / "deepsvg"
if _dsvg_dbg.exists() and str(_dsvg_dbg) not in _sys_dbg_inject.path:
    _sys_dbg_inject.path.insert(0, str(_dsvg_dbg))
# ------------------------------------------------------------------

import argparse
import json
import sqlite3
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# --- Project imports ---
try:
    from src.data.contour_parser import parse_contours, ContourCommand
    from src.data.normalization import (
        NormalizationConfig,
        Strategy,
        apply_normalization,
    )
    from src.model.svgtensor_builder import (
        SVGTensorBuilder,
        build_default_builder_from_cfg,
    )
    from deepsvg.model.config import _DefaultConfig
except Exception as e:
    print("[FATAL] Failed to import project modules:", e, file=sys.stderr)
    raise


# ---------------------------------------------------------------------------
# SQLite utilities
# ---------------------------------------------------------------------------


def connect_ro(path: str) -> sqlite3.Connection:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Database not found: {p}")
    try:
        return sqlite3.connect(f"file:{p}?mode=ro", uri=True)
    except sqlite3.OperationalError:
        return sqlite3.connect(str(p))


@dataclass
class GlyphRow:
    id: int
    font_hash: str
    label: str
    contours: str
    upem: Optional[int]


def fetch_by_ids(conn: sqlite3.Connection, ids: Sequence[int]) -> List[GlyphRow]:
    placeholders = ",".join("?" for _ in ids)
    sql = f"""
        SELECT g.id, g.f_id, g.label, g.contours, f.upem
        FROM glyphs g
        JOIN fonts f ON f.file_hash = g.f_id
        WHERE g.id IN ({placeholders})
    """
    cur = conn.execute(sql, list(ids))
    rows = [
        GlyphRow(id=r[0], font_hash=r[1], label=r[2], contours=r[3], upem=r[4])
        for r in cur.fetchall()
    ]
    return rows


def fetch_random(conn: sqlite3.Connection, n: int) -> List[GlyphRow]:
    sql = f"""
        SELECT g.id, g.f_id, g.label, g.contours, f.upem
        FROM glyphs g
        JOIN fonts f ON f.file_hash = g.f_id
        WHERE g.contours IS NOT NULL AND length(g.contours) > 0
        ORDER BY RANDOM()
        LIMIT ?
    """
    cur = conn.execute(sql, (n,))
    rows = [
        GlyphRow(id=r[0], font_hash=r[1], label=r[2], contours=r[3], upem=r[4])
        for r in cur.fetchall()
    ]
    return rows


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def summarize_parsed(parsed) -> Dict[str, Any]:
    num_subpaths = len(parsed)
    total_cmds = sum(len(s) for s in parsed)
    cmd_freq: Dict[str, int] = {}
    for sub in parsed:
        for c in sub:
            cmd_freq[c.cmd] = cmd_freq.get(c.cmd, 0) + 1
    return {
        "num_subpaths": num_subpaths,
        "total_commands": total_cmds,
        "command_frequency": cmd_freq,
        "avg_cmds_per_subpath": (total_cmds / num_subpaths) if num_subpaths else 0.0,
    }


def normalize_wrapper(
    parsed_like, strategy: str, upem: Optional[int], flip_y: bool = True
) -> Tuple[Any, Dict[str, Any]]:
    cfg = NormalizationConfig(strategy=strategy, flip_y=flip_y)
    return apply_normalization(parsed_like, cfg, upem)


def build_tensors(parsed_norm) -> Tuple[Any, Any, Dict[str, Any]]:
    # Use a default config (mirrors _DefaultConfig) for builder size
    cfg = _DefaultConfig()
    builder = build_default_builder_from_cfg(cfg)
    commands_g, args_g = builder.glyph_to_group_tensors(parsed_norm)
    stats = {
        "commands_shape": tuple(commands_g.shape),
        "args_shape": tuple(args_g.shape),
        "first_group_commands": commands_g[0, : min(12, commands_g.shape[1])].tolist(),
    }
    return commands_g, args_g, stats


def rewrap_for_normalization(parsed):
    """
    The normalization module expects objects with .cmd and .points (tuples).
    parse_contours already returns ContourCommand dataclass instances (after fix).
    To be safe (immutability / detach), rewrap into lightweight mirrors.
    """
    wrapped = []
    for sub in parsed:
        new_sub = []
        for cmd in sub:

            class Wrapper:
                __slots__ = ("cmd", "points")

                def __init__(self, c, pts):
                    self.cmd = c
                    self.points = pts

            new_sub.append(Wrapper(cmd.cmd, cmd.points))
        wrapped.append(new_sub)
    return wrapped


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: List[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Debug parser / normalization / tensor build for a glyph."
    )
    ap.add_argument("--db", required=True, help="Path to glyphs.db")
    ap.add_argument("--glyph-id", type=int, help="Single glyph id")
    ap.add_argument(
        "--glyph-ids",
        type=str,
        default=None,
        help="Comma-separated list of glyph ids (overrides --glyph-id)",
    )
    ap.add_argument(
        "--random", type=int, default=0, help="Sample N random glyphs (if no ids)"
    )
    ap.add_argument(
        "--qcurve-mode",
        choices=("midpoint", "naive"),
        default="midpoint",
        help="Quadratic expansion strategy",
    )
    ap.add_argument(
        "--strategy",
        choices=("norm_v1", "norm_v2"),
        default="norm_v2",
        help="Normalization strategy",
    )
    ap.add_argument("--show-raw", action="store_true", help="Print raw JSON string")
    ap.add_argument("--show-parsed", action="store_true", help="Print parsed commands")
    ap.add_argument("--show-norm", action="store_true", help="Print normalization meta")
    ap.add_argument(
        "--build-tensors", action="store_true", help="Build command/arg tensors"
    )
    ap.add_argument(
        "--out-json",
        type=str,
        default=None,
        help="Write a JSON summary (list) of results to this path",
    )
    ap.add_argument("--limit-subpaths", type=int, default=2, help="Print cap (parsed)")
    ap.add_argument("--limit-cmds", type=int, default=12, help="Print cap (commands)")
    return ap.parse_args(argv)


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------


def process_row(
    row: GlyphRow, args: argparse.Namespace
) -> Dict[str, Any]:  # returns summary dict
    summary: Dict[str, Any] = {
        "glyph_id": row.id,
        "font_hash": row.font_hash,
        "label": row.label,
        "upem": row.upem,
    }

    if args.show_raw:
        summary["raw_prefix"] = row.contours[:160]

    # Parse
    q_stats: Dict[str, int] = {}
    try:
        parsed = parse_contours(
            row.contours, qcurve_mode=args.qcurve_mode, qcurve_stats=q_stats
        )
        summary["parse_ok"] = True
        summary["parse_stats"] = summarize_parsed(parsed)
        summary["qcurve_stats"] = q_stats
    except Exception as e:
        summary["parse_ok"] = False
        summary["error"] = f"parse_error: {e}"
        return summary

    if args.show_parsed:
        printed = []
        for s_idx, sub in enumerate(parsed[: args.limit_subpaths]):
            for c_idx, cmd in enumerate(sub[: args.limit_cmds]):
                printed.append(
                    {
                        "subpath": s_idx,
                        "cmd_index": c_idx,
                        "cmd": cmd.cmd,
                        "points": cmd.points,
                    }
                )
        summary["parsed_preview"] = printed

    # Rewrap & normalize
    wrapped = rewrap_for_normalization(parsed)
    try:
        norm_contours, norm_meta = normalize_wrapper(
            wrapped, args.strategy, row.upem, flip_y=True
        )
        summary["norm_ok"] = True
        if args.show_norm:
            summary["norm_meta"] = norm_meta
    except Exception as e:
        summary["norm_ok"] = False
        summary["error"] = f"norm_error: {e}"
        return summary

    # Optionally build tensors
    if args.build_tensors:
        try:
            cmds_g, args_g, tstats = build_tensors(norm_contours)
            summary["tensor_ok"] = True
            summary["tensor_stats"] = tstats
            # Add a tiny slice of args for the first few commands of group 0
            g0_cmds = cmds_g[0, : min(8, cmds_g.shape[1])].tolist()
            g0_args = args_g[0, : min(4, args_g.shape[1]), :].tolist()
            summary["tensor_preview"] = {
                "g0_commands_first": g0_cmds,
                "g0_args_first_rows": g0_args,
            }
        except Exception as e:
            summary["tensor_ok"] = False
            summary["error"] = f"tensor_error: {e}"

    return summary


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    # Collect glyph rows
    conn = connect_ro(args.db)
    rows: List[GlyphRow] = []
    if args.glyph_ids:
        ids = [int(x) for x in args.glyph_ids.split(",") if x.strip()]
        rows = fetch_by_ids(conn, ids)
    elif args.glyph_id is not None:
        rows = fetch_by_ids(conn, [args.glyph_id])
    elif args.random > 0:
        rows = fetch_random(conn, args.random)
    else:
        print("Provide --glyph-id / --glyph-ids / --random", file=sys.stderr)
        return 2

    if not rows:
        print("No glyph rows found for given criteria.", file=sys.stderr)
        return 1

    results: List[Dict[str, Any]] = []
    for r in rows:
        print(f"[INFO] Processing glyph_id={r.id} label={r.label}")
        summary = process_row(r, args)
        results.append(summary)
        # Console output (succinct)
        status_parts = []
        status_parts.append("PARSE:OK" if summary.get("parse_ok") else "PARSE:FAIL")
        status_parts.append("NORM:OK" if summary.get("norm_ok") else "NORM:FAIL")
        if args.build_tensors:
            status_parts.append(
                "TENSOR:OK" if summary.get("tensor_ok") else "TENSOR:FAIL"
            )
        print("  Status:", " ".join(status_parts))
        if "error" in summary:
            print("  Error:", summary["error"])
        if summary.get("parse_ok") and args.show_parsed:
            pstats = summary["parse_stats"]
            print(
                f"  Parsed: subpaths={pstats['num_subpaths']} cmds={pstats['total_commands']} freq={pstats['command_frequency']}"
            )
        if summary.get("norm_ok") and args.show_norm:
            nm = summary.get("norm_meta", {})
            print(
                "  Norm: width_em={:.3f} height_em={:.3f} bbox_raw={}".format(
                    nm.get("width_em", -1),
                    nm.get("height_em", -1),
                    nm.get("bbox_raw"),
                )
            )
        if summary.get("tensor_ok") and args.build_tensors:
            ts = summary.get("tensor_stats", {})
            print(
                f"  Tensor: commands_shape={ts.get('commands_shape')} args_shape={ts.get('args_shape')}"
            )
        print()

    if args.out_json:
        outp = Path(args.out_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Wrote JSON summary to {outp}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
