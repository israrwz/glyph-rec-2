"""
Contour parsing and normalization utilities for glyph vector data.

(Updated: Added robust qCurveTo midpoint inference & statistics.)

Overview
--------
The SQLite `glyphs.contours` field stores a JSON string representing a list
of subpaths. Each subpath is a list of commands in the form:
    ["moveTo", [x, y]]
    ["lineTo", [x, y]]
    ["curveTo", [[x1, y1], [x2, y2], [x3, y3]]]      # cubic Bézier (c1, c2, end)
    ["qCurveTo", [[qx, qy], [ex, ey]]]               # quadratic (control, end)  (short form)
    ["qCurveTo", [[qx, qy], [mx, my], [ex, ey]]]     # quadratic variants (needs empirical verification)
    ["closePath", null]

This module:
1. Parses the JSON structure into canonical internal commands:
       'm' (move), 'l' (line), 'c' (cubic), 'z' (close)
2. Converts any quadratic Bézier segments (qCurveTo) to cubic equivalents.
3. Normalizes coordinates:
       - Optionally EM-normalized using font units per EM (upem).
       - Centers glyph coordinates at origin (or (0,0) by bounding box midpoint).
       - Scales uniformly to fit a target range (default roughly [-1, 1]).
4. Provides a (future) bridge to DeepSVG's SVG / SVGTensor structures
   without importing heavy dependencies at module import time.

Design Goals
------------
- Lazy import heavy deps (torch, DeepSVG) to keep CLI start fast.
- Pure functions for easier unit testing.
- Clear separation: parsing (raw -> canonical) vs normalization (canonical -> normalized).

NOTE: DeepSVG expects cubic curves; we convert quadratics to cubics immediately.
      Conversion formula for a quadratic (P0, Q1, P2):
          C1 = P0 + 2/3 * (Q1 - P0)
          C2 = P2 + 2/3 * (Q1 - P2)

TODO (future enhancements)
--------------------------
- Robust TrueType-style qCurveTo sequences with implied on-curve points.
- Orientation / winding-based differentiation of holes vs solids.
- Optional caching (hash(raw_contours) -> parsed tensor).
- Visual regression test harness for verifying shape fidelity.

Author: Project scaffolding phase (Phase 1)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
)

__all__ = [
    "ContourCommand",
    "ParsedSubpath",
    "ParsedGlyphContours",
    "FontMetricsCache",
    "parse_contours",
    "normalize_contours",
    "contours_to_svg",
    "svg_to_svgtensor",
    "contours_to_svgtensor",
]


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


from dataclasses import dataclass


@dataclass(frozen=True)
class ContourCommand:
    """
    Canonical contour command.

    cmd:
        'm' = moveTo
        'l' = lineTo
        'c' = cubic Bézier
        'z' = closePath

    points:
        - For 'm'/'l': [(x, y)]
        - For 'c': [(x1, y1), (x2, y2), (x3, y3)]
        - For 'z': [] (or empty tuple)
    """

    cmd: str
    points: Tuple[Tuple[float, float], ...]


ParsedSubpath = List[ContourCommand]
ParsedGlyphContours = List[ParsedSubpath]


# ---------------------------------------------------------------------------
# Font Metrics Cache
# ---------------------------------------------------------------------------


class FontMetricsCache:
    """
    Lightweight cache for font metrics (currently only upem).

    Usage:
        cache = FontMetricsCache(load_fn)
        upem = cache.get_upem(file_hash)

    Where `load_fn` is a callable that, given a font hash, returns an integer upem.
    """

    def __init__(self, loader_fn):
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
# Parsing Logic
# ---------------------------------------------------------------------------


def parse_contours(
    raw_text: str,
    *,
    qcurve_mode: str = "midpoint",
    qcurve_stats: Optional[Dict[str, int]] = None,
) -> ParsedGlyphContours:
    """
    Parse a glyph's raw JSON contour string into canonical contour commands.

    Parameters
    ----------
    raw_text : str
        JSON encoded string representing list of subpaths.

    Returns
    -------
    List[List[ContourCommand]]
        A list of subpaths; each subpath is a sequence of ContourCommand.

    Notes
    -----
    - Quadratic segments ('qCurveTo') are converted to cubic segments.
    - Minimal validation is performed; malformed entries are skipped.
    - If a subpath does not start with a move command, an implicit move
      will be inserted at the first encountered coordinate (cautious approach).
    """
    try:
        outer = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse contour JSON: {e}") from e

    if not isinstance(outer, list):
        raise ValueError("Contours root must be a list of subpaths")

    parsed: ParsedGlyphContours = []

    for subpath in outer:
        if not isinstance(subpath, list):
            continue

        current_point: Optional[Tuple[float, float]] = None
        sub_commands: ParsedSubpath = []

        for entry in subpath:
            if (
                not isinstance(entry, list)
                or len(entry) != 2
                or not isinstance(entry[0], str)
            ):
                continue

            op = entry[0]
            payload = entry[1]

            if op == "moveTo":
                # Expect single [x, y]
                if (
                    isinstance(payload, list)
                    and len(payload) == 2
                    and all(isinstance(v, (int, float)) for v in payload)
                ):
                    pt = (float(payload[0]), float(payload[1]))
                    sub_commands.append(ContourCommand("m", (pt,)))
                    current_point = pt
                continue

            if op == "lineTo":
                if (
                    isinstance(payload, list)
                    and len(payload) == 2
                    and all(isinstance(v, (int, float)) for v in payload)
                ):
                    pt = (float(payload[0]), float(payload[1]))
                    # Insert implicit move if needed
                    if current_point is None:
                        sub_commands.append(ContourCommand("m", (pt,)))
                    else:
                        sub_commands.append(ContourCommand("l", (pt,)))
                    current_point = pt
                continue

            if op == "curveTo":
                # Expect [[x1,y1],[x2,y2],[x3,y3]]
                if (
                    isinstance(payload, list)
                    and len(payload) == 3
                    and all(isinstance(p, list) and len(p) == 2 for p in payload)
                ):
                    try:
                        c1 = (float(payload[0][0]), float(payload[0][1]))
                        c2 = (float(payload[1][0]), float(payload[1][1]))
                        p3 = (float(payload[2][0]), float(payload[2][1]))
                    except (TypeError, ValueError):
                        continue
                    if current_point is None:
                        # Provide implicit move to first control? More correct to move to p3 start?
                        # We cannot recover original start, so we skip implicit for now.
                        # Optionally we could treat c1 as start, but that distorts the curve.
                        # Skip if no current_point (logically incomplete).
                        continue
                    sub_commands.append(ContourCommand("c", (c1, c2, p3)))
                    current_point = p3
                continue

            if op == "qCurveTo":
                # Enhanced quadratic handling with midpoint inference.
                if not isinstance(payload, list):
                    continue

                if current_point is None:
                    continue

                # Gather raw points
                raw_pts: List[Tuple[float, float]] = []
                valid = True
                for p in payload:
                    if (
                        isinstance(p, list)
                        and len(p) == 2
                        and all(isinstance(v, (int, float)) for v in p)
                    ):
                        raw_pts.append((float(p[0]), float(p[1])))
                    else:
                        valid = False
                        break
                if not valid or not raw_pts:
                    continue

                if qcurve_stats is not None:
                    key = f"qcurve_payload_len_{len(raw_pts)}"
                    qcurve_stats[key] = qcurve_stats.get(key, 0) + 1

                if qcurve_mode == "naive":
                    # Original naive pairing fallback
                    if len(raw_pts) == 2:
                        control, end = raw_pts
                        cubics = _convert_qcurve_to_cubics(
                            current_point, [(control, end)]
                        )
                    else:
                        pairs: List[
                            Tuple[Tuple[float, float], Tuple[float, float]]
                        ] = []
                        for i in range(0, len(raw_pts) - 1, 2):
                            pairs.append((raw_pts[i], raw_pts[i + 1]))
                        cubics = _convert_qcurve_to_cubics(current_point, pairs)
                else:
                    # midpoint inference mode
                    cubics: List[
                        Tuple[
                            Tuple[float, float],
                            Tuple[float, float],
                            Tuple[float, float],
                        ]
                    ] = []
                    work_start = current_point
                    pts = raw_pts[:]
                    if len(pts) == 1:
                        # Not meaningful (single point); skip
                        continue
                    # Assume final point is explicit on-curve
                    final_on = pts[-1]
                    intermediates = pts[:-1]

                    # Heuristic: treat all intermediates as off-curve controls.
                    i = 0
                    while i < len(intermediates):
                        if i == len(intermediates) - 1:
                            # Last control before final on-curve
                            control = intermediates[i]
                            end = final_on
                            quad_segments = [(control, end)]
                            cubics.extend(
                                _convert_qcurve_to_cubics(work_start, quad_segments)
                            )
                            work_start = end
                            i += 1
                        else:
                            c1 = intermediates[i]
                            c2 = intermediates[i + 1]
                            # Midpoint implied on-curve
                            mid = ((c1[0] + c2[0]) / 2.0, (c1[1] + c2[1]) / 2.0)
                            quad_segments = [(c1, mid)]
                            cubics.extend(
                                _convert_qcurve_to_cubics(work_start, quad_segments)
                            )
                            work_start = mid
                            i += 1
                            # Do not advance extra here; next loop iteration uses c2 as first of next pair
                    if work_start != final_on:
                        # Final segment to explicit on-curve
                        quad_segments = [(intermediates[-1], final_on)]
                        cubics.extend(
                            _convert_qcurve_to_cubics(work_start, quad_segments)
                        )

                for c1, c2, p3 in cubics:
                    sub_commands.append(ContourCommand("c", (c1, c2, p3)))
                    current_point = p3
                if qcurve_stats is not None:
                    qcurve_stats["qcurve_segments_to_cubic"] = qcurve_stats.get(
                        "qcurve_segments_to_cubic", 0
                    ) + len(cubics)
                continue

            if op == "closePath":
                sub_commands.append(ContourCommand("z", ()))
                continue

            # Unknown op: ignore for now

        # Ensure subpath starts with 'm' (if not empty)
        if sub_commands:
            if sub_commands[0].cmd != "m":
                # Insert a dummy move to first coordinate found in subsequent commands.
                for cmd in sub_commands:
                    if cmd.points:
                        sub_commands.insert(0, ContourCommand("m", (cmd.points[0],)))
                        break

        if sub_commands:
            parsed.append(sub_commands)

    return parsed


def _convert_qcurve_to_cubics(
    start_point: Tuple[float, float],
    quad_segments: Sequence[Tuple[Tuple[float, float], Tuple[float, float]]],
) -> List[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]]:
    """
    Convert a sequence of quadratic segments sharing continuity into cubic segments.

    Parameters
    ----------
    start_point : (float, float)
        Starting point of the first quadratic segment (P0).
    quad_segments : sequence of (control, end)
        Control point (Q1), end point (P2) pairs for each quadratic segment.

    Returns
    -------
    List of (c1, c2, p3) cubic control & end points.

    Notes
    -----
    Formula:
        C1 = P0 + 2/3 * (Q1 - P0)
        C2 = P2 + 2/3 * (Q1 - P2)
        P3 = P2
    """
    cubics: List[
        Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
    ] = []
    current = start_point
    for control, end in quad_segments:
        p0x, p0y = current
        qx, qy = control
        p2x, p2y = end

        c1x = p0x + (2.0 / 3.0) * (qx - p0x)
        c1y = p0y + (2.0 / 3.0) * (qy - p0y)
        c2x = p2x + (2.0 / 3.0) * (qx - p2x)
        c2y = p2y + (2.0 / 3.0) * (qy - p2y)

        c1 = (c1x, c1y)
        c2 = (c2x, c2y)
        p3 = (p2x, p2y)
        cubics.append((c1, c2, p3))
        current = end
    return cubics


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def normalize_contours(
    parsed: ParsedGlyphContours,
    upem: Optional[int] = None,
    em_normalize: bool = True,
    center_origin: bool = True,
    scale_to_unit: bool = True,
    target_range: float = 1.0,
    min_dimension_eps: float = 1e-6,
    flip_y: bool = False,
) -> ParsedGlyphContours:
    """
    Normalize glyph contours (in-place-safe) returning a new structure.

    Steps:
        1. Optionally divide all coordinates by upem (EM normalization).
        2. Compute overall bounding box.
        3. Center at midpoint if center_origin=True.
        4. Uniform scale so that max(|x|,|y|) ~ target_range if scale_to_unit=True.
        5. Optional vertical flip.

    Parameters
    ----------
    parsed : ParsedGlyphContours
        Parsed contour commands.
    upem : int or None
        Units per EM. If provided and em_normalize=True, divides coordinates by this.
    em_normalize : bool
        Whether to apply upem division.
    center_origin : bool
        Center the bounding box midpoint at (0,0).
    scale_to_unit : bool
        Uniformly scale so the maximum extent fits target_range.
    target_range : float
        Desired half-extent bound (e.g., 1.0 -> coordinates in roughly [-1,1]).
    min_dimension_eps : float
        Avoid division by zero for degenerate glyphs.
    flip_y : bool
        If True, invert Y axis (y' = -y); depends on coordinate system conventions.

    Returns
    -------
    ParsedGlyphContours
        New list with normalized coordinates.
    """
    # Collect all coordinates
    xs: List[float] = []
    ys: List[float] = []
    for sub in parsed:
        for cmd in sub:
            for x, y in cmd.points:
                xs.append(x)
                ys.append(y)

    if not xs or not ys:
        return parsed

    # Prepare scaling factors
    def _apply_point(x: float, y: float) -> Tuple[float, float]:
        if em_normalize and upem and upem > 0:
            x_local = x / upem
            y_local = y / upem
        else:
            x_local = x
            y_local = y

        return (x_local, -y_local if flip_y else y_local)

    # First pass: EM scaling & optional flip for bounds
    scaled_points: List[Tuple[float, float]] = []
    idx = 0
    for sub in parsed:
        for cmd in sub:
            for x, y in cmd.points:
                scaled_points.append(_apply_point(x, y))
                idx += 1

    sx = [p[0] for p in scaled_points]
    sy = [p[1] for p in scaled_points]
    min_x, max_x = min(sx), max(sx)
    min_y, max_y = min(sy), max(sy)

    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    width = max_x - min_x
    height = max_y - min_y
    max_dim = max(width, height)

    if max_dim < min_dimension_eps:
        max_dim = 1.0  # Degenerate glyph (e.g., single point)

    scale_factor = 1.0
    if scale_to_unit and max_dim > 0:
        scale_factor = (target_range * 2.0) / max_dim  # so half-extent ~ target_range

    # Rebuild normalized structure
    new_parsed: ParsedGlyphContours = []
    point_iter = iter(scaled_points)
    for sub in parsed:
        new_sub: ParsedSubpath = []
        for cmd in sub:
            new_pts: List[Tuple[float, float]] = []
            for _ in cmd.points:
                ox, oy = next(point_iter)
                if center_origin:
                    ox -= cx
                    oy -= cy
                ox *= scale_factor
                oy *= scale_factor
                new_pts.append((ox, oy))
            new_sub.append(ContourCommand(cmd.cmd, tuple(new_pts)))
        new_parsed.append(new_sub)

    return new_parsed


# ---------------------------------------------------------------------------
# DeepSVG Integration Placeholders
# ---------------------------------------------------------------------------


def contours_to_svg(parsed: ParsedGlyphContours) -> Any:
    """
    Convert parsed contours to a DeepSVG SVG object.

    Placeholder:
        Implement by importing deepsvg.svglib.svg.SVG lazily, constructing path
        objects, and returning an SVG instance.

    Returns
    -------
    Any
        A placeholder or actual SVG object (future implementation).

    NOTE: For Phase 1 skeleton, returns the parsed structure unchanged.
    """
    # TODO: Implement actual conversion using deepsvg.svglib.svg.SVG
    return parsed


def svg_to_svgtensor(svg_obj: Any) -> Any:
    """
    Convert an SVG object (or parsed fallback) into a DeepSVG SVGTensor.

    Placeholder:
        Will wrap internal DeepSVG utilities for tokenization & tensor creation.

    Returns
    -------
    Any
        SVGTensor-like object suitable for model encoder input.

    NOTE: For Phase 1 skeleton, returns input unchanged.
    """
    # TODO: Implement using DeepSVG's tokenization pipeline
    return svg_obj


def contours_to_svgtensor(
    raw_text: str,
    upem: Optional[int] = None,
    em_normalize: bool = True,
    center_origin: bool = True,
    scale_to_unit: bool = True,
    target_range: float = 1.0,
    flip_y: bool = False,
) -> Any:
    """
    Convenience pipeline: raw contour JSON -> parsed -> normalized -> SVG -> SVGTensor.

    Parameters mirror normalize_contours arguments.
    """
    parsed = parse_contours(raw_text)
    norm = normalize_contours(
        parsed,
        upem=upem,
        em_normalize=em_normalize,
        center_origin=center_origin,
        scale_to_unit=scale_to_unit,
        target_range=target_range,
        flip_y=flip_y,
    )
    svg_obj = contours_to_svg(norm)
    return svg_to_svgtensor(svg_obj)


# ---------------------------------------------------------------------------
# Simple Stats Utility (Optional Helper)
# ---------------------------------------------------------------------------


def contour_stats(parsed: ParsedGlyphContours) -> Dict[str, Any]:
    """
    Generate basic statistics for a parsed glyph.

    Returns
    -------
    dict with:
        subpaths: int
        commands: int
        move: int
        line: int
        cubic: int
        close: int
        max_subpath_length: int
    """
    move = line = cubic = close = 0
    max_len = 0
    total_cmds = 0
    for sub in parsed:
        max_len = max(max_len, len(sub))
        for cmd in sub:
            total_cmds += 1
            if cmd.cmd == "m":
                move += 1
            elif cmd.cmd == "l":
                line += 1
            elif cmd.cmd == "c":
                cubic += 1
            elif cmd.cmd == "z":
                close += 1
    return {
        "subpaths": len(parsed),
        "commands": total_cmds,
        "move": move,
        "line": line,
        "cubic": cubic,
        "close": close,
        "max_subpath_length": max_len,
    }


# ---------------------------------------------------------------------------
# Debug Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Minimal manual test (replace with an actual sample string for quick dev checks)
    sample = json.dumps(
        [
            [
                ["moveTo", [0, 0]],
                ["lineTo", [100, 0]],
                ["qCurveTo", [[150, 50], [100, 100]]],
                ["curveTo", [[80, 110], [60, 90], [0, 100]]],
                ["closePath", None],
            ]
        ]
    )
    parsed = parse_contours(sample)
    norm = normalize_contours(parsed, upem=1000)
    stats = contour_stats(norm)
    print("Parsed:", parsed)
    print("Normalized:", norm)
    print("Stats:", stats)
