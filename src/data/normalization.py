# glyph-rec-2/src/data/normalization.py
"""
Normalization strategies for glyph contours.

This module centralizes normalization logic so we can switch strategies
(naming versioned as norm_v1, norm_v2, …) without scattering conditionals
across parsing / embedding code.

Current Strategies
------------------
norm_v1 (deprecated for embeddings, kept for experimentation):
    - Per-glyph size invariant.
    - Steps:
        1. Optional EM scale (x/upem, y/upem) if available.
        2. Center glyph by bbox midpoint.
        3. Uniformly scale so max dimension fits target_range (default 1.0).
        4. Optional Y flip.
    - Effect: removes size discriminative features (not ideal for Arabic diacritics
      vs base forms classification).

norm_v2 (default for embeddings):
    - Size-preserving across glyphs (after EM normalization).
    - Steps:
        1. EM normalize: (x, y) -> (x/upem, y/upem) if upem is known (else raw units).
        2. Center by bbox midpoint (translation only).
        3. NO per-glyph scaling to unit box (preserves relative size across glyphs).
        4. Optional single Y flip (to move from font y-up to model y-down).
        5. (Optional) Global scaling / clamping (not enabled by default): apply a
           fixed factor to keep 99.x percentile of coordinates within [-1, 1].
    - Retains width/height signals (critical for distinguishing diacritics vs
      initial or medial forms).

API Overview
------------
- NormalizationConfig: dataclass describing normalization parameters.
- get_normalizer(cfg): returns a callable performing normalization on a list of
  subpaths (ParsedGlyphContours-like structure) producing:
     normalized_contours, meta_dict
  where meta_dict includes bounding box and scaling metadata.

- apply_normalization(parsed, cfg): convenience wrapper.

Integration
-----------
Existing contour parsing returns: List[List[ContourCommand]]
Each ContourCommand has:
    cmd: str in {'m','l','c','z'}
    points: tuple of (x, y) points (empty for 'z')

We operate purely on numeric coordinates, preserving command structure.

Meta Fields Produced
--------------------
bbox_raw: (min_x, min_y, max_x, max_y) pre-normalization (after EM divide if used)
width_em, height_em: width/height after EM normalization (before centering)
center_em: (cx, cy)
scale_applied: per-glyph scale factor (norm_v1) or 1.0 (norm_v2)
global_scale: global post factor (if configured)
normalization_version: 'norm_v1' or 'norm_v2'
y_flipped: bool
upem_used: int or None

Usage Example
-------------
    from src.data.normalization import NormalizationConfig, apply_normalization, Strategy
    cfg = NormalizationConfig(strategy="norm_v2", flip_y=True)
    normalized, meta = apply_normalization(parsed_contours, cfg)

Future Extensions
-----------------
- norm_v3: hybrid add explicit size tokens
- norm_v4: baseline + stroke weight heuristics (if later extracted)
- Option to encode vertical alignment relative to a statistically inferred baseline

Author: Phase 1
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional

# Type Aliases (lightweight, to avoid import cycles)
Point = Tuple[float, float]


class Strategy:
    NORM_V1 = "norm_v1"
    NORM_V2 = "norm_v2"


ALLOWED_STRATEGIES = {Strategy.NORM_V1, Strategy.NORM_V2}


@dataclass
class NormalizationConfig:
    """
    Configuration for glyph coordinate normalization.

    Attributes
    ----------
    strategy : str
        One of 'norm_v1' (size-invariant) or 'norm_v2' (size-preserving, default).
    flip_y : bool
        Whether to invert Y axis (font y-up to model y-down). Recommended True.
    target_range : float
        Half-extent for norm_v1 scaling (ignored in norm_v2).
    em_normalize : bool
        Divide coordinates by upem when available. Should generally be True.
    apply_global_scale : bool
        Whether to apply a global scaling factor after strategy-specific steps.
    global_scale : Optional[float]
        If provided and apply_global_scale=True, multiply all coords by this factor.
        If None and apply_global_scale=True, no-op (placeholder for adaptive pipeline).
    clamp_after : Optional[float]
        If set, clamp coordinates to [-clamp_after, clamp_after] at the end.
    """

    strategy: str = Strategy.NORM_V2
    flip_y: bool = True
    target_range: float = 1.0
    em_normalize: bool = True
    apply_global_scale: bool = False
    global_scale: Optional[float] = None
    clamp_after: Optional[float] = None

    # Reserved for extra metadata / version bumping.
    extra: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.strategy not in ALLOWED_STRATEGIES:
            raise ValueError(f"Unsupported normalization strategy: {self.strategy}")
        if self.strategy == Strategy.NORM_V1 and self.target_range <= 0:
            raise ValueError("target_range must be positive for norm_v1")
        if self.clamp_after is not None and self.clamp_after <= 0:
            raise ValueError("clamp_after must be positive")


# ContourCommand-like minimal structure expected
class ContourCommandLike:
    def __init__(self, cmd: str, points: Tuple[Point, ...]):
        self.cmd = cmd
        self.points = points

    def replace_points(self, new_points: List[Point]) -> "ContourCommandLike":
        return ContourCommandLike(self.cmd, tuple(new_points))


ParsedGlyphContoursLike = List[List[ContourCommandLike]]


def _extract_all_points(parsed: ParsedGlyphContoursLike) -> List[Point]:
    pts: List[Point] = []
    for sub in parsed:
        for cmd in sub:
            pts.extend(cmd.points)
    return pts


def _compute_bbox(points: List[Point]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def _center_points(points: List[Point], cx: float, cy: float) -> List[Point]:
    return [(x - cx, y - cy) for x, y in points]


def _scale_points(points: List[Point], scale: float) -> List[Point]:
    if scale == 1.0:
        return points
    return [(x * scale, y * scale) for x, y in points]


def _flip_y_points(points: List[Point]) -> List[Point]:
    return [(x, -y) for x, y in points]


def _clamp_points(points: List[Point], limit: float) -> List[Point]:
    return [
        (
            max(-limit, min(limit, x)),
            max(-limit, min(limit, y)),
        )
        for x, y in points
    ]


def _reconstruct(
    parsed: ParsedGlyphContoursLike, new_points_iter
) -> ParsedGlyphContoursLike:
    """
    Rebuild contour structure with a flat iterator of new points.
    """
    new_parsed: ParsedGlyphContoursLike = []
    it = iter(new_points_iter)
    for sub in parsed:
        new_sub: List[ContourCommandLike] = []
        for cmd in sub:
            if cmd.points:
                n_pts = [next(it) for _ in cmd.points]
                new_sub.append(cmd.replace_points(n_pts))
            else:
                new_sub.append(cmd)  # 'z' or empty
        new_parsed.append(new_sub)
    return new_parsed


def _normalize_v1(
    parsed: ParsedGlyphContoursLike,
    cfg: NormalizationConfig,
    upem: Optional[int],
) -> Tuple[ParsedGlyphContoursLike, Dict[str, Any]]:
    points = _extract_all_points(parsed)
    if not points:
        return parsed, {
            "normalization_version": Strategy.NORM_V1,
            "bbox_raw": None,
            "width_em": 0.0,
            "height_em": 0.0,
            "center_em": (0.0, 0.0),
            "scale_applied": 1.0,
            "global_scale": None,
            "y_flipped": cfg.flip_y,
            "upem_used": upem,
        }

    # EM divide first (if available)
    em_factor = float(upem) if (cfg.em_normalize and upem and upem > 0) else 1.0
    scaled_em = [(x / em_factor, y / em_factor) for x, y in points]

    min_x, min_y, max_x, max_y = _compute_bbox(scaled_em)
    width = max_x - min_x
    height = max_y - min_y
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    centered = _center_points(scaled_em, cx, cy)

    # Scale to fit target_range
    max_dim = max(width, height) if max(width, height) > 0 else 1.0
    # We want half-extent ~ target_range => total span 2 * target_range
    per_glyph_scale = (cfg.target_range * 2.0) / max_dim

    sized = _scale_points(centered, per_glyph_scale)

    if cfg.flip_y:
        sized = _flip_y_points(sized)

    if cfg.apply_global_scale and cfg.global_scale:
        sized = _scale_points(sized, cfg.global_scale)

    if cfg.clamp_after:
        sized = _clamp_points(sized, cfg.clamp_after)

    rebuilt = _reconstruct(parsed, iter(sized))

    meta = {
        "normalization_version": Strategy.NORM_V1,
        "bbox_raw": (min_x, min_y, max_x, max_y),
        "width_em": width,
        "height_em": height,
        "center_em": (cx, cy),
        "scale_applied": per_glyph_scale,
        "global_scale": cfg.global_scale if cfg.apply_global_scale else None,
        "y_flipped": cfg.flip_y,
        "upem_used": upem,
    }
    return rebuilt, meta


def _normalize_v2(
    parsed: ParsedGlyphContoursLike,
    cfg: NormalizationConfig,
    upem: Optional[int],
) -> Tuple[ParsedGlyphContoursLike, Dict[str, Any]]:
    points = _extract_all_points(parsed)
    if not points:
        return parsed, {
            "normalization_version": Strategy.NORM_V2,
            "bbox_raw": None,
            "width_em": 0.0,
            "height_em": 0.0,
            "center_em": (0.0, 0.0),
            "scale_applied": 1.0,
            "global_scale": None,
            "y_flipped": cfg.flip_y,
            "upem_used": upem,
        }

    # 1. EM normalization
    em_factor = float(upem) if (cfg.em_normalize and upem and upem > 0) else 1.0
    em_pts = [(x / em_factor, y / em_factor) for x, y in points]

    # 2. Compute bbox & center
    min_x, min_y, max_x, max_y = _compute_bbox(em_pts)
    width = max_x - min_x
    height = max_y - min_y
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    centered = _center_points(em_pts, cx, cy)

    # 3. No per-glyph scale
    per_glyph_scale = 1.0
    sized = centered

    # 4. Optional flip
    if cfg.flip_y:
        sized = _flip_y_points(sized)

    # 5. Optional global scale
    if cfg.apply_global_scale and cfg.global_scale:
        sized = _scale_points(sized, cfg.global_scale)

    # 6. Optional clamp
    if cfg.clamp_after:
        sized = _clamp_points(sized, cfg.clamp_after)

    rebuilt = _reconstruct(parsed, iter(sized))

    meta = {
        "normalization_version": Strategy.NORM_V2,
        "bbox_raw": (min_x, min_y, max_x, max_y),
        "width_em": width,
        "height_em": height,
        "center_em": (cx, cy),
        "scale_applied": per_glyph_scale,
        "global_scale": cfg.global_scale if cfg.apply_global_scale else None,
        "y_flipped": cfg.flip_y,
        "upem_used": upem,
    }
    return rebuilt, meta


def apply_normalization(
    parsed: ParsedGlyphContoursLike,
    cfg: NormalizationConfig,
    upem: Optional[int],
) -> Tuple[ParsedGlyphContoursLike, Dict[str, Any]]:
    """
    Apply selected normalization strategy to parsed contours.
    """
    cfg.validate()
    if cfg.strategy == Strategy.NORM_V1:
        return _normalize_v1(parsed, cfg, upem)
    elif cfg.strategy == Strategy.NORM_V2:
        return _normalize_v2(parsed, cfg, upem)
    else:  # should not happen due to validate()
        raise ValueError(f"Unsupported strategy: {cfg.strategy}")


# Convenience factory for default embedding normalization
def default_embedding_config() -> NormalizationConfig:
    """
    Returns the default config (currently norm_v2, size-preserving).
    """
    return NormalizationConfig(strategy=Strategy.NORM_V2, flip_y=True)


# Debug / manual test
if __name__ == "__main__":
    # Minimal synthetic test
    square = [
        [
            ContourCommandLike("m", ((0.0, 0.0),)),
            ContourCommandLike("l", ((500.0, 0.0),)),
            ContourCommandLike("l", ((500.0, 500.0),)),
            ContourCommandLike("l", ((0.0, 500.0),)),
            ContourCommandLike("z", ()),
        ]
    ]
    cfg_v1 = NormalizationConfig(strategy=Strategy.NORM_V1, flip_y=True)
    cfg_v2 = NormalizationConfig(strategy=Strategy.NORM_V2, flip_y=True)
    norm1, meta1 = apply_normalization(square, cfg_v1, upem=1000)
    norm2, meta2 = apply_normalization(square, cfg_v2, upem=1000)
    print("norm_v1 meta:", meta1)
    print("norm_v2 meta:", meta2)
