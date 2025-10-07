"""
Raster Glyph Embedding Package

This package contains the raster-based glyph embedding pipeline:
- rasterize.py      : Contour → normalized grayscale tensor rendering
- dataset.py        : On-the-fly raster dataset with light augmentations
- model.py          : LeViT_128S (img_size=128) wrapper + embedding head
- train.py          : Minimal training loop (classification + retrieval metrics)
- embed.py          : Batch embedding extraction + metadata export
- eval_similarity.py: Lightweight retrieval / effect size evaluation
- visualize.py      : Grid visualization & polyline overlay diagnostics
- plan.md           : Design / implementation plan & status

NOTE ON EXECUTION CONTEXT
-------------------------
If you invoke scripts via:
    python raster/visualize.py
the relative imports inside those scripts can fail because Python sets
sys.path[0] to the *raster/* directory, not its parent, so the package
name "raster" is not discoverable.

Preferred invocation (ensures proper package context):
    python -m raster.visualize        ...
    python -m raster.train            ...
    python -m raster.embed            ...
    python -m raster.eval_similarity  ...

Alternatively, you can add the project root to PYTHONPATH:
    export PYTHONPATH="$(pwd):$PYTHONPATH"
    python raster/visualize.py ...

Import Strategy
---------------
Historically this module unconditionally imported the model wrapper which
pulls in the LeViT repo and its dependency `timm`. Visualization and
raster-only utilities do NOT require `timm`; failing early due to a missing
`timm` install blocked raster QA.

We now:
- Attempt to import model exports in a try/except.
- Expose a flag `HAS_MODEL` for callers.
- Only add model symbols to __all__ if successfully imported.

Convenience Re-exports
----------------------
We re-export a few primary entry points for ergonomic interactive use:

    from raster import build_glyph_levit_128s, GlyphLeViTConfig, Rasterizer

Versioning
----------
No semantic versioning yet; treat as 0.x experimental while Phase 1
metrics are being validated.
"""

# Always available (no heavy deps)
from .rasterize import Rasterizer, RasterizerConfig  # noqa: F401

_HAS_MODEL = False
try:
    from .model import build_glyph_levit_128s, GlyphLeViTConfig  # noqa: F401

    _HAS_MODEL = True
except Exception:  # Broad catch: missing timm, LeViT path, etc.
    # Graceful degradation: rasterization & visualization still work.
    build_glyph_levit_128s = None  # type: ignore
    GlyphLeViTConfig = None  # type: ignore

HAS_MODEL = _HAS_MODEL  # Public flag

__all__ = [
    "Rasterizer",
    "RasterizerConfig",
    "HAS_MODEL",
]

if _HAS_MODEL:
    __all__.extend(
        [
            "build_glyph_levit_128S",
            "build_glyph_levit_128s",  # prefer snake case
            "GlyphLeViTConfig",
        ]
    )

# Backwards compatibility: expose canonical name if model present
if _HAS_MODEL and "build_glyph_levit_128S" not in globals():
    # Provide a capital S alias (harmless if not used)
    build_glyph_levit_128S = build_glyph_levit_128s  # type: ignore
