"""
Top-level package for glyph recognition embedding pipeline.

This `src` package contains project-specific code that layers on top of the
vendored DeepSVG repository (located at `./deepsvg/`). We intentionally keep
our domain logic isolated here so that upstream DeepSVG code can be updated
or patched with minimal merge friction.

Subpackages (planned / in-progress):
- data:    Glyph + font data access, contour parsing, normalization utilities.
- model:   Pretrained DeepSVG encoder loading, embedding extraction helpers.
- index:   (Future) Embedding storage abstractions (memmap / vector DB).
- eval:    (Future) Metrics, nearest-neighbor quality reports.
- scripts: Entry points / CLI drivers for embedding + evaluation workflows.

Design Notes:
- No heavy imports at package import time; defer expensive dependencies
  (e.g., torch, sqlite3) to module-level functions to keep CLI startup snappy.
- All random / deterministic behaviors should pass an explicit seed parameter.
- Avoid modifying the vendored DeepSVG code; wrap it instead.

Environment Assumptions:
- Python >= 3.9 (macOS 2019 MBP CPU workflow).
- Initially compatible with torch==1.4.0; future upgrade path to newer PyTorch.

Utility:
`get_version()` returns a semantic-ish internal version string for logging.

"""

from __future__ import annotations

__all__ = [
    "get_version",
]

# Internal version for our project layer (unrelated to DeepSVG's version).
_PROJECT_VERSION = "0.1.0-phase1-planning"


def get_version() -> str:
    """
    Return the current project package version.

    This is distinct from any model versioning or embedding schema versions
    (which will be tracked separately in metadata artifacts).
    """
    return _PROJECT_VERSION
