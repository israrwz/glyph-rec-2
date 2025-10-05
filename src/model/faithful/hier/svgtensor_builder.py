"""
Faithful hierarchical SVGTensor builder for DeepSVG encoder input.

This module converts normalized glyph contour subpaths into a list of
`SVGTensor` objects (one per group / subpath) that conform to the original
DeepSVG data layout and semantics:

Command vocabulary (DeepSVG.COMMANDS_SIMPLIFIED):
    ["m", "l", "c", "a", "EOS", "SOS", "z"]

We currently emit only: m, l, c, z  (arcs 'a' are skipped; can be added later)

Argument slot ordering (n_args = 11):
    0: rx              (arc radius x)      -> unused (PAD)
    1: ry              (arc radius y)      -> unused (PAD)
    2: phi             (arc rotation)      -> unused (PAD)
    3: large_arc_flag  (arc flag)          -> unused (PAD)
    4: sweep_flag      (arc flag)          -> unused (PAD)
    5: control1.x      (cubic only)
    6: control1.y
    7: control2.x
    8: control2.y
    9: end_pos.x       (m, l, c)
   10: end_pos.y       (m, l, c)

PAD handling:
  - We store PAD_VAL = -1 for all inactive or absent argument slots.
  - The DeepSVG embedding layer applies (args + 1) before lookup, mapping PAD to 0.

EOS:
  - For each subpath, we append an EOS (via SVGTensor.add_eos()) then pad to
    max_seq_len with EOS + PAD (SVGTensor.pad()).
  - We *truncate* before adding EOS if the raw command count would exceed
    (max_seq_len - 1).

Optional synthetic 'z':
  - If ensure_close=True and the last emitted command is not 'z', we append a
    'z' (if capacity remains before truncation/EOS insertion).

Quantization:
  - Coordinates are assumed normalized (e.g., from a prior normalization step).
  - We clip each coordinate to [-clip_range, +clip_range] (default 1.2),
    then map to integer bins in [0, args_dim - 2].
  - These discrete bins become the raw absolute argument values (not relative).
  - Relative argument encoding (SVGTensor.get_relative_args()) can be added later
    based on configuration flags if needed.

High-Level Flow:
  1. Receive canonical normalized contour data: List[List[CommandLike]].
     Each CommandLike must have:
         .cmd   in {'m','l','c','z'}
         .points tuple-of-tuples:
             m,l : ((x,y),)
             c   : ((c1x,c1y),(c2x,c2y),(ex,ey))
             z   : ()
  2. For each subpath (in order):
       - Ensure a leading 'm' (synthesize from first encountered coordinate if missing).
       - Emit commands; skip unsupported.
       - Optionally append 'z'.
       - Truncate so raw length <= max_seq_len - 1 (reserve one slot for EOS).
       - Build stacked argument tensor (N, 11) and wrap in SVGTensor.
       - Call add_eos().pad(max_seq_len).
  3. Limit total subpaths to max_num_groups; ignore extras.
  4. Return list[SVGTensor] and stats.

Stats Recorded:
  - groups_built: number of SVGTensors created (non-empty)
  - truncated_subpaths: number of subpaths truncated due to length overflow
  - truncated_tokens: total tokens dropped across all subpaths
  - added_close: number of synthetic 'z' appended
  - empty_subpaths: count of subpaths that produced zero tokens (all skipped)
  - max_seq_len, max_num_groups
  - per_group_raw_lengths: raw (pre-EOS) lengths (post-truncation, pre-padding)

Author: Faithful hierarchical implementation phase
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple, Any, Dict, Optional

import torch

try:
    from deepsvg.difflib.tensor import SVGTensor
except ImportError as e:  # pragma: no cover
    raise RuntimeError(
        "DeepSVG not importable. Ensure 'deepsvg' repo or package is on PYTHONPATH."
    ) from e


# ---------------------------------------------------------------------------
# Configuration & Stats Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class HierBuilderConfig:
    max_num_groups: int
    max_seq_len: int
    args_dim: int = 256
    ensure_close: bool = True
    clip_range: float = 1.2
    allow_empty: bool = False  # if False, completely empty subpaths are skipped
    limit_subpaths: bool = True  # enforce max_num_groups ceiling


@dataclass
class HierBuildStats:
    groups_built: int = 0
    truncated_subpaths: int = 0
    truncated_tokens: int = 0
    added_close: int = 0
    empty_subpaths: int = 0
    per_group_raw_lengths: List[int] = field(default_factory=list)
    max_seq_len: int = 0
    max_num_groups: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "groups_built": self.groups_built,
            "truncated_subpaths": self.truncated_subpaths,
            "truncated_tokens": self.truncated_tokens,
            "added_close": self.added_close,
            "empty_subpaths": self.empty_subpaths,
            "per_group_raw_lengths": self.per_group_raw_lengths,
            "max_seq_len": self.max_seq_len,
            "max_num_groups": self.max_num_groups,
        }


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class HierSVGTensorBuilder:
    """
    Faithful hierarchical SVGTensor builder.

    Usage:
        builder = HierSVGTensorBuilder(HierBuilderConfig(...))
        svgt_list, stats = builder.build_glyph(parsed_norm)

    Where parsed_norm is:
        List[ List[CmdLike] ] with
            CmdLike.cmd in {'m','l','c','z'}
            CmdLike.points structured as described in module docstring.
    """

    _SUPPORTED = {"m", "l", "c", "z"}
    _CMD_INDEX = {c: i for i, c in enumerate(SVGTensor.COMMANDS_SIMPLIFIED)}

    def __init__(self, cfg: HierBuilderConfig):
        self.cfg = cfg
        # Keep a reserved max bin (args_dim-1) consistent with typical DeepSVG usage.
        self._value_max_bin = cfg.args_dim - 2

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def build_glyph(
        self, parsed_subpaths: Sequence[Sequence[Any]]
    ) -> Tuple[List[SVGTensor], HierBuildStats]:
        """
        Build SVGTensors for the given glyph contour subpaths.

        Returns:
            (svgt_list, stats)
        """
        stats = HierBuildStats(
            max_seq_len=self.cfg.max_seq_len, max_num_groups=self.cfg.max_num_groups
        )
        svgt_list: List[SVGTensor] = []

        # Iterate subpaths in order
        for sub_idx, sub in enumerate(parsed_subpaths):
            if self.cfg.limit_subpaths and len(svgt_list) >= self.cfg.max_num_groups:
                break

            svgt = self._build_subpath_svgtensor(sub, stats)
            if svgt is None:
                stats.empty_subpaths += 1
                if not self.cfg.allow_empty:
                    continue
            else:
                svgt_list.append(svgt)
                stats.groups_built += 1

        return svgt_list, stats

    # ------------------------------------------------------------------ #
    # Subpath → SVGTensor
    # ------------------------------------------------------------------ #
    def _build_subpath_svgtensor(
        self, sub: Sequence[Any], stats: HierBuildStats
    ) -> Optional[SVGTensor]:
        """
        Convert a single subpath into an SVGTensor (or None if empty / unsupported).
        """
        if not sub:
            return None

        # Accumulate linear command sequence
        cmds: List[str] = []
        c1_list: List[Tuple[float, float]] = []
        c2_list: List[Tuple[float, float]] = []
        end_list: List[Tuple[float, float]] = []

        # Helper to emit placeholders for control points for non-cubic
        def append_placeholders():
            c1_list.append((0.0, 0.0))
            c2_list.append((0.0, 0.0))
            end_list.append((0.0, 0.0))

        # First pass – gather raw commands
        for cmd_obj in sub:
            c = getattr(cmd_obj, "cmd", None)
            pts = getattr(cmd_obj, "points", ())
            if c not in self._SUPPORTED:
                continue
            if c in ("m", "l"):
                if not pts:
                    continue
                (x, y) = pts[0]
                cmds.append(c)
                c1_list.append((0.0, 0.0))
                c2_list.append((0.0, 0.0))
                end_list.append((x, y))
            elif c == "c":
                if len(pts) != 3:
                    continue
                (c1x, c1y), (c2x, c2y), (ex, ey) = pts
                cmds.append("c")
                c1_list.append((c1x, c1y))
                c2_list.append((c2x, c2y))
                end_list.append((ex, ey))
            elif c == "z":
                # close path
                cmds.append("z")
                append_placeholders()

        # Ensure at least one 'm' at start
        if cmds and cmds[0] != "m":
            # Attempt to synthesize from first end point or control point
            synth_pt = None
            for e in end_list:
                if e != (0.0, 0.0):
                    synth_pt = e
                    break
            if synth_pt is None:
                # fallback (0,0) if nothing else
                synth_pt = (0.0, 0.0)
            cmds.insert(0, "m")
            c1_list.insert(0, (0.0, 0.0))
            c2_list.insert(0, (0.0, 0.0))
            end_list.insert(0, synth_pt)

        if not cmds:
            return None

        # Optional synthetic close 'z'
        added_close = 0
        if (
            self.cfg.ensure_close
            and cmds[-1] != "z"
            and len(cmds) < self.cfg.max_seq_len
        ):
            cmds.append("z")
            c1_list.append((0.0, 0.0))
            c2_list.append((0.0, 0.0))
            end_list.append((0.0, 0.0))
            added_close = 1

        # Truncation (reserve 1 slot for EOS)
        max_payload = self.cfg.max_seq_len - 1
        truncated_tokens = 0
        if len(cmds) > max_payload:
            truncated_tokens = len(cmds) - max_payload
            cmds = cmds[:max_payload]
            c1_list = c1_list[:max_payload]
            c2_list = c2_list[:max_payload]
            end_list = end_list[:max_payload]
            stats.truncated_subpaths += 1
            stats.truncated_tokens += truncated_tokens
            # If we truncated a synthetic close, reduce the count
            if added_close and cmds[-1] != "z":
                stats.added_close -= 1
                added_close = 0

        stats.added_close += added_close
        stats.per_group_raw_lengths.append(len(cmds))

        # Build command indices
        cmd_indices = torch.tensor(
            [self._CMD_INDEX.get(c, self._CMD_INDEX["EOS"]) for c in cmds],
            dtype=torch.long,
        )

        # Quantize coordinates
        q_c1 = [self._quantize_xy(x, y) for (x, y) in c1_list]
        q_c2 = [self._quantize_xy(x, y) for (x, y) in c2_list]
        q_end = [self._quantize_xy(x, y) for (x, y) in end_list]

        # Construct argument matrix shape (len(cmds), 11)
        # rx,ry,phi,fA,fS,(c1x,c1y),(c2x,c2y),(ex,ey)
        n = len(cmds)
        radius = torch.zeros(n, 2)
        phi = torch.zeros(n, 1)
        large_arc = torch.zeros(n, 1)
        sweep = torch.zeros(n, 1)
        control1 = torch.tensor(q_c1, dtype=torch.float32)
        control2 = torch.tensor(q_c2, dtype=torch.float32)
        end_pos = torch.tensor(q_end, dtype=torch.float32)

        # Assemble args: (n, 11)
        args = torch.cat(
            [
                radius,  # 0:2
                phi,  # 2
                large_arc,  # 3
                sweep,  # 4
                control1,  # 5:7
                control2,  # 7:9
                end_pos,  # 9:11
            ],
            dim=1,
        )

        # Convert to SVGTensor
        svgt = SVGTensor.from_cmd_args(
            cmd_indices,
            args,
            PAD_VAL=-1,
            ARGS_DIM=self.cfg.args_dim,
        )
        svgt.add_eos().pad(seq_len=self.cfg.max_seq_len)
        return svgt

    # ------------------------------------------------------------------ #
    # Quantization Helpers
    # ------------------------------------------------------------------ #
    def _quantize_xy(self, x: float, y: float) -> Tuple[float, float]:
        """
        Quantize (x,y) to discrete bins in [0, value_max_bin] as floats
        (SVGTensor stores raw floats; the later embedding shifts & bins).
        """
        return float(self._quantize_scalar(x)), float(self._quantize_scalar(y))

    def _quantize_scalar(self, v: float) -> int:
        cr = self.cfg.clip_range
        if v <= -cr:
            return 0
        if v >= cr:
            return self._value_max_bin
        ratio = (v + cr) / (2 * cr)  # map [-cr,cr] -> [0,1]
        q = int(round(ratio * self._value_max_bin))
        if q < 0:
            return 0
        if q > self._value_max_bin:
            return self._value_max_bin
        return q


# ---------------------------------------------------------------------------
# Convenience Factory
# ---------------------------------------------------------------------------


def build_hier_builder_from_model_cfg(model_cfg) -> HierSVGTensorBuilder:
    """
    Construct a HierSVGTensorBuilder from a DeepSVG model cfg object.
    Expects attributes: max_num_groups, max_seq_len, args_dim.
    """
    cfg = HierBuilderConfig(
        max_num_groups=getattr(model_cfg, "max_num_groups"),
        max_seq_len=getattr(model_cfg, "max_seq_len"),
        args_dim=getattr(model_cfg, "args_dim", 256),
        ensure_close=True,
        clip_range=1.2,
    )
    return HierSVGTensorBuilder(cfg)


# ---------------------------------------------------------------------------
# Self-Test (Optional)
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover

    @dataclass
    class DummyCmd:
        cmd: str
        points: Tuple[Tuple[float, float], ...]

    # Two subpaths: one cubic heavy, one simple
    glyph_subpaths = [
        [
            DummyCmd("m", ((0.0, 0.0),)),
            DummyCmd("c", ((0.2, 0.3), (0.4, 0.6), (0.5, 0.7))),
            DummyCmd("l", ((0.9, -0.2),)),
        ],
        [
            DummyCmd("m", ((-0.5, 0.5),)),
            DummyCmd("l", ((-0.1, 0.2),)),
            DummyCmd("z", ()),
        ],
    ]

    builder = HierSVGTensorBuilder(
        HierBuilderConfig(max_num_groups=4, max_seq_len=16, args_dim=256)
    )
    svgs, st = builder.build_glyph(glyph_subpaths)
    print("Built groups:", len(svgs))
    for i, svgt in enumerate(svgs):
        print(f" Group {i} seq_len(raw padded) =", int(svgt.seq_len.item()))
    print("Stats:", st.to_dict())
