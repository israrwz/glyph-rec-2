"""
Repo-faithful preprocessing for DeepSVG encoder inputs.

Goal
----
Produce grouped (commands, args) tensors that closely match what DeepSVG's
dataset / SVGTensor preprocessing generates for the encoder, eliminating
ad-hoc shortcuts that led to NaNs or degenerate embeddings.

Key Semantics Replicated
------------------------
1. Command vocabulary (fixed order):
       ["m", "l", "c", "a", "EOS", "SOS", "z"]
   - We currently generate only m, l, c, z, EOS (no arcs, no SOS for encoder).
   - EOS also doubles as PAD for commands (DeepSVG uses cumulative EOS mask).

2. Argument tensor layout (n_args = 11), slots:
       0: rx                (arc radius x)    - unused => PAD
       1: ry                (arc radius y)    - unused => PAD
       2: phi               (arc rotation)    - unused => PAD
       3: large_arc_flag    (arc flag)        - unused => PAD
       4: sweep_flag        (arc flag)        - unused => PAD
       5: control1.x        (cubic only)
       6: control1.y        (cubic only)
       7: control2.x        (cubic only)
       8: control2.y        (cubic only)
       9: end_pos.x         (m,l,c)
      10: end_pos.y         (m,l,c)

3. PAD handling:
   - Raw args use PAD_VAL = -1 for any inactive slot (matches repository).
   - The encoder embedding layer internally adds +1 ( (args + 1) ) so PAD (-1) -> 0
     while real bins become >=1. We must ensure no value < -1 or >= args_dim.

4. Quantization:
   - We map normalized continuous coordinates (already roughly in [-1, 1] or a
     configured window) into integer bins [0, args_dim-2].
     (args_dim-1 reserved conceptually; consistent with original practice of
      leaving headroom for relative encoding variants.)
   - After embedding shift inside model, bins occupy [1 .. args_dim-1].

5. Grouping & Padding:
   - We produce exactly cfg.max_num_groups groups.
   - Each group sequence has length cfg.max_seq_len.
   - Pre-filled with EOS; real commands overwrite prefix.
   - Remaining tail positions remain EOS.

6. Closing Paths:
   - If ensure_z=True and a subpath’s last emitted command is not 'z' and there
     is still capacity, we append a 'z'.

7. Truncation:
   - If a subpath exceeds max_seq_len, extra tokens are dropped (logged in stats).
   - If total subpaths exceed max_num_groups, extras are dropped.

8. Stages:
   - For encode_stages == 2 (hierarchical), grouped tensors (G,S) are expected.
   - For encode_stages == 1 (flat), we optionally flatten groups into a single
     sequence up to cfg.max_total_len if provided (not yet required, but hooks
     provided).

Inputs
------
parsed_contours: List[List[ContourCommandLike]]
  Where each ContourCommandLike has:
       .cmd   in {'m','l','c','z'}
       .points: tuple of points:
           m,l : ((x,y),)
           c   : ((c1x,c1y),(c2x,c2y),(ex,ey))
           z   : ()

You can integrate with your existing contour_parser output which already
normalizes and converts qCurveTo -> cubic.

Outputs
-------
Group-level tensors shaped:
  commands_grouped : (N, G, S)
  args_grouped     : (N, G, S, n_args)
with raw PAD_VAL = -1 for args.

Stats:
  Per glyph:
     {
       'groups_used',
       'tokens_encoded',
       'truncated_subpaths',
       'truncated_tokens',
       'empty' (bool),
       'per_group_lengths': [...],
       'added_close_commands': count
     }

Validation Helpers:
  - ensure_no_invalid_indices() quickly asserts index bounds.
  - summarize_batch() returns aggregate utilization info.

Usage Example
-------------
    cfg = FaithfulBuilderConfig(
        max_num_groups=model_cfg.max_num_groups,
        max_seq_len=model_cfg.max_seq_len,
        max_total_len=getattr(model_cfg, 'max_total_len', model_cfg.max_seq_len * model_cfg.max_num_groups),
        n_args=model_cfg.n_args,
        args_dim=model_cfg.args_dim,
        n_commands=len(COMMANDS_SIMPLIFIED),
        encode_stages=model_cfg.encode_stages,
    )
    builder = RepoFaithfulPreprocessor(cfg)
    commands_g, args_g, stats = builder.glyph_to_group_tensors(parsed_norm)
    batch = builder.collate([(commands_g, args_g, stats)])

Caveats / Future Fidelity Enhancements
--------------------------------------
- Arc commands ('a') omitted (font focus rarely requires).
- Relative argument encoding (rel_targets / rel_args) not implemented yet.
- SOS tokens not added (encoder path does not require them).
- Potential improvement: replicate exact relative coordinate derivation logic
  from SVGTensor.get_relative_args when config.rel_targets is True.

Author: Phase 1F (Faithful preprocessing)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Sequence, Dict, Any, Optional

import math
import torch


# ---------------------------------------------------------------------------
# Constants & Command Vocabulary
# ---------------------------------------------------------------------------

COMMANDS_SIMPLIFIED = ["m", "l", "c", "a", "EOS", "SOS", "z"]
IDX_CMD = {c: i for i, c in enumerate(COMMANDS_SIMPLIFIED)}
EOS_IDX = IDX_CMD["EOS"]
SUPPORTED = {"m", "l", "c", "z"}  # subset we emit

PAD_VAL = -1  # raw PAD for args BEFORE implicit +1 shift inside model embedding


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class FaithfulBuilderConfig:
    max_num_groups: int
    max_seq_len: int
    max_total_len: int  # For potential encode_stages == 1 flattening
    n_args: int
    args_dim: int
    n_commands: int
    encode_stages: int = 2  # 2 = hierarchical (groups+sequence), 1 = flat
    rel_args: bool = False
    rel_targets: bool = False
    ensure_close: bool = True
    allow_truncate: bool = True
    clip_range: float = 1.2  # coordinate clipping window [-clip_range, clip_range]
    auto_drop_empty: bool = True
    collect_stats: bool = True
    # Future toggles / placeholders:
    add_sos: bool = False  # Not used currently for encoder side
    add_trailing_eos: bool = False  # Already implicitly padded with EOS


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass
class GlyphBuildStats:
    groups_used: int = 0
    tokens_encoded: int = 0
    per_group_lengths: List[int] = field(default_factory=list)
    truncated_subpaths: int = 0
    truncated_tokens: int = 0
    added_close_commands: int = 0
    empty: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "groups_used": self.groups_used,
            "tokens_encoded": self.tokens_encoded,
            "per_group_lengths": self.per_group_lengths,
            "truncated_subpaths": self.truncated_subpaths,
            "truncated_tokens": self.truncated_tokens,
            "added_close_commands": self.added_close_commands,
            "empty": self.empty,
        }


# ---------------------------------------------------------------------------
# Core Preprocessor
# ---------------------------------------------------------------------------


class RepoFaithfulPreprocessor:
    """
    Repo-faithful builder producing grouped command & args tensors for the DeepSVG encoder.

    This class emits raw (-1 padded) argument indices. The model embedding layer
    performs the (+1) shift internally.

    Primary method:
        glyph_to_group_tensors(parsed_norm) -> (commands_g, args_g, stats)
    """

    def __init__(self, cfg: FaithfulBuilderConfig):
        self.cfg = cfg
        self.cmd_to_idx = IDX_CMD
        self.EOS = EOS_IDX
        self.value_max_bin = cfg.args_dim - 2  # Reserve top slot & PAD semantics

        if self.cfg.n_commands != len(COMMANDS_SIMPLIFIED):
            raise ValueError(
                f"Config n_commands={self.cfg.n_commands} != len(COMMANDS_SIMPLIFIED)={len(COMMANDS_SIMPLIFIED)}"
            )

        if self.cfg.n_args != 11:
            raise ValueError(
                f"Faithful pipeline expects n_args=11 (got {self.cfg.n_args}). Adapt slot mapping if changed."
            )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def glyph_to_group_tensors(
        self, parsed_norm: Sequence[Sequence[Any]]
    ) -> Tuple[torch.Tensor, torch.Tensor, GlyphBuildStats]:
        """
        Convert normalized parsed contour subpaths into grouped command & args tensors.

        parsed_norm: List[List[CommandLike]]
            Each command-like object has `.cmd` and `.points`.

        Returns:
            commands_g : LongTensor (G, S)
            args_g     : LongTensor (G, S, n_args) (PAD_VAL = -1 for unused)
            stats      : GlyphBuildStats
        """
        # Branch by encode stages:
        if self.cfg.encode_stages == 1:
            # ---------- Single-stage (flatten) ----------
            T = self.cfg.max_total_len
            n_args = self.cfg.n_args
            commands_flat = torch.full((1, T), self.EOS, dtype=torch.long)
            args_flat = torch.full((1, T, n_args), PAD_VAL, dtype=torch.long)
            stats = GlyphBuildStats()

            cursor = 0
            added_close_total = 0

            for sub in parsed_norm:
                if cursor >= T:
                    break
                # Filter supported commands
                filtered = [c for c in sub if getattr(c, "cmd", None) in SUPPORTED]
                if not filtered:
                    continue

                # Ensure leading move
                if filtered[0].cmd != "m":
                    first_pt = None
                    for c in filtered:
                        pts = getattr(c, "points", None)
                        if pts:
                            first_pt = pts[0]
                            break
                    if first_pt and cursor < T:
                        commands_flat[0, cursor] = self.cmd_to_idx["m"]
                        qx, qy = self._quantize_xy(first_pt[0], first_pt[1])
                        args_flat[0, cursor, 9] = qx
                        args_flat[0, cursor, 10] = qy
                        cursor += 1

                for cmd_obj in filtered:
                    if cursor >= T:
                        break
                    c = cmd_obj.cmd
                    if c == "m":
                        (x, y) = cmd_obj.points[0]
                        commands_flat[0, cursor] = self.cmd_to_idx["m"]
                        qx, qy = self._quantize_xy(x, y)
                        args_flat[0, cursor, 9] = qx
                        args_flat[0, cursor, 10] = qy
                        cursor += 1
                    elif c == "l":
                        (x, y) = cmd_obj.points[0]
                        commands_flat[0, cursor] = self.cmd_to_idx["l"]
                        qx, qy = self._quantize_xy(x, y)
                        args_flat[0, cursor, 9] = qx
                        args_flat[0, cursor, 10] = qy
                        cursor += 1
                    elif c == "c" and len(cmd_obj.points) == 3:
                        (c1x, c1y), (c2x, c2y), (ex, ey) = cmd_obj.points
                        commands_flat[0, cursor] = self.cmd_to_idx["c"]
                        q1x, q1y = self._quantize_xy(c1x, c1y)
                        q2x, q2y = self._quantize_xy(c2x, c2y)
                        qex, qey = self._quantize_xy(ex, ey)
                        args_flat[0, cursor, 5] = q1x
                        args_flat[0, cursor, 6] = q1y
                        args_flat[0, cursor, 7] = q2x
                        args_flat[0, cursor, 8] = q2y
                        args_flat[0, cursor, 9] = qex
                        args_flat[0, cursor, 10] = qey
                        cursor += 1
                    elif c == "z":
                        commands_flat[0, cursor] = self.cmd_to_idx["z"]
                        cursor += 1

                if (
                    self.cfg.ensure_close
                    and cursor < T
                    and (
                        cursor == 0
                        or commands_flat[0, cursor - 1] != self.cmd_to_idx["z"]
                    )
                ):
                    commands_flat[0, cursor] = self.cmd_to_idx["z"]
                    cursor += 1
                    added_close_total += 1

            stats.tokens_encoded = cursor
            stats.groups_used = 1 if cursor > 0 else 0
            stats.per_group_lengths = [cursor]
            stats.added_close_commands = added_close_total
            stats.empty = cursor == 0
            return commands_flat, args_flat, stats

        # ---------- Hierarchical (encode_stages == 2) ----------
        G, S, n_args = (
            self.cfg.max_num_groups,
            self.cfg.max_seq_len,
            self.cfg.n_args,
        )

        commands_g = torch.full((G, S), self.EOS, dtype=torch.long)
        args_g = torch.full((G, S, n_args), PAD_VAL, dtype=torch.long)
        stats = GlyphBuildStats()

        subpaths = list(parsed_norm)
        if len(subpaths) > G:
            stats.truncated_subpaths = len(subpaths) - G
            subpaths = subpaths[:G]

        g_out = 0
        for sub in subpaths:
            if g_out >= G:
                break
            seq_len_used, added_close = self._encode_subpath(
                sub, commands_g, args_g, g_out
            )
            if seq_len_used > 0:
                stats.per_group_lengths.append(seq_len_used)
                stats.tokens_encoded += seq_len_used
                stats.groups_used += 1
                stats.added_close_commands += added_close
                g_out += 1

        stats.empty = stats.tokens_encoded == 0
        return commands_g, args_g, stats

    def collate(
        self,
        glyph_triplets: Sequence[
            Tuple[torch.Tensor, torch.Tensor, GlyphBuildStats]
            | Tuple[torch.Tensor, torch.Tensor]
        ],
        device: Optional[str] = None,
        aggregate_stats: bool = True,
    ) -> Dict[str, Any]:
        """
        Collate per-glyph command/arg tensors (and optional stats) into batch.

        Returns:
            {
              'commands_grouped': (N,G,S),
              'args_grouped': (N,G,S,n_args),
              'stats': {...} (if aggregate_stats & stats present)
            }
        """
        if not glyph_triplets:
            raise ValueError("Empty glyph_triplets in collate")

        normalized: List[
            Tuple[torch.Tensor, torch.Tensor, Optional[GlyphBuildStats]]
        ] = []
        for item in glyph_triplets:
            if len(item) == 2:
                c, a = item  # type: ignore
                normalized.append((c, a, None))
            elif len(item) == 3:
                c, a, s = item  # type: ignore
                normalized.append((c, a, s))
            else:
                raise ValueError("Each item must be length 2 or 3.")

        # Shape consistency
        c0, a0, _ = normalized[0]
        for i, (c, a, _) in enumerate(normalized):
            if c.shape != c0.shape or a.shape != a0.shape:
                raise ValueError(
                    f"Inconsistent shapes at index {i}: {tuple(c.shape)} vs {tuple(c0.shape)}, {tuple(a.shape)} vs {tuple(a0.shape)}"
                )

        batch_cmds = torch.stack([t[0] for t in normalized], dim=0)
        batch_args = torch.stack([t[1] for t in normalized], dim=0)

        if device:
            batch_cmds = batch_cmds.to(device)
            batch_args = batch_args.to(device)

        out: Dict[str, Any] = {
            "commands_grouped": batch_cmds,
            "args_grouped": batch_args,
        }

        if aggregate_stats:
            collected = [t[2] for t in normalized if t[2] is not None]
            if collected:
                out["stats"] = self._aggregate_stats(collected)

        return out

    # ------------------------------------------------------------------ #
    # Internal Encoding
    # ------------------------------------------------------------------ #

    def _encode_subpath(
        self,
        subpath: Sequence[Any],
        commands_g: torch.Tensor,
        args_g: torch.Tensor,
        g_idx: int,
    ) -> Tuple[int, int]:
        """
        Encode a single subpath into group row g_idx.

        Returns:
            (seq_len_used, added_close_flag)
        """
        S = self.cfg.max_seq_len
        seq_i = 0
        added_close = 0

        if not subpath:
            return 0, added_close

        # Ensure first command is a move (if not present, synthesize from first point)
        if subpath[0].cmd != "m":
            first_pt = None
            for cmd in subpath:
                if getattr(cmd, "points", None):
                    pts = cmd.points
                    if pts:
                        first_pt = pts[0]
                        break
            if first_pt:
                seq_i = self._emit_move(commands_g, args_g, g_idx, seq_i, first_pt)
            # continue encoding remainder

        for cmd_obj in subpath:
            if seq_i >= S:
                # Truncation: remaining tokens ignored
                break
            c = cmd_obj.cmd
            if c not in SUPPORTED:
                continue
            if c == "m":
                if cmd_obj.points:
                    seq_i = self._emit_move(
                        commands_g, args_g, g_idx, seq_i, cmd_obj.points[0]
                    )
            elif c == "l":
                if cmd_obj.points:
                    seq_i = self._emit_line(
                        commands_g, args_g, g_idx, seq_i, cmd_obj.points[0]
                    )
            elif c == "c":
                if len(cmd_obj.points) == 3:
                    seq_i = self._emit_cubic(
                        commands_g,
                        args_g,
                        g_idx,
                        seq_i,
                        cmd_obj.points[0],
                        cmd_obj.points[1],
                        cmd_obj.points[2],
                    )
            elif c == "z":
                seq_i = self._emit_close(commands_g, g_idx, seq_i)

        # Append synthetic close if requested:
        if (
            self.cfg.ensure_close
            and seq_i < S
            and seq_i > 0
            and (seq_i == 0 or commands_g[g_idx, seq_i - 1] != self.cmd_to_idx["z"])
        ):
            commands_g[g_idx, seq_i] = self.cmd_to_idx["z"]
            seq_i += 1
            added_close = 1

        return seq_i, added_close

    # -------------------- Emit Helpers -------------------- #

    def _emit_move(
        self,
        commands: torch.Tensor,
        args: torch.Tensor,
        g: int,
        s: int,
        pt: Tuple[float, float],
    ) -> int:
        commands[g, s] = self.cmd_to_idx["m"]
        qx, qy = self._quantize_xy(pt[0], pt[1])
        args[g, s, 9] = qx
        args[g, s, 10] = qy
        return s + 1

    def _emit_line(
        self,
        commands: torch.Tensor,
        args: torch.Tensor,
        g: int,
        s: int,
        pt: Tuple[float, float],
    ) -> int:
        commands[g, s] = self.cmd_to_idx["l"]
        qx, qy = self._quantize_xy(pt[0], pt[1])
        args[g, s, 9] = qx
        args[g, s, 10] = qy
        return s + 1

    def _emit_cubic(
        self,
        commands: torch.Tensor,
        args: torch.Tensor,
        g: int,
        s: int,
        c1: Tuple[float, float],
        c2: Tuple[float, float],
        end_pt: Tuple[float, float],
    ) -> int:
        commands[g, s] = self.cmd_to_idx["c"]
        q_c1x, q_c1y = self._quantize_xy(c1[0], c1[1])
        q_c2x, q_c2y = self._quantize_xy(c2[0], c2[1])
        q_ex, q_ey = self._quantize_xy(end_pt[0], end_pt[1])
        args[g, s, 5] = q_c1x
        args[g, s, 6] = q_c1y
        args[g, s, 7] = q_c2x
        args[g, s, 8] = q_c2y
        args[g, s, 9] = q_ex
        args[g, s, 10] = q_ey
        return s + 1

    def _emit_close(
        self,
        commands: torch.Tensor,
        g: int,
        s: int,
    ) -> int:
        commands[g, s] = self.cmd_to_idx["z"]
        return s + 1

    # ------------------------------------------------------------------ #
    # Quantization / Utility
    # ------------------------------------------------------------------ #

    def _quantize_xy(self, x: float, y: float) -> Tuple[int, int]:
        cr = self.cfg.clip_range
        return self._quantize_scalar(x, -cr, cr), self._quantize_scalar(y, -cr, cr)

    def _quantize_scalar(self, v: float, vmin: float, vmax: float) -> int:
        if v <= vmin:
            return 0
        if v >= vmax:
            return self.value_max_bin
        ratio = (v - vmin) / (vmax - vmin)
        q = int(round(ratio * self.value_max_bin))
        if q < 0:
            return 0
        if q > self.value_max_bin:
            return self.value_max_bin
        return q

    # ------------------------------------------------------------------ #
    # Stats Aggregation
    # ------------------------------------------------------------------ #

    def _aggregate_stats(
        self, glyph_stats: Sequence[GlyphBuildStats]
    ) -> Dict[str, Any]:
        agg: Dict[str, Any] = {}
        if not glyph_stats:
            return agg

        def collect(field: str) -> List[float]:
            vals = []
            for s in glyph_stats:
                v = getattr(s, field, None)
                if isinstance(v, (int, float)):
                    vals.append(float(v))
            return vals

        for field in [
            "groups_used",
            "tokens_encoded",
            "truncated_subpaths",
            "truncated_tokens",
            "added_close_commands",
        ]:
            nums = collect(field)
            if nums:
                agg[field] = {
                    "mean": sum(nums) / len(nums),
                    "min": min(nums),
                    "max": max(nums),
                    "total": sum(nums),
                    "count": len(nums),
                }

        # Flatten per_group_lengths
        per_group_lengths_all: List[int] = []
        for s in glyph_stats:
            per_group_lengths_all.extend(s.per_group_lengths)
        if per_group_lengths_all:
            ag = per_group_lengths_all
            agg["per_group_lengths"] = {
                "mean": sum(ag) / len(ag),
                "min": min(ag),
                "max": max(ag),
                "count": len(ag),
            }

        empties = sum(1 for s in glyph_stats if s.empty)
        agg["empty_glyphs"] = empties

        return agg

    # ------------------------------------------------------------------ #
    # Debug / Validation Helpers
    # ------------------------------------------------------------------ #

    def ensure_no_invalid_indices(self, commands: torch.Tensor, args: torch.Tensor):
        """
        Quick validation: commands in [0,n_commands-1], args >= -1, args < args_dim.
        """
        if commands.min().item() < 0 or commands.max().item() >= self.cfg.n_commands:
            raise ValueError(
                f"Invalid command index range: min={commands.min().item()} max={commands.max().item()}"
            )
        if args.min().item() < -1:
            raise ValueError(f"Args contain value < -1 (min={args.min().item()})")
        if (args >= self.cfg.args_dim).any():
            raise ValueError("Args contain value >= args_dim (out of range)")

    def summarize_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        cmds = batch["commands_grouped"]
        args = batch["args_grouped"]
        N, G, S = cmds.shape
        non_eos_per_sample = (cmds != self.EOS).sum(dim=(1, 2))
        utilization = non_eos_per_sample.float() / float(G * S)
        return {
            "N": N,
            "G": G,
            "S": S,
            "mean_tokens": float(non_eos_per_sample.float().mean().item()),
            "min_tokens": int(non_eos_per_sample.min().item()),
            "max_tokens": int(non_eos_per_sample.max().item()),
            "mean_utilization": float(utilization.mean().item()),
        }


# ---------------------------------------------------------------------------
# Minimal Self-Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    @dataclass
    class DummyCmd:
        cmd: str
        points: Tuple[Tuple[float, float], ...]

    # Two glyphs: one simple, one multi-subpath with truncation test
    glyph1 = [
        [
            DummyCmd("m", ((0.0, 0.0),)),
            DummyCmd("l", ((0.3, 0.1),)),
            DummyCmd("c", ((0.1, 0.2), (0.2, 0.4), (0.5, 0.6))),
            DummyCmd("z", ()),
        ]
    ]
    glyph2 = [
        [
            DummyCmd("m", ((-0.5, -0.5),)),
            DummyCmd("c", ((-0.3, -0.3), (0.3, 0.3), (0.4, 0.4))),
            DummyCmd("l", ((0.9, 1.3),)),
        ],
        [
            DummyCmd("m", ((0.2, 0.2),)),
            DummyCmd("l", ((0.6, 0.6),)),
        ],
    ]

    cfg = FaithfulBuilderConfig(
        max_num_groups=4,
        max_seq_len=10,
        max_total_len=40,
        n_args=11,
        args_dim=256,
        n_commands=len(COMMANDS_SIMPLIFIED),
        encode_stages=2,
    )
    builder = RepoFaithfulPreprocessor(cfg)

    g1_cmds, g1_args, g1_stats = builder.glyph_to_group_tensors(glyph1)
    g2_cmds, g2_args, g2_stats = builder.glyph_to_group_tensors(glyph2)

    batch = builder.collate(
        [(g1_cmds, g1_args, g1_stats), (g2_cmds, g2_args, g2_stats)]
    )

    print("Glyph1 stats:", g1_stats.to_dict())
    print("Glyph2 stats:", g2_stats.to_dict())
    print("Batch summary:", builder.summarize_batch(batch))
    builder.ensure_no_invalid_indices(batch["commands_grouped"], batch["args_grouped"])
