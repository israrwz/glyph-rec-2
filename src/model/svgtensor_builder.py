"""
svgtensor_builder.py

Refactored SVGTensor builder with:
  - Stats-aware glyph conversion
  - Explicit compaction of real (non-EOS) tokens
  - Detection & signaling of empty glyphs (so caller can skip)
  - Backward compatible API (glyph_to_group_tensors still works)
  - Optional aggregated stats at batch collation

Rationale
---------
Previous implementation pre-filled every (group, sequence) slot with EOS.
This interacted poorly with DeepSVG's padding mask logic:
    padding_mask = (commands == EOS).cumsum(dim=seq) == 0
If an entire group (sequence) was all EOS, the valid token count became zero and
downstream pooling produced division-by-zero -> NaNs.

This refactor ensures:
  - Only subpaths with at least one supported command produce a group
  - Groups after the last real group remain EOS (fine)
  - A glyph with 0 valid commands raises a ValueError -> caller can skip it
  - Rich per‑glyph stats enable diagnostics and dataset quality filtering

Supported Commands
------------------
Subset: m, l, c, z
(Arcs and others excluded in Phase 1 scope.)

Arguments Layout (n_args = 11)
------------------------------
Indices follow DeepSVG canonical order:
  0: rx       (unused -> -1)
  1: ry       (unused -> -1)
  2: phi      (unused -> -1)
  3: fA       (unused -> -1)
  4: fS       (unused -> -1)
  5: qx1  (cubic c1.x)
  6: qy1  (cubic c1.y)
  7: qx2  (cubic c2.x)
  8: qy2  (cubic c2.y)
  9: x1   (terminal / move / line target x)
 10: x2   (terminal / move / line target y)

Quantization
------------
Coordinates assumed already normalized (norm_v2). We clip to [-clip_range,+clip_range]
and bin to [0, args_dim - 2]. Unused args remain -1 (PAD placeholder; after +1 shift
inside model they map to 0 index).

Stats Schema (per glyph)
------------------------
{
  'subpaths_input': int,
  'subpaths_used': int,
  'commands_total': int,          # raw supported commands counted
  'commands_encoded': int,        # actually placed into tensor (<= total due to truncation)
  'groups_nonempty': int,
  'groups_full_eos': int,
  'tokens_non_eos': int,          # commands_encoded
  'fraction_utilization': float,  # tokens_non_eos / (G * S)
  'seq_len_per_group': List[int], # length (non-EOS) for each used group
  'truncated_groups': int,        # number of subpaths dropped due to max_num_groups
  'truncated_tokens_last_group': int, # tokens dropped because seq length > max_seq_len
}

Batch Collation
---------------
collate_glyph_tensors can optionally aggregate per-glyph stats if provided.

Backward Compatibility
----------------------
Existing call sites using:
    commands_g, args_g = builder.glyph_to_group_tensors(parsed)
continue to work (returns tensors only). Internally it now delegates to the
stats method.

Author: Phase 1B (NaN remediation + diagnostics)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Sequence, Optional, Dict, Any

import torch

try:
    # If DeepSVG installed / vendored
    from deepsvg.difflib.tensor import SVGTensor  # type: ignore
except Exception:  # pragma: no cover

    class _DummySVGTensor:
        COMMANDS_SIMPLIFIED = ["m", "l", "c", "a", "EOS", "SOS", "z"]
        CMD_ARGS_MASK = None

    SVGTensor = _DummySVGTensor()  # type: ignore


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class BuilderConfig:
    max_num_groups: int
    max_seq_len: int
    n_args: int = 11
    args_dim: int = 256
    clip_range: float = 1.2
    ensure_z_close: bool = True
    include_explicit_eos: bool = False
    drop_empty: bool = (
        True  # Raise if glyph has no encodable commands (so caller skips)
    )
    # Future toggles
    allow_truncate_group: bool = True
    allow_truncate_groups: bool = True
    # Diagnostics
    collect_stats: bool = True


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class SVGTensorBuilder:
    """
    Convert normalized parsed contours (list of subpaths; each list of ContourCommand-like)
    into grouped command & argument tensors for DeepSVG encoder consumption.

    Key invariants after refactor:
      - Non-empty glyph => at least one group with >=1 non-EOS token.
      - Empty glyph (all unsupported or zero commands) -> ValueError if drop_empty.
      - No group will be partially filled then followed by a fully-EOS group that
        logically still belongs to the glyph (i.e., we compact the real groups
        to the front).
    """

    def __init__(
        self,
        max_num_groups: int,
        max_seq_len: int,
        n_args: int = 11,
        args_dim: int = 256,
        clip_range: float = 1.2,
        ensure_z_close: bool = True,
        include_explicit_eos: bool = False,
        drop_empty: bool = True,
        collect_stats: bool = True,
    ):
        self.cfg = BuilderConfig(
            max_num_groups=max_num_groups,
            max_seq_len=max_seq_len,
            n_args=n_args,
            args_dim=args_dim,
            clip_range=clip_range,
            ensure_z_close=ensure_z_close,
            include_explicit_eos=include_explicit_eos,
            drop_empty=drop_empty,
            collect_stats=collect_stats,
        )

        # Token mapping
        self._cmd_to_idx = {
            c: i
            for i, c in enumerate(
                getattr(
                    SVGTensor,
                    "COMMANDS_SIMPLIFIED",
                    ["m", "l", "c", "a", "EOS", "SOS", "z"],
                )
            )
        }
        self._idx_eos = self._cmd_to_idx.get("EOS", 4)
        self._supported = {"m", "l", "c", "z"}

        # Argument slots
        self.IDX_QX1 = 5
        self.IDX_QY1 = 6
        self.IDX_QX2 = 7
        self.IDX_QY2 = 8
        self.IDX_X1 = 9
        self.IDX_X2 = 10

        self._value_max_bin = self.cfg.args_dim - 2  # reserve last + PAD semantics

    # ---------------------------------------------------------------------
    # Public convenience (backward compatible)
    # ---------------------------------------------------------------------

    def glyph_to_group_tensors(
        self, parsed_contours
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Backward compatible wrapper (tensors only).
        Raises ValueError if glyph empty and drop_empty=True.
        """
        commands, args, _stats = self.glyph_to_group_tensors_with_stats(parsed_contours)
        return commands, args

    # ---------------------------------------------------------------------
    # Primary new API
    # ---------------------------------------------------------------------

    def glyph_to_group_tensors_with_stats(
        self, parsed_contours
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Convert one glyph (list of subpaths) into (commands_g, args_g, stats).

        Returns
        -------
        commands_g : LongTensor (G, S)
        args_g     : LongTensor (G, S, n_args)
        stats      : dict (see module doc)
        """
        G = self.cfg.max_num_groups
        S = self.cfg.max_seq_len
        n_args = self.cfg.n_args

        # Initialize with EOS / PAD sentinel (-1)
        commands_g = torch.full((G, S), self._idx_eos, dtype=torch.long)
        args_g = torch.full((G, S, n_args), -1, dtype=torch.long)

        subpaths_input = len(parsed_contours)
        truncated_groups = 0
        subpaths_used = 0
        commands_total_supported = 0
        commands_encoded = 0
        seq_len_per_group: List[int] = []
        truncated_tokens_last_group = 0

        # Pre-truncate subpaths to max groups
        if subpaths_input > G:
            truncated_groups = subpaths_input - G
        subpaths_slice = parsed_contours[:G]

        next_group_index = 0
        for sub_idx, subpath in enumerate(subpaths_slice):
            # Filter / transform commands into supported tokens
            encodable_cmds = self._prepare_subpath_commands(subpath)
            if not encodable_cmds:
                continue  # empty subpath after filtering; skip (compaction)

            # Optionally ensure trailing 'z'
            if (
                self.cfg.ensure_z_close
                and encodable_cmds[-1][0] != "z"
                and len(encodable_cmds) < S
            ):
                encodable_cmds.append(("z", ()))  # add synthetic close

            # Encode into tensor rows
            g_row = next_group_index
            seq_cursor = 0
            for c_token, pts in encodable_cmds:
                commands_total_supported += 1
                if seq_cursor >= S:
                    truncated_tokens_last_group += len(encodable_cmds) - seq_cursor
                    break
                self._encode_command(
                    commands_g, args_g, g_row, seq_cursor, c_token, pts
                )
                seq_cursor += 1

            if seq_cursor == 0:
                # Should not happen (we ensured non-empty), but keep defensive guard
                continue

            # Optional explicit EOS after last real command
            if self.cfg.include_explicit_eos and seq_cursor < S:
                commands_g[g_row, seq_cursor] = self._idx_eos
                seq_cursor += 1

            seq_len_per_group.append(seq_cursor)
            commands_encoded += seq_cursor
            subpaths_used += 1
            next_group_index += 1

        # Final stats
        groups_nonempty = subpaths_used
        groups_full_eos = G - groups_nonempty
        tokens_non_eos = commands_encoded
        frac_util = float(tokens_non_eos) / float(G * S) if G * S > 0 else 0.0

        if self.cfg.drop_empty and commands_encoded == 0:
            raise ValueError(
                "Glyph has zero encodable commands (empty after filtering)."
            )

        stats = {
            "subpaths_input": subpaths_input,
            "subpaths_used": subpaths_used,
            "commands_total": commands_total_supported,
            "commands_encoded": commands_encoded,
            "groups_nonempty": groups_nonempty,
            "groups_full_eos": groups_full_eos,
            "tokens_non_eos": tokens_non_eos,
            "fraction_utilization": frac_util,
            "seq_len_per_group": seq_len_per_group,
            "truncated_groups": truncated_groups,
            "truncated_tokens_last_group": truncated_tokens_last_group,
        }

        return commands_g, args_g, stats

    # ---------------------------------------------------------------------
    # Collation
    # ---------------------------------------------------------------------

    def collate_glyph_tensors(
        self,
        glyph_tensors: Sequence[
            Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, Any]]]
            | Tuple[torch.Tensor, torch.Tensor]
        ],
        device: Optional[str] = None,
        aggregate_stats: bool = True,
    ) -> Dict[str, Any]:
        """
        Collate per-glyph tensors (and optional stats) into batch tensors.

        Accepts a mixed sequence where each element is either:
          (commands_g, args_g) or (commands_g, args_g, stats_dict)

        Returns
        -------
        {
          'commands_grouped': (N, G, S) LongTensor,
          'args_grouped'    : (N, G, S, n_args) LongTensor,
          'stats'           : {...}  # Only if stats present & aggregate_stats=True
        }
        """
        if not glyph_tensors:
            raise ValueError("Empty glyph_tensors cannot be collated.")

        # Normalize inputs to triple
        normalized: List[
            Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, Any]]]
        ] = []
        for item in glyph_tensors:
            if len(item) == 2:
                c, a = item  # type: ignore
                normalized.append((c, a, None))
            elif len(item) == 3:
                c, a, s = item  # type: ignore
                normalized.append((c, a, s))
            else:
                raise ValueError("Glyph tensor tuple must have length 2 or 3.")
        commands_list = [t[0] for t in normalized]
        args_list = [t[1] for t in normalized]
        stats_list = [t[2] for t in normalized]

        g_shape = commands_list[0].shape
        a_shape = args_list[0].shape
        for i, (c, a, _) in enumerate(normalized):
            if c.shape != g_shape or a.shape != a_shape:
                raise ValueError(
                    f"Inconsistent glyph tensor shapes at index {i}: {c.shape} vs {g_shape}, {a.shape} vs {a_shape}"
                )

        batch_commands = torch.stack(commands_list, dim=0)
        batch_args = torch.stack(args_list, dim=0)

        if device:
            batch_commands = batch_commands.to(device)
            batch_args = batch_args.to(device)

        out: Dict[str, Any] = {
            "commands_grouped": batch_commands,
            "args_grouped": batch_args,
        }

        if aggregate_stats and any(s is not None for s in stats_list):
            agg: Dict[str, Any] = {}
            numeric_accumulators: Dict[str, List[float]] = {}
            list_fields: Dict[str, List[List[int]]] = {}

            for s in stats_list:
                if not s:
                    continue
                for k, v in s.items():
                    if isinstance(v, (int, float)):
                        numeric_accumulators.setdefault(k, []).append(float(v))
                    elif isinstance(v, list):
                        list_fields.setdefault(k, []).append(v)

            # Aggregate numeric
            for k, vals in numeric_accumulators.items():
                if not vals:
                    continue
                agg[k] = {
                    "mean": float(sum(vals) / len(vals)),
                    "min": float(min(vals)),
                    "max": float(max(vals)),
                    "total": float(sum(vals)),
                    "count": len(vals),
                }
            # Aggregate list-of-lists (flatten)
            for k, list_groups in list_fields.items():
                flat = [x for group in list_groups for x in group]
                if flat:
                    agg[k] = {
                        "all_values": flat[:1000],  # cap to avoid huge dumps
                        "min": min(flat),
                        "max": max(flat),
                        "mean": sum(flat) / len(flat),
                        "count": len(flat),
                    }
            out["stats"] = agg

        return out

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    def _prepare_subpath_commands(self, subpath) -> List[Tuple[str, Tuple]]:
        """
        Filter and adapt a raw subpath (list of ContourCommand-like objects)
        into a list of (cmd_token, points_tuple).
        """
        prepared: List[Tuple[str, Tuple]] = []
        for cmd_obj in subpath:
            c = getattr(cmd_obj, "cmd", None)
            pts = getattr(cmd_obj, "points", ())
            if c not in self._supported:
                continue
            # Basic validation for required point counts
            if c in ("m", "l"):
                if not pts or len(pts[0]) != 2:
                    continue
            elif c == "c":
                if not pts or len(pts) != 3:
                    continue
            # 'z' has no points
            prepared.append((c, pts))
        return prepared

    def _encode_command(
        self,
        commands_g: torch.Tensor,
        args_g: torch.Tensor,
        g: int,
        s: int,
        c_token: str,
        pts,
    ) -> None:
        """
        Write a single command into tensors at group g, sequence index s.
        """
        commands_g[g, s] = self._cmd_to_idx.get(c_token, self._idx_eos)

        if c_token in ("m", "l"):
            (x, y) = pts[0]
            qx, qy = self._quantize_xy(x, y)
            args_g[g, s, self.IDX_X1] = qx
            args_g[g, s, self.IDX_X2] = qy
        elif c_token == "c":
            (c1x, c1y), (c2x, c2y), (ex, ey) = pts
            q_c1x, q_c1y = self._quantize_xy(c1x, c1y)
            q_c2x, q_c2y = self._quantize_xy(c2x, c2y)
            q_ex, q_ey = self._quantize_xy(ex, ey)
            args_g[g, s, self.IDX_QX1] = q_c1x
            args_g[g, s, self.IDX_QY1] = q_c1y
            args_g[g, s, self.IDX_QX2] = q_c2x
            args_g[g, s, self.IDX_QY2] = q_c2y
            args_g[g, s, self.IDX_X1] = q_ex
            args_g[g, s, self.IDX_X2] = q_ey
        # 'z' has no args

    # Quantization utilities
    def _quantize_xy(self, x: float, y: float) -> Tuple[int, int]:
        cr = self.cfg.clip_range
        return self._quantize_scalar(x, -cr, cr), self._quantize_scalar(y, -cr, cr)

    def _quantize_scalar(self, v: float, vmin: float, vmax: float) -> int:
        if v <= vmin:
            return 0
        if v >= vmax:
            return self._value_max_bin
        ratio = (v - vmin) / (vmax - vmin)
        q = int(round(ratio * self._value_max_bin))
        if q < 0:
            return 0
        if q > self._value_max_bin:
            return self._value_max_bin
        return q


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def build_default_builder_from_cfg(cfg) -> SVGTensorBuilder:
    """
    Build builder from a DeepSVG config instance (keeping parity with previous helper).
    """
    return SVGTensorBuilder(
        max_num_groups=cfg.max_num_groups,
        max_seq_len=cfg.max_seq_len,
        n_args=cfg.n_args,
        args_dim=cfg.args_dim,
        clip_range=1.2,
        ensure_z_close=True,
        include_explicit_eos=False,
        drop_empty=True,
        collect_stats=True,
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    @dataclass
    class ContourCommand:
        cmd: str
        points: Tuple[Tuple[float, float], ...]

    # Two subpaths; second will be truncated if max_num_groups=1
    glyph = [
        [
            ContourCommand("m", ((0.0, 0.0),)),
            ContourCommand("c", ((0.1, 0.2), (0.2, 0.3), (0.4, 0.5))),
            ContourCommand("l", ((0.6, -0.2),)),
        ],
        [
            ContourCommand("m", ((-0.2, 0.1),)),
            ContourCommand("l", ((-0.3, 0.2),)),
            ContourCommand("z", ()),
        ],
    ]

    builder = SVGTensorBuilder(3, 8)
    cg, ag, st = builder.glyph_to_group_tensors_with_stats(glyph)
    print("commands_g shape:", cg.shape)
    print("args_g shape:", ag.shape)
    print("stats:", st)
    print("Group 0 commands:", cg[0])
    print("Group 1 commands:", cg[1])
    print("Group 2 commands (should be all EOS):", cg[2])
