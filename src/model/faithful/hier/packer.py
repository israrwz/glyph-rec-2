"""
Hierarchical group packer for faithful DeepSVG preprocessing.

This module takes a list of SVGTensor objects (one per subpath / group) and
packs them into batched grouped command / argument tensors compatible with the
hierarchical (encode_stages=2) DeepSVG encoder.

Expected Upstream:
------------------
- Each glyph subpath has been converted into an SVGTensor using a faithful
  builder (see `svgtensor_builder.py`) with:
    * .commands  (shape [seq_len, 1])
    * .args()    (shape [seq_len, 11]) absolute or relative depending on config
    * .seq_len   (length INCLUDING EOS token; already padded to max_seq_len)

- All SVGTensors share the same `max_seq_len` (padding applied).
- Each SVGTensor already contains EOS/PAD tokens in the tail.

Core Responsibilities:
----------------------
1. Truncate / limit the list of SVGTensors to `max_num_groups`.
2. Allocate grouped tensors:
       commands_grouped : (G, S)
       args_grouped     : (G, S, n_args)
   filled with EOS / PAD (-1) where no group exists.
3. Copy each group's sequence up to S.
4. Provide lightweight stats for diagnostics.

Assumptions:
------------
- SVGTensor.COMMANDS_SIMPLIFIED matches the model's vocabulary.
- EOS index corresponds to SVGTensor.COMMANDS_SIMPLIFIED.index("EOS").
- Args use PAD_VAL = -1, and the embedding layer will apply (+1) shift.

Output:
-------
pack_groups(svgt_list, max_num_groups, max_seq_len, n_args) -> (commands, args, stats)
    commands : LongTensor (G, S)
    args     : LongTensor (G, S, n_args)
    stats    : dict with:
        {
          'groups_used': int,
          'truncated_groups': int,
          'token_counts': List[int],   # raw (seq_len) per included group (post-truncation)
          'max_seq_len': int,
          'max_num_groups': int
        }

Future Extensions:
------------------
- Optionally compute utilization percentage (non-EOS tokens / (G*S)).
- Support arc commands ('a') once builder emits them.
- Support relative argument vs absolute switch in stats.

Author: Faithful hierarchical implementation phase.
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Any
import torch

try:
    from deepsvg.difflib.tensor import SVGTensor
except ImportError as e:  # pragma: no cover
    raise RuntimeError(
        "DeepSVG not importable. Ensure repository or package is on PYTHONPATH."
    ) from e


def pack_groups(
    svgt_list: List[SVGTensor],
    max_num_groups: int,
    max_seq_len: int,
    n_args: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Pack a list of SVGTensor objects into grouped command / arg tensors.

    Parameters
    ----------
    svgt_list : list[SVGTensor]
        Ordered per-subpath tensors (already padded) for one glyph.
    max_num_groups : int
        Number of group slots (G).
    max_seq_len : int
        Target sequence length (S) each SVGTensor must already be padded to.
    n_args : int
        Number of argument slots (usually 11).

    Returns
    -------
    commands_grouped : LongTensor (G, S)
    args_grouped     : LongTensor (G, S, n_args)
    stats            : dict (see module docstring)
    """
    eos_idx = SVGTensor.COMMANDS_SIMPLIFIED.index("EOS")
    G, S = max_num_groups, max_seq_len

    commands = torch.full((G, S), eos_idx, dtype=torch.long)
    args = torch.full((G, S, n_args), -1, dtype=torch.long)

    # Truncate group list if necessary
    truncated_groups = 0
    if len(svgt_list) > G:
        truncated_groups = len(svgt_list) - G
        svgt_list = svgt_list[:G]

    token_counts: List[int] = []
    g_out = 0
    for svgt in svgt_list:
        if svgt is None:  # safety
            continue
        if g_out >= G:
            break

        # SVGTensor.seq_len includes EOS (after add_eos) but may be shorter than S if padded.
        seq_len = int(svgt.seq_len.item())
        copy_len = min(seq_len, S)

        raw_cmds = svgt.commands.long().view(-1)[:copy_len]  # shape (L,)
        raw_args = svgt.args()[:copy_len]  # shape (L, n_args)

        commands[g_out, :copy_len] = raw_cmds
        args[g_out, :copy_len] = raw_args
        token_counts.append(copy_len)
        g_out += 1

    stats = {
        "groups_used": g_out,
        "truncated_groups": truncated_groups,
        "token_counts": token_counts,
        "max_seq_len": max_seq_len,
        "max_num_groups": max_num_groups,
    }
    return commands, args, stats


def batch_collate_grouped(
    glyph_group_tensors: List[Tuple[torch.Tensor, torch.Tensor]],
    device: str | None = None,
) -> Dict[str, torch.Tensor]:
    """
    Collate a list of (commands_grouped, args_grouped) pairs into a batch.

    Parameters
    ----------
    glyph_group_tensors : list of ( (G,S) LongTensor, (G,S,n_args) LongTensor )
        Output of pack_groups for each glyph.
    device : optional str
        If provided, move batch tensors to this device.

    Returns
    -------
    dict with:
      commands_grouped : (N, G, S)
      args_grouped     : (N, G, S, n_args)
    """
    if not glyph_group_tensors:
        raise ValueError("Empty glyph_group_tensors in batch_collate_grouped()")

    cmds0, args0 = glyph_group_tensors[0]
    G, S = cmds0.shape
    n_args = args0.shape[2]

    cmds_batch = []
    args_batch = []
    for cg, ag in glyph_group_tensors:
        if cg.shape != (G, S) or ag.shape != (G, S, n_args):
            raise ValueError(
                f"Inconsistent grouped tensor shapes. Expected {(G, S)} & {(G, S, n_args)}, got {tuple(cg.shape)} & {tuple(ag.shape)}"
            )
        cmds_batch.append(cg)
        args_batch.append(ag)

    batch_cmds = torch.stack(cmds_batch, dim=0)  # (N,G,S)
    batch_args = torch.stack(args_batch, dim=0)  # (N,G,S,n_args)

    if device:
        batch_cmds = batch_cmds.to(device)
        batch_args = batch_args.to(device)

    return {
        "commands_grouped": batch_cmds,
        "args_grouped": batch_args,
    }


# ---------------------------------------------------------------------------
# Self-test (optional)
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    # Minimal smoke test using synthetic SVGTensors.
    from dataclasses import dataclass

    @dataclass
    class DummyCmd:
        cmd: str
        points: Tuple[Tuple[float, float], ...]

    # Build two fake SVGTensors (directly)
    # In practice you'd use the hierarchical builder.
    def make_simple_svgt(cmd_seq):
        cmds = torch.tensor(
            [SVGTensor.COMMANDS_SIMPLIFIED.index(c) for c in cmd_seq], dtype=torch.long
        )
        # Make all-zero args except end_pos
        n = len(cmd_seq)
        args = torch.zeros(n, 11)
        svgt = SVGTensor.from_cmd_args(cmds, args, PAD_VAL=-1, ARGS_DIM=256)
        svgt.add_eos().pad(seq_len=12)
        return svgt

    sv1 = make_simple_svgt(["m", "l", "z"])
    sv2 = make_simple_svgt(["m", "c", "c", "z"])

    cmds_g, args_g, st = pack_groups(
        [sv1, sv2], max_num_groups=4, max_seq_len=12, n_args=11
    )
    print("Grouped commands shape:", cmds_g.shape)
    print("Grouped args shape:", args_g.shape)
    print("Stats:", st)
    batch = batch_collate_grouped([(cmds_g, args_g)])
    print("Batch keys:", {k: v.shape for k, v in batch.items()})
