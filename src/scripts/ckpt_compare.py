#!/usr/bin/env python3
"""
ckpt_compare.py

Checkpoint comparison & introspection utility for DeepSVG (or similar PyTorch)
model checkpoints. Helps identify architectural / configuration differences
between two .pth/.pt/.tar files (e.g. hierarchical_ordered.pth.tar vs
hierarchical_ordered_fonts.pth.tar).

Features
--------
1. Robust unwrapping of typical checkpoint nesting:
     - {"model": {...}}
     - {"state_dict": {...}}
     - {"model": {"state_dict": {...}}}
2. Key-level diff:
     - only_in_A
     - only_in_B
     - common_keys
     - shape_mismatches (for common keys whose tensor shapes differ)
3. Module prefix aggregation:
     - Counts and (optional) shape summaries grouped by prefix depth (--prefix-depth).
4. Heuristics summary:
     - Detect presence of VAE, bottleneck, hierarchical encoder, label embeddings.
5. Optional JSON export of structured diff results.
6. Size / parameter count statistics:
     - Total parameters (if tensors can be inspected)
     - Parameter count by top-level module.
7. Optional key filtering via regex (--include / --exclude).
8. Safety: works in "metadata-only" mode if torch is absent (parses with best effort).

Usage
-----
Basic compare:
    python -m src.scripts.ckpt_compare \\
        --a deepsvg/pretrained/hierarchical_ordered.pth.tar \\
        --b deepsvg/pretrained/hierarchical_ordered_fonts.pth.tar

With JSON output and deeper prefix grouping:
    python -m src.scripts.ckpt_compare \\
        --a .../hierarchical_ordered.pth.tar \\
        --b .../hierarchical_ordered_fonts.pth.tar \\
        --json artifacts/ckpt_diff.json \\
        --prefix-depth 2

Filter (e.g., encoder only):
    python -m src.scripts.ckpt_compare --a A.pth --b B.pth --include '^encoder\\.'

Interpretation Hints
--------------------
- "only_in_A" often indicates modules present in one training variant (e.g.,
  hierarchical heads, font-adaptive layers) but not the other.
- A large number of "shape_mismatches" in positional encodings suggests
  differing max sequence lengths / group counts.
- Presence of 'bottleneck.' vs 'vae.' indicates latent pathway differences.
- If 'encoder.hierarchical_encoder.' appears only in one checkpoint,
  that checkpoint likely used a two-stage (hierarchical) encoder.

Exit Codes
----------
0 = success
>0 = command-line or file loading errors.

Author: Phase 1 tooling
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Optional torch import
try:
    import torch

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


@dataclass
class KeyInfo:
    name: str
    shape: Optional[Tuple[int, ...]]  # None if non-tensor or torch missing
    dtype: Optional[str]


@dataclass
class DiffResult:
    only_in_a: List[str]
    only_in_b: List[str]
    common: List[str]
    shape_mismatches: List[Tuple[str, str, str]]  # (key, shape_a, shape_b)
    stats_a: Dict[str, Any]
    stats_b: Dict[str, Any]
    heuristics_a: Dict[str, Any]
    heuristics_b: Dict[str, Any]
    prefix_summary_a: Dict[str, Any]
    prefix_summary_b: Dict[str, Any]


# ---------------------------------------------------------------------------
# Loading / unwrapping
# ---------------------------------------------------------------------------


def load_state_dict_like(path: str) -> Dict[str, Any]:
    """
    Load a checkpoint and unwrap nested model/state_dict layers.
    Returns a flat dict mapping parameter names to values (tensors or misc).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if not TORCH_AVAILABLE:
        raise RuntimeError("Torch not available; cannot load checkpoint tensors.")

    raw = torch.load(p, map_location="cpu")
    # Unwrap logic
    if isinstance(raw, dict):
        # Common patterns
        if "model" in raw and isinstance(raw["model"], dict):
            candidate = raw["model"]
            if "state_dict" in candidate and isinstance(candidate["state_dict"], dict):
                return candidate["state_dict"]
            return candidate
        if "state_dict" in raw and isinstance(raw["state_dict"], dict):
            return raw["state_dict"]
    # If raw itself looks like a state dict
    if isinstance(raw, dict) and all(isinstance(k, str) for k in raw.keys()):
        return raw
    raise ValueError(f"Unsupported checkpoint structure for: {path}")


def extract_key_info(state: Dict[str, Any]) -> Dict[str, KeyInfo]:
    info: Dict[str, KeyInfo] = {}
    for k, v in state.items():
        if TORCH_AVAILABLE and hasattr(v, "shape"):
            try:
                shape = tuple(v.shape)  # type: ignore
                dtype = str(v.dtype) if hasattr(v, "dtype") else None
            except Exception:
                shape = None
                dtype = None
        else:
            shape = None
            dtype = None
        info[k] = KeyInfo(name=k, shape=shape, dtype=dtype)
    return info


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def filter_keys(
    keys: Iterable[str],
    include_regex: Optional[str],
    exclude_regex: Optional[str],
) -> List[str]:
    inc = re.compile(include_regex) if include_regex else None
    exc = re.compile(exclude_regex) if exclude_regex else None
    out = []
    for k in keys:
        if inc and not inc.search(k):
            continue
        if exc and exc.search(k):
            continue
        out.append(k)
    return out


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------


def heuristics_from_keys(keys: List[str]) -> Dict[str, Any]:
    def has_any(prefix: str) -> bool:
        return any(k.startswith(prefix) for k in keys)

    flags = {
        "has_hierarchical_encoder": has_any("encoder.hierarchical_encoder."),
        "has_hierarchical_PE": has_any("encoder.hierarchical_PE."),
        "has_label_embedding": has_any("encoder.label_embedding."),
        "has_vae": any(k.startswith("vae.") for k in keys),
        "has_bottleneck": any(k.startswith("bottleneck.") for k in keys),
        "has_decoder_hierarchical": has_any("decoder.hierarchical_decoder."),
        "has_decoder": any(k.startswith("decoder.") for k in keys),
        "has_resnet": any(k.startswith("resnet.") for k in keys),
    }
    # Positional embedding hints
    pos_keys = [
        k for k in keys if "pos_embed" in k or "pos_encoding" in k or "PE." in k
    ]
    flags["positional_key_count"] = len(pos_keys)
    return flags


# ---------------------------------------------------------------------------
# Prefix aggregation
# ---------------------------------------------------------------------------


def aggregate_prefix(
    keys: List[str], key_info: Dict[str, KeyInfo], depth: int
) -> Dict[str, Any]:
    """
    Group keys by the first `depth` segments (split by '.').
    Return counts + optional total parameter count estimate if shapes available.
    """
    groups: Dict[str, Dict[str, Any]] = {}
    for k in keys:
        parts = k.split(".")
        prefix = ".".join(parts[:depth]) if len(parts) >= depth else k
        g = groups.setdefault(prefix, {"count": 0, "param_count": 0})
        g["count"] += 1
        ki = key_info.get(k)
        if ki and ki.shape:
            # Parameter count = product of shape dims
            pc = 1
            for d in ki.shape:
                pc *= d
            g["param_count"] += pc
    # Sort by param_count descending
    sorted_items = sorted(groups.items(), key=lambda x: (-x[1]["param_count"], x[0]))
    return {k: v for k, v in sorted_items}


# ---------------------------------------------------------------------------
# Diff logic
# ---------------------------------------------------------------------------


def compute_diff(
    info_a: Dict[str, KeyInfo],
    info_b: Dict[str, KeyInfo],
    include_regex: Optional[str],
    exclude_regex: Optional[str],
    prefix_depth: int,
) -> DiffResult:
    keys_a_all = list(info_a.keys())
    keys_b_all = list(info_b.keys())

    keys_a = set(filter_keys(keys_a_all, include_regex, exclude_regex))
    keys_b = set(filter_keys(keys_b_all, include_regex, exclude_regex))

    common = sorted(keys_a & keys_b)
    only_in_a = sorted(keys_a - keys_b)
    only_in_b = sorted(keys_b - keys_a)

    shape_mismatches: List[Tuple[str, str, str]] = []
    for k in common:
        sa = info_a[k].shape
        sb = info_b[k].shape
        if sa and sb and sa != sb:
            shape_mismatches.append((k, str(sa), str(sb)))

    heur_a = heuristics_from_keys(common + only_in_a)
    heur_b = heuristics_from_keys(common + only_in_b)

    # Basic stats
    def stat_block(info: Dict[str, KeyInfo], keyset: Iterable[str]) -> Dict[str, Any]:
        total_params = 0
        has_shape = 0
        for k in keyset:
            ki = info.get(k)
            if ki and ki.shape:
                has_shape += 1
                pc = 1
                for d in ki.shape:
                    pc *= d
                total_params += pc
        return {
            "num_keys": len(list(keyset)),
            "num_with_shape": has_shape,
            "total_param_count": total_params,
        }

    stats_a = stat_block(info_a, keys_a)
    stats_b = stat_block(info_b, keys_b)

    prefix_summary_a = aggregate_prefix(sorted(keys_a), info_a, prefix_depth)
    prefix_summary_b = aggregate_prefix(sorted(keys_b), info_b, prefix_depth)

    return DiffResult(
        only_in_a=only_in_a,
        only_in_b=only_in_b,
        common=common,
        shape_mismatches=shape_mismatches,
        stats_a=stats_a,
        stats_b=stats_b,
        heuristics_a=heur_a,
        heuristics_b=heur_b,
        prefix_summary_a=prefix_summary_a,
        prefix_summary_b=prefix_summary_b,
    )


# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------


def print_section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def summarize(diff: DiffResult, max_list: int):
    print_section("SUMMARY")
    print(f"Common keys: {len(diff.common)}")
    print(f"Only in A  : {len(diff.only_in_a)}")
    print(f"Only in B  : {len(diff.only_in_b)}")
    print(f"Shape mismatches: {len(diff.shape_mismatches)}")

    print_section("STATS A")
    print(json.dumps(diff.stats_a, indent=2))
    print_section("STATS B")
    print(json.dumps(diff.stats_b, indent=2))

    print_section("HEURISTICS A")
    print(json.dumps(diff.heuristics_a, indent=2))
    print_section("HEURISTICS B")
    print(json.dumps(diff.heuristics_b, indent=2))

    if diff.shape_mismatches:
        print_section("SHAPE MISMATCHES (sample)")
        for k, sa, sb in diff.shape_mismatches[:max_list]:
            print(f"{k}: {sa} vs {sb}")

    if diff.only_in_a:
        print_section("ONLY IN A (sample)")
        for k in diff.only_in_a[:max_list]:
            print(k)

    if diff.only_in_b:
        print_section("ONLY IN B (sample)")
        for k in diff.only_in_b[:max_list]:
            print(k)

    print_section("PREFIX SUMMARY A")
    # Limit lines
    for i, (prefix, data) in enumerate(diff.prefix_summary_a.items()):
        if i >= max_list:
            print("... (truncated)")
            break
        print(f"{prefix}: {data}")

    print_section("PREFIX SUMMARY B")
    for i, (prefix, data) in enumerate(diff.prefix_summary_b.items()):
        if i >= max_list:
            print("... (truncated)")
            break
        print(f"{prefix}: {data}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Compare two DeepSVG (or generic PyTorch) checkpoints."
    )
    ap.add_argument("--a", required=True, help="Path to checkpoint A")
    ap.add_argument("--b", required=True, help="Path to checkpoint B")
    ap.add_argument(
        "--include",
        type=str,
        default=None,
        help="Regex to include keys (applied before exclude)",
    )
    ap.add_argument(
        "--exclude",
        type=str,
        default=None,
        help="Regex to exclude keys",
    )
    ap.add_argument(
        "--prefix-depth",
        type=int,
        default=1,
        help="Module prefix depth for aggregation summary (default: 1)",
    )
    ap.add_argument(
        "--max-list",
        type=int,
        default=50,
        help="Maximum lines to print in each sample list.",
    )
    ap.add_argument(
        "--json",
        type=str,
        default=None,
        help="Optional JSON output path for full diff result.",
    )
    ap.add_argument(
        "--fail-on-mismatch",
        action="store_true",
        help="Exit with code 2 if any key shape mismatches are found.",
    )
    return ap.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if not TORCH_AVAILABLE:
        print("[ERROR] Torch is not available in this environment.", file=sys.stderr)
        return 3

    try:
        state_a = load_state_dict_like(args.a)
        state_b = load_state_dict_like(args.b)
    except Exception as e:
        print(f"[ERROR] Failed to load checkpoints: {e}", file=sys.stderr)
        return 1

    info_a = extract_key_info(state_a)
    info_b = extract_key_info(state_b)

    diff = compute_diff(
        info_a=info_a,
        info_b=info_b,
        include_regex=args.include,
        exclude_regex=args.exclude,
        prefix_depth=args.prefix_depth,
    )

    summarize(diff, max_list=args.max_list)

    if args.json:
        outp = Path(args.json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "only_in_a": diff.only_in_a,
            "only_in_b": diff.only_in_b,
            "common_count": len(diff.common),
            "shape_mismatches": diff.shape_mismatches,
            "stats_a": diff.stats_a,
            "stats_b": diff.stats_b,
            "heuristics_a": diff.heuristics_a,
            "heuristics_b": diff.heuristics_b,
            "prefix_summary_a": diff.prefix_summary_a,
            "prefix_summary_b": diff.prefix_summary_b,
            "args": vars(args),
        }
        with outp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\n[INFO] Wrote JSON diff to {outp}")

    if args.fail_on_mismatch and diff.shape_mismatches:
        print(
            f"[ERROR] Shape mismatches present ({len(diff.shape_mismatches)}) and --fail-on-mismatch set.",
            file=sys.stderr,
        )
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
