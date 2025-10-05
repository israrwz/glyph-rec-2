#!/usr/bin/env python3
"""
diagnose_embeds.py

Diagnostic utility for inspecting a saved embedding tensor (produced by run_embed.py)
to identify numerical pathologies (NaNs / Infs / zero vectors) and summarize label
cluster health.

Motivation
----------
Similarity evaluation surfaced NaNs in aggregate stats (mean/std became NaN). This
script pinpoints:
  * Which rows (indices) contain NaN or Inf values.
  * Per-dimension NaN / Inf incidence.
  * Row norm distribution (treating NaNs/Infs as zeros for norm calc).
  * Optional metadata correlation (counts per label for affected rows).
  * Ability to write:
      - A cleaned embedding tensor (replacing NaNs/Infs row-wise and re-normalizing).
      - A filtered metadata file excluding invalid rows.
      - A JSON summary report.

Cleaning Policy
---------------
Default cleaning strategy (when --out-clean is specified):
  1. For each affected row:
       - Replace NaN / Inf components with 0.
       - If resulting norm == 0 => fill with small random normal vector then L2 normalize.
       - Else L2 normalize (if --renorm specified).
  2. Leave unaffected rows unchanged (unless --renorm-all).

You can choose a different fill strategy via:
  --fill-strategy zero | mean | random
    zero   : leave zeros where NaNs occurred (may shrink norm)
    mean   : replace each NaN with the per-dimension finite mean (computed globally)
    random : replace with small Gaussian noise (mean=0, std=1e-3) before renorm

CLI Examples
------------
Basic stats:
  python -m src.scripts.diagnose_embeds --embeds artifacts/embeddings/embeds_1k.pt

With metadata correlation + JSON report:
  python -m src.scripts.diagnose_embeds \
      --embeds artifacts/embeddings/embeds_1k.pt \
      --meta artifacts/embeddings/meta_1k.jsonl \
      --report-json artifacts/embeddings/diagnostics.json

Produce cleaned outputs (re-normalizing all rows):
  python -m src.scripts.diagnose_embeds \
      --embeds artifacts/embeddings/embeds_1k.pt \
      --meta artifacts/embeddings/meta_1k.jsonl \
      --out-clean artifacts/embeddings/embeds_1k_clean.pt \
      --out-meta-clean artifacts/embeddings/meta_1k_clean.jsonl \
      --renorm-all --fill-strategy random

Exit Codes
----------
0 = success
>0 = error conditions (missing file, invalid serialization, etc.)

Author: Phase 1 Diagnostics
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Metadata dataclass
# ---------------------------------------------------------------------------


@dataclass
class MetaRow:
    glyph_id: int
    font_hash: str
    label: str
    width_em: float
    height_em: float
    normalization_version: str
    upem: Optional[int]
    embedding_index: int


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def load_embeddings(path: str) -> torch.Tensor:
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Embeddings file not found: {p}")
    obj = torch.load(p, map_location="cpu")
    if not isinstance(obj, torch.Tensor):
        raise ValueError("File does not contain a tensor.")
    if obj.ndim != 2:
        raise ValueError(f"Expected 2-D tensor (N,D); got shape={tuple(obj.shape)}")
    return obj


def load_metadata(path: Optional[str]) -> List[MetaRow]:
    if not path:
        return []
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Metadata file not found: {p}")
    rows: List[MetaRow] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
                rows.append(
                    MetaRow(
                        glyph_id=o["glyph_id"],
                        font_hash=o["font_hash"],
                        label=o["label"],
                        width_em=o.get("width_em", float("nan")),
                        height_em=o.get("height_em", float("nan")),
                        normalization_version=o.get("normalization_version", ""),
                        upem=o.get("upem"),
                        embedding_index=o.get("embedding_index", -1),
                    )
                )
            except Exception:
                continue
    return rows


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


@dataclass
class EmbedDiagnostics:
    n_rows: int
    dim: int
    nan_rows: List[int]
    inf_rows: List[int]
    zero_norm_rows: List[int]
    finite_row_norm_stats: Dict[str, float]
    per_dim_nan_frac: List[float]
    per_dim_inf_frac: List[float]
    any_nan: bool
    any_inf: bool


def compute_diagnostics(emb: torch.Tensor) -> EmbedDiagnostics:
    N, D = emb.shape
    finite_mask = torch.isfinite(emb)
    nan_mask = torch.isnan(emb)
    inf_mask = torch.isinf(emb)

    row_has_nan = nan_mask.view(N, -1).any(dim=1)
    row_has_inf = inf_mask.view(N, -1).any(dim=1)

    clean = emb.clone()
    clean[~finite_mask] = 0.0
    row_norms = clean.norm(dim=1)

    zero_norm_rows = (row_norms < 1e-12).nonzero(as_tuple=True)[0].tolist()
    finite_norms = [row_norms[i].item() for i in range(N) if i not in zero_norm_rows]

    if finite_norms:
        norm_stats = {
            "min": float(min(finite_norms)),
            "max": float(max(finite_norms)),
            "mean": float(sum(finite_norms) / len(finite_norms)),
            "median": float(statistics.median(finite_norms)),
            "p10": float(statistics.quantiles(finite_norms, n=10)[0])
            if len(finite_norms) >= 10
            else float("nan"),
            "p90": float(statistics.quantiles(finite_norms, n=10)[-1])
            if len(finite_norms) >= 10
            else float("nan"),
        }
    else:
        norm_stats = {
            k: float("nan") for k in ["min", "max", "mean", "median", "p10", "p90"]
        }

    per_dim_nan_frac = nan_mask.float().mean(dim=0).tolist()
    per_dim_inf_frac = inf_mask.float().mean(dim=0).tolist()

    return EmbedDiagnostics(
        n_rows=N,
        dim=D,
        nan_rows=row_has_nan.nonzero(as_tuple=True)[0].tolist(),
        inf_rows=row_has_inf.nonzero(as_tuple=True)[0].tolist(),
        zero_norm_rows=zero_norm_rows,
        finite_row_norm_stats=norm_stats,
        per_dim_nan_frac=per_dim_nan_frac,
        per_dim_inf_frac=per_dim_inf_frac,
        any_nan=row_has_nan.any().item() == 1,
        any_inf=row_has_inf.any().item() == 1,
    )


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------


def compute_dim_means_finite(emb: torch.Tensor) -> torch.Tensor:
    # Replace non-finite with 0 for sum, count finite for division
    finite_mask = torch.isfinite(emb)
    masked = emb.clone()
    masked[~finite_mask] = 0.0
    counts = finite_mask.sum(dim=0).clamp_min(1)
    sums = masked.sum(dim=0)
    return sums / counts


def clean_embeddings(
    emb: torch.Tensor,
    diag: EmbedDiagnostics,
    fill_strategy: str,
    renorm_all: bool,
    renorm: bool,
    seed: int,
) -> Tuple[torch.Tensor, List[int]]:
    """
    Returns (cleaned_tensor, dropped_rows).

    We do not drop rows unless they are entirely non-finite (all components nan/inf).
    Instead, we attempt to repair; drop condition is conservative and rare.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    cleaned = emb.clone()
    finite_mask = torch.isfinite(cleaned)
    nan_or_inf = ~finite_mask

    dim_means = None
    if fill_strategy == "mean":
        dim_means = compute_dim_means_finite(cleaned)

    affected_rows = []
    dropped_rows: List[int] = []

    for i in range(cleaned.shape[0]):
        if not nan_or_inf[i].any():
            continue
        affected_rows.append(i)
        row = cleaned[i]
        mask = nan_or_inf[i]

        if mask.all():  # Extremely pathological
            # Replace entire row
            if fill_strategy == "random":
                row[:] = torch.randn_like(row) * 1e-3
            elif fill_strategy == "mean" and dim_means is not None:
                row[:] = dim_means
            else:
                row[:] = 0.0
            mask = torch.zeros_like(mask, dtype=torch.bool)

        else:
            if fill_strategy == "random":
                noise = torch.randn(mask.sum().item()) * 1e-3
                row[mask] = noise.to(row.dtype)
            elif fill_strategy == "mean" and dim_means is not None:
                row[mask] = dim_means[mask]
            else:  # zero
                row[mask] = 0.0

        # Optional renorm per affected row
        if renorm or renorm_all:
            norm = row.norm()
            if norm < 1e-12:
                # Provide small random vector then norm
                row[:] = torch.randn_like(row)
                norm = row.norm()
            row /= norm

    # Renorm all rows (even unaffected) if requested
    if renorm_all and not renorm:
        norms = cleaned.norm(dim=1, keepdim=True).clamp_min(1e-12)
        cleaned = cleaned / norms

    return cleaned, dropped_rows


# ---------------------------------------------------------------------------
# Metadata correlation
# ---------------------------------------------------------------------------


def correlate_with_metadata(
    diag: EmbedDiagnostics, meta: List[MetaRow]
) -> Dict[str, Dict[str, int]]:
    if not meta:
        return {}
    # Map embedding_index -> metadata row(s)
    index_to_labels: Dict[int, List[str]] = {}
    for m in meta:
        if 0 <= m.embedding_index < diag.n_rows:
            index_to_labels.setdefault(m.embedding_index, []).append(m.label)

    def count_labels(indices: List[int]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for idx in indices:
            for lab in index_to_labels.get(idx, []):
                counts[lab] = counts.get(lab, 0) + 1
        return counts

    return {
        "nan_rows_label_counts": count_labels(diag.nan_rows),
        "inf_rows_label_counts": count_labels(diag.inf_rows),
        "zero_norm_rows_label_counts": count_labels(diag.zero_norm_rows),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def build_summary_dict(
    diag: EmbedDiagnostics,
    meta: List[MetaRow],
    label_corr: Dict[str, Dict[str, int]],
    fill_strategy: Optional[str],
    cleaned_path: Optional[str],
) -> Dict:
    # Basic label cluster stats (only if metadata)
    label_stats = {}
    if meta:
        label_freq: Dict[str, int] = {}
        for m in meta:
            label_freq[m.label] = label_freq.get(m.label, 0) + 1
        sizes = sorted(label_freq.values())
        if sizes:
            label_stats = {
                "num_labels": len(sizes),
                "avg_cluster_size": sum(sizes) / len(sizes),
                "median_cluster_size": statistics.median(sizes),
                "p90_cluster_size": sizes[int(0.9 * len(sizes)) - 1]
                if len(sizes) >= 10
                else sizes[-1],
                "largest_cluster": sizes[-1],
            }

    summary = {
        "num_rows": diag.n_rows,
        "dim": diag.dim,
        "num_nan_rows": len(diag.nan_rows),
        "num_inf_rows": len(diag.inf_rows),
        "num_zero_norm_rows": len(diag.zero_norm_rows),
        "finite_row_norm_stats": diag.finite_row_norm_stats,
        "any_nan": diag.any_nan,
        "any_inf": diag.any_inf,
        "per_dim_nan_fraction_sample": diag.per_dim_nan_frac[
            :10
        ],  # truncate sample for brevity
        "per_dim_inf_fraction_sample": diag.per_dim_inf_frac[:10],
        "label_stats": label_stats,
        "label_correlations": label_corr,
        "fill_strategy": fill_strategy,
        "cleaned_output": cleaned_path,
    }
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Diagnose embedding tensor for NaNs / Infs / zero norms and optionally clean."
    )
    ap.add_argument("--embeds", required=True, help="Path to embeddings .pt file")
    ap.add_argument("--meta", type=str, default=None, help="Optional metadata JSONL")
    ap.add_argument(
        "--report-json", type=str, default=None, help="Write summary JSON here"
    )
    ap.add_argument(
        "--out-clean", type=str, default=None, help="Write cleaned embeddings tensor"
    )
    ap.add_argument(
        "--out-meta-clean",
        type=str,
        default=None,
        help="Write filtered metadata excluding bad rows",
    )
    ap.add_argument(
        "--fill-strategy",
        choices=["zero", "mean", "random"],
        default="random",
        help="Value fill for NaN/Inf replacements (default random)",
    )
    ap.add_argument(
        "--renorm", action="store_true", help="Renormalize only affected rows"
    )
    ap.add_argument(
        "--renorm-all", action="store_true", help="Renormalize all rows after cleaning"
    )
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument(
        "--show-nan-rows", action="store_true", help="Print list of NaN row indices"
    )
    ap.add_argument(
        "--show-zero-norm-rows",
        action="store_true",
        help="Print list of zero-norm row indices",
    )
    return ap.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    args = parse_args(argv)

    try:
        emb = load_embeddings(args.embeds)
    except Exception as e:
        print(f"[ERROR] Failed to load embeddings: {e}")
        return 2

    try:
        meta = load_metadata(args.meta)
    except Exception as e:
        print(f"[WARN] Failed to load metadata: {e}")
        meta = []

    diag = compute_diagnostics(emb)

    print(f"[INFO] Embeddings: rows={diag.n_rows} dim={diag.dim}")
    print(
        f"[INFO] Rows with NaNs: {len(diag.nan_rows)} | Rows with Infs: {len(diag.inf_rows)} | Zero-norm rows: {len(diag.zero_norm_rows)}"
    )
    print("[INFO] Row norm stats (finite subset):", diag.finite_row_norm_stats)

    if args.show_nan_rows and diag.nan_rows:
        print(f"[DETAIL] NaN row indices ({len(diag.nan_rows)}): {diag.nan_rows}")
    if args.show_zero_norm_rows and diag.zero_norm_rows:
        print(
            f"[DETAIL] Zero-norm row indices ({len(diag.zero_norm_rows)}): {diag.zero_norm_rows}"
        )

    label_corr = correlate_with_metadata(diag, meta)
    if meta:
        print(f"[INFO] Metadata rows loaded: {len(meta)}")
        if diag.nan_rows:
            top_labels_nan = sorted(
                label_corr.get("nan_rows_label_counts", {}).items(), key=lambda x: -x[1]
            )[:10]
            print(f"[INFO] Top labels among NaN rows (up to 10): {top_labels_nan}")

    cleaned_path = None
    cleaned_emb = emb
    dropped_rows: List[int] = []

    if args.out_clean:
        cleaned_emb, dropped_rows = clean_embeddings(
            emb=emb,
            diag=diag,
            fill_strategy=args.fill_strategy,
            renorm_all=args.renorm_all,
            renorm=args.renorm,
            seed=args.seed,
        )
        cleaned_path = args.out_clean
        outp = Path(args.out_clean)
        outp.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cleaned_emb, outp)
        print(
            f"[INFO] Wrote cleaned embeddings: {outp} (dropped_rows={len(dropped_rows)})"
        )

    if args.out_meta_clean and meta:
        # Filter metadata: exclude rows whose embedding index is in dropped_rows (currently usually empty)
        drop_set = set(dropped_rows)
        filtered_meta = [m for m in meta if m.embedding_index not in drop_set]
        outm = Path(args.out_meta_clean)
        outm.parent.mkdir(parents=True, exist_ok=True)
        with outm.open("w", encoding="utf-8") as f:
            for m in filtered_meta:
                f.write(
                    json.dumps(
                        {
                            "glyph_id": m.glyph_id,
                            "font_hash": m.font_hash,
                            "label": m.label,
                            "width_em": m.width_em,
                            "height_em": m.height_em,
                            "normalization_version": m.normalization_version,
                            "upem": m.upem,
                            "embedding_index": m.embedding_index,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        print(f"[INFO] Wrote cleaned metadata: {outm} (rows={len(filtered_meta)})")

    summary = build_summary_dict(
        diag=diag,
        meta=meta,
        label_corr=label_corr,
        fill_strategy=args.fill_strategy if args.out_clean else None,
        cleaned_path=cleaned_path,
    )

    if args.report_json:
        rpt = Path(args.report_json)
        rpt.parent.mkdir(parents=True, exist_ok=True)
        with rpt.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Wrote diagnostics JSON: {rpt}")

    # If NaNs detected but not cleaned, hint next steps
    if diag.any_nan and not args.out_clean:
        print(
            "[HINT] NaNs detected. Consider re-running with --out-clean and a fill strategy."
        )
        print(
            "[HINT] Probable cause: encoder pooled over zero valid tokens (empty/EOS-only sequence)."
        )
        print(
            "[HINT] Mitigation upstream: skip glyphs whose command groups contain no non-EOS tokens."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
