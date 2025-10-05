#!/usr/bin/env python3
"""
PCA Post-Processing Utility for Glyph Embeddings.

Features:
  1. Fit PCA (SVD-based) on a 2-D embeddings tensor (N, D).
  2. Optionally remove top-K principal components (common component removal).
  3. Optionally project to lower dimensional PCA space (with optional whitening).
  4. Save / load PCA model (mean, components, singular values) for reuse.
  5. Apply previously fit PCA to new embeddings.

Typical Use Cases:
  A. Fit + Project:
     python -m src.scripts.pca_postprocess \
         --embeds artifacts/hier_auto3000_embeds.pt \
         --fit-pca --pca-dim 128 \
         --out-dir artifacts/pca/hier_auto3000 \
         --out artifacts/hier_auto3000_pca128.pt

  B. Common Component Removal (no projection):
     python -m src.scripts.pca_postprocess \
         --embeds artifacts/hier_auto3000_embeds.pt \
         --fit-pca --pca-dim 128 --remove-top 2 --no-project \
         --out-dir artifacts/pca/hier_auto3000_ccr \
         --out artifacts/hier_auto3000_ccr.pt

  C. Apply Existing PCA Model:
     python -m src.scripts.pca_postprocess \
         --embeds artifacts/hier_auto3000_embeds.pt \
         --model-dir artifacts/pca/hier_auto3000 \
         --apply --remove-top 1 \
         --out artifacts/hier_auto3000_pca128_rt1.pt

Arguments:
  --embeds        Path to input embeddings (.pt, 2-D tensor).
  --fit-pca       Fit a new PCA model (requires --out-dir).
  --pca-dim       Number of principal components to keep (default 128).
  --remove-top    Remove top-K PCs from (centered) embeddings before optional projection.
  --no-project    Skip projection; only (optionally) remove PCs in original space.
  --whiten        Divide projected components by singular values (variance normalization).
  --out-dir       Directory to save newly fit PCA model (mean.pt, components.pt, singular_values.pt, meta.json).
  --model-dir     Existing PCA model directory (when using --apply).
  --apply         Apply an existing PCA model (must supply --model-dir).
  --out           Output path for processed embeddings (.pt).

Notes:
  - If both --fit-pca and --apply are specified, the newly fit model is used (applied in same run).
  - remove-top happens in original space (after centering) BEFORE projection.
  - Whitening should follow research needs; for cosine similarity spaces whitening can sometimes hurt if you later L2-normalize again.

Output:
  Saves processed tensor to --out if provided.

Author: Projection / Post-Processing Module
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import torch


# ----------------------------- Core Utilities ----------------------------- #


def load_embeddings(path: str) -> torch.Tensor:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Embeddings file not found: {p}")
    t = torch.load(p, map_location="cpu")
    if not isinstance(t, torch.Tensor):
        raise TypeError("Loaded object is not a torch.Tensor")
    if t.ndim != 2:
        raise ValueError(f"Embeddings must be 2-D (N,D); got shape={tuple(t.shape)}")
    return t.float()


def fit_pca(
    X: torch.Tensor,
    dim: int,
    center: bool = True,
) -> dict:
    """
    Fit PCA using SVD (X = U S V^T).

    Returns:
      {
        'mean': (1,D) tensor
        'components': (dim, D) tensor (top PC first)
        'singular_values': (dim,)
        'orig_dim': D
        'pca_dim': dim
        'centered': bool
      }
    """
    if dim <= 0 or dim > X.shape[1]:
        raise ValueError(f"Invalid pca-dim={dim} for input dim={X.shape[1]}")

    if center:
        mean = X.mean(dim=0, keepdim=True)
        Xc = X - mean
    else:
        mean = torch.zeros(1, X.shape[1], dtype=X.dtype)
        Xc = X

    # SVD
    # Note: torch.linalg.svd returns U (N,N), S (min(N,D)), Vh (D,D) when full_matrices=False => shapes trimmed to rank.
    U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    comps = Vh[:dim].contiguous()  # (dim, D)

    return {
        "mean": mean,
        "components": comps,
        "singular_values": S[:dim].contiguous(),
        "orig_dim": X.shape[1],
        "pca_dim": dim,
        "centered": center,
    }


def remove_top_components(X: torch.Tensor, pcs: torch.Tensor, k: int) -> torch.Tensor:
    """
    Remove the projection of X onto the first k principal components.

    pcs: (pca_dim, D) with PCs sorted descending by variance.
    """
    if k <= 0:
        return X
    if k > pcs.shape[0]:
        raise ValueError(f"remove-top={k} exceeds available PCs ({pcs.shape[0]})")
    top = pcs[:k]  # (k, D)
    # Project and subtract: X_proj = (X * top)top^T
    # Equivalent: (X @ top.T) -> (N,k) then multiply (N,k) @ (k,D) = (N,D)
    coeff = X @ top.t()  # (N,k)
    recon = coeff @ top  # (N,D)
    return X - recon


def project(X: torch.Tensor, model: dict, whiten: bool = False) -> torch.Tensor:
    """
    Project centered embeddings onto PCA subspace.

    model['components']: (pca_dim, D)
    """
    mean = model["mean"]
    comps = model["components"]  # (pca_dim, D)
    Xc = X - mean
    Z = Xc @ comps.t()  # (N, pca_dim)
    if whiten:
        S = model["singular_values"]
        S_safe = torch.where(S > 0, S, torch.ones_like(S))
        Z = Z / S_safe
    return Z


def save_model(model: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model["mean"], out_dir / "mean.pt")
    torch.save(model["components"], out_dir / "components.pt")
    torch.save(model["singular_values"], out_dir / "singular_values.pt")
    meta = {
        "orig_dim": model["orig_dim"],
        "pca_dim": model["pca_dim"],
        "centered": model["centered"],
    }
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def load_model(model_dir: Path) -> dict:
    if not model_dir.exists():
        raise FileNotFoundError(f"PCA model dir not found: {model_dir}")
    with (model_dir / "meta.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)
    mean = torch.load(model_dir / "mean.pt", map_location="cpu")
    comps = torch.load(model_dir / "components.pt", map_location="cpu")
    sv = torch.load(model_dir / "singular_values.pt", map_location="cpu")
    meta.update({"mean": mean, "components": comps, "singular_values": sv})
    return meta


# ----------------------------- CLI Handling ------------------------------ #


def parse_args():
    ap = argparse.ArgumentParser(
        "PCA post-processing for glyph embeddings (fit/apply, remove top PCs, project, whiten)."
    )
    ap.add_argument("--embeds", required=True, help="Path to input embeddings (.pt)")
    ap.add_argument("--fit-pca", action="store_true", help="Fit a new PCA model")
    ap.add_argument("--apply", action="store_true", help="Apply existing PCA model")
    ap.add_argument(
        "--pca-dim",
        type=int,
        default=128,
        help="Target PCA dimensionality when fitting (default=128)",
    )
    ap.add_argument(
        "--remove-top",
        type=int,
        default=0,
        help="Remove top-K principal components before optional projection",
    )
    ap.add_argument(
        "--no-project",
        action="store_true",
        help="Skip projection (stay in original D after optional removal)",
    )
    ap.add_argument(
        "--whiten",
        action="store_true",
        help="Whiten projected components by singular values (PCA basis).",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory to save a newly fit PCA model (required if --fit-pca).",
    )
    ap.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Directory of existing PCA model (required if --apply).",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output path (.pt) for processed embeddings (optional).",
    )
    return ap.parse_args()


# ------------------------------- Main Flow ------------------------------- #


def main():
    args = parse_args()
    X = load_embeddings(args.embeds)  # (N,D)
    model = None

    if args.fit_pca:
        if not args.out_dir:
            raise ValueError("--out-dir required when using --fit-pca")
        model = fit_pca(X, dim=args.pca_dim)
        save_model(model, Path(args.out_dir))
        print(
            f"[INFO] Fitted PCA: orig_dim={model['orig_dim']} pca_dim={model['pca_dim']} centered={model['centered']}"
        )
        print(f"[INFO] Saved PCA model to: {args.out_dir}")

    if args.apply:
        if not args.model_dir:
            raise ValueError("--model-dir required when using --apply")
        model = load_model(Path(args.model_dir))
        print(
            f"[INFO] Loaded PCA model: orig_dim={model['orig_dim']} pca_dim={model['pca_dim']} centered={model['centered']}"
        )

    if model is None:
        raise ValueError(
            "Nothing to do: specify at least --fit-pca or --apply (or both)."
        )

    # Always center relative to model mean if centered
    if model["centered"]:
        X_centered = X - model["mean"]
    else:
        X_centered = X

    # Remove top PCs (common component removal)
    if args.remove_top > 0:
        if args.remove_top > model["pca_dim"]:
            print(
                f"[WARN] remove-top={args.remove_top} > available pca_dim={model['pca_dim']}; clipping to {model['pca_dim']}"
            )
        k = min(args.remove_top, model["pca_dim"])
        X_centered = remove_top_components(X_centered, model["components"], k)
        print(f"[INFO] Removed top {k} PCs (common component removal)")

    if args.no_project:
        X_out = X_centered
        print(
            "[INFO] Skipping projection (--no-project set); retaining original dimensionality."
        )
    else:
        # Project using PCA basis
        X_out = project(X_centered + model["mean"], model, whiten=args.whiten)
        print(
            f"[INFO] Projected embeddings: new_dim={X_out.shape[1]} whiten={args.whiten}"
        )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(X_out, out_path)
        print(f"[INFO] Wrote processed embeddings to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
