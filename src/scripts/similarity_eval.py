#!/usr/bin/env python3
"""
similarity_eval.py

Phase 1 similarity evaluation & diagnostics for extracted glyph embeddings.

Purpose
-------
Given:
  1. A tensor file of embeddings (e.g. produced by run_embed.py) saved via torch.save
     containing a FloatTensor of shape (N, D).
  2. A metadata JSONL file where each line contains:
        {
          "glyph_id": int,
          "font_hash": str,
          "label": str,
          "width_em": float,
          "height_em": float,
          "normalization_version": str,
          "upem": int | null,
          "embedding_index": int
        }

This script computes:
  * Global embedding statistics (mean/std per-dimension summary aggregates).
  * Cosine similarity matrix (optionally in chunks to conserve RAM).
  * Top-k neighbor statistics:
      - Top-k accuracy: fraction where at least one of the k nearest neighbors
        (excluding self) shares the same label.
      - Mean reciprocal rank (MRR) of first correct label match.
      - Average intra-label cosine vs inter-label cosine (sampled).
  * Label cluster diagnostics: distribution of cluster sizes for labels present.
  * Optional filtered evaluation by label prefix / regex.

All metrics are printed; optional JSON export.

Usage
-----
  Basic:
    python -m src.scripts.similarity_eval \
        --embeds artifacts/embeddings/test_embeds.pt \
        --meta artifacts/embeddings/test_meta.jsonl

  With top-k selection and JSON output:
    python -m src.scripts.similarity_eval \
        --embeds artifacts/embeddings/embeds.pt \
        --meta artifacts/embeddings/meta.jsonl \
        --k 10 \
        --json-out artifacts/embeddings/similarity_metrics.json

  Limit evaluated set (speed / quick smoke):
    python -m src.scripts.similarity_eval \
        --embeds .../embeds.pt \
        --meta .../meta.jsonl \
        --limit 500

  Regex label filter (e.g. only Arabic diacritic / isol / medi labels):
    python -m src.scripts.similarity_eval \
        --embeds ... \
        --meta ... \
        --label-regex '_(isol|medi|fina)$'

Notes
-----
- Cosine similarity matrix is O(N^2). For large N you can:
    * Provide --limit.
    * Use --chunk-size to compute top-k incrementally without holding full matrix.
- Embeddings are assumed already L2-normalized (if not, this script can apply it
  via --force-l2).
- For large scale (N > ~50k), consider an ANN library (FAISS, ScaNN, etc.) – out
  of Phase 1 scope.

Outputs
-------
Printed summary + optional JSON file with keys:
  {
    "N": ...,
    "D": ...,
    "k": ...,
    "topk_accuracy": ...,
    "mrr": ...,
    "avg_intra_cos": ...,
    "avg_inter_cos": ...,
    "intra_pairs_sampled": ...,
    "inter_pairs_sampled": ...,
    "label_stats": {
        "num_labels": ...,
        "avg_cluster_size": ...,
        "median_cluster_size": ...,
        "p90_cluster_size": ...,
        "largest_cluster": ...
    }
  }

Author: Phase 1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional, Any

import torch


# ---------------------------------------------------------------------------
# Data Classes
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
    joining_group: Optional[str]
    group_count: Optional[int]
    tokens_non_eos: Optional[int]
    embedding_index: int


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_embeddings(path: str) -> torch.Tensor:
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Embeddings file not found: {p}")
    t = torch.load(p, map_location="cpu")
    if not isinstance(t, torch.Tensor):
        raise ValueError("Loaded object is not a torch.Tensor")
    if t.ndim != 2:
        raise ValueError(f"Embeddings must be 2-D (N,D); got shape={tuple(t.shape)}")
    return t


def load_metadata(path: str) -> List[MetaRow]:
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
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            try:
                rows.append(
                    MetaRow(
                        glyph_id=obj["glyph_id"],
                        font_hash=obj["font_hash"],
                        label=obj["label"],
                        width_em=obj.get("width_em", float("nan")),
                        height_em=obj.get("height_em", float("nan")),
                        normalization_version=obj.get("normalization_version", ""),
                        upem=obj.get("upem"),
                        joining_group=obj.get("joining_group"),
                        group_count=obj.get("group_count"),
                        tokens_non_eos=obj.get("tokens_non_eos"),
                        embedding_index=obj.get("embedding_index", -1),
                    )
                )
            except KeyError:
                continue
    return rows


# ---------------------------------------------------------------------------
# Filtering & Subsetting
# ---------------------------------------------------------------------------


def apply_filters(
    embeds: torch.Tensor,
    meta: List[MetaRow],
    limit: Optional[int],
    label_regex: Optional[str],
    min_cluster: int,
) -> Tuple[torch.Tensor, List[MetaRow]]:
    filtered_meta = meta
    if label_regex:
        pattern = re.compile(label_regex)
        filtered_meta = [m for m in filtered_meta if pattern.search(m.label)]
    # Filter out labels with cluster size < min_cluster (rare labels often produce noisy metrics)
    label_counts = defaultdict(int)
    for m in filtered_meta:
        label_counts[m.label] += 1
    filtered_meta = [m for m in filtered_meta if label_counts[m.label] >= min_cluster]
    if limit and len(filtered_meta) > limit:
        filtered_meta = filtered_meta[:limit]
    # Reindex embeddings accordingly (embedding_index points to original order)
    idxs = [m.embedding_index for m in filtered_meta]
    if any(i < 0 or i >= embeds.shape[0] for i in idxs):
        raise ValueError("Invalid embedding_index found in metadata subset.")
    sub_embeds = embeds[idxs]
    return sub_embeds, filtered_meta


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def ensure_l2(emb: torch.Tensor) -> torch.Tensor:
    norms = emb.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return emb / norms


def cosine_topk_chunked(
    emb: torch.Tensor,
    k: int,
    chunk_size: int = 2048,
) -> Tuple[List[List[int]], List[List[float]]]:
    """
    Compute top-k (excluding self) neighbors using chunked cosine similarity.

    Returns
    -------
    indices: List[N][k]
    scores:  List[N][k]
    """
    N, D = emb.shape
    if k >= N:
        raise ValueError("k must be < number of embeddings")

    emb_t = emb.t().contiguous()
    results_idx: List[List[int]] = [[] for _ in range(N)]
    results_val: List[List[float]] = [[] for _ in range(N)]

    # We'll maintain candidate lists per row; for moderate N this is fine.
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk = emb[start:end]  # (C,D)
        sims = torch.mm(chunk, emb_t)  # (C,N)
        for i_local in range(chunk.shape[0]):
            i_global = start + i_local
            row = sims[i_local]
            # Exclude self index
            row[i_global] = -2.0  # sentinel less than any cosine
            # Merge existing candidates with current row
            if not results_idx[i_global]:
                # Fast path initial fill
                vals, idxs = torch.topk(row, k, largest=True, sorted=True)
                results_idx[i_global] = idxs.tolist()
                results_val[i_global] = vals.tolist()
            else:
                # Combine existing + new row candidates (all N but we only keep k)
                existing_pairs = list(zip(results_val[i_global], results_idx[i_global]))
                # Extract top from current row
                vals, idxs = torch.topk(row, k, largest=True, sorted=True)
                combined = existing_pairs + list(zip(vals.tolist(), idxs.tolist()))
                # Deduplicate keeping max score per index
                temp: Dict[int, float] = {}
                for v, idx in combined:
                    if idx not in temp or v > temp[idx]:
                        temp[idx] = v
                # Remove sentinel if present
                temp.pop(i_global, None)
                # Sort and keep top k
                top = sorted(temp.items(), key=lambda x: x[1], reverse=True)[:k]
                results_idx[i_global] = [i for i, _ in top]
                results_val[i_global] = [s for _, s in top]

    return results_idx, results_val


def topk_accuracy_and_mrr(
    neighbor_indices: List[List[int]],
    labels: List[str],
) -> Tuple[float, float]:
    """
    Compute:
      - Top-k accuracy: at least one neighbor shares label.
      - MRR based on first correct neighbor rank.
    """
    N = len(labels)
    assert N == len(neighbor_indices)
    hits = 0
    rr_sum = 0.0
    for i in range(N):
        target = labels[i]
        neigh = neighbor_indices[i]
        first_rr = 0.0
        found = False
        for rank, j in enumerate(neigh, start=1):
            if labels[j] == target:
                hits += 1
                first_rr = 1.0 / rank
                found = True
                break
        rr_sum += first_rr
        if not found:
            pass
    topk_acc = hits / N
    mrr = rr_sum / N
    return topk_acc, mrr


def intra_inter_cosine_samples(
    emb: torch.Tensor,
    labels: List[str],
    max_samples: int = 20000,
    seed: int = 42,
) -> Tuple[
    float,
    float,
    int,
    int,
    Dict[str, float],
    Dict[str, float],
]:
    """
    Randomly sample pairs to estimate average intra-label vs inter-label cosine similarity.

    Returns
    -------
    avg_intra, avg_inter : float
    n_intra, n_inter     : number of sampled pairs
    intra_stats, inter_stats : dicts with keys (std, p10, p90)
    """
    rng = random.Random(seed)
    N = emb.shape[0]
    label_to_indices: Dict[str, List[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        label_to_indices[lab].append(i)

    emb_t = emb  # already normalized

    # Intra-label sampling
    intra_pairs: List[Tuple[int, int]] = []
    for lab, idxs in label_to_indices.items():
        if len(idxs) >= 2:
            possible = len(idxs) * (len(idxs) - 1) // 2
            target = min(possible, max_samples // max(1, len(label_to_indices)))
            tried = 0
            while len(intra_pairs) < max_samples and tried < target * 3:
                a, b = rng.sample(idxs, 2)
                if a > b:
                    a, b = b, a
                intra_pairs.append((a, b))
                tried += 1
            if len(intra_pairs) >= max_samples:
                break
    if len(intra_pairs) > max_samples:
        intra_pairs = intra_pairs[:max_samples]

    # Inter-label sampling
    inter_pairs: List[Tuple[int, int]] = []
    all_indices = list(range(N))
    tried = 0
    while len(inter_pairs) < max_samples and tried < max_samples * 10:
        a, b = rng.sample(all_indices, 2)
        if labels[a] != labels[b]:
            if a > b:
                a, b = b, a
            inter_pairs.append((a, b))
        tried += 1

    def batch_cos(pairs: List[Tuple[int, int]]) -> torch.Tensor:
        if not pairs:
            return torch.tensor([], dtype=torch.float32)
        a_idx = torch.tensor([p[0] for p in pairs], dtype=torch.long)
        b_idx = torch.tensor([p[1] for p in pairs], dtype=torch.long)
        sims = (emb_t[a_idx] * emb_t[b_idx]).sum(dim=1)
        return sims

    intra_sims = batch_cos(intra_pairs)
    inter_sims = batch_cos(inter_pairs)

    def summarize(t: torch.Tensor) -> Dict[str, float]:
        if t.numel() == 0:
            return {
                "std": float("nan"),
                "p10": float("nan"),
                "p90": float("nan"),
            }
        sorted_vals = t.sort().values
        p10_idx = int(0.10 * (sorted_vals.numel() - 1))
        p90_idx = int(0.90 * (sorted_vals.numel() - 1))
        return {
            "std": float(t.std(unbiased=False).item()),
            "p10": float(sorted_vals[p10_idx].item()),
            "p90": float(sorted_vals[p90_idx].item()),
        }

    intra_stats = summarize(intra_sims)
    inter_stats = summarize(inter_sims)

    avg_intra = float(intra_sims.mean().item()) if intra_sims.numel() else float("nan")
    avg_inter = float(inter_sims.mean().item()) if inter_sims.numel() else float("nan")
    return (
        avg_intra,
        avg_inter,
        len(intra_pairs),
        len(inter_pairs),
        intra_stats,
        inter_stats,
    )


def summarize_labels(meta: List[MetaRow]) -> Dict[str, Any]:
    counts = defaultdict(int)
    for m in meta:
        counts[m.label] += 1
    sizes = list(counts.values())
    if not sizes:
        return {
            "num_labels": 0,
            "avg_cluster_size": 0,
            "median_cluster_size": 0,
            "p90_cluster_size": 0,
            "largest_cluster": 0,
        }
    return {
        "num_labels": len(sizes),
        "avg_cluster_size": sum(sizes) / len(sizes),
        "median_cluster_size": statistics.median(sizes),
        "p90_cluster_size": sorted(sizes)[int(0.9 * len(sizes)) - 1],
        "largest_cluster": max(sizes),
    }


# ---------------------------------------------------------------------------
# CLI & Main
# ---------------------------------------------------------------------------


def parse_args(argv: List[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Compute cosine similarity & basic clustering metrics for glyph embeddings (fine label + optional coarse grouping, optional PCA & feature augmentation)."
    )
    ap.add_argument("--embeds", required=True, help="Path to embeddings .pt file")
    ap.add_argument("--meta", required=True, help="Path to metadata JSONL file")
    ap.add_argument(
        "--limit", type=int, default=None, help="Limit number of rows (early slice)"
    )
    ap.add_argument(
        "--k", type=int, default=5, help="Top-k for accuracy / MRR computation"
    )
    ap.add_argument(
        "--chunk-size",
        type=int,
        default=2048,
        help="Chunk size for incremental similarity (memory control)",
    )
    ap.add_argument(
        "--force-l2",
        action="store_true",
        help="Force L2 normalization (if embeddings not already normalized)",
    )
    ap.add_argument(
        "--label-regex",
        type=str,
        default=None,
        help="Regex to filter labels (keep matches only)",
    )
    ap.add_argument(
        "--min-cluster",
        type=int,
        default=2,
        help="Minimum cluster size (labels below are dropped)",
    )
    ap.add_argument(
        "--intra-inter-samples",
        type=int,
        default=20000,
        help="Max sampled pairs for intra/inter cosine estimation",
    )
    ap.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Optional JSON output path for metrics",
    )
    ap.add_argument(
        "--seed", type=int, default=42, help="Random seed for sampling operations"
    )
    ap.add_argument(
        "--coarse-field",
        type=str,
        default="joining_group",
        help="Metadata field to use for coarse grouping (e.g. joining_group).",
    )
    ap.add_argument(
        "--dual-metrics",
        action="store_true",
        help="If set, compute both fine (label) and coarse (coarse-field) metrics.",
    )
    ap.add_argument(
        "--augment-features",
        action="store_true",
        help="Concatenate simple scalar glyph features (group_count, tokens_non_eos, width_em, height_em) after z-score before similarity.",
    )
    ap.add_argument(
        "--pca-dim",
        type=int,
        default=None,
        help="If set, apply PCA projection to this dimensionality after optional feature augmentation.",
    )
    ap.add_argument(
        "--no-center",
        action="store_true",
        help="Do not mean-center embeddings before PCA.",
    )
    ap.add_argument(
        "--whiten",
        action="store_true",
        help="After PCA, divide projected components by singular values (variance whitening).",
    )
    return ap.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    embeds = load_embeddings(args.embeds)
    meta = load_metadata(args.meta)
    if not meta:
        print("[ERROR] No metadata rows loaded.")
        return 2

    if len(meta) != embeds.shape[0]:
        print(
            f"[WARN] Embedding count ({embeds.shape[0]}) != metadata rows ({len(meta)}). Using embedding_index for re-selection."
        )

    # Filter & subset
    embeds_sub, meta_sub = apply_filters(
        embeds,
        meta,
        limit=args.limit,
        label_regex=args.label_regex,
        min_cluster=args.min_cluster,
    )
    N, D = embeds_sub.shape
    if N == 0:
        print("[ERROR] No rows after filtering.")
        return 3
    if args.force_l2:
        embeds_sub = ensure_l2(embeds_sub)

    labels = [m.label for m in meta_sub]

    print(f"[INFO] Using N={N} embeddings, D={D}, k={args.k}")
    label_summary = summarize_labels(meta_sub)
    print(
        "[INFO] Label stats: num_labels={num_labels} avg_cluster={avg_cluster_size:.2f} "
        "median={median_cluster_size:.1f} p90={p90_cluster_size} largest={largest_cluster}".format(
            **label_summary
        )
    )

    # Embedding global stats
    mean_vec = embeds_sub.mean(dim=0)
    std_vec = embeds_sub.std(dim=0, unbiased=False)
    mean_norm = float(embeds_sub.norm(dim=1).mean().item())
    print(
        "[INFO] Embedding dim stats: mean(|mean_vec|)={:.4f} mean(std_dim)={:.4f} mean(norm_row)={:.4f}".format(
            mean_vec.abs().mean().item(), std_vec.mean().item(), mean_norm
        )
    )

    # ------------------------------------------------------------------
    # Optional feature augmentation (scalar glyph metadata features)
    # ------------------------------------------------------------------
    feature_augmented = False
    if args.augment_features:
        # Collect available scalar features; skip if all None
        feat_cols: List[torch.Tensor] = []

        def col_from_attr(getter, name: str):
            vals = []
            has_any = False
            for m in meta_sub:
                v = getter(m)
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    vals.append(float("nan"))
                else:
                    has_any = True
                    vals.append(float(v))
            if not has_any:
                return None
            t = torch.tensor(vals, dtype=torch.float32).view(-1, 1)
            # z-score (ignore NaNs)
            mask = ~torch.isnan(t)
            if mask.any():
                valid = t[mask]
                if valid.numel() > 0:
                    mean = valid.mean()
                    std = valid.std(unbiased=False)
                    if std > 0:
                        t = (t - mean) / std
                    else:
                        t = t * 0.0
                else:
                    t = t * 0.0
            # Replace remaining NaNs with 0 after normalization
            t = torch.nan_to_num(t, nan=0.0)
            return t

        feature_specs = [
            (lambda m: getattr(m, "group_count", None), "group_count"),
            (lambda m: getattr(m, "tokens_non_eos", None), "tokens_non_eos"),
            (lambda m: m.width_em, "width_em"),
            (lambda m: m.height_em, "height_em"),
        ]
        for getter, name in feature_specs:
            col = col_from_attr(getter, name)
            if col is not None:
                feat_cols.append(col)
        if feat_cols:
            aug = torch.cat(feat_cols, dim=1)
            embeds_sub = torch.cat([embeds_sub, aug], dim=1)
            feature_augmented = True
            print(
                f"[INFO] Feature augmentation applied: +{aug.shape[1]} dims -> {embeds_sub.shape[1]}"
            )
        else:
            print("[INFO] Feature augmentation skipped (no usable scalar features).")

    # ------------------------------------------------------------------
    # Optional PCA (after feature augmentation)
    # ------------------------------------------------------------------
    pca_dim_used = None
    if args.pca_dim is not None:
        target_d = int(args.pca_dim)
        if target_d <= 0 or target_d > embeds_sub.shape[1]:
            print(
                f"[WARN] Ignoring invalid --pca-dim={target_d} (current dim={embeds_sub.shape[1]})."
            )
        else:
            X = embeds_sub
            if not args.no_center:
                mean_vec_full = X.mean(dim=0, keepdim=True)
                Xc = X - mean_vec_full
            else:
                Xc = X
            # SVD-based PCA
            try:
                U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
            except RuntimeError:
                # Fallback to CPU if needed
                Xc_cpu = Xc.to("cpu")
                U, S, Vh = torch.linalg.svd(Xc_cpu, full_matrices=False)
            V = Vh  # (D,D)
            comps = V[:target_d].t()  # (D, target_d)
            Xp = Xc @ comps  # (N, target_d)
            if args.whiten:
                # Avoid divide-by-zero
                scale = S[:target_d].clone()
                scale[scale == 0] = 1.0
                Xp = Xp / scale
            embeds_sub = Xp
            pca_dim_used = target_d
            print(
                f"[INFO] PCA applied: original_dim={X.shape[1]} -> pca_dim={target_d} whiten={args.whiten} centered={not args.no_center}"
            )

    if N <= args.chunk_size:
        # Simpler path: full matrix
        sims = embeds_sub @ embeds_sub.t()
        # Replace diagonal with -inf sentinel to exclude self
        sims.fill_diagonal_(-2.0)
        vals, idxs = torch.topk(sims, k=args.k, dim=1, largest=True, sorted=True)
        neighbor_indices = idxs.tolist()
    else:
        neighbor_indices, _ = cosine_topk_chunked(
            embeds_sub, k=args.k, chunk_size=args.chunk_size
        )

    topk_acc, mrr = topk_accuracy_and_mrr(neighbor_indices, labels)
    print(
        f"[INFO] Top-{args.k} Accuracy={topk_acc:.4f}  MRR={mrr:.4f} (higher is better)"
    )

    (
        avg_intra,
        avg_inter,
        n_intra,
        n_inter,
        intra_stats,
        inter_stats,
    ) = intra_inter_cosine_samples(
        embeds_sub,
        labels,
        max_samples=args.intra_inter_samples,
        seed=args.seed,
    )
    sep = float("nan")
    eff = float("nan")
    if not math.isnan(avg_intra) and not math.isnan(avg_inter):
        sep = avg_intra - avg_inter
        denom = inter_stats.get("std", float("nan"))
        if denom and not math.isnan(denom) and denom > 0:
            eff = sep / denom
    print(
        f"[INFO] Intra-label cosine (avg {avg_intra:.4f}, std {intra_stats['std']:.4f}, p10 {intra_stats['p10']:.4f}, p90 {intra_stats['p90']:.4f}, n={n_intra}) | "
        f"Inter-label cosine (avg {avg_inter:.4f}, std {inter_stats['std']:.4f}, p10 {inter_stats['p10']:.4f}, p90 {inter_stats['p90']:.4f}, n={n_inter})"
    )
    if not math.isnan(sep):
        print(f"[INFO] Intra-Inter separation = {sep:.4f}  effect_size={eff:.4f}")

    metrics = {
        "N": N,
        "D": D,
        "k": args.k,
        "topk_accuracy": topk_acc,
        "mrr": mrr,
        "avg_intra_cos": avg_intra,
        "avg_inter_cos": avg_inter,
        "intra_pairs_sampled": n_intra,
        "inter_pairs_sampled": n_inter,
        "intra_std": intra_stats["std"],
        "inter_std": inter_stats["std"],
        "intra_p10": intra_stats["p10"],
        "intra_p90": intra_stats["p90"],
        "inter_p10": inter_stats["p10"],
        "inter_p90": inter_stats["p90"],
        "separation": sep,
        "effect_size": eff,
        "label_stats": label_summary,
        "force_l2": args.force_l2,
        "limit": args.limit,
        "label_regex": args.label_regex,
        "min_cluster": args.min_cluster,
        "chunk_size": args.chunk_size,
        "feature_augmented": feature_augmented,
        "pca_dim_used": pca_dim_used,
    }

    # Optional coarse (joining_group or other field) metrics
    if args.dual_metrics:
        coarse_labels = []
        coarse_field = args.coarse_field
        for m in meta_sub:
            v = getattr(m, coarse_field, None)
            if v is None or (isinstance(v, str) and v.strip() == ""):
                v = m.label  # fallback
            coarse_labels.append(v)

        topk_acc_c, mrr_c = topk_accuracy_and_mrr(neighbor_indices, coarse_labels)
        (
            avg_intra_c,
            avg_inter_c,
            n_intra_c,
            n_inter_c,
            intra_stats_c,
            inter_stats_c,
        ) = intra_inter_cosine_samples(
            embeds_sub,
            coarse_labels,
            max_samples=args.intra_inter_samples,
            seed=args.seed,
        )
        sep_c = float("nan")
        eff_c = float("nan")
        if not math.isnan(avg_intra_c) and not math.isnan(avg_inter_c):
            sep_c = avg_intra_c - avg_inter_c
            denom_c = inter_stats_c.get("std", float("nan"))
            if denom_c and not math.isnan(denom_c) and denom_c > 0:
                eff_c = sep_c / denom_c
        print(
            f"[INFO] (Coarse:{coarse_field}) Top-{args.k} Accuracy={topk_acc_c:.4f} MRR={mrr_c:.4f} sep={sep_c:.4f} eff={eff_c:.4f}"
        )
        metrics["coarse"] = {
            "field": coarse_field,
            "topk_accuracy": topk_acc_c,
            "mrr": mrr_c,
            "avg_intra_cos": avg_intra_c,
            "avg_inter_cos": avg_inter_c,
            "intra_pairs_sampled": n_intra_c,
            "inter_pairs_sampled": n_inter_c,
            "intra_std": intra_stats_c["std"],
            "inter_std": inter_stats_c["std"],
            "intra_p10": intra_stats_c["p10"],
            "intra_p90": intra_stats_c["p90"],
            "inter_p10": inter_stats_c["p10"],
            "inter_p90": inter_stats_c["p90"],
            "separation": sep_c,
            "effect_size": eff_c,
        }

    if args.json_out:
        outp = Path(args.json_out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Wrote metrics JSON: {outp}")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv[1:]))
