#!/usr/bin/env python3
"""
eval_similarity.py
==================

Lightweight similarity / retrieval evaluation for raster glyph embeddings.

Purpose
-------
Given:
  1. Embeddings tensor file produced by raster/embed.py
       - torch.FloatTensor of shape (N, D) (ideally L2-normalized)
  2. Metadata JSONL file produced by raster/embed.py with lines:
       {"glyph_id": int, "font_hash": str, "label": str, "embedding_index": int}

This script computes:
  * Top-k Accuracy (at least one of k nearest neighbors shares label)
  * Mean Reciprocal Rank (MRR) of first matching label
  * Intra vs Inter label cosine statistics (means)
  * Approximate effect size (Cohen-like): (mean_intra - mean_inter) / pooled_std
  * Label cluster statistics (count distribution)
  * Optional label filtering via regex
  * Optional minimum cluster size filtering
  * Optional limiting of evaluation subset for quicker runs

Supports two modes:
  - Full (in-memory) similarity matrix (fast but O(N^2) memory)
  - Chunked top-k search (less RAM; limited metrics—effect size computed via sampling)

Usage
-----
Basic:
  python raster/eval_similarity.py \
    --embeds raster/artifacts/raster_embeddings.pt \
    --meta raster/artifacts/raster_meta.jsonl

Specify k and JSON output:
  python raster/eval_similarity.py \
    --embeds .../raster_embeddings.pt \
    --meta .../raster_meta.jsonl \
    --k 10 \
    --json-out raster/artifacts/raster_similarity.json

Limit evaluation (speed):
  python raster/eval_similarity.py \
    --embeds ... --meta ... --limit 5000

Regex filter (example keeps labels ending in '_isol' or '_medi'):
  python raster/eval_similarity.py \
    --embeds ... --meta ... --label-regex '_(isol|medi)$'

Chunked mode (lower memory; recommended if N > ~30k):
  python raster/eval_similarity.py \
    --embeds ... --meta ... --chunk-size 4096

Notes
-----
- Assumes embeddings are already L2-normalized; pass --force-l2 if not sure.
- If using chunked mode, effect size & intra/inter means are computed from sampled pairs.
- For very large datasets consider ANN libraries (FAISS) — out of scope here.

Output JSON Structure:
{
  "N": int,
  "D": int,
  "k": int,
  "topk_accuracy": float,
  "mrr": float,
  "intra_mean": float,
  "inter_mean": float,
  "effect_size": float,
  "label_stats": { ... },
  "settings": { ... }
}

Author: Raster Phase 1
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class MetaRow:
    glyph_id: int
    font_hash: str
    label: str
    embedding_index: int


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_embeddings(path: str) -> torch.Tensor:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Embeddings file not found: {p}")
    obj = torch.load(p, map_location="cpu")
    if not isinstance(obj, torch.Tensor):
        raise ValueError("Embeddings file must contain a torch.Tensor")
    if obj.ndim != 2:
        raise ValueError(f"Embeddings tensor must be 2-D (N,D); got {tuple(obj.shape)}")
    return obj.float()


def load_metadata(path: str) -> List[MetaRow]:
    p = Path(path)
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
            except json.JSONDecodeError:
                continue
            try:
                rows.append(
                    MetaRow(
                        glyph_id=int(o["glyph_id"]),
                        font_hash=str(o["font_hash"]),
                        label=str(o["label"]),
                        embedding_index=int(o["embedding_index"]),
                    )
                )
            except (KeyError, ValueError):
                continue
    return rows


# ---------------------------------------------------------------------------
# Filtering / Subsetting
# ---------------------------------------------------------------------------


def subset_by_metadata(
    embeds: torch.Tensor,
    meta: List[MetaRow],
    limit: Optional[int],
    label_regex: Optional[str],
    min_cluster: int,
) -> Tuple[torch.Tensor, List[MetaRow]]:
    if label_regex:
        pattern = re.compile(label_regex)
        meta = [m for m in meta if pattern.search(m.label)]
    # Filter small clusters
    counts = defaultdict(int)
    for m in meta:
        counts[m.label] += 1
    meta = [m for m in meta if counts[m.label] >= min_cluster]
    if limit and len(meta) > limit:
        meta = meta[:limit]
    # Reindex embeddings
    idxs = [m.embedding_index for m in meta]
    if any(i < 0 or i >= embeds.shape[0] for i in idxs):
        raise ValueError("Invalid embedding_index encountered after filtering.")
    sub_embeds = embeds[idxs]
    return sub_embeds, meta


# ---------------------------------------------------------------------------
# Retrieval Metrics
# ---------------------------------------------------------------------------


def ensure_l2(emb: torch.Tensor) -> torch.Tensor:
    norms = emb.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return emb / norms


def cosine_topk_full(emb: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    sims = emb @ emb.t()  # (N,N)
    sims.fill_diagonal_(-2.0)  # exclude self
    top_vals, top_idx = sims.topk(k, dim=1)
    return top_idx, top_vals


def cosine_topk_chunked(
    emb: torch.Tensor, k: int, chunk_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Memory-aware top-k (approx same results as full)
    N = emb.shape[0]
    emb_t = emb.t().contiguous()
    top_idx = torch.empty(N, k, dtype=torch.long)
    top_val = torch.empty(N, k, dtype=emb.dtype)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk = emb[start:end]  # (C,D)
        sims = chunk @ emb_t  # (C,N)
        batch_size = sims.size(0)
        arange_local = torch.arange(start, end)
        sims[torch.arange(batch_size), arange_local] = -2.0  # mask self
        v, i = sims.topk(k, dim=1)
        top_idx[start:end] = i
        top_val[start:end] = v
    return top_idx, top_val


def compute_topk_accuracy_mrr(
    top_idx: torch.Tensor, labels: List[str]
) -> Tuple[float, float]:
    N, K = top_idx.shape
    hits = 0
    mrr = 0.0
    for i in range(N):
        li = labels[i]
        row = top_idx[i].tolist()
        rank_found = None
        for r, j in enumerate(row):
            if labels[j] == li:
                hits += 1
                rank_found = r + 1
                break
        if rank_found:
            mrr += 1.0 / rank_found
    return hits / N, mrr / N


def sample_intra_inter(
    emb: torch.Tensor,
    labels: List[str],
    max_pairs: int = 10000,
    seed: int = 123,
) -> Tuple[List[float], List[float]]:
    import random

    rng = random.Random(seed)
    label_to_indices: Dict[str, List[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        label_to_indices[lab].append(i)

    intra: List[float] = []
    inter: List[float] = []
    # Intra
    for lab, idxs in label_to_indices.items():
        if len(idxs) < 2:
            continue
        pairs_to_sample = min(len(idxs) // 2, 50)
        for _ in range(pairs_to_sample):
            a, b = rng.sample(idxs, 2)
            intra.append(float((emb[a] * emb[b]).sum().item()))
            if len(intra) >= max_pairs:
                break
        if len(intra) >= max_pairs:
            break
    # Inter
    all_idx = list(range(len(labels)))
    while len(inter) < min(max_pairs, len(intra)):
        a, b = rng.sample(all_idx, 2)
        if labels[a] != labels[b]:
            inter.append(float((emb[a] * emb[b]).sum().item()))
    return intra, inter


def effect_size_from_samples(
    intra: List[float], inter: List[float]
) -> Tuple[float, float, float]:
    if not intra or not inter:
        return 0.0, 0.0, 0.0
    mu_in = statistics.mean(intra)
    mu_out = statistics.mean(inter)
    all_vals = intra + inter
    sigma = statistics.pstdev(all_vals) if len(all_vals) > 1 else 1.0
    eff = (mu_in - mu_out) / sigma if sigma > 1e-9 else 0.0
    return eff, mu_in, mu_out


# ---------------------------------------------------------------------------
# Label Stats
# ---------------------------------------------------------------------------


def label_statistics(meta: List[MetaRow]) -> Dict[str, float]:
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
    sizes_sorted = sorted(sizes)
    p90_index = int(0.9 * (len(sizes_sorted) - 1))
    return {
        "num_labels": len(sizes),
        "avg_cluster_size": sum(sizes) / len(sizes),
        "median_cluster_size": float(sizes_sorted[len(sizes_sorted) // 2]),
        "p90_cluster_size": float(sizes_sorted[p90_index]),
        "largest_cluster": max(sizes),
    }


# ---------------------------------------------------------------------------
# Main Evaluation
# ---------------------------------------------------------------------------


def evaluate(
    embeds: torch.Tensor,
    meta: List[MetaRow],
    k: int,
    chunk_size: Optional[int],
    force_l2: bool,
    sample_pairs: int,
) -> Dict[str, float]:
    labels = [m.label for m in meta]
    if force_l2:
        embeds = ensure_l2(embeds)

    if k >= embeds.shape[0]:
        raise ValueError("k must be less than number of embeddings")

    if chunk_size and chunk_size < embeds.shape[0]:
        top_idx, _ = cosine_topk_chunked(embeds, k=k, chunk_size=chunk_size)
    else:
        top_idx, _ = cosine_topk_full(embeds, k=k)

    topk_acc, mrr = compute_topk_accuracy_mrr(top_idx, labels)

    intra, inter = sample_intra_inter(embeds, labels, max_pairs=sample_pairs)
    effect, intra_mean, inter_mean = effect_size_from_samples(intra, inter)

    return {
        "topk_accuracy": topk_acc,
        "mrr": mrr,
        "intra_mean": intra_mean,
        "inter_mean": inter_mean,
        "effect_size": effect,
        "intra_pairs_sampled": len(intra),
        "inter_pairs_sampled": len(inter),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Raster glyph embedding similarity eval.")
    ap.add_argument("--embeds", required=True, help="Embeddings .pt file (N,D)")
    ap.add_argument("--meta", required=True, help="Metadata JSONL (glyph_id,label,...)")
    ap.add_argument("--k", type=int, default=10, help="Top-k for retrieval metrics")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of rows (0=all)")
    ap.add_argument(
        "--label-regex", type=str, default=None, help="Regex to filter labels pre-eval"
    )
    ap.add_argument(
        "--min-cluster",
        type=int,
        default=2,
        help="Minimum label cluster size to keep after filtering",
    )
    ap.add_argument(
        "--chunk-size",
        type=int,
        default=0,
        help="Chunk size for memory-aware top-k (0 = full in-memory)",
    )
    ap.add_argument(
        "--force-l2",
        action="store_true",
        help="Force L2 normalization of embeddings before metrics",
    )
    ap.add_argument(
        "--sample-pairs",
        type=int,
        default=10000,
        help="Max intra/inter pairs sampled for effect size stats",
    )
    ap.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Optional path to write metrics JSON",
    )
    ap.add_argument("--seed", type=int, default=123, help="Sampling seed")
    return ap.parse_args(argv)


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    args = parse_args(argv)

    embeds = load_embeddings(args.embeds)
    meta = load_metadata(args.meta)

    print(f"[INFO] Loaded embeddings shape={tuple(embeds.shape)} meta_rows={len(meta)}")

    embeds_sub, meta_sub = subset_by_metadata(
        embeds,
        meta,
        limit=args.limit if args.limit > 0 else None,
        label_regex=args.label_regex,
        min_cluster=args.min_cluster,
    )
    print(
        f"[INFO] After filtering: N={embeds_sub.shape[0]} labels={len({m.label for m in meta_sub})}"
    )

    if embeds_sub.shape[0] <= args.k:
        print("[WARN] Not enough samples after filtering for chosen k.")
        return 1

    label_stats = label_statistics(meta_sub)

    metrics = evaluate(
        embeds_sub,
        meta_sub,
        k=args.k,
        chunk_size=args.chunk_size if args.chunk_size > 0 else None,
        force_l2=args.force_l2,
        sample_pairs=args.sample_pairs,
    )

    summary = {
        "N": embeds_sub.shape[0],
        "D": embeds_sub.shape[1],
        "k": args.k,
        **metrics,
        "label_stats": label_stats,
        "settings": {
            "limit": args.limit,
            "label_regex": args.label_regex,
            "min_cluster": args.min_cluster,
            "chunk_size": args.chunk_size,
            "force_l2": args.force_l2,
            "sample_pairs": args.sample_pairs,
        },
    }

    # Print concise summary
    print(
        f"[RESULT] topk_acc={metrics['topk_accuracy']:.4f} "
        f"mrr={metrics['mrr']:.4f} effect={metrics['effect_size']:.4f} "
        f"intra_mean={metrics['intra_mean']:.4f} inter_mean={metrics['inter_mean']:.4f}"
    )

    if args.json_out:
        outp = Path(args.json_out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Wrote metrics JSON: {outp}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
