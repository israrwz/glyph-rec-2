#!/usr/bin/env python3
"""
Train Projection Head with Curriculum, Temperature Scheduling, Cluster Weighting,
Variance Logging and (optional) Hard Negative Emphasis.

Enhancements vs previous version:
---------------------------------
1. Curriculum training phases:
     --curriculum "coarse:50,fine:100,hybrid:350"
   Each segment sets the active label-mode for its epoch span (overrides --label-mode).
   If not provided, the static --label-mode behavior (legacy) is used.

2. Temperature schedule:
     --temp-start 0.18 --temp-end 0.05 --temp-mode linear
   Supports linear | cosine | exp. Temperature is recomputed each epoch.

3. Cluster-size reweighting:
     --cluster-weighting inv_sqrt (or inv_log / none)
   Applies per-sample weighting (1 / sqrt(freq(label))) or 1 / log(1+freq).

4. Variance logging:
     --log-var-every 50 --var-topk 10
   Computes top-K explained variance ratios (SVD on centered embeddings) for
   (a) initial base embeddings after PCA removal / augmentation,
   (b) periodically for projected embeddings (in eval mode).
   Optionally writes all logs to JSON (--log-json).

5. Hybrid loss alpha scheduling:
     --hybrid-alpha-start 0.7 --hybrid-alpha-end 0.5 (interpolated across hybrid epochs).
   If static value preferred, set both to same number.

6. Hard negative emphasis (hybrid only):
     --hard-neg --hard-neg-scale 1.5
   In hybrid phase, negatives that share coarse label but differ fine label
   are up-weighted by scaling their similarity exponent (stronger competition).

7. PCA top-K removal retained (before feature augmentation).

8. Structural feature augmentation unchanged (optionally appended).

9. Deterministic seed control via --seed.

Example:
--------
python -m src.scripts.train_projection_head \
  --embeds artifacts/hier_auto3000_embeds.pt \
  --meta artifacts/hier_auto3000_meta.jsonl \
  --pca-model artifacts/pca/hier_auto3000 --remove-top 5 \
  --augment-features \
  --epochs 500 \
  --curriculum "coarse:50,fine:100,hybrid:350" \
  --hybrid-alpha-start 0.8 --hybrid-alpha-end 0.6 \
  --temp-start 0.18 --temp-end 0.05 --temp-mode cosine \
  --cluster-weighting inv_sqrt \
  --hard-neg --hard-neg-scale 1.8 \
  --batch-size 256 \
  --log-var-every 50 \
  --out-proj artifacts/projection/head.pt \
  --out-embeds artifacts/projection/hier_auto3000_proj.pt \
  --log-json artifacts/projection/train_log.json

Outputs:
--------
- Projection head state_dict (if requested)
- Projected embeddings (if requested)
- Console logs + optional JSON log (variance, losses, temperatures, etc.)

NOTE:
-----
Be mindful of memory cost when computing variance on very large datasets.
Adjust --var-sample if needed (random subset size). Default = 0 (use all).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# Local imports (unchanged core components)
from src.model.projection_head import (
    ProjectionHead,
    ProjectionHeadConfig,
    supervised_contrastive_loss as base_supervised_contrastive_loss,
)

# ---------------------------------------------------------------------------
# Metadata Loading
# ---------------------------------------------------------------------------


class MetaRow:
    __slots__ = (
        "glyph_id",
        "font_hash",
        "label",
        "width_em",
        "height_em",
        "normalization_version",
        "upem",
        "embedding_index",
        "joining_group",
        "group_count",
        "tokens_non_eos",
    )


def load_metadata(path: str) -> List[MetaRow]:
    rows: List[MetaRow] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            m = MetaRow()
            m.glyph_id = obj["glyph_id"]
            m.font_hash = obj["font_hash"]
            m.label = obj["label"]
            m.width_em = obj.get("width_em", float("nan"))
            m.height_em = obj.get("height_em", float("nan"))
            m.normalization_version = obj.get("normalization_version", "")
            m.upem = obj.get("upem")
            m.embedding_index = obj.get("embedding_index", -1)
            m.joining_group = obj.get("joining_group")
            m.group_count = obj.get("group_count")
            m.tokens_non_eos = obj.get("tokens_non_eos")
            rows.append(m)
    return rows


# ---------------------------------------------------------------------------
# PCA Model Loading
# ---------------------------------------------------------------------------


def load_pca_model(pca_dir: Optional[str]):
    if not pca_dir:
        return None
    p = Path(pca_dir)
    if not p.exists():
        raise FileNotFoundError(f"PCA model dir not found: {p}")
    with (p / "meta.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)
    mean = torch.load(p / "mean.pt", map_location="cpu")
    comps = torch.load(p / "components.pt", map_location="cpu")  # (pca_dim, D)
    return {"mean": mean, "components": comps, "meta": meta}


def remove_top_k(X: torch.Tensor, comps: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0:
        return X
    if k > comps.shape[0]:
        k = comps.shape[0]
    top = comps[:k]  # (k, D)
    coeff = X @ top.t()  # (N, k)
    return X - coeff @ top


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class EmbeddingDataset(Dataset):
    def __init__(
        self,
        embeds: torch.Tensor,
        labels_fine: List[str],
        labels_coarse: List[str],
    ):
        self.embeds = embeds
        self.labels_fine = labels_fine
        self.labels_coarse = labels_coarse

        # Map labels to integer IDs
        self.fine_vocab = {lab: i for i, lab in enumerate(sorted(set(labels_fine)))}
        self.coarse_vocab = {lab: i for i, lab in enumerate(sorted(set(labels_coarse)))}

        self.ids_fine = torch.tensor([self.fine_vocab[x] for x in labels_fine])
        self.ids_coarse = torch.tensor([self.coarse_vocab[x] for x in labels_coarse])

    def __len__(self):
        return self.embeds.shape[0]

    def __getitem__(self, idx):
        return (
            self.embeds[idx],
            self.ids_fine[idx],
            self.ids_coarse[idx],
        )


# ---------------------------------------------------------------------------
# Feature Augmentation
# ---------------------------------------------------------------------------


def _norm_feature_column(values: List[float]) -> torch.Tensor:
    t = torch.tensor(values, dtype=torch.float32)
    mask = torch.isfinite(t)
    if mask.any():
        mean = t[mask].mean()
        std = t[mask].std(unbiased=False)
        if std > 0:
            t[mask] = (t[mask] - mean) / std
        else:
            t[mask] = 0.0
    t[~mask] = 0.0
    return t.view(-1, 1)


def augment_structural_features(
    embeds: torch.Tensor, meta: List[MetaRow]
) -> torch.Tensor:
    cols = []
    cols.append(
        _norm_feature_column(
            [
                getattr(m, "group_count", float("nan"))
                if getattr(m, "group_count", None) is not None
                else float("nan")
                for m in meta
            ]
        )
    )
    cols.append(
        _norm_feature_column(
            [
                getattr(m, "tokens_non_eos", float("nan"))
                if getattr(m, "tokens_non_eos", None) is not None
                else float("nan")
                for m in meta
            ]
        )
    )
    cols.append(_norm_feature_column([m.width_em for m in meta]))
    cols.append(_norm_feature_column([m.height_em for m in meta]))
    aug = torch.cat(cols, dim=1)
    return torch.cat([embeds, aug], dim=1)


# ---------------------------------------------------------------------------
# Variance / PCA Utility (Top-K explained variance via SVD)
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_topk_variance(
    X: torch.Tensor, k: int = 10, sample: int = 0, seed: int = 42
) -> List[float]:
    """
    Returns top-k explained variance ratios (not cumulative) of centered X.
    If sample > 0 and X is larger, randomly subsample rows.
    """
    if sample > 0 and X.shape[0] > sample:
        g = torch.Generator(device=X.device)
        g.manual_seed(seed)
        idx = torch.randperm(X.shape[0], generator=g)[:sample]
        X = X[idx]
    X = X.float()
    mu = X.mean(dim=0, keepdim=True)
    Xc = X - mu
    # Economy SVD; for large dims fallback to torch.linalg.svd
    try:
        U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    except RuntimeError:
        # Fallback to CPU if needed
        Xc_cpu = Xc.cpu()
        U, S, Vh = torch.linalg.svd(Xc_cpu, full_matrices=False)
        S = S.to(X.device)
    var_total = (S**2).sum().item() + 1e-12
    r = (S**2) / var_total
    top = r[:k].tolist()
    # Pad if fewer singular values
    while len(top) < k:
        top.append(0.0)
    return top


# ---------------------------------------------------------------------------
# Loss Wrappers (Hybrid, Hard Negatives, Weighting)
# ---------------------------------------------------------------------------


def make_temperature_scheduler(
    epochs: int, t_start: float, t_end: float, mode: str
) -> List[float]:
    mode = mode.lower()
    temps = []
    for e in range(epochs):
        p = e / max(1, epochs - 1)
        if mode == "linear":
            t = t_start + (t_end - t_start) * p
        elif mode == "cosine":
            # cosine decay from start to end
            t = t_end + 0.5 * (t_start - t_end) * (1 + math.cos(math.pi * p))
        elif mode == "exp":
            # exponential (geometric) interpolation
            if t_start <= 0 or t_end <= 0:
                raise ValueError("Exponential schedule requires positive temps.")
            t = t_start * (t_end / t_start) ** p
        else:
            raise ValueError(f"Unsupported temp schedule mode: {mode}")
        temps.append(float(t))
    return temps


def supervised_contrastive_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    label_weights: Optional[torch.Tensor] = None,
    hard_neg_mask: Optional[torch.Tensor] = None,
    hard_neg_scale: float = 1.0,
) -> torch.Tensor:
    """
    Extension of base supervised contrastive loss supporting:
      - label_weights (per-sample)
      - hard_neg_mask (B,B) bool for negatives to emphasize; scaling similarity.

    Approach:
      1. Compute similarity matrix sim / temp (excluding self).
      2. For hard negatives, scale their exp(sim) by hard_neg_scale (>1).
    """
    B = z.shape[0]
    if B <= 1:
        return z.new_tensor(0.0)

    # similarity (expected normalized)
    sim = z @ z.t()
    sim = sim / temperature

    eye = torch.eye(B, device=z.device, dtype=torch.bool)
    sim.masked_fill_(eye, -float("inf"))

    labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
    labels_eq.masked_fill_(eye, False)

    # Row-wise stabilization
    sim_max, _ = sim.max(dim=1, keepdim=True)
    sim_stable = sim - sim_max
    exp_sim = torch.exp(sim_stable)

    # Hard negatives: scale their contribution
    if hard_neg_mask is not None and hard_neg_scale > 1.0:
        # Only for negatives (ensure we don't touch positives)
        neg_mask = (~labels_eq) & (~eye)
        scale_mask = hard_neg_mask & neg_mask
        exp_sim = exp_sim + (hard_neg_scale - 1.0) * exp_sim * scale_mask

    pos_mask = labels_eq
    pos_sum = (exp_sim * pos_mask).sum(dim=1)
    denom = exp_sim.sum(dim=1) + 1e-12

    valid_mask = (pos_mask.sum(dim=1) > 0).float()
    loss_vec = -torch.log((pos_sum + 1e-12) / denom)

    if label_weights is not None:
        loss_vec = loss_vec * label_weights

    loss = (loss_vec * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)
    return loss


def hybrid_contrastive_loss(
    z: torch.Tensor,
    fine_ids: torch.Tensor,
    coarse_ids: torch.Tensor,
    temp: float,
    alpha: float,
    fine_weights: Optional[torch.Tensor] = None,
    coarse_weights: Optional[torch.Tensor] = None,
    hard_neg_mask: Optional[torch.Tensor] = None,
    hard_neg_scale: float = 1.0,
) -> torch.Tensor:
    """
    Hybrid supervised contrastive loss:

      L = alpha * L_fine + (1-alpha) * L_coarse

    Hard negative mask (if provided) is applied only on the fine loss
    (focusing sharper discrimination within coarse groups).
    """
    lf = supervised_contrastive_loss(
        z,
        fine_ids,
        temperature=temp,
        label_weights=fine_weights,
        hard_neg_mask=hard_neg_mask,
        hard_neg_scale=hard_neg_scale,
    )
    lc = supervised_contrastive_loss(
        z, coarse_ids, temperature=temp, label_weights=coarse_weights
    )
    return alpha * lf + (1.0 - alpha) * lc


# ---------------------------------------------------------------------------
# Cluster Weighting
# ---------------------------------------------------------------------------


def compute_cluster_weights(
    ids: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    """
    ids : (B,) int64
    mode: 'none' | 'inv_sqrt' | 'inv_log'
    Returns per-sample weight vector normalized to mean ≈ 1.0.
    """
    if mode == "none":
        return torch.ones_like(ids, dtype=torch.float32)

    unique, counts = torch.unique(ids, return_counts=True)
    freq = dict(zip(unique.tolist(), counts.tolist()))
    w = []
    for i in ids.tolist():
        c = freq[i]
        if mode == "inv_sqrt":
            w.append(1.0 / math.sqrt(c))
        elif mode == "inv_log":
            w.append(1.0 / math.log(1.0 + c))
        else:
            raise ValueError(f"Unsupported cluster weighting mode: {mode}")
    w_t = torch.tensor(w, dtype=torch.float32, device=ids.device)
    # Normalize so average weight ~1
    w_t = w_t * (len(w) / w_t.sum().clamp_min(1e-6))
    return w_t


# ---------------------------------------------------------------------------
# Curriculum Parsing
# ---------------------------------------------------------------------------


def parse_curriculum(spec: Optional[str]) -> Optional[List[Tuple[str, int]]]:
    """
    spec format: "coarse:50,fine:100,hybrid:350"
    Returns list of (mode, epochs)
    """
    if not spec:
        return None
    parts = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(f"Invalid curriculum segment (missing colon): {token}")
        m, e_str = token.split(":", 1)
        m = m.strip().lower()
        if m not in ("coarse", "fine", "hybrid"):
            raise ValueError(f"Unknown curriculum mode: {m}")
        try:
            e = int(e_str)
        except ValueError:
            raise ValueError(f"Invalid epoch count in curriculum: {e_str}")
        if e <= 0:
            raise ValueError(f"Epoch count must be positive in: {token}")
        parts.append((m, e))
    return parts if parts else None


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_projection(
    embeds: torch.Tensor,
    meta: List[MetaRow],
    args,
):
    device = torch.device(args.device)

    # Build fine & coarse label lists
    fine_labels = [m.label for m in meta]
    coarse_labels = []
    for m in meta:
        cg = m.joining_group
        if not cg or (isinstance(cg, str) and not cg.strip()):
            cg = m.label  # fallback
        coarse_labels.append(cg)

    # PCA debias BEFORE feature augmentation
    if args.pca_model:
        pm = load_pca_model(args.pca_model)
        mean = pm["mean"]
        comps = pm["components"]
        if mean.shape[1] != embeds.shape[1]:
            raise ValueError(
                f"PCA mean dim mismatch: mean_dim={mean.shape[1]} embeds_dim={embeds.shape[1]} "
                "— ensure PCA was fit on embeddings pre-augmentation."
            )
        embeds = embeds - mean
        if args.remove_top > 0:
            embeds = remove_top_k(embeds, comps, args.remove_top)

    # Feature augmentation (after PCA)
    if args.augment_features:
        embeds = augment_structural_features(embeds, meta)

    ds = EmbeddingDataset(embeds, fine_labels, coarse_labels)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    input_dim = embeds.shape[1]
    cfg = ProjectionHeadConfig(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.out_dim,
        dropout=args.dropout,
        activation=args.activation,
        norm=args.norm,
        residual=True,
    )
    head = ProjectionHead(cfg, norm_last=True).to(device)

    opt = torch.optim.AdamW(
        head.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.fp16))

    # Curriculum & scheduling
    curriculum = parse_curriculum(args.curriculum)
    if curriculum:
        total_curr_epochs = sum(e for _, e in curriculum)
        if total_curr_epochs != args.epochs:
            raise ValueError(
                f"Curriculum epochs ({total_curr_epochs}) must sum to --epochs ({args.epochs})"
            )

    temps = make_temperature_scheduler(
        args.epochs, args.temp_start, args.temp_end, args.temp_mode
    )

    # Hybrid alpha schedule (only applied during hybrid epochs)
    if args.hybrid_alpha_start < 0 or args.hybrid_alpha_end < 0:
        raise ValueError("Hybrid alpha must be non-negative.")
    # We'll compute per hybrid epoch later (linear interpolation across hybrid span).
    hybrid_total_epochs = (
        sum(e for m, e in curriculum if m == "hybrid") if curriculum else 0
    )
    hybrid_alphas = []
    if hybrid_total_epochs > 0:
        for i in range(hybrid_total_epochs):
            p = i / max(1, hybrid_total_epochs - 1)
            a = (
                args.hybrid_alpha_start
                + (args.hybrid_alpha_end - args.hybrid_alpha_start) * p
            )
            hybrid_alphas.append(float(a))

    # Initial variance logging
    variance_logs: List[Dict] = []
    if args.log_var_every > 0:
        topk_initial = compute_topk_variance(
            embeds.cpu(), k=args.var_topk, sample=args.var_sample, seed=args.seed
        )
        variance_logs.append(
            {
                "epoch": 0,
                "stage": "initial",
                "temperature": temps[0],
                "topk_var": topk_initial,
            }
        )
        print(f"[VAR] Epoch 0 (initial) top-{args.var_topk} var: {topk_initial}")

    print(
        f"[INFO] Training projection head: input_dim={input_dim} hidden_dim={cfg.hidden_dim} out_dim={cfg.output_dim}"
    )
    print(
        f"[INFO] Total epochs={args.epochs} batches/epoch={math.ceil(len(ds) / args.batch_size)}"
    )
    if curriculum:
        print(f"[INFO] Curriculum: {curriculum}")
    else:
        print(f"[INFO] Static label mode: {args.label_mode}")

    # Main training loop
    epoch_global = 0
    hybrid_epoch_index = 0  # counter only for hybrid epochs

    logs: List[Dict] = []

    for epoch in range(1, args.epochs + 1):
        # Determine active phase label mode
        if curriculum:
            cumulative = 0
            active_mode = None
            for mode_name, length in curriculum:
                if epoch_global < cumulative + length:
                    active_mode = mode_name
                    break
                cumulative += length
            if active_mode is None:
                active_mode = curriculum[-1][0]
        else:
            active_mode = args.label_mode

        # Determine temperature
        temperature = temps[epoch - 1]

        # Determine hybrid alpha (if in hybrid phase)
        if active_mode == "hybrid" and hybrid_total_epochs > 0:
            alpha = hybrid_alphas[hybrid_epoch_index]
        else:
            alpha = args.hybrid_alpha_start  # fallback or irrelevant

        head.train()
        running = 0.0
        steps = 0

        for batch in dl:
            x, fine_ids, coarse_ids = batch
            x = x.to(device, non_blocking=True)
            fine_ids = fine_ids.to(device, non_blocking=True)
            coarse_ids = coarse_ids.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and args.fp16)):
                z = head(x)

                # Cluster weights
                fine_weights = None
                coarse_weights = None
                if args.cluster_weighting != "none":
                    if active_mode in ("fine", "hybrid"):
                        fine_weights = compute_cluster_weights(
                            fine_ids, args.cluster_weighting
                        )
                    if active_mode in ("coarse", "hybrid"):
                        coarse_weights = compute_cluster_weights(
                            coarse_ids, args.cluster_weighting
                        )

                # Hard negative mask (only hybrid if enabled)
                hard_neg_mask = None
                if (
                    args.hard_neg
                    and active_mode == "hybrid"
                    and args.hard_neg_scale > 1.0
                ):
                    # Hard negatives: same coarse but different fine
                    coarse_eq = coarse_ids.unsqueeze(0) == coarse_ids.unsqueeze(1)
                    fine_eq = fine_ids.unsqueeze(0) == fine_ids.unsqueeze(1)
                    eye = torch.eye(
                        coarse_ids.shape[0], device=coarse_ids.device
                    ).bool()
                    hard_neg_mask = coarse_eq & (~fine_eq) & (~eye)

                if active_mode == "hybrid":
                    loss = hybrid_contrastive_loss(
                        z,
                        fine_ids,
                        coarse_ids,
                        temp=temperature,
                        alpha=alpha,
                        fine_weights=fine_weights,
                        coarse_weights=coarse_weights,
                        hard_neg_mask=hard_neg_mask,
                        hard_neg_scale=args.hard_neg_scale,
                    )
                elif active_mode == "fine":
                    # Use extended supervised loss for consistency
                    loss = supervised_contrastive_loss(
                        z,
                        fine_ids,
                        temperature=temperature,
                        label_weights=fine_weights,
                    )
                elif active_mode == "coarse":
                    loss = supervised_contrastive_loss(
                        z,
                        coarse_ids,
                        temperature=temperature,
                        label_weights=coarse_weights,
                    )
                else:
                    raise ValueError(f"Unexpected active_mode: {active_mode}")

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()

            steps += 1
            running += loss.item()

        avg_loss = running / max(1, steps)
        epoch_log = {
            "epoch": epoch,
            "mode": active_mode,
            "loss": avg_loss,
            "temperature": temperature,
        }
        if active_mode == "hybrid":
            epoch_log["hybrid_alpha"] = alpha
        logs.append(epoch_log)

        msg = f"[EPOCH {epoch}/{args.epochs}] mode={active_mode} loss={avg_loss:.4f} temp={temperature:.4f}"
        if active_mode == "hybrid":
            msg += f" alpha={alpha:.4f}"
        print(msg)

        # Variance logging
        if args.log_var_every > 0 and (
            epoch % args.log_var_every == 0 or epoch == args.epochs
        ):
            head.eval()
            with torch.no_grad():
                B = 4096
                proj_list = []
                for start in range(0, embeds.shape[0], B):
                    z_all = head(embeds[start : start + B].to(device))
                    proj_list.append(z_all.cpu())
                proj_all = torch.cat(proj_list, dim=0)
                topk_proj = compute_topk_variance(
                    proj_all, k=args.var_topk, sample=args.var_sample, seed=args.seed
                )
                variance_logs.append(
                    {
                        "epoch": epoch,
                        "stage": "projected",
                        "temperature": temperature,
                        "topk_var": topk_proj,
                        "mode": active_mode,
                    }
                )
                print(
                    f"[VAR] Epoch {epoch} projected top-{args.var_topk} var: {topk_proj}"
                )

        # Increment counters
        epoch_global += 1
        if active_mode == "hybrid":
            hybrid_epoch_index += 1

    # Save model
    if args.out_proj:
        out_p = Path(args.out_proj)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": head.state_dict(), "config": asdict(cfg)}, out_p)
        print(f"[INFO] Saved projection head: {out_p}")

    # Export projected embeddings
    if args.out_embeds:
        head.eval()
        proj_list = []
        with torch.no_grad():
            B = 4096
            for start in range(0, embeds.shape[0], B):
                z = head(embeds[start : start + B].to(device))
                proj_list.append(z.cpu())
        proj_all = torch.cat(proj_list, dim=0)
        out_e = Path(args.out_embeds)
        out_e.parent.mkdir(parents=True, exist_ok=True)
        torch.save(proj_all, out_e)
        print(f"[INFO] Saved projected embeddings: {out_e}")

    # Save JSON log
    if args.log_json:
        out_j = Path(args.log_json)
        out_j.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "epochs": args.epochs,
            "curriculum": curriculum,
            "temperature_schedule": temps,
            "logs": logs,
            "variance_logs": variance_logs,
            "config": {
                "cluster_weighting": args.cluster_weighting,
                "temp_mode": args.temp_mode,
                "temp_start": args.temp_start,
                "temp_end": args.temp_end,
                "hybrid_alpha_start": args.hybrid_alpha_start,
                "hybrid_alpha_end": args.hybrid_alpha_end,
                "hard_neg": args.hard_neg,
                "hard_neg_scale": args.hard_neg_scale,
                "remove_top": args.remove_top,
                "augment_features": args.augment_features,
            },
        }
        with out_j.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[INFO] Wrote training log JSON: {out_j}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    ap = argparse.ArgumentParser(
        "Curriculum + contrastive projection head trainer for glyph embeddings."
    )
    ap.add_argument("--embeds", required=True, help="Input base embeddings (.pt)")
    ap.add_argument("--meta", required=True, help="Metadata JSONL path")
    ap.add_argument(
        "--pca-model",
        type=str,
        default=None,
        help="Optional PCA model directory (mean.pt, components.pt, meta.json).",
    )
    ap.add_argument(
        "--remove-top",
        type=int,
        default=0,
        help="Remove top-K principal components after centering.",
    )
    ap.add_argument(
        "--augment-features",
        action="store_true",
        help="Append structural features (group_count, tokens_non_eos, width_em, height_em).",
    )
    # Legacy static mode (still supported if no curriculum)
    ap.add_argument(
        "--label-mode",
        choices=("fine", "coarse", "hybrid"),
        default="hybrid",
        help="Static label mode (ignored if --curriculum provided).",
    )
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden-dim", type=int, default=512)
    ap.add_argument("--out-dim", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--activation", type=str, default="gelu")
    ap.add_argument("--norm", type=str, default="layer")

    # Temperatures / scheduling
    ap.add_argument("--temp-start", type=float, default=0.07)
    ap.add_argument("--temp-end", type=float, default=0.07)
    ap.add_argument(
        "--temp-mode",
        type=str,
        default="linear",
        choices=("linear", "cosine", "exp"),
        help="Temperature interpolation mode across epochs.",
    )

    # Hybrid alpha schedule
    ap.add_argument("--hybrid-alpha-start", type=float, default=0.5)
    ap.add_argument("--hybrid-alpha-end", type=float, default=0.5)

    # Cluster weighting
    ap.add_argument(
        "--cluster-weighting",
        type=str,
        default="none",
        choices=("none", "inv_sqrt", "inv_log"),
        help="Per-sample weighting based on cluster size.",
    )

    # Curriculum
    ap.add_argument(
        "--curriculum",
        type=str,
        default=None,
        help='Comma spec, e.g. "coarse:50,fine:100,hybrid:350" summing to --epochs.',
    )

    # Hard negatives (hybrid only)
    ap.add_argument(
        "--hard-neg",
        action="store_true",
        help="Enable additional emphasis on negatives sharing coarse but diff fine label (hybrid phase).",
    )
    ap.add_argument(
        "--hard-neg-scale",
        type=float,
        default=1.5,
        help="Scaling factor (>1) for hard negative exp(sim).",
    )

    # Variance logging
    ap.add_argument(
        "--log-var-every",
        type=int,
        default=0,
        help="Log top-K variance every N epochs (0 disables).",
    )
    ap.add_argument(
        "--var-topk",
        type=int,
        default=10,
        help="Number of leading variance ratios to log.",
    )
    ap.add_argument(
        "--var-sample",
        type=int,
        default=0,
        help="If >0, subsample this many rows for variance computation.",
    )

    ap.add_argument("--fp16", action="store_true", help="Enable mixed precision")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument(
        "--out-proj",
        type=str,
        default=None,
        help="Path to save projection head state_dict (.pt).",
    )
    ap.add_argument(
        "--out-embeds",
        type=str,
        default=None,
        help="Path to save projected embeddings (.pt).",
    )
    ap.add_argument(
        "--log-json",
        type=str,
        default=None,
        help="Optional path to save JSON training logs.",
    )
    ap.add_argument("--seed", type=int, default=42, help="RNG seed.")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    embeds = torch.load(args.embeds, map_location="cpu")
    if not isinstance(embeds, torch.Tensor) or embeds.ndim != 2:
        raise ValueError("Embeddings must be a 2-D torch.Tensor")
    embeds = embeds.float()
    meta = load_metadata(args.meta)
    if len(meta) != embeds.shape[0]:
        raise ValueError(
            f"Meta rows ({len(meta)}) != embeddings rows ({embeds.shape[0]}); ensure matching extraction."
        )
    train_projection(embeds, meta, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
