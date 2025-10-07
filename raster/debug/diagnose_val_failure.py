#!/usr/bin/env python3
"""
diagnose_val_failure.py
=======================

Diagnostic script to understand why validation loss diverges while training loss improves.

Usage:
    python -m raster.debug.diagnose_val_failure \
      --checkpoint raster/checkpoints/best.pt \
      --db dataset/glyphs.db \
      --limit 528000 \
      --pre-raster-mmap-path preraster_full_528k_u8.dat
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from raster.model import load_checkpoint, build_glyph_levit_128s, GlyphLeViTConfig
from raster.dataset import (
    GlyphRasterDataset,
    DatasetConfig,
    make_train_val_split,
    simple_collate,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    ap.add_argument("--db", required=True, help="Path to glyphs.db")
    ap.add_argument("--limit", type=int, default=528000)
    ap.add_argument("--pre-raster-mmap-path", type=str, default=None)
    ap.add_argument(
        "--num-samples", type=int, default=100, help="Samples to analyze per split"
    )
    return ap.parse_args()


def analyze_predictions(model, loader, device, num_samples=100, split_name="train"):
    """Analyze model predictions in detail."""
    model.eval()

    all_logits = []
    all_labels = []
    all_embeddings = []
    all_glyph_ids = []
    all_raw_labels = []

    print(f"\n[{split_name.upper()}] Analyzing {num_samples} samples...")

    with torch.no_grad():
        total = 0
        for batch in loader:
            imgs = batch["images"].to(device)
            labels = batch["labels"].to(device)

            out = model(imgs)
            logits = out["logits"]
            embeddings = out["embedding"]

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            all_embeddings.append(embeddings.cpu())
            all_glyph_ids.extend(batch["glyph_ids"])
            all_raw_labels.extend(batch["raw_labels"])

            total += imgs.size(0)
            if total >= num_samples:
                break

    logits = torch.cat(all_logits, dim=0)[:num_samples]
    labels = torch.cat(all_labels, dim=0)[:num_samples]
    embeddings = torch.cat(all_embeddings, dim=0)[:num_samples]

    # Compute statistics
    probs = F.softmax(logits, dim=1)
    max_probs, preds = probs.max(dim=1)

    # Top-k accuracy
    top1_correct = (preds == labels).sum().item()
    _, top5_preds = logits.topk(5, dim=1)
    top5_correct = sum((labels[i] in top5_preds[i]) for i in range(len(labels)))

    # Entropy of predictions
    entropy = -(probs * (probs + 1e-10).log()).sum(dim=1)

    # Logit statistics
    logit_max = logits.max(dim=1)[0]
    logit_min = logits.min(dim=1)[0]
    logit_range = logit_max - logit_min

    # Embedding statistics
    emb_norms = embeddings.norm(dim=1)

    print(f"  Top-1 Accuracy: {top1_correct / num_samples * 100:.2f}%")
    print(f"  Top-5 Accuracy: {top5_correct / num_samples * 100:.2f}%")
    print(f"  Mean Max Probability: {max_probs.mean():.4f}")
    print(
        f"  Mean Entropy: {entropy.mean():.4f} (uniform={torch.log(torch.tensor(float(logits.size(1)))):.4f})"
    )
    print(f"  Logit Range (mean): {logit_range.mean():.4f}")
    print(f"  Logit Max (mean): {logit_max.mean():.4f}")
    print(f"  Logit Min (mean): {logit_min.mean():.4f}")
    print(f"  Embedding Norm (mean): {emb_norms.mean():.4f}")

    # Check for degenerate predictions
    unique_preds = preds.unique()
    print(f"  Unique predictions: {len(unique_preds)} / {logits.size(1)} classes")

    # Most common predictions
    from collections import Counter

    pred_counts = Counter(preds.tolist())
    print(f"  Top-5 most common predictions:")
    for pred_idx, count in pred_counts.most_common(5):
        print(f"    Class {pred_idx}: {count} times ({count / num_samples * 100:.1f}%)")

    # Show a few example predictions
    print(f"\n  Sample predictions:")
    for i in range(min(5, num_samples)):
        pred_idx = preds[i].item()
        true_idx = labels[i].item()
        prob = max_probs[i].item()
        print(
            f"    Sample {i}: true={true_idx} ('{all_raw_labels[i]}'), pred={pred_idx}, prob={prob:.4f}"
        )

    return {
        "top1_acc": top1_correct / num_samples,
        "top5_acc": top5_correct / num_samples,
        "mean_max_prob": max_probs.mean().item(),
        "mean_entropy": entropy.mean().item(),
        "unique_preds": len(unique_preds),
        "logit_range_mean": logit_range.mean().item(),
        "embedding_norm_mean": emb_norms.mean().item(),
    }


def check_bn_stats(model):
    """Check BatchNorm running statistics."""
    print("\n[BATCHNORM] Checking BN layer statistics...")

    bn_layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d) or isinstance(
            module, torch.nn.BatchNorm1d
        ):
            bn_layers.append((name, module))

    if not bn_layers:
        print("  No BatchNorm layers found")
        return

    print(f"  Found {len(bn_layers)} BatchNorm layers")

    for name, bn in bn_layers[:5]:  # Show first 5
        if hasattr(bn, "running_mean") and bn.running_mean is not None:
            mean_abs = bn.running_mean.abs().mean().item()
            var_mean = bn.running_var.mean().item() if bn.running_var is not None else 0
            print(
                f"    {name}: running_mean_abs={mean_abs:.4f}, running_var_mean={var_mean:.4f}"
            )


def check_weight_stats(model):
    """Check classifier head weight statistics."""
    print("\n[WEIGHTS] Checking classifier head weights...")

    # Find the classification head
    head = getattr(model.backbone, "head", None)
    if head is None:
        print("  No classification head found")
        return

    # Get linear layer weights
    linear = None
    if hasattr(head, "l"):
        linear = head.l
    elif isinstance(head, torch.nn.Linear):
        linear = head

    if linear is None:
        print("  Could not find linear layer in head")
        return

    W = linear.weight  # (num_classes, feature_dim)
    if linear.bias is not None:
        b = linear.bias  # (num_classes,)
    else:
        b = None

    print(f"  Weight shape: {W.shape}")
    print(f"  Weight mean: {W.mean():.6f}")
    print(f"  Weight std: {W.std():.6f}")
    print(f"  Weight abs mean: {W.abs().mean():.6f}")
    print(f"  Weight max: {W.max():.6f}")
    print(f"  Weight min: {W.min():.6f}")

    if b is not None:
        print(f"  Bias mean: {b.mean():.6f}")
        print(f"  Bias std: {b.std():.6f}")
        print(f"  Bias abs mean: {b.abs().mean():.6f}")

    # Check for degenerate weights (too small)
    small_weights = (W.abs() < 1e-4).float().mean()
    print(f"  Fraction of weights < 1e-4: {small_weights:.4f}")

    # Check weight norms per class
    weight_norms = W.norm(dim=1)
    print(f"  Weight norm per class (mean): {weight_norms.mean():.6f}")
    print(f"  Weight norm per class (std): {weight_norms.std():.6f}")
    print(f"  Weight norm per class (min): {weight_norms.min():.6f}")
    print(f"  Weight norm per class (max): {weight_norms.max():.6f}")


def main():
    args = parse_args()

    print("=" * 80)
    print("VALIDATION FAILURE DIAGNOSTIC")
    print("=" * 80)

    # Load datasets
    print("\n[SETUP] Loading datasets...")
    ds_cfg = DatasetConfig(
        db_path=args.db,
        limit=args.limit,
        randomize_query=True,
        image_size=128,
        supersample=2,
        augment=True,  # Will be disabled for val split
        seed=42,
        pre_rasterize=True,
        pre_raster_dtype="uint8",
        pre_raster_mmap=True,
        pre_raster_mmap_path=args.pre_raster_mmap_path,
        min_label_count=2,
        drop_singletons=True,
        verbose_stats=False,
    )

    full_ds = GlyphRasterDataset(ds_cfg)
    train_ds, val_ds = make_train_val_split(full_ds, val_fraction=0.1, seed=42)

    print(f"  Train size: {len(train_ds)}")
    print(f"  Val size: {len(val_ds)}")
    print(f"  Num classes: {len(train_ds.label_to_index)}")
    print(f"  Train augment: {train_ds.cfg.augment}")
    print(f"  Val augment: {val_ds.cfg.augment}")

    train_loader = DataLoader(
        train_ds,
        batch_size=256,
        shuffle=False,
        num_workers=0,
        collate_fn=simple_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=256, shuffle=False, num_workers=0, collate_fn=simple_collate
    )

    # Load model
    print("\n[SETUP] Loading model...")
    device = "cpu"
    num_classes = len(train_ds.label_to_index)

    cfg = GlyphLeViTConfig(
        img_size=128,
        num_classes=num_classes,
        embedding_out_dim=128,
        hidden_dim=256,
        activation="gelu",
        distillation=False,
        replicate_gray=True,
    )

    from raster.model import build_glyph_levit_128s

    model = build_glyph_levit_128s(cfg)
    model.to(device)

    load_checkpoint(model, args.checkpoint, strict=False)
    print(f"  Loaded checkpoint: {args.checkpoint}")

    # Run diagnostics
    check_weight_stats(model)
    check_bn_stats(model)

    train_stats = analyze_predictions(
        model, train_loader, device, args.num_samples, "TRAIN"
    )
    val_stats = analyze_predictions(model, val_loader, device, args.num_samples, "VAL")

    # Compare train vs val
    print("\n" + "=" * 80)
    print("SUMMARY: Train vs Val Comparison")
    print("=" * 80)
    print(
        f"  Top-1 Accuracy:    Train {train_stats['top1_acc'] * 100:.2f}%  |  Val {val_stats['top1_acc'] * 100:.2f}%"
    )
    print(
        f"  Top-5 Accuracy:    Train {train_stats['top5_acc'] * 100:.2f}%  |  Val {val_stats['top5_acc'] * 100:.2f}%"
    )
    print(
        f"  Mean Max Prob:     Train {train_stats['mean_max_prob']:.4f}  |  Val {val_stats['mean_max_prob']:.4f}"
    )
    print(
        f"  Mean Entropy:      Train {train_stats['mean_entropy']:.4f}  |  Val {val_stats['mean_entropy']:.4f}"
    )
    print(
        f"  Unique Preds:      Train {train_stats['unique_preds']}  |  Val {val_stats['unique_preds']}"
    )
    print(
        f"  Logit Range:       Train {train_stats['logit_range_mean']:.4f}  |  Val {val_stats['logit_range_mean']:.4f}"
    )
    print(
        f"  Embedding Norm:    Train {train_stats['embedding_norm_mean']:.4f}  |  Val {val_stats['embedding_norm_mean']:.4f}"
    )

    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)

    # Diagnose based on patterns
    if val_stats["top1_acc"] < 0.01 and train_stats["top1_acc"] > 0.05:
        print("⚠️  SEVERE TRAIN-VAL GAP detected")
        print("    - Training accuracy significantly better than validation")
        print("    - Model is either overfitting or there's a distribution mismatch")

    if val_stats["mean_entropy"] > 7.0:  # Close to log(1517) = 7.32
        print("⚠️  NEAR-UNIFORM PREDICTIONS on validation")
        print("    - Model outputs are close to uniform distribution")
        print("    - Classifier head is not learning validation patterns")

    if val_stats["unique_preds"] < 100:
        print("⚠️  DEGENERATE PREDICTIONS detected")
        print(
            f"    - Only {val_stats['unique_preds']} unique predictions out of {num_classes} classes"
        )
        print("    - Model is collapsing to a few dominant classes")

    if val_stats["logit_range_mean"] < 5.0:
        print("⚠️  WEAK LOGIT SEPARATION")
        print("    - Logits are not well separated between classes")
        print("    - Classifier confidence is low")

    if abs(train_stats["embedding_norm_mean"] - val_stats["embedding_norm_mean"]) > 0.2:
        print("⚠️  EMBEDDING NORM MISMATCH")
        print("    - Train and val embeddings have different norms")
        print("    - Possible BN statistics mismatch")

    print("\nRECOMMENDATIONS:")
    print("  1. Check if train/val splits have different font distributions")
    print("  2. Try training with NO augmentation on train split")
    print("  3. Reduce weight decay (try 0.001 or 0.0)")
    print("  4. Increase head learning rate (try 0.005-0.01)")
    print("  5. Try freezing backbone and only training classifier head")
    print("  6. Check if label mapping is consistent across splits")


if __name__ == "__main__":
    main()
