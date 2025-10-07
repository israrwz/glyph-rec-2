#!/usr/bin/env python3
"""
test_gradient_flow.py
=====================

Test script to verify that gradients flow correctly from both classifier and embedding heads
back to the backbone after the gradient flow bug fix.

Usage:
    python -m raster.debug.test_gradient_flow
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from raster.model import build_glyph_levit_128s, GlyphLeViTConfig


def test_gradient_flow():
    """
    Test that gradients flow from both heads back to backbone.
    """
    print("=" * 80)
    print("GRADIENT FLOW VALIDATION TEST")
    print("=" * 80)

    # Build model
    print("\n[SETUP] Building model...")
    cfg = GlyphLeViTConfig(
        img_size=128,
        num_classes=10,  # Small for testing
        embedding_out_dim=128,
        hidden_dim=256,
        distillation=False,
        replicate_gray=True,
    )
    model = build_glyph_levit_128s(cfg)
    model.train()

    # Create dummy input
    batch_size = 4
    x = torch.randn(batch_size, 1, 128, 128)
    labels = torch.randint(0, 10, (batch_size,))

    print(f"  Input shape: {x.shape}")
    print(f"  Num classes: {cfg.num_classes}")
    print(f"  Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Forward pass
    print("\n[FORWARD] Computing outputs...")
    out = model(x)
    logits = out["logits"]
    embeddings = out["embedding"]

    print(f"  Logits shape: {logits.shape}")
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Embedding norm: {embeddings.norm(dim=1).mean():.4f}")

    # Compute losses
    print("\n[LOSS] Computing losses...")
    ce_loss = F.cross_entropy(logits, labels)

    # Simple embedding loss (intra-class similarity)
    # For testing, just use a dummy target
    emb_target = torch.randn_like(embeddings)
    emb_target = F.normalize(emb_target, dim=1)
    emb_loss = (1 - (embeddings * emb_target).sum(dim=1)).mean()

    total_loss = ce_loss + 0.5 * emb_loss

    print(f"  CE Loss: {ce_loss.item():.4f}")
    print(f"  Embedding Loss: {emb_loss.item():.4f}")
    print(f"  Total Loss: {total_loss.item():.4f}")

    # Backward pass
    print("\n[BACKWARD] Computing gradients...")
    total_loss.backward()

    # Check gradients in key components
    print("\n[GRADIENTS] Checking gradient presence...")

    backbone_grads = []
    classifier_grads = []
    embedding_grads = []

    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"  ⚠️  {name}: NO GRADIENT")
            continue

        grad_norm = param.grad.norm().item()

        if name.startswith("backbone."):
            backbone_grads.append(grad_norm)
        elif "head" in name and "embed" not in name:
            classifier_grads.append(grad_norm)
        elif "embed_head" in name:
            embedding_grads.append(grad_norm)

    print(f"\n  Backbone gradients: {len(backbone_grads)} params")
    print(f"    Mean grad norm: {sum(backbone_grads) / len(backbone_grads):.6f}")
    print(f"    Max grad norm: {max(backbone_grads):.6f}")
    print(f"    Min grad norm: {min(backbone_grads):.6f}")

    print(f"\n  Classifier head gradients: {len(classifier_grads)} params")
    if classifier_grads:
        print(
            f"    Mean grad norm: {sum(classifier_grads) / len(classifier_grads):.6f}"
        )
        print(f"    Max grad norm: {max(classifier_grads):.6f}")

    print(f"\n  Embedding head gradients: {len(embedding_grads)} params")
    if embedding_grads:
        print(f"    Mean grad norm: {sum(embedding_grads) / len(embedding_grads):.6f}")
        print(f"    Max grad norm: {max(embedding_grads):.6f}")

    # Validation
    print("\n" + "=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)

    success = True

    if not backbone_grads:
        print("❌ FAIL: No gradients in backbone!")
        success = False
    elif sum(backbone_grads) < 1e-8:
        print("❌ FAIL: Backbone gradients are near-zero!")
        success = False
    else:
        print("✅ PASS: Backbone has non-zero gradients")

    if not classifier_grads:
        print("❌ FAIL: No gradients in classifier head!")
        success = False
    else:
        print("✅ PASS: Classifier head has gradients")

    if not embedding_grads:
        print("❌ FAIL: No gradients in embedding head!")
        success = False
    else:
        print("✅ PASS: Embedding head has gradients")

    # Test feature sharing
    print("\n[FEATURE SHARING] Verifying shared features...")
    model.zero_grad()
    out = model(x)

    # Check that _last_backbone_features is set
    if not hasattr(model, "_last_backbone_features"):
        print("❌ FAIL: _last_backbone_features not set!")
        success = False
    else:
        feats = model._last_backbone_features
        print(f"  ✅ Feature tensor shape: {feats.shape}")
        print(f"  ✅ Feature requires_grad: {feats.requires_grad}")

        if not feats.requires_grad:
            print("  ❌ FAIL: Features do not require gradients!")
            success = False
        else:
            print("  ✅ PASS: Features are part of computation graph")

    # Test that both heads get gradients from the same features
    print("\n[DUAL-HEAD GRADIENT TEST]")
    model.zero_grad()
    x_test = torch.randn(2, 1, 128, 128)
    labels_test = torch.randint(0, 10, (2,))

    out = model(x_test)

    # Only classifier loss
    model.zero_grad()
    ce_only = F.cross_entropy(out["logits"], labels_test)
    ce_only.backward()
    backbone_grad_ce = sum(
        p.grad.norm().item() for p in model.backbone.parameters() if p.grad is not None
    )

    # Only embedding loss
    model.zero_grad()
    out2 = model(x_test)
    emb_target2 = torch.randn_like(out2["embedding"])
    emb_target2 = F.normalize(emb_target2, dim=1)
    emb_only = (1 - (out2["embedding"] * emb_target2).sum(dim=1)).mean()
    emb_only.backward()
    backbone_grad_emb = sum(
        p.grad.norm().item() for p in model.backbone.parameters() if p.grad is not None
    )

    print(f"  Backbone grad norm (CE only): {backbone_grad_ce:.6f}")
    print(f"  Backbone grad norm (Emb only): {backbone_grad_emb:.6f}")

    if backbone_grad_ce < 1e-8:
        print("  ❌ FAIL: Classifier loss does not update backbone!")
        success = False
    else:
        print("  ✅ PASS: Classifier loss updates backbone")

    if backbone_grad_emb < 1e-8:
        print("  ❌ FAIL: Embedding loss does not update backbone!")
        success = False
    else:
        print("  ✅ PASS: Embedding loss updates backbone")

    # Final verdict
    print("\n" + "=" * 80)
    if success:
        print("✅ ALL TESTS PASSED - Gradient flow is correct!")
        print("=" * 80)
        return 0
    else:
        print("❌ TESTS FAILED - Gradient flow is broken!")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    exit_code = test_gradient_flow()
    sys.exit(exit_code)
