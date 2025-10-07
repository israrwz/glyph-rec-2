"""
model.py
========
Lightweight LeViT_128S wrapper specialized for raster glyph embeddings.

Goals:
- Instantiate a LeViT_128S backbone at img_size=128 (rather than the upstream 224),
  disabling distillation and heavy augmentation assumptions.
- Provide a compact embedding head that yields a normalized 128-D vector.
- Support grayscale (1-channel) glyph inputs by simple replication to 3 channels
  (avoids modifying original conv weights).
- Expose a single forward() returning a dict:
      {
        "embedding": <B, embed_dim_out>,
        "logits": <B, num_classes>  # only if num_classes > 0
      }

Source Alignment:
- This file reads the upstream LeViT definitions directly (no code duplication of
  internal attention blocks). We call the LeViT constructor directly rather than
  model_factory to customize img_size=128.
- Architecture parameters (C, D, N, X) are taken from specification['LeViT_128S'].

Pretrained Weights Notice:
- We deliberately avoid loading ImageNet-pretrained weights for 224-resolution models
  because relative attention bias tables and token spatial resolutions differ
  (8x8 initial tokens at 128 vs 14x14 at 224). Any direct load would mismatch or
  require remapping of bias indices. Phase 1 uses random initialization.

Checkpoint Format (save_checkpoint):
{
  "backbone": backbone_state_dict,
  "embed_head": embedding_head_state_dict,
  "num_classes": int,
  "embedding_dim": int,
  "config": { ... build args ... },
  "step": optional_int
}

Example Usage:
    from raster.model import build_glyph_levit_128s

    model = build_glyph_levit_128s(num_classes=1000)
    batch = torch.randn(16, 1, 128, 128)  # grayscale glyphs
    out = model(batch)
    emb = out["embedding"]   # (16, 128)
    logits = out["logits"]   # (16, 1000)

"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import weakref

# Import upstream LeViT components
# The repository is vendored in ../LeViT so this relative import should work
# --- LeViT import handling --------------------------------------------------
# The upstream LeViT repo (vendored in ../LeViT) references a local `utils.py`
# via a plain `import utils` inside `levit.py`. Because the `LeViT` directory
# is not a Python package (no __init__.py), a direct
# `from LeViT.levit import ...` can fail to satisfy that relative import.
#
# We patch sys.path to include the LeViT directory explicitly, then import.
import sys as _sys, pathlib as _pathlib

_LEVIT_DIR = _pathlib.Path(__file__).resolve().parent.parent / "LeViT"
if _LEVIT_DIR.exists():
    p_str = str(_LEVIT_DIR)
    if p_str not in _sys.path:
        _sys.path.append(p_str)
try:
    # After path injection, `import levit` resolves and its internal `import utils`
    # succeeds because both live in the same directory on sys.path.
    from levit import LeViT, specification, b16  # type: ignore
except ModuleNotFoundError as e:
    raise ImportError(
        "Failed to import LeViT modules. Ensure the 'LeViT' directory exists at project root."
    ) from e
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Configuration Dataclass
# ---------------------------------------------------------------------------


@dataclass
class GlyphLeViTConfig:
    img_size: int = 128
    embedding_out_dim: int = 128
    hidden_dim: int = 256
    dropout: float = 0.1
    use_layernorm_head: bool = False
    replicate_gray: bool = True  # If input is (B,1,H,W) replicate to 3 channels
    activation: str = "gelu"  # "gelu" or "hardswish"
    num_classes: int = 0  # 0 => no classification head
    distillation: bool = False  # Always disabled for Phase 1
    # Reserved for future: margin losses, partial freeze, etc.


# ---------------------------------------------------------------------------
# Embedding Head
# ---------------------------------------------------------------------------


class EmbeddingHead(nn.Module):
    """
    Projection head producing an L2-normalized embedding.

    Input: (B, in_dim)
    Output: (B, out_dim) with unit L2 norm (unless disabled)
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        activation: str = "gelu",
        dropout: float = 0.1,
        layernorm_first: bool = False,
        l2_normalize: bool = True,
    ):
        super().__init__()
        self.l2_normalize = l2_normalize
        act_layer: nn.Module
        if activation.lower() == "gelu":
            act_layer = nn.GELU()
        elif activation.lower() == "hardswish":
            act_layer = nn.Hardswish()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        norm = nn.LayerNorm(in_dim) if layernorm_first else nn.Identity()
        self.net = nn.Sequential(
            norm,
            nn.Linear(in_dim, hidden_dim),
            act_layer,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use a mild truncated normal-like init
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        if self.l2_normalize:
            x = F.normalize(x, dim=-1, eps=1e-8)
        return x


# ---------------------------------------------------------------------------
# Wrapper Model
# ---------------------------------------------------------------------------


class GlyphLeViT(nn.Module):
    """
    Wrapper around LeViT_128S backbone (custom img_size=128) plus
    an embedding head and an optional classification head.

    Forward returns:
      {
        "embedding": (B, embedding_out_dim),
        "logits": (B, num_classes)  # only if num_classes > 0
      }

    Notes:
    - If input is grayscale and replicate_gray=True, we replicate channels to 3.
    - Pooled feature taken after mean over spatial tokens (as in original LeViT).
    """

    def __init__(
        self,
        backbone: LeViT,
        embed_head: EmbeddingHead,
        num_classes: int,
        replicate_gray: bool,
    ):
        super().__init__()
        self.backbone = backbone
        self.embed_head = embed_head
        self.num_classes = num_classes
        self.replicate_gray = replicate_gray

        # If num_classes == 0, backbone.head is Identity but still exists;
        # we keep it around in case we later fine-tune.
        # Provide a direct handle for classification logits:
        self._class_head = backbone.head if num_classes > 0 else None

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Handle grayscale replication
        if self.replicate_gray and x.dim() == 4 and x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        # Upstream LeViT forward returns either:
        # - logits (Tensor) if distillation=False
        # - (logits, dist_logits) if distillation=True
        # We forced distillation=False in build.
        logits_or_tuple = self.backbone(x)
        if isinstance(logits_or_tuple, tuple):
            # Defensive: average if user accidentally set distillation
            logits = sum(logits_or_tuple) / len(logits_or_tuple)
        else:
            logits = logits_or_tuple

        # We need the pooled features pre-class-head to build the embedding.
        # The original forward path:
        #   x = patch_embed -> flatten -> transpose -> blocks -> mean(1) -> head(...)
        # We cannot easily intercept inside backbone without modifying it;
        # so we re-run minimal logic OR store an intermediate hook.
        # Easiest: register a forward hook on backbone.head input during build.
        # For simplicity here, we attach a hook populated in build function.
        feats = getattr(self, "_last_backbone_features", None)
        if feats is None:
            # Fallback: if hook not set, raise explicit error to avoid silent mismatch.
            raise RuntimeError(
                "Backbone feature hook not initialized. Ensure build_glyph_levit_128s() was used."
            )

        embedding = self.embed_head(feats)

        out = {"embedding": embedding}
        if self.num_classes > 0:
            out["logits"] = logits
        return out


# ---------------------------------------------------------------------------
# Build Function
# ---------------------------------------------------------------------------


def build_glyph_levit_128s(
    config: Optional[GlyphLeViTConfig] = None,
) -> GlyphLeViT:
    """
    Construct a GlyphLeViT model using the LeViT_128S specification
    with a custom image size (default 128).

    Parameters
    ----------
    config : GlyphLeViTConfig
        Configuration dataclass. If None, defaults are used.

    Returns
    -------
    GlyphLeViT
        Initialized model (random weights).
    """
    if config is None:
        config = GlyphLeViTConfig()

    spec = specification["LeViT_128S"]
    # Parse spec fields
    embed_dim = [int(x) for x in spec["C"].split("_")]  # [128,256,384]
    num_heads = [int(x) for x in spec["N"].split("_")]  # [4,6,8]
    depth = [int(x) for x in spec["X"].split("_")]  # [2,3,4]
    D = spec["D"]  # key dim = 16
    drop_path = spec["drop_path"]

    # Build the LeViT backbone directly to override img_size
    act = nn.Hardswish  # consistent with upstream
    backbone = LeViT(
        img_size=config.img_size,
        patch_size=16,
        in_chans=3,
        num_classes=config.num_classes if config.num_classes > 0 else 0,
        embed_dim=embed_dim,
        key_dim=[D] * 3,
        depth=depth,
        num_heads=num_heads,
        attn_ratio=[2, 2, 2],
        mlp_ratio=[2, 2, 2],
        hybrid_backbone=b16(embed_dim[0], activation=act, resolution=config.img_size),
        down_ops=[
            # ['Subsample', key_dim, num_heads, attn_ratio, mlp_ratio, stride]
            ["Subsample", D, embed_dim[0] // D, 4, 2, 2],
            ["Subsample", D, embed_dim[1] // D, 4, 2, 2],
        ],
        attention_activation=act,
        mlp_activation=act,
        distillation=config.distillation,
        drop_path=drop_path,
    )

    # Feature hook: capture pooled (pre-head) features.
    # Implementation detail: we register a forward hook on backbone.head's
    # BatchNorm (first module inside BN_Linear) input is AFTER pooling + BN,
    # but we want just BEFORE classification BN? To keep it simple,
    # we capture right after the global mean (with minimal intrusion) by
    # patching backbone.forward. Slight override wrapper below.

    original_forward = backbone.forward

    def patched_forward(x: torch.Tensor):
        # Copy of original logic with insertion of feature stash
        x_ = backbone.patch_embed(x)
        x_ = x_.flatten(2).transpose(1, 2)
        x_ = backbone.blocks(x_)
        x_ = x_.mean(1)

        # CRITICAL: Store features BEFORE head consumption to maintain gradient flow
        # Use weakref to avoid creating a module cycle (which caused recursion in .to())
        wr_ref = getattr(backbone, "_wrapper_ref", None)
        if wr_ref is not None:
            wrapper = wr_ref()
            if wrapper is not None:
                # Store x_ while it's still part of the computation graph
                setattr(wrapper, "_last_backbone_features", x_)

        # Now compute logits using the same x_ (gradients will flow back through both paths)
        if hasattr(backbone, "distillation") and backbone.distillation:
            logits = backbone.head(x_), backbone.head_dist(x_)
        else:
            logits = backbone.head(x_)
        return logits

    backbone.forward = patched_forward  # type: ignore

    # Build embedding head
    embed_head = EmbeddingHead(
        in_dim=embed_dim[-1],
        hidden_dim=config.hidden_dim,
        out_dim=config.embedding_out_dim,
        activation=config.activation,
        dropout=config.dropout,
        layernorm_first=config.use_layernorm_head,
        l2_normalize=True,
    )

    model = GlyphLeViT(
        backbone=backbone,
        embed_head=embed_head,
        num_classes=config.num_classes,
        replicate_gray=config.replicate_gray,
    )
    # Provide weakref back reference (avoid recursion through module graph)
    setattr(backbone, "_wrapper_ref", weakref.ref(model))

    return model


# ---------------------------------------------------------------------------
# Checkpoint Utilities
# ---------------------------------------------------------------------------


def save_checkpoint(
    model: GlyphLeViT,
    path: str,
    step: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
):
    """
    Save model weights (backbone + embedding head) and minimal metadata.
    """
    payload: Dict[str, Any] = {
        "backbone": model.backbone.state_dict(),
        "embed_head": model.embed_head.state_dict(),
        "num_classes": model.num_classes,
        "embedding_dim": model.embed_head.net[-1].out_features
        if isinstance(model.embed_head.net[-1], nn.Linear)
        else None,
        "step": step,
    }
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)


def load_checkpoint(model: GlyphLeViT, path: str, strict: bool = True):
    """
    Load model weights. Ignores mismatch in classification head if num_classes differs
    (when strict=False).
    """
    ckpt = torch.load(path, map_location="cpu")
    bb_state = ckpt.get("backbone", {})
    eh_state = ckpt.get("embed_head", {})

    # If classification dimension changed and strict=False, drop keys
    if not strict and model.num_classes > 0:
        head_w = "head.l.weight"
        head_b = "head.l.bias"
        for k in [head_w, head_b]:
            if k in bb_state:
                if bb_state[k].shape != model.backbone.state_dict()[k].shape:
                    bb_state.pop(k, None)

    model.backbone.load_state_dict(bb_state, strict=strict)
    model.embed_head.load_state_dict(eh_state, strict=strict)
    return ckpt


# ---------------------------------------------------------------------------
# Self-Test (Optional)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = GlyphLeViTConfig(num_classes=100, embedding_out_dim=128)
    m = build_glyph_levit_128s(cfg)
    x = torch.randn(4, 1, cfg.img_size, cfg.img_size)
    out = m(x)
    emb = out["embedding"]
    print("Embedding shape:", emb.shape, "norm mean:", emb.norm(dim=1).mean().item())
    if "logits" in out:
        print("Logits shape:", out["logits"].shape)
    # Save & load cycle
    save_checkpoint(m, "tmp_ckpt.pt", step=1)
    load_checkpoint(m, "tmp_ckpt.pt", strict=False)
    print("Checkpoint round-trip OK.")
