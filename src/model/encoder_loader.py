"""
DeepSVG Encoder Loader (Enhanced)

Purpose
-------
Utility functions and a lightweight wrapper to:
1. Locate & load a pretrained DeepSVG model (small variant by default).
2. Extract ONLY the encoder portion for embedding generation.
3. Provide a stable API to encode batches of SVG tensors into latent vectors.

This module is intentionally defensive:
- It degrades gracefully if DeepSVG or torch are not available at import time.
- Actual DeepSVG model loading is deferred until `load_encoder(...)` is called.

Phase 1 Scope
-------------
We only need forward-pass embeddings (no fine-tuning yet). Thus:
- The model parameters can be frozen (gradient disabled).
- We pick a single latent representation (e.g., final encoder hidden state pooled).
- We do not (yet) manage multi-scale hierarchical latents; a TODO note is left.

Recent Additions
----------------
- Checkpoint structure heuristic: inspects keys to auto-toggle `use_vae` vs. bottleneck.
- Filename and key-based heuristic for hierarchical (two-stage) vs one-stage config.
- Expanded debug logging summarizing configuration decisions before state load.
- More robust encoder-only forward path & tensor shape normalization.

Future Enhancements
-------------------
- Allow selection of intermediate layer outputs.
- Fine-tuning / projection heads (contrastive training).
- Embedding schema versioning & metadata injection.
- Optional half precision (fp16 / bf16) on capable hardware.

Usage Example
-------------
    from model.encoder_loader import load_encoder

    encoder = load_encoder(
        pretrained_root="deepsvg/pretrained",
        variant="deepsvg-small",
        device="cpu"
    )

    # svgtensor_batch: project-specific tensor object prepared elsewhere
    with torch.no_grad():
        emb = encoder.encode(svgtensor_batch)   # -> (batch, embed_dim)

Notes
-----
This file started as a scaffold; it now contains heuristics to better align
pretrained checkpoints whose configs are not explicitly serialized.

Author: Phase 1 scaffolding + heuristic enhancements.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Optional Torch import (lazy-checked later)
try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - handled gracefully
    torch = None
    nn = object  # type: ignore


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class EncoderLoaderError(RuntimeError):
    """Raised when the encoder cannot be properly initialized."""


# ---------------------------------------------------------------------------
# Configuration & Metadata
# ---------------------------------------------------------------------------

DEFAULT_VARIANT = "deepsvg-small"

# Hypothetical variant registry. Adjust values after inspecting actual configs.
# Keys can be expanded to include expected weight filename patterns.
VARIANT_REGISTRY: Dict[str, Dict[str, Any]] = {
    "deepsvg-small": {
        "expected_weight_substring": "small",
        "embedding_dim": 512,  # Placeholder; confirm with actual model
        "description": "Small DeepSVG variant (fast inference).",
    },
    "deepsvg-medium": {
        "expected_weight_substring": "medium",
        "embedding_dim": 768,
        "description": "Medium DeepSVG variant.",
    },
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class EncoderInfo:
    variant: str
    embedding_dim: int
    weight_path: str
    device: str
    frozen: bool
    model_class: str
    notes: str


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------


class DeepSVGEncoderWrapper:
    """
    Wraps a loaded DeepSVG model, exposing an `encode(...)` method that returns
    a latent embedding for each input element in a batch.

    Expected Model Attributes (to verify post-load):
        - A forward method that accepts a batch (SVGTensor-like).
        - Internal encoder stack producing per-token / per-command hidden states.

    For now we implement a generic strategy:
        1. Run model forward to obtain an output dict or tensor.
        2. If dict: attempt keys in order: ("latent", "z", "encoder_out", "final").
        3. If sequence hidden states produced: apply mean pooling across time (mask-aware TBD).
    """

    def __init__(
        self,
        model: Any,
        variant: str,
        weight_path: str,
        device: str,
        embedding_dim: int,
        frozen: bool = True,
        nan_guard: bool = True,
    ):
        self._model = model
        self._variant = variant
        self._weight_path = weight_path
        self._device = device
        self._embedding_dim = embedding_dim
        self._frozen = frozen
        self._nan_guard = nan_guard

        # Sanity freeze
        if frozen and hasattr(model, "parameters"):
            for p in model.parameters():
                p.requires_grad_(False)

        self._model.eval()

    # ------------------------------------------------------------------ #
    # Introspection / Metadata
    # ------------------------------------------------------------------ #
    @property
    def info(self) -> EncoderInfo:
        return EncoderInfo(
            variant=self._variant,
            embedding_dim=self._embedding_dim,
            weight_path=self._weight_path,
            device=self._device,
            frozen=self._frozen,
            model_class=type(self._model).__name__,
            notes="Phase1 wrapper (pooling strategy: mean over last hidden).",
        )

    # ------------------------------------------------------------------ #
    # Forward Encoding
    # ------------------------------------------------------------------ #
    def encode(self, batch: Any) -> Any:
        """
        Faithful hierarchical (or one-stage) encoder path.

        Required batch format (dict):
            commands_grouped : LongTensor (N, G, S)
            args_grouped     : LongTensor (N, G, S, n_args)

        Returns:
            Tensor (N, embedding_dim)
        """
        if torch is None:
            raise EncoderLoaderError("Torch not available; cannot run encode.")

        if not isinstance(batch, dict):
            raise EncoderLoaderError("encode expects batch dict with grouped tensors.")
        if "commands_grouped" not in batch or "args_grouped" not in batch:
            raise EncoderLoaderError(
                "Missing 'commands_grouped' or 'args_grouped' in batch."
            )

        cmds = batch["commands_grouped"]
        args = batch["args_grouped"]

        if not (isinstance(cmds, torch.Tensor) and isinstance(args, torch.Tensor)):
            raise EncoderLoaderError("Grouped tensors must be torch.Tensor instances.")
        if cmds.ndim != 3 or args.ndim != 4:
            raise EncoderLoaderError(
                f"Invalid ranks: commands {tuple(cmds.shape)} (expect 3D), args {tuple(args.shape)} (expect 4D)."
            )
        if args.shape[:3] != cmds.shape:
            raise EncoderLoaderError(
                f"Shape mismatch between commands {tuple(cmds.shape)} and args {tuple(args.shape)} (first 3 dims)."
            )

        N, G, S = cmds.shape

        # Permute to (S, G, N)
        cmds_seq_first = cmds.permute(2, 1, 0).contiguous()
        args_seq_first = args.permute(2, 1, 0, 3).contiguous()

        # Encoder forward (strict – no fallback)

        try:
            z = self._model.encoder(cmds_seq_first, args_seq_first, label=None)
        except Exception as e:
            raise EncoderLoaderError(f"Encoder forward failed: {e}") from e

        # Post-encoder modules
        if getattr(self._model.cfg, "use_resnet", False) and hasattr(
            self._model, "resnet"
        ):
            try:
                z = self._model.resnet(z)
            except Exception as e:
                raise EncoderLoaderError(f"ResNet forward failed: {e}") from e

        if getattr(self._model.cfg, "use_vae", False) and hasattr(self._model, "vae"):
            try:
                _, mu, _ = self._model.vae(z)  # deterministic mean
                z = mu
            except Exception as e:
                raise EncoderLoaderError(f"VAE forward failed: {e}") from e
        elif hasattr(self._model, "bottleneck"):
            try:
                z = self._model.bottleneck(z)
            except Exception as e:
                raise EncoderLoaderError(f"Bottleneck forward failed: {e}") from e

        if not isinstance(z, torch.Tensor):
            raise EncoderLoaderError(f"Encoder output is not a tensor (got {type(z)}).")

        # Simplified shape normalization:
        # Accept (1,N,D), (N,D), or (1,1,N,D) (edge-case: extra singleton group dimension).
        if z.ndim == 4 and z.shape[0] == 1 and z.shape[1] == 1:
            z = z.squeeze(0).squeeze(0)
        elif z.ndim == 3 and z.shape[0] == 1:
            z = z.squeeze(0)
        elif z.ndim != 2:
            raise EncoderLoaderError(
                f"Unexpected encoder output shape (expected (1,N,D), (N,D) or (1,1,N,D)): {tuple(z.shape)}"
            )

        if torch.isnan(z).any():
            if self._nan_guard:
                z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
                norms = z.norm(dim=1, keepdim=True).clamp_min(1e-8)
                z = z / norms
            else:
                raise EncoderLoaderError("NaNs in encoder output (nan_guard disabled).")

        if getattr(self, "_debug_stats", False):
            with torch.no_grad():
                norms = z.norm(dim=1)
                print(
                    "[ENCDBG] Encode stats: N=%d dim=%d row_norm_mean=%.4e min=%.4e max=%.4e"
                    % (
                        z.shape[0],
                        z.shape[1],
                        norms.mean().item(),
                        norms.min().item(),
                        norms.max().item(),
                    )
                )

        return z


# ---------------------------------------------------------------------------
# Loader Utilities
# ---------------------------------------------------------------------------


def _import_deepsvg_model() -> Tuple[Any, Any]:
    """
    Dynamically import DeepSVG model & config modules.

    Returns
    -------
    (model_module, config_module)

    Raises
    ------
    EncoderLoaderError if imports fail.
    """
    try:
        # Typical structure after adding project root to PYTHONPATH
        import importlib

        model_mod = importlib.import_module("deepsvg.model.model")
        config_mod = importlib.import_module("deepsvg.model.config")
        return model_mod, config_mod
    except Exception as e:  # pragma: no cover
        raise EncoderLoaderError(
            "Failed to import DeepSVG modules. Ensure 'deepsvg' "
            "repository is on PYTHONPATH."
        ) from e


def _find_pretrained_weight_file(
    pretrained_root: str,
    variant: str,
    expected_substring: str,
) -> Optional[str]:
    """
    Scan a directory for a weight file matching a substring heuristic.

    Parameters
    ----------
    pretrained_root : str
        Directory to search.
    variant : str
        Variant name (for logging only).
    expected_substring : str
        A substring expected to appear in the weight filename.

    Returns
    -------
    str | None
        Path to weight file or None if not found.
    """
    root = Path(pretrained_root)
    if not root.exists():
        return None

    candidates = []
    for p in root.glob("**/*"):
        if p.is_file():
            name = p.name.lower()
            if expected_substring in name and name.endswith((".pt", ".pth", ".bin")):
                candidates.append(p)

    # Heuristic: pick the shortest path (or first) if multiple
    if not candidates:
        return None
    candidates.sort(key=lambda x: len(str(x)))
    return str(candidates[0])


def _load_state_dict_safely(model: Any, weight_path: str, strict: bool = False) -> None:
    """
    Attempt to load a state dict with optional relaxed strictness.

    Parameters
    ----------
    model : nn.Module
    weight_path : str
    strict : bool
        If False, will ignore non-matching keys with a warning.

    Raises
    ------
    EncoderLoaderError on fatal failure.
    """
    if torch is None:
        raise EncoderLoaderError("Torch not available to load state dict.")

    if not os.path.isfile(weight_path):
        raise EncoderLoaderError(f"Weight file not found: {weight_path}")

    try:
        state = torch.load(weight_path, map_location="cpu")

        # Unwrap common checkpoint nesting patterns
        if isinstance(state, dict):
            if "model" in state and isinstance(state["model"], dict):
                inner = state["model"]
                # Some checkpoints: {"model": {"state_dict": {...}}}
                if "state_dict" in inner and isinstance(inner["state_dict"], dict):
                    state = inner["state_dict"]
                else:
                    state = inner
            elif "state_dict" in state and isinstance(state["state_dict"], dict):
                state = state["state_dict"]

        if not isinstance(state, dict):
            raise EncoderLoaderError("Loaded checkpoint is not a dict-like state dict.")

        # Optionally strip leading prefixes (e.g., 'module.' / 'model.')
        def _strip_prefix(k: str) -> str:
            for pref in ("module.", "model."):
                if k.startswith(pref):
                    return k[len(pref) :]
            return k

        processed_state = {}
        for k, v in state.items():
            if not isinstance(k, str):
                continue
            processed_state[_strip_prefix(k)] = v
        state = processed_state

        model_keys = set(model.state_dict().keys())
        loaded_keys = set(state.keys())

        # Shape-mismatch filtering (only when strict=False)
        if not strict:
            model_state_full = model.state_dict()
            to_drop = []
            for k, v in list(state.items()):
                if k in model_state_full:
                    mv = model_state_full[k]
                    if (
                        hasattr(v, "shape")
                        and hasattr(mv, "shape")
                        and v.shape != mv.shape
                    ):
                        to_drop.append((k, v.shape, mv.shape))
                        state.pop(k)
            if to_drop:
                print(
                    "[EncoderLoader] Dropping %d mismatched shape keys (strict=False). First 5: %s"
                    % (
                        len(to_drop),
                        [f"{k}:{src}->{tgt}" for k, src, tgt in to_drop[:5]],
                    )
                )

        # Load with provided strict flag (default False) (after filtering)
        missing, unexpected = model.load_state_dict(state, strict=strict)

        matched = len(model_keys) - len(missing)
        match_ratio = matched / max(1, len(model_keys))

        summary = (
            f"[EncoderLoader] State dict load summary: "
            f"matched={matched}/{len(model_keys)} ({match_ratio:.2%}), "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )
        print(summary)

        # Show a few sample keys for quick inspection
        if missing:
            print(f"[EncoderLoader] Sample missing keys (first 5): {missing[:5]}")
        if unexpected:
            print(f"[EncoderLoader] Sample unexpected keys (first 5): {unexpected[:5]}")

        # Helpful debug: list unmatched loaded keys (rare case)
        unmatched_loaded = sorted(list(loaded_keys - model_keys))
        if unmatched_loaded:
            print(
                f"[EncoderLoader] Loaded keys with no target match (first 5): {unmatched_loaded[:5]}"
            )

        if strict and (missing or unexpected):
            raise EncoderLoaderError(
                f"Strict load failed: {len(missing)} missing, {len(unexpected)} unexpected keys."
            )

    except Exception as e:
        raise EncoderLoaderError(
            f"Failed to load state dict from {weight_path}: {e}"
        ) from e


# ---------------------------------------------------------------------------
# Public Loader
# ---------------------------------------------------------------------------


def load_encoder(
    pretrained_root: str = "deepsvg/pretrained",
    variant: str = DEFAULT_VARIANT,
    device: str = "cpu",
    freeze: bool = True,
    strict_state: bool = False,
    custom_weight_path: Optional[str] = None,
    auto_config: bool = True,
    verbose: bool = True,
    force_one_stage: bool = False,
    nan_guard: bool = True,
) -> DeepSVGEncoderWrapper:
    """
    Load a DeepSVG encoder wrapper.

    Parameters
    ----------
    pretrained_root : str
        Directory containing pretrained weights (heuristic search).
    variant : str
        Variant key from VARIANT_REGISTRY.
    device : str
        Torch device (e.g., 'cpu', 'cuda:0').
    freeze : bool
        Whether to disable gradients.
    strict_state : bool
        Pass strict=True to model.load_state_dict.
    custom_weight_path : str | None
        Explicit path to weights (overrides heuristic search).
    force_one_stage : bool
        If True, override heuristic and use OneStageOneShot config (encode_stages=1).
    nan_guard : bool
        If True, replace any NaNs in final embeddings with zero + re-normalize.
    """
    if torch is None:
        raise EncoderLoaderError("Torch not available; install PyTorch first.")

    if variant not in VARIANT_REGISTRY:
        raise EncoderLoaderError(
            f"Unknown variant '{variant}'. Available: {list(VARIANT_REGISTRY)}"
        )

    registry_entry = VARIANT_REGISTRY[variant]
    expected_substring = registry_entry["expected_weight_substring"]
    embedding_dim = registry_entry["embedding_dim"]

    # Resolve weight path
    weight_path = custom_weight_path or _find_pretrained_weight_file(
        pretrained_root=pretrained_root,
        variant=variant,
        expected_substring=expected_substring,
    )
    if weight_path is None:
        raise EncoderLoaderError(
            f"Could not locate weights for variant '{variant}' under '{pretrained_root}'. "
            "Provide --custom-weight-path or download the pretrained files."
        )

    model_module, config_module = _import_deepsvg_model()

    # Pre-scan checkpoint (without mutating) to detect structural hints if auto_config
    detected = {
        "has_vae_keys": False,
        "has_bottleneck_keys": False,
        "hierarchical_in_name": False,
        "matched_encode_stages": None,
    }
    if auto_config:
        try:
            raw_state = torch.load(weight_path, map_location="cpu")
            # Shallow unwrap to access keys
            if isinstance(raw_state, dict):
                if "model" in raw_state and isinstance(raw_state["model"], dict):
                    probe = raw_state["model"]
                    if "state_dict" in probe and isinstance(probe["state_dict"], dict):
                        probe = probe["state_dict"]
                elif "state_dict" in raw_state and isinstance(
                    raw_state["state_dict"], dict
                ):
                    probe = raw_state["state_dict"]
                else:
                    probe = raw_state
                if isinstance(probe, dict):
                    klist = list(probe.keys())
                    for k in klist:
                        if (
                            k.startswith("vae.")
                            or ".enc_mu_fcn" in k
                            or ".enc_sigma_fcn" in k
                        ):
                            detected["has_vae_keys"] = True
                        if k.startswith("bottleneck.") or ".bottleneck." in k:
                            detected["has_bottleneck_keys"] = True
        except Exception:
            pass
        detected["hierarchical_in_name"] = (
            "hierarchical" in os.path.basename(weight_path).lower()
        )

    # Instantiate configuration (choose appropriate concrete config) with heuristic
    try:
        weight_lower = os.path.basename(weight_path).lower()
        if force_one_stage:
            ConfigClass = getattr(config_module, "OneStageOneShot", None)
        elif detected["hierarchical_in_name"]:
            ConfigClass = getattr(config_module, "Hierarchical", None)
        else:
            ConfigClass = getattr(config_module, "OneStageOneShot", None)
        if ConfigClass is None:
            ConfigClass = getattr(config_module, "_DefaultConfig")
        model_cfg = ConfigClass()
        # Heuristic toggles
        if auto_config:
            if detected["has_bottleneck_keys"] and not detected["has_vae_keys"]:
                # Use bottleneck, disable VAE
                if getattr(model_cfg, "use_vae", None) is not None:
                    model_cfg.use_vae = False
            elif detected["has_vae_keys"]:
                if getattr(model_cfg, "use_vae", None) is not None:
                    model_cfg.use_vae = True
            # If hierarchical name absent but we somehow have stage-2 style keys (future heuristic), could adjust encode_stages/decode_stages here.
        if verbose:
            print(
                "[EncoderLoader] Config heuristic: hierarchical=%s has_vae_keys=%s has_bottleneck_keys=%s -> use_vae=%s encode_stages=%s decode_stages=%s"
                % (
                    detected["hierarchical_in_name"],
                    detected["has_vae_keys"],
                    detected["has_bottleneck_keys"],
                    getattr(model_cfg, "use_vae", "?"),
                    getattr(model_cfg, "encode_stages", "?"),
                    getattr(model_cfg, "decode_stages", "?"),
                )
            )
    except Exception as e:
        raise EncoderLoaderError(f"Failed to build model config: {e}") from e

    # Build model object
    try:
        # Locate actual DeepSVG model class (prefer SVGTransformer, fallback to Model)
        ModelClass = getattr(model_module, "SVGTransformer", None) or getattr(
            model_module, "Model"
        )
    except AttributeError as e:
        raise EncoderLoaderError(
            "DeepSVG model class not found (tried: 'SVGTransformer', 'Model')."
        ) from e

    try:
        model = ModelClass(model_cfg)
    except Exception as e:
        raise EncoderLoaderError(
            f"Failed to instantiate model with provided config: {e}"
        ) from e

    _load_state_dict_safely(model, weight_path, strict=strict_state)

    model.to(device)

    # Override embedding_dim with model latent dimension if available
    embedding_dim = getattr(model_cfg, "dim_z", embedding_dim)

    if verbose:
        print(
            "[EncoderLoader] Final encoder wrapper: variant=%s dim_z=%s d_model=%s use_vae=%s use_resnet=%s"
            % (
                variant,
                getattr(model_cfg, "dim_z", "?"),
                getattr(model_cfg, "d_model", "?"),
                getattr(model_cfg, "use_vae", "?"),
                getattr(model_cfg, "use_resnet", "?"),
            )
        )
    wrapper = DeepSVGEncoderWrapper(
        model=model,
        variant=variant,
        weight_path=weight_path,
        device=device,
        embedding_dim=embedding_dim,
        frozen=freeze,
        nan_guard=nan_guard,
    )
    return wrapper


# ---------------------------------------------------------------------------
# CLI / Debug
# ---------------------------------------------------------------------------


def _cli() -> int:
    """
    Minimal CLI for smoke-testing the loader.
    """
    import argparse

    parser = argparse.ArgumentParser(description="DeepSVG encoder loader test.")
    parser.add_argument("--pretrained-root", default="deepsvg/pretrained")
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--weight", default=None, help="Explicit weight file path.")
    parser.add_argument(
        "--unfrozen", action="store_true", help="Do not freeze parameters."
    )
    args = parser.parse_args()

    try:
        enc = load_encoder(
            pretrained_root=args.pretrained_root,
            variant=args.variant,
            device=args.device,
            freeze=not args.unfrozen,
            custom_weight_path=args.weight,
        )
    except Exception as e:
        print(f"[ERROR] Failed to load encoder: {e}", file=sys.stderr)
        traceback.print_exc()
        return 2

    info = enc.info
    print("[INFO] Encoder loaded:")
    print(json.dumps(info.__dict__, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_cli())
