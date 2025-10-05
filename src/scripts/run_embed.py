#!/usr/bin/env python3
from __future__ import annotations

# --- Dynamic path injection for local deepsvg repository ---
# Ensures 'deepsvg' can be imported when running this script directly without installing as a package.
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[2]  # .../glyph-rec-2
_deepsvg_dir = _project_root / "deepsvg"
if _deepsvg_dir.exists() and str(_deepsvg_dir) not in sys.path:
    sys.path.insert(0, str(_deepsvg_dir))
# ----------------------------------------------------------------
"""
run_embed.py

Phase 1: Extract glyph embeddings using a pretrained DeepSVG encoder (one-stage)
with size-preserving (norm_v2) normalization and the simplified SVGTensor
builder.

Pipeline (per glyph):
1. Load raw contour JSON from SQLite (glyphs table).
2. Parse + quadratic qCurveTo midpoint expansion (parse_contours).
3. Apply norm_v2 normalization (EM-scale, center, single y-flip, no per-glyph scale).
4. Convert normalized contours to grouped command & argument tensors via
   simplified SVGTensorBuilder (m / l / c / z subset, cubic args encoded).
5. Collate a batch, feed through encoder to obtain latent embedding z.
6. L2-normalize embeddings (optional flag) and store:
   - In-memory list
   - (Optional future) Memory-mapped file / metadata JSON.

This script focuses on correctness and traceable metadata rather than maximal
throughput. It is CPU-friendly (no GPU assumptions on 2019 MacBook Pro).

Note: The mapping from our simplified command/argument space to the original
DeepSVG training distribution is approximate; embeddings are still valuable for
relative similarity / nearest-neighbor experiments.

Usage Examples
--------------
Extract 500 glyph embeddings from random fonts:

    python -m src.scripts.run_embed \
        --db dataset/glyphs.db \
        --limit 500 \
        --batch-size 32 \
        --pretrained deepsvg/pretrained/hierarchical_ordered.pth.tar \
        --out artifacts/embeddings/embeds.pt \
        --meta artifacts/embeddings/meta.json

Limit to specific font hashes (comma separated):

    python -m src.scripts.run_embed \
        --db dataset/glyphs.db \
        --font-hashes abcd1234...,efgh5678... \
        --limit 300

Export a memmap (future placeholder):

    python -m src.scripts.run_embed \
        --memmap-dir artifacts/embeddings_memmap

Outputs
-------
- Embedding tensor saved (if --out specified): shape (N, D)
- Metadata JSON (if --meta specified) with glyph_id, font_hash, label, width_em, height_em, normalization_version.
- Console timing + stats.

Dependencies
------------
Relies on existing project modules:
- src/data/contour_parser.py
- src/data/normalization.py
- src/model/encoder_loader.py
- src/model/svgtensor_builder.py

Author: Phase 1
"""


import argparse
import json
import math
import os
import sqlite3
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import torch

# Local imports (assumes project root added to PYTHONPATH)
try:
    from src.data.contour_parser import parse_contours
    from src.data.normalization import (
        NormalizationConfig,
        Strategy,
        apply_normalization,
        default_embedding_config,
        NormalizationConfig as NormCfg,
        ContourCommandLike,
    )
    from src.model.encoder_loader import load_encoder
    from src.model.svgtensor_builder import (
        build_default_builder_from_cfg,
        SVGTensorBuilder,
    )
except ImportError as e:
    print(
        "[ERROR] Failed to import project modules. Ensure PYTHONPATH includes project root.",
        file=sys.stderr,
    )
    raise


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass
class GlyphRow:
    glyph_id: int
    font_hash: str
    label: str
    contours: str
    upem: Optional[int]
    # Note: keep joining_group after all non-default fields to satisfy dataclass ordering
    joining_group: Optional[str] = None


@dataclass
class GlyphMeta:
    glyph_id: int
    font_hash: str
    label: str
    width_em: float
    height_em: float
    normalization_version: str
    upem: Optional[int]
    group_count: int
    tokens_non_eos: int
    embedding_index: int
    # Note: joining_group placed after embedding_index with a default to avoid default-before-non-default error
    joining_group: Optional[str] = None


# ---------------------------------------------------------------------------
# SQLite Helpers
# ---------------------------------------------------------------------------


def connect_readonly(db_path: str) -> sqlite3.Connection:
    p = Path(db_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Database not found: {p}")
    try:
        return sqlite3.connect(f"file:{p}?mode=ro", uri=True)
    except sqlite3.OperationalError:
        return sqlite3.connect(str(p))


def fetch_glyph_rows(
    conn: sqlite3.Connection,
    limit: int,
    font_hashes: Optional[List[str]] = None,
    randomize: bool = True,
) -> List[GlyphRow]:
    where = "WHERE g.contours IS NOT NULL AND length(g.contours)>0"
    params: List[Any] = []
    if font_hashes:
        placeholders = ",".join("?" for _ in font_hashes)
        where += f" AND g.f_id IN ({placeholders})"
        params.extend(font_hashes)
    order_clause = "ORDER BY RANDOM()" if randomize else ""
    sql = f"""
        SELECT g.id, g.f_id, g.label, g.contours, f.upem, g.joining_group
        FROM glyphs g
        JOIN fonts f ON f.file_hash = g.f_id
        {where}
        {order_clause}
        LIMIT ?
    """
    params.append(limit)
    cur = conn.execute(sql, params)
    rows = [
        GlyphRow(
            glyph_id=r[0],
            font_hash=r[1],
            label=r[2],
            contours=r[3],
            upem=r[4],
            joining_group=r[5],
        )
        for r in cur.fetchall()
    ]
    return rows


# ---------------------------------------------------------------------------
# Embedding Extraction
# ---------------------------------------------------------------------------


class EmbedExtractor:
    def __init__(
        self,
        encoder,
        builder: SVGTensorBuilder,
        norm_cfg: NormalizationConfig,
        device: str = "cpu",
        l2_normalize: bool = True,
        qcurve_mode: str = "midpoint",
    ):
        self.encoder = encoder
        # Normalize hierarchical builder dict -> object
        if isinstance(builder, dict) and builder.get("mode") == "hier":
            hier_builder_obj = builder["builder"]
            hier_cfg = builder["cfg"]
            # Ensure expected n_args attribute (hier config omits it; model uses 11)
            if not hasattr(hier_cfg, "n_args"):
                setattr(hier_cfg, "n_args", 11)
            # Attach helper attributes for downstream process_batch logic
            hier_builder_obj._is_hier = True
            hier_builder_obj._packer = builder.get("packer", None)
            self.builder = hier_builder_obj
        else:
            self.builder = builder
        self.norm_cfg = norm_cfg
        self.device = device
        self.l2_normalize = l2_normalize
        self.qcurve_mode = qcurve_mode

        self.encoder_info = encoder.info
        self.embedding_dim = self.encoder_info.embedding_dim

    def process_batch(
        self, glyph_rows: List[GlyphRow]
    ) -> Tuple[torch.Tensor, List[GlyphMeta]]:
        glyph_tensors = []
        metas: List[GlyphMeta] = []
        skipped = 0
        skipped_ids: List[int] = []

        for gr in glyph_rows:
            try:
                q_stats = {}
                parsed = parse_contours(
                    gr.contours, qcurve_mode=self.qcurve_mode, qcurve_stats=q_stats
                )
                # Convert parsed ContourCommand dataclasses to ContourCommandLike objects
                # required by normalization (they provide replace_points()).
                parsed_like = [
                    [ContourCommandLike(cmd.cmd, cmd.points) for cmd in sub]
                    for sub in parsed
                ]

                # Apply normalization (norm_v2)
                norm_contours, meta_norm = apply_normalization(
                    parsed_like, self.norm_cfg, gr.upem
                )

                # Convert to builder-friendly structure (reuse wrappers)
                # Phase B: stats-aware builder usage with empty-glyph skip
                try:
                    if getattr(self.builder, "_is_hier", False):
                        # Hierarchical path: build per-subpath SVGTensors then pack
                        svgt_list, hier_stats = self.builder.build_glyph(norm_contours)
                        if not svgt_list:
                            raise ValueError("Hierarchical builder produced no groups.")
                        packer = getattr(self.builder, "_packer", None)
                        if packer is None:
                            raise ValueError(
                                "Hierarchical builder missing packer function."
                            )
                        commands_g, args_g, pack_stats = packer(
                            svgt_list,
                            self.builder.cfg.max_num_groups,
                            self.builder.cfg.max_seq_len,
                            getattr(self.builder.cfg, "n_args", 11),
                        )
                        build_stats = {
                            "hier_stats": hier_stats.to_dict(),
                            "pack_stats": pack_stats,
                        }
                    elif hasattr(self.builder, "glyph_to_group_tensors_with_stats"):
                        # Legacy / stats-aware builder
                        commands_g, args_g, build_stats = (
                            self.builder.glyph_to_group_tensors_with_stats(
                                norm_contours
                            )
                        )
                        # Expect dict-like stats
                        if (
                            isinstance(build_stats, dict)
                            and build_stats.get("commands_encoded", 0) == 0
                        ):
                            raise ValueError(
                                "No encodable commands (all-EOS) after builder."
                            )
                    else:
                        # Faithful preprocessor path (returns a GlyphBuildStats dataclass)
                        commands_g, args_g, build_stats_obj = (
                            self.builder.glyph_to_group_tensors(norm_contours)
                        )
                        if getattr(build_stats_obj, "tokens_encoded", 0) == 0:
                            raise ValueError(
                                "No encodable commands (all-EOS) after faithful builder."
                            )
                        # Keep the stats object (do NOT convert to dict so collate() can aggregate)
                        build_stats = build_stats_obj
                except Exception:
                    skipped += 1
                    skipped_ids.append(gr.glyph_id)
                    continue

                # Append triplet; faithful path supplies GlyphBuildStats instance, legacy supplies dict
                glyph_tensors.append((commands_g, args_g, build_stats))
                metas.append(
                    GlyphMeta(
                        glyph_id=gr.glyph_id,
                        font_hash=gr.font_hash,
                        label=gr.label,
                        width_em=meta_norm["width_em"],
                        height_em=meta_norm["height_em"],
                        normalization_version=meta_norm["normalization_version"],
                        upem=gr.upem,
                        group_count=-1,
                        tokens_non_eos=-1,
                        embedding_index=-1,  # fill later
                        joining_group=gr.joining_group,
                    )
                )
            except Exception as e:
                skipped += 1
                skipped_ids.append(gr.glyph_id)
                continue

        if not glyph_tensors:
            raise RuntimeError(
                f"All glyphs failed parsing/normalization; skipped={skipped} ids={skipped_ids[:10]}{'...' if len(skipped_ids) > 10 else ''}"
            )

        if skipped > 0:
            print(
                f"[DEBUG] Skipped {skipped} glyphs in batch (first 10 ids: {skipped_ids[:10]}{'...' if len(skipped_ids) > 10 else ''})"
            )
        if getattr(self.builder, "_is_hier", False):
            # Bucket glyphs by actual used group count (G_used) to avoid padding
            eos_idx = 4  # DeepSVG.COMMANDS_SIMPLIFIED.index("EOS")
            per_glyph_used = []
            for cmds_g, args_g, _stats in glyph_tensors:
                non_empty_mask = (cmds_g != eos_idx).any(dim=1)
                used_idx = torch.nonzero(non_empty_mask, as_tuple=False).view(-1)
                if used_idx.numel() == 0:
                    # Safeguard: retain first row (should not happen with a valid glyph)
                    used_idx = torch.tensor([0])
                per_glyph_used.append(
                    (
                        cmds_g[used_idx],  # (G_used, S)
                        args_g[used_idx],  # (G_used, S, n_args)
                        used_idx.shape[0],
                    )
                )

            # Build buckets: group_count -> list of (cmds_used, args_used, original_index)
            buckets: Dict[int, List[Tuple[torch.Tensor, torch.Tensor, int]]] = {}
            for idx, (c_used, a_used, g_used) in enumerate(per_glyph_used):
                buckets.setdefault(g_used, []).append((c_used, a_used, idx))

            # Encode each bucket separately (no empty columns inside a bucket)
            bucket_embeddings: Dict[int, torch.Tensor] = {}
            embedding_dim = self.encoder_info.embedding_dim
            N_total = len(per_glyph_used)
            z_out = torch.zeros(N_total, embedding_dim, dtype=torch.float32)

            for g_used, items in buckets.items():
                # Stack without padding along group axis (uniform g_used within bucket)
                cmds_stack = torch.stack([it[0] for it in items], dim=0).to(
                    self.device
                )  # (Nb, g_used, S)
                args_stack = torch.stack([it[1] for it in items], dim=0).to(
                    self.device
                )  # (Nb, g_used, S, n_args)

                # Debug stats per bucket
                try:
                    non_eos_tokens = int((cmds_stack != eos_idx).sum().item())
                    total_slots = cmds_stack.numel()
                    util = non_eos_tokens / total_slots if total_slots else 0.0
                    if getattr(self, "l2_normalize", False):
                        sample_cmds = cmds_stack[
                            0, 0, : min(12, cmds_stack.shape[2])
                        ].tolist()
                        print(
                            "[HIER-DBG] bucket g_used=%d size=%d cmds_shape=%s util=%.4f sample_g0=%s"
                            % (
                                g_used,
                                len(items),
                                tuple(cmds_stack.shape),
                                util,
                                sample_cmds,
                            )
                        )
                except Exception:
                    pass

                enc_batch = {
                    "commands_grouped": cmds_stack,
                    "args_grouped": args_stack,
                }
                with torch.no_grad():
                    z_bucket = self.encoder.encode(enc_batch)  # (Nb, D)
                if z_bucket.ndim == 3 and z_bucket.shape[0] == 1:
                    z_bucket = z_bucket.squeeze(0)

                # NaN guard (already applied inside encode if enabled, but double-check)
                if torch.isnan(z_bucket).any():
                    z_bucket = torch.nan_to_num(
                        z_bucket, nan=0.0, posinf=0.0, neginf=0.0
                    )

                # Assign into global output respecting original glyph order
                for (c_used, a_used, orig_idx), row in zip(items, z_bucket):
                    z_out[orig_idx] = row

            # Optional L2 normalization
            if self.l2_normalize:
                with torch.no_grad():
                    norms = z_out.norm(dim=1, keepdim=True).clamp_min(1e-12)
                    z_out = z_out / norms
                    mean_norm = norms.mean().item()
                    zero_rows = int((norms.squeeze(1) == 0).sum().item())
                    print(
                        "[HIER-DBG] post-bucket norms mean=%.4e zero_rows=%d/%d"
                        % (mean_norm, zero_rows, z_out.shape[0])
                    )

            # Set embedding indices and populate group/token stats in metas
            for i, m in enumerate(metas):
                m.embedding_index = i
                try:
                    # per_glyph_used[i] = (commands_used, args_used, g_used)
                    cmds_used, _args_used, g_used = per_glyph_used[i]
                    m.group_count = g_used
                    m.tokens_non_eos = int((cmds_used != eos_idx).sum().item())
                except Exception:
                    # Fallback if any unexpected shape/lookup issue
                    pass

            return z_out.to(self.device), metas
        elif hasattr(self.builder, "collate"):
            batch = self.builder.collate(
                glyph_tensors, device=self.device, aggregate_stats=True
            )
        else:
            batch = self.builder.collate_glyph_tensors(
                glyph_tensors, device=self.device, aggregate_stats=True
            )
        # Optional builder diagnostics (token utilization / padding efficiency)
        if "stats" in batch and isinstance(batch["stats"], dict):
            util_mean = batch["stats"].get("fraction_utilization", {}).get("mean")
            tokens_mean = batch["stats"].get("tokens_non_eos", {}).get("mean")
            if util_mean is not None:
                print(
                    f"[DEBUG] Builder batch stats: tokens_non_eos.mean={tokens_mean} utilization.mean={util_mean:.4f}"
                )
        # Strip stats before encoding
        enc_batch = {
            "commands_grouped": batch["commands_grouped"],
            "args_grouped": batch["args_grouped"],
        }

        # Additional pre-encode diagnostics for hierarchical path
        if getattr(self.builder, "_is_hier", False):
            try:
                cg = enc_batch["commands_grouped"]
                ag = enc_batch["args_grouped"]
                eos_idx = 4
                non_eos = cg != eos_idx
                print(
                    "[HIER-DBG] pre-encode: cmds_non_eos=%d/%d (%.4f) arg_pad_frac=%.4f"
                    % (
                        int(non_eos.sum().item()),
                        cg.numel(),
                        float(non_eos.float().mean().item()),
                        float((ag == -1).float().mean().item()),
                    )
                )
                # Show first glyph / first group raw command indices
                fg = cg[0, 0, : min(16, cg.shape[2])].tolist()
                print("[HIER-DBG] first glyph first group cmds (truncated):", fg)
            except Exception as _e:
                print(f"[HIER-DBG] (warning) pre-encode stats failed: {_e}")

        with torch.no_grad():
            z = self.encoder.encode(enc_batch)  # (1, N, d) or (N, d)
        if z.ndim == 3:
            # Expect shape (1, N, D)
            z = z.squeeze(0)
        # Debug: pre-L2 norms (B)
        if self.l2_normalize:
            try:
                norms_before = z.norm(dim=1)
                zero_rows = (norms_before == 0).sum().item()
                print(
                    "[DEBUG] Pre-L2 embedding norms: mean={:.4e} min={:.4e} max={:.4e} zero_rows={}/{}".format(
                        norms_before.mean().item(),
                        norms_before.min().item(),
                        norms_before.max().item(),
                        zero_rows,
                        norms_before.numel(),
                    )
                )
            except Exception:
                pass
            z = torch.nn.functional.normalize(z, dim=1)

        for i, m in enumerate(metas):
            m.embedding_index = i

        return z, metas


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_embeddings(embeds: torch.Tensor, out_path: Optional[str]):
    if out_path:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(embeds, p)


def save_metadata(metas: List[GlyphMeta], meta_path: Optional[str]):
    if meta_path:
        p = Path(meta_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for m in metas:
                f.write(json.dumps(asdict(m), ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: List[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Extract DeepSVG encoder embeddings for glyph contours (norm_v2 size-preserving)."
    )
    ap.add_argument("--db", required=True, help="Path to glyphs.db SQLite")
    ap.add_argument(
        "--pretrained", required=True, help="Path to pretrained .pth/.pt/.tar weights"
    )
    ap.add_argument(
        "--faithful",
        action="store_true",
        help="Use repo-faithful DeepSVG preprocessing pipeline (commands/args with proper PAD/EOS and arg shift semantics).",
    )
    ap.add_argument(
        "--faithful-hier",
        action="store_true",
        help="Use hierarchical faithful DeepSVG pipeline (encode_stages=2) with SVGTensor-based grouping.",
    )
    ap.add_argument("--limit", type=int, default=500, help="Number of glyphs to sample")
    ap.add_argument(
        "--font-hashes",
        type=str,
        default=None,
        help="Comma-separated font file_hash values to restrict sampling",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for embedding pass (logical grouping after parse)",
    )
    ap.add_argument(
        "--no-random",
        action="store_true",
        help="Disable random sampling (take first rows)",
    )
    ap.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda:0)")
    ap.add_argument("--out", type=str, default=None, help="Output tensor file (.pt)")
    ap.add_argument("--meta", type=str, default=None, help="Metadata JSONL file path")
    ap.add_argument(
        "--no-l2", action="store_true", help="Disable L2 normalization of embeddings"
    )
    ap.add_argument(
        "--qcurve-mode",
        choices=("midpoint", "naive"),
        default="midpoint",
        help="Quadratic expansion strategy",
    )
    ap.add_argument(
        "--strategy",
        choices=("norm_v1", "norm_v2"),
        default="norm_v2",
        help="Normalization strategy version",
    )
    ap.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Status log frequency (glyph count)",
    )
    # --- New debug / instrumentation flags ---
    ap.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug logging (shapes, sample metas, timing).",
    )
    ap.add_argument(
        "--debug-first",
        type=int,
        default=5,
        help="When --debug is set, number of first glyph metas to print.",
    )
    ap.add_argument(
        "--dump-failed",
        action="store_true",
        help="When --debug set, log IDs of glyphs that failed parsing/normalization.",
    )
    ap.add_argument(
        "--print-batch-shapes",
        action="store_true",
        help="Log per-batch tensor shapes before encoder call.",
    )
    ap.add_argument(
        "--limit-encoder-batch",
        type=int,
        default=None,
        help="Debug: hard cap number of glyphs per batch actually sent to encoder.",
    )
    ap.add_argument(
        "--max-parse-errors",
        type=int,
        default=None,
        help="Abort if more than this number of glyphs are skipped (debug safeguard).",
    )
    ap.add_argument(
        "--force-one-stage",
        action="store_true",
        help="Force one-stage encoder config (override hierarchical auto-detection).",
    )
    ap.add_argument(
        "--no-nan-guard",
        action="store_true",
        help="Disable NaN guard (debug; keep raw NaNs in embeddings).",
    )
    return ap.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    font_hashes = args.font_hashes.split(",") if args.font_hashes else None

    print(
        f"[INFO] Loading glyph rows (limit={args.limit}, random={not args.no_random})"
    )
    conn = connect_readonly(args.db)
    rows = fetch_glyph_rows(
        conn, limit=args.limit, font_hashes=font_hashes, randomize=not args.no_random
    )
    if not rows:
        print("[WARN] No glyph rows fetched.")
        return 0

    # Load model / config
    print("[INFO] Loading pretrained encoder...")
    encoder = load_encoder(
        pretrained_root=str(Path(args.pretrained).parent),
        variant="deepsvg-small",  # variant registry placeholder; adjust if needed
        device=args.device,
        freeze=True,
        custom_weight_path=args.pretrained,
        auto_config=not args.force_one_stage,
        verbose=True if args.debug else False,
        force_one_stage=args.force_one_stage,
        nan_guard=not args.no_nan_guard,
    )
    if args.debug:
        setattr(encoder, "_debug_stats", True)
    cfg_model = encoder._model.cfg if hasattr(encoder, "_model") else None
    # ------------------------------------------------------------------
    # Builder selection precedence:
    # 1. --faithful-hier (hierarchical faithful pipeline)
    # 2. --faithful (one-stage faithful simplified path already present)
    # 3. Legacy simplified builder
    # ------------------------------------------------------------------
    if args.debug:
        print(
            "[DEBUG] Builder selection: faithful_hier=%s faithful=%s"
            % (getattr(args, "faithful_hier", False), getattr(args, "faithful", False))
        )
    builder = None

    # ------------------------------------------------------------------
    # Auto-switch: if the loaded model is hierarchical (encode_stages=2)
    # and user requested flat faithful (--faithful) but not --faithful-hier,
    # promote to hierarchical faithful to match repo behavior.
    # ------------------------------------------------------------------
    if (
        cfg_model is not None
        and getattr(cfg_model, "encode_stages", 2) == 2
        and getattr(args, "faithful", False)
        and not getattr(args, "faithful_hier", False)
    ):
        print(
            "[INFO] Hierarchical model detected (encode_stages=2). Flat faithful mode is DEPRECATED under hierarchical checkpoints and will be ignored. Switching to hierarchical faithful (--faithful-hier). Use --force-one-stage ONLY if you explicitly need a legacy single-stage path."
        )
        # Deprecate flat faithful in hierarchical context
        args.faithful_hier = True
        args.faithful = False
        # Ensure any later conditional blocks see updated flags

    if getattr(args, "faithful_hier", False) and cfg_model is not None:
        try:
            from src.model.faithful.hier.svgtensor_builder import (
                HierSVGTensorBuilder,
                HierBuilderConfig,
            )
            from src.model.faithful.hier.packer import (
                pack_groups,
                batch_collate_grouped,
            )

            hier_cfg = HierBuilderConfig(
                max_num_groups=cfg_model.max_num_groups,
                max_seq_len=cfg_model.max_seq_len,
                args_dim=getattr(cfg_model, "args_dim", 256),
                ensure_close=True,
                clip_range=1.2,
            )
            builder = {
                "mode": "hier",
                "builder": HierSVGTensorBuilder(hier_cfg),
                "packer": pack_groups,
                "collate": batch_collate_grouped,
                "cfg": hier_cfg,
            }
            # Enforce hierarchical encode_stages=2 expectation; warn if mismatch
            if getattr(cfg_model, "encode_stages", 2) != 2 and args.debug:
                print(
                    "[WARN] encode_stages in model cfg != 2 while using --faithful-hier. Proceeding but results may be invalid."
                )
        except Exception as e:
            print(
                f"[WARN] Failed to initialize hierarchical faithful builder ({e}); falling back to other options."
            )

    # If we auto-promoted to faithful_hier above, skip flat faithful branch.
    if (
        builder is None
        and getattr(args, "faithful", False)
        and not getattr(args, "faithful_hier", False)
        and cfg_model is not None
    ):
        try:
            from src.model.faithful.preprocess import (
                FaithfulBuilderConfig,
                RepoFaithfulPreprocessor,
                COMMANDS_SIMPLIFIED as _FAITHFUL_CMDS,
            )

            fb_cfg = FaithfulBuilderConfig(
                max_num_groups=cfg_model.max_num_groups,
                max_seq_len=cfg_model.max_seq_len,
                max_total_len=getattr(
                    cfg_model,
                    "max_total_len",
                    cfg_model.max_num_groups * cfg_model.max_seq_len,
                ),
                n_args=cfg_model.n_args,
                args_dim=cfg_model.args_dim,
                n_commands=len(_FAITHFUL_CMDS),
                encode_stages=getattr(cfg_model, "encode_stages", 2),
                rel_args=getattr(cfg_model, "rel_args", False),
                rel_targets=getattr(cfg_model, "rel_targets", False),
                ensure_close=True,
            )
            builder = {
                "mode": "faithful_flat",
                "builder": RepoFaithfulPreprocessor(fb_cfg),
                "cfg": fb_cfg,
            }
            if args.debug:
                print(
                    "[DEBUG] Faithful (flat) builder cfg: G=%d S=%d total_len=%d"
                    % (
                        fb_cfg.max_num_groups,
                        fb_cfg.max_seq_len,
                        fb_cfg.max_total_len,
                    )
                )
        except Exception as e:
            print(
                f"[WARN] Failed to initialize faithful flat builder ({e}); will use legacy builder."
            )

    if builder is None:
        from src.model.svgtensor_builder import (
            build_default_builder_from_cfg,
            SVGTensorBuilder,
        )

        legacy = (
            build_default_builder_from_cfg(cfg_model)
            if cfg_model is not None
            else SVGTensorBuilder(8, 30)
        )
        builder = {"mode": "legacy", "builder": legacy}

    if args.debug:
        print(f"[DEBUG] Selected builder mode: {builder['mode']}")
    # Builder selection: faithful (repo-aligned) vs legacy simplified
    if args.debug:
        print(
            "[DEBUG] Selecting builder (faithful=%s)" % getattr(args, "faithful", False)
        )
    # Guard: if a hierarchical builder dict was already selected, skip further builder overrides
    is_hier_dict = isinstance(builder, dict) and builder.get("mode") == "hier"
    if (not is_hier_dict) and getattr(args, "faithful", False) and cfg_model:
        from src.model.faithful.preprocess import (
            FaithfulBuilderConfig,
            RepoFaithfulPreprocessor,
            COMMANDS_SIMPLIFIED as _FAITHFUL_CMDS,
        )

        fb_cfg = FaithfulBuilderConfig(
            max_num_groups=cfg_model.max_num_groups,
            max_seq_len=cfg_model.max_seq_len,
            max_total_len=getattr(
                cfg_model,
                "max_total_len",
                cfg_model.max_num_groups * cfg_model.max_seq_len,
            ),
            n_args=cfg_model.n_args,
            args_dim=cfg_model.args_dim,
            n_commands=len(_FAITHFUL_CMDS),
            encode_stages=getattr(cfg_model, "encode_stages", 2),
            rel_args=getattr(cfg_model, "rel_args", False),
            rel_targets=getattr(cfg_model, "rel_targets", False),
            ensure_close=True,
        )
        builder = RepoFaithfulPreprocessor(fb_cfg)
        if args.debug:
            print(
                "[DEBUG] Faithful builder cfg: G=%d S=%d total_len=%d n_args=%d args_dim=%d encode_stages=%d"
                % (
                    fb_cfg.max_num_groups,
                    fb_cfg.max_seq_len,
                    fb_cfg.max_total_len,
                    fb_cfg.n_args,
                    fb_cfg.args_dim,
                    fb_cfg.encode_stages,
                )
            )
    elif not is_hier_dict:
        builder = (
            build_default_builder_from_cfg(cfg_model)
            if cfg_model
            else SVGTensorBuilder(8, 30)
        )
    if args.debug:
        print("[DEBUG] Encoder info:", encoder.info)
        if cfg_model:
            print(
                "[DEBUG] Model cfg: dim_z=%s d_model=%s max_num_groups=%s max_seq_len=%s use_vae=%s use_resnet=%s"
                % (
                    getattr(cfg_model, "dim_z", "?"),
                    getattr(cfg_model, "d_model", "?"),
                    getattr(cfg_model, "max_num_groups", "?"),
                    getattr(cfg_model, "max_seq_len", "?"),
                    getattr(cfg_model, "use_vae", "?"),
                    getattr(cfg_model, "use_resnet", "?"),
                )
            )
        # Support dict-based hierarchical builder or object-based builders
        if isinstance(builder, dict):
            bcfg = builder.get("cfg")
            mode = builder.get("mode", "unknown")
        else:
            bcfg = getattr(builder, "cfg", None)
            mode = "hier" if getattr(builder, "_is_hier", False) else "legacy/faithful"
        print(
            "[DEBUG] Builder cfg: mode=%s G=%s S=%s n_args=%s args_dim=%s clip_range=%s"
            % (
                mode,
                getattr(bcfg, "max_num_groups", "?"),
                getattr(bcfg, "max_seq_len", "?"),
                getattr(bcfg, "n_args", "?"),
                getattr(bcfg, "args_dim", "?"),
                getattr(bcfg, "clip_range", "?"),
            )
        )

    # Normalization config
    norm_cfg = NormalizationConfig(strategy=args.strategy, flip_y=True)

    extractor = EmbedExtractor(
        encoder=encoder,
        builder=builder,
        norm_cfg=norm_cfg,
        device=args.device,
        l2_normalize=not args.no_l2,
        qcurve_mode=args.qcurve_mode,
    )

    all_embeds: List[torch.Tensor] = []
    all_meta: List[GlyphMeta] = []

    batch_size = max(1, args.batch_size)
    t0 = time.time()
    parse_fail_total = 0
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        effective_batch_rows = batch_rows
        if (
            args.limit_encoder_batch
            and len(effective_batch_rows) > args.limit_encoder_batch
        ):
            effective_batch_rows = effective_batch_rows[: args.limit_encoder_batch]
        t_batch0 = time.time()
        try:
            embeds, metas = extractor.process_batch(effective_batch_rows)
        except Exception as e:
            print(f"[ERROR] Batch starting at {start} failed: {e}")
            if args.debug:
                import traceback as _tb

                _tb.print_exc()
            continue
        batch_time = time.time() - t_batch0
        all_embeds.append(embeds.cpu())
        all_meta.extend(metas)
        if args.debug and args.print_batch_shapes:
            print(
                "[DEBUG] Batch %d -> embeds %s (dtype=%s) metas=%d time=%.3fs"
                % (
                    start // batch_size,
                    tuple(embeds.shape),
                    embeds.dtype,
                    len(metas),
                    batch_time,
                )
            )
            # Show coordinate width/height stats for this batch
            if metas:
                widths = [m.width_em for m in metas]
                heights = [m.height_em for m in metas]
                import statistics as _stats

                print(
                    "[DEBUG]   width_em mean=%.4f std=%.4f min=%.4f max=%.4f | height_em mean=%.4f std=%.4f min=%.4f max=%.4f"
                    % (
                        _stats.mean(widths),
                        (_stats.pstdev(widths) if len(widths) > 1 else 0.0),
                        min(widths),
                        max(widths),
                        _stats.mean(heights),
                        (_stats.pstdev(heights) if len(heights) > 1 else 0.0),
                        min(heights),
                        max(heights),
                    )
                )
        if args.debug and start == 0:
            # Print a sample of first metas
            sample_n = min(args.debug_first, len(metas))
            print("[DEBUG] First batch sample metas:")
            for m in metas[:sample_n]:
                print(
                    f"         glyph_id={m.glyph_id} font={m.font_hash} label={m.label} w={m.width_em:.3f} h={m.height_em:.3f}"
                )

        processed = start + len(batch_rows)
        if processed % args.progress_every == 0 or processed == len(rows):
            elapsed = time.time() - t0
            print(
                f"[INFO] Processed {processed}/{len(rows)} glyphs in {elapsed:.1f}s (last batch {batch_time:.2f}s)"
            )
        if (
            args.max_parse_errors is not None
            and parse_fail_total > args.max_parse_errors
        ):
            print(
                f"[ERROR] Abort: parse/normalization failures exceeded --max-parse-errors ({args.max_parse_errors})."
            )
            break

    embeddings_tensor = torch.cat(all_embeds, dim=0)
    print(
        f"[INFO] Final embeddings shape: {tuple(embeddings_tensor.shape)} (dim={embeddings_tensor.shape[1]})"
    )

    save_embeddings(embeddings_tensor, args.out)
    save_metadata(all_meta, args.meta)

    print("[INFO] Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
