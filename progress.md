# Glyph Recognition via DeepSVG Encoder – Progress & Plan
<!-- Normalization default updated: norm_v2 (size-preserving EM-based). norm_v1 retained only for experiments. Section 21 reflects this change. -->

### Quick Start: Contour Sampling Script (Added)
Use this script early to understand quadratic curve payload shapes before finalizing the qCurveTo → cubic conversion logic.

Run (from project root):
```
python -m src.scripts.sample_contours \
  --db dataset/glyphs.db \
  --limit 1000 \
  --output artifacts/reports/qcurve_payload_stats.json
```

Key optional flags:
- `--no-random` : take the first N glyphs instead of random sampling.
- `--max-qcurve-examples 10` : store more example glyph ids per qCurveTo payload length.
- `--max-raw-examples 20` : keep more raw JSON prefixes for manual inspection.

Output (JSON) fields of interest:
- `qcurve_payload_lengths`: Maps payload length (number of coordinate pairs) → count + example glyph ids.
- `command_type_counts`: Global counts of moveTo / lineTo / curveTo / qCurveTo / closePath.
- `sequence_length_distribution`: Helps decide padding / truncation thresholds.
- `errors`: Any malformed or unexpected patterns encountered.

Recommended next step after running:
1. Inspect `artifacts/reports/qcurve_payload_stats.json`.
2. If payload lengths > 2 appear frequently for qCurveTo, refine pairing logic in `parse_contours` to correctly interpret chained quadratics.
3. Re-run with a larger `--limit` (e.g., 5000) once logic is stable to confirm distribution stability.

(Section inserted near top for quick visibility.)

## 1. Project Overview
Goal: Automatically infer Unicode codepoint labels for glyphs in fonts that lack proper `cmap` / `GSUB` tables by leveraging shape similarity.  
Near-term Objective (Phase 1): Build an embedding pipeline using the encoder portion of a pretrained `deepsvg-small` model, generate vector embeddings for glyph contours from a curated subset of fonts (≈10–20), persist them (in-memory + memory-mapped), and perform similarity/search experiments to validate embedding quality.

## 2. Current Assets
- SQLite DB: `dataset/glyphs.db`
  - Tables:
    - `fonts` (id, metadata)
    - `glyphs` (id, f_id (FK), label, contours, …)
- Class metadata: `dataset/chars.csv` (column of interest: `label`)
- DeepSVG repository already present at `./deepsvg/`
- Machine: 2019 MacBook Pro 16" (Intel CPU + AMD GPU; PyTorch GPU acceleration not realistically available — treat as CPU-only)

## 3. Constraints & Considerations
- CPU-bound embedding generation (optimize for batching and minimal Python overhead).
- Potential version friction: DeepSVG pins `torch==1.4.0`; we may want a modern PyTorch for performance & maintenance. Need to weigh:
  - Option A: Use original environment for guaranteed compatibility (lowest risk).
  - Option B: Incrementally port to modern PyTorch (≥2.x) — medium risk, possible API adjustments in `nn.Transformer`, layer norm, etc.
- Contours format in DB (TBD): Must parse into a canonical path command sequence matching the minimal subset (`m`, `l`, `c`, `z`) DeepSVG expects.
- Unicode label granularity: Some codepoints have multiple presentation forms (isol/medi/init/final). Treat each `label` row (e.g., `1595_init`) as distinct training/evaluation class tokens for now; unify later if needed.

## 4. Phase 1 Deliverables
1. Environment bootstrap script(s).
2. Utility to download / place pretrained `deepsvg-small` weights (if not already present).
3. SQLite glyph loader:
   - Configurable font sampling (by number or explicit font ids).
   - Parse + transform `contours` → DeepSVG `SVG` / `SVGTensor` input.
4. Encoder-only embedding extraction module:
   - Batch processing.
   - Deterministic preprocessing (scaling, normalization).
5. Embedding store:
   - In-memory (Python dict or structured NumPy array).
   - Memory-mapped file(s) on disk for persistence & fast reload.
6. Similarity search prototype:
   - Cosine similarity.
   - Basic metrics: top-k same-label match rate, cluster purity, qualitative nearest-neighbor dumps.
7. Documentation: usage instructions + notes on assumptions + pitfalls.
8. `progress.md` (this file) maintained with status.

## 5. High-Level Architecture (Phase 1)
```
SQLite glyphs.db
   ↓ (SQL query, subset fonts/glyphs)
Raw contour strings
   ↓ (Parser / Normalizer)
Canonical path representation
   ↓ (DeepSVG svglib conversion)
SVGTensor (tokenized sequence)
   ↓ (Batch collate)
DeepSVG Encoder (frozen weights)
   ↓
Latent Embedding (vector)
   ↙                ↘
In-Memory Index     Memory-Mapped Array (disk)
                         ↓
                  Reload on subsequent runs
                         ↓
                  Similarity / Evaluation
```

## 6. Detailed Task Breakdown

### 6.1 Environment & Dependencies
- Create `env/requirements_base.txt` for our code.
- Keep original `deepsvg/requirements.txt` intact to avoid breaking the vendor code.
- Evaluate if `torch==1.4.0` suffices; if slow, plan migration branch later.
- macOS dependencies: Ensure CairoSVG prerequisites (`brew install cairo libffi`).

### 6.2 Pretrained Model Acquisition
- Inspect `deepsvg/pretrained/download.sh` for model references.
- Identify target file for `deepsvg-small` (naming should match config; if not provided, fetch from upstream project release assets).
- Store weights in `deepsvg/pretrained/` (keep vendor layout).
- Write loader helper:
  - Load config (YAML or Python config object).
  - Initialize model.
  - Load state dict.
  - Set to `eval()`.
  - Expose function: `encode_svgtensor(batch_svgtensor) -> embedding`.

### 6.3 Contour Parsing & Normalization
- Inspect sample `contours` rows (future step: query the DB).
- Determine format:
  - If serialized list of path segments (e.g., moveto/lineto/cubic). If proprietary, implement parser to unify to `(cmd, points[])`.
- Normalize:
  - Translate glyph so bounding box center at origin.
  - Uniform scale to fit standard viewbox (e.g. [-1,1] or [0,1]).
  - Ensure direction & closure rules (supply `z` if shape closed).
- Convert to DeepSVG `SVG` object:
  - Build minimal path(s).

  ## 21. Visualization & Normalization Defaults (Updated – norm_v2 Active)
  
  (Addition) Normalization strategy versions:
  - norm_v1 (size-invariant, per-glyph unit scaling) – retained only for experiments.
  - norm_v2 (ACTIVE) size-preserving after EM normalization – preserves relative scale so diacritics (e.g. ء) remain distinguishable from larger base / initial forms (e.g. ع in its contextual shapes).
  
  Rationale for adopting norm_v2:
  - Arabic script classification relies on relative size & vertical extent (diacritics vs base glyphs, short vs elongated contextual forms).
  - Per-glyph unit scaling in norm_v1 collapses these cues → risk of confusing small diacritics with cropped larger letters.
  - EM normalization (divide by upem) harmonizes cross-font units while keeping natural proportions.
  - Center-only translation + single y-flip gives positional consistency without discarding scale information.
  
  What norm_v2 does:
  1. EM normalize coordinates: (x, y) → (x/upem, y/upem) if upem available.
  2. Compute glyph bbox in EM space and translate center to origin (0,0).
  3. Do NOT per-glyph scale (scale_applied = 1.0).
  4. Flip Y once (font y-up → model y-down) if flip_y=True.
  5. (Optional, currently disabled) Apply a global_scale if later we find extreme outliers; this would be a single scalar for ALL glyphs, preserving relative differences.
  6. (Optional clamp disabled) Could clamp to [-C, C] if extreme coordinates appear after broader sampling.
  
  Metadata captured (per glyph) under norm_v2:
  - normalization_version = "norm_v2"
  - width_em, height_em, bbox_raw, center_em
  - upem_used
  - scale_applied (always 1.0 here)
  - global_scale (None unless enabled)
  - y_flipped (bool)
  
  Future extension (planned, not implemented):
  - norm_v3: hybrid (concatenate size-invariant embedding + size-preserving embedding)
  - norm_v4: inject explicit size tokens or auxiliary numeric features (width_em, height_em) before encoder
  - Optional similarity re-ranking factor using size_ratio penalty
  
  Decision lock:
  - All Phase 1 embeddings will use norm_v2 unless a deliberate experiment requires norm_v1; any change triggers a new normalization_version tag.

## 22. Model Integration Status (DeepSVG Encoder) – Current Findings (Updated)

### 22.1 Repository Structure Clarification
- The full DeepSVG source (including `model`, `svglib`, `utils`, `difflib`, etc.) resides under the nested path `deepsvg/deepsvg/`.
- Path injection now standardized across scripts (`run_embed.py`, `debug_parse.py`, `similarity_eval.py`) so local repo usage requires no install step.

### 22.2 Checkpoint Structure Issue (Resolved)
- `hierarchical_ordered.pth.tar` wraps weights under a `"model"` key; loader now unwraps `"model"` and/or `"state_dict"` recursively.
- Additional normalization of key prefixes (`module.` / `model.`) implemented.
- Result: zero unexpected keys after heuristic alignment (see 22.11).

### 22.3 Config / Architecture Mismatch (Mitigated via Heuristics)
- Filename `"hierarchical"` triggers selection of `Hierarchical` config (encode_stages=2, decode_stages=2).
- New heuristic scans checkpoint keys:
  - Presence of `bottleneck.*` and absence of `vae.*` → enforce `use_vae=False`.
  - Presence of `vae.` keys → enforce `use_vae=True`.
- For the shipped hierarchical checkpoint: bottleneck present, VAE absent → auto sets `use_vae=False` (eliminating previous 4 missing + 2 unexpected keys).

### 22.4 Embedding Extraction Failures (Root Causes & Fixes)
Original failure (“All glyphs skipped”) traced to:
1. `ContourCommand` lacked dataclass constructor → instantiation errors silently swallowed.
2. Normalization expected `ContourCommandLike.replace_points`; ad‑hoc wrapper lacked this method.
3. Lack of debug output hid exceptions.
Fixes:
- Added `@dataclass(frozen=True)` to `ContourCommand`.
- Explicit conversion to `ContourCommandLike` prior to normalization.
- Added batch-level debug logging and skip summaries.
- Implemented encoder-only fast path; robust shape squeezing for (1,1,N,D) → (N,D).

### 22.5 Normalization Strategy Alignment
- Still using `norm_v2` (EM-scale, size-preserving, center, y-flip).
- Width/height EM statistics logged per batch (useful for later size-token experiments).
- Observed width_em range ~0.07–1.52 in small samples — indicates outliers; will revisit after larger run (potential mild global scale clamp).

### 22.6 Risks (Re-evaluated)
| Risk | Status | Mitigation |
|------|--------|------------|
| Unwrapped checkpoint `model` key | Resolved | Loader unwrap logic |
| VAE vs bottleneck mismatch | Resolved | Auto-config heuristic |
| Silent glyph parse errors | Mitigated | Verbose debug + failure IDs |
| Arg tokenization mismatch | Open | Phase 2 authentic tokenizer plan |
| Large coordinate magnitudes | Monitoring | Collect stats; consider global scale |
| Hierarchical pooling ambiguity | Acceptable | Output shape normalized & validated |
| All‑EOS group padding → attention NaNs | Resolved | Diagnosed root cause (all-masked attention rows); implemented dynamic group bucketing to eliminate zero-length columns |

### 22.7 Immediate Next Steps (Actionable – Updated)
- Add Section 24 (Hierarchical Faithful Embedding Pipeline Plan) implementing a fully repo-faithful DeepSVG preprocessing + hierarchical two-stage encoder path (replacing interim one-stage fallback).

DONE:
1. Loader unwrapping & safe state dict load w/ stats.
2. Encoder-only path & deterministic latent (bottleneck).
3. Verbose parse/normalize error diagnostics.
4. State dict match ratio logging (now 100%).
5. Embedding dimensionality assertion & shape normalization.
6. Metadata capture (width_em, height_em).
7. Similarity evaluation tooling (Section 23).
8. Hierarchical NaN diagnosis (all‑EOS columns) + dynamic group bucketing implementation.
9. Stability confirmation over 256 glyph sample (no NaNs; zero zero-norm rows post-bucketing).
10. Removal of deep internal probe instrumentation (kept lightweight debug only).

PENDING / UPCOMING:
11. Larger-scale embedding run (≥1k–5k) & aggregate similarity metrics (cluster cohesion / separation).
12. Intra/inter label distance tracking over bigger sample (persist summary stats).
13. Memmap (or parquet / Arrow) persistence layer for scalable embedding storage.
14. Optional global coordinate scale evaluation (decide after distribution analysis of larger run).
15. Authentic DeepSVG argument tokenizer parity plan draft (Phase 2).
16. Add group count distribution & per-bucket utilization reporting to progress logs.
17. Baseline similarity report population in Section 23 (after ≥1k run).

### 22.8 Longer-Term Enhancements
(unchanged – retained for continuity)

### 22.9 Acceptance Criteria (Baseline)
Remains valid; criteria (A) now achieved (100% encoder key match). Progress toward (B) pending larger batch job.

### 22.10 Work Queue (Concrete Tasks – Status Updated)
- [x] Unwrap checkpoint & reload (update loader).
- [x] Add encoder-only forward path (bypass VAE/decoder).
- [x] Implement verbose parse error logging.
- [x] Collect coordinate & command length stats in run_embed (initial batch-level stats).
- [x] Save embeddings + metadata with width_em/height_em (area derivable).
- [x] Add similarity_eval script (cosine top-k metrics).
- [x] Auto-config heuristic for VAE vs bottleneck & hierarchical detection.
- [ ] Larger evaluation run (≥500 glyphs) + baseline similarity metrics.
- [ ] Implement memmap persistence.
- [ ] Document baseline evaluation results (populate Section 23 metrics table).
- [ ] Decide if global scale needed (post distribution review).
- [ ] Draft plan for authentic arg tokenization parity (Phase 2).
- [ ] Dual normalization A/B exploratory run (optional).

### 22.11 Heuristic Loader Outcome Summary
| Checkpoint | Hierarchical Detected | VAE Keys | Bottleneck Keys | use_vae Chosen | Match Ratio |
|------------|-----------------------|----------|-----------------|----------------|-------------|
| hierarchical_ordered.pth.tar | Yes | No | Yes | False | 100.00% (247/247) |

#### 22.11.1 Dynamic Group Bucketing Summary (New)
| Metric | Observation (256 sample) |
|--------|---------------------------|
| Distinct G_used buckets | {1,2,3,4,5,6,7,8} (skewed toward 1–3) |
| Max G_used in sample | 8 |
| Mean G_used | ~2.6 (approx; precise stat to log in larger run) |
| Empty-first columns (pre-fix) | ~31–69% (earlier small runs) |
| Empty-first columns (post-fix) | 0% |
| NaN occurrences | 0 after bucketing |
| Zero-norm embeddings | 0 after bucketing & L2 |
| Row norm (mean) | ~18.6–19.1 across buckets |
| Attention stability | Confirmed (no all-masked rows) |

Rationale: Bucketing by actual group count eliminates fully masked attention columns without injecting synthetic tokens, preserving encoder behavior while preventing NaNs.


Notes:
- Previous mismatch (98.39%) eliminated by switching off VAE to align with bottleneck-only weights.
- Deterministic embeddings: VAE sampling skipped entirely; stable runs ensured.

## 23. Similarity Evaluation & Tooling (Initial / Updated with NaN Findings)
NOTE: Post Section 24 implementation we will regenerate embeddings with the hierarchical (encode_stages=2) faithful pipeline and re-run similarity evaluation using cluster size filters (e.g. --min-cluster ≥3) to obtain more meaningful top-k metrics and intra/inter separation.
A new script `src/scripts/similarity_eval.py` was added.
 
Capabilities:
- Loads embeddings (.pt) + metadata (JSONL).
- Optional label regex & minimum cluster size filtering.
- Computes:
  - Top-k (k configurable) accuracy & MRR (first correct label rank).
  - Intra-label vs inter-label cosine separation (sampled).
  - Per-dimension mean/std and row L2 norm stats.
  - Label cluster size distribution (mean / median / p90 / max).
- Chunked top-k computation for memory control (O(N * chunk_size * D)).
 
Recent Findings (1K embedding run – pre Phase B fix):
- 1000 glyph extraction produced (1000, 256) tensor but 985 rows contained NaNs (diagnosed post-run).
- similarity_eval consumed only 760 rows (cluster filter) and all aggregate stats became NaN (embedding mean/std, intra/inter cosines).
- Top-5 accuracy ≈ 1.18% (near random), confirming embeddings were numerically invalid rather than semantically weak.
- Root cause traced to encoder pooling division by zero (padding / visibility masks treating EOS-filled groups as empty → denominator 0 → NaNs).
- Confirmed hierarchical checkpoint integrator was fine; issue is builder + masking semantics (see Section 22.12).
 
Planned Usage Milestones (Revised):
1. Re-run a clean 128 glyph sample after masking fix (Phase B) – verify zero NaNs, report token stats.
2. Run ≥1000 glyphs with hierarchical_ordered (bottleneck; use_vae=False) and recompute:
   - k=5 and k=10 top-k accuracy
   - MRR
   - Intra vs inter mean cosine + separation Δ
3. Establish baseline acceptance thresholds (post-fix) and lock “Baseline v1”.
 
Upcoming Additions (if needed):
- Persist neighbor indices for offline inspection.
- Optional per-label confusion summary.
- Size-aware re-ranking experiment (penalize large size mismatch).
- Finite-only embedding filter in similarity_eval (drop rows with embedding_valid == false once metadata flag added).
 
Immediate Next Similarity Actions (Post Phase B):
- Implement builder refactor + pooling safety (see 22.12).
- Regenerate 1K embeddings (clean).
- Produce new sim_metrics_1k_clean.json (k=5,10).
- Append metrics excerpt below (in 23.1).
 
### 23.1 Baseline Metrics (Hierarchical Faithful Default – Post NaN Fix & Initial Projection Attempt)

Run: hier_auto3000 (limit=3000; hierarchical faithful auto-promoted; min-cluster=3; k=10)  
Sources: artifacts/hier_auto3000_embeds.pt + artifacts/reports/hier_auto3000_similarity_dual_k10.json

Raw / Filter:
- Raw embeddings: 3000
- Retained after cluster filter (≥3 per label): 2534
- Embedding dim: 256

Fine Label Metrics:
- Top-10 accuracy: 2.45%
- MRR@10: 0.0081
- Intra cosine mean: 0.96475 (std 0.03152, p10 0.92388, p90 0.99230)
- Inter cosine mean: 0.96440 (std 0.03115, p10 0.92051, p90 0.99212)
- Separation Δ (intra - inter): 0.00035
- Effect size (Δ / inter_std): 0.0113

Coarse (joining_group) Metrics:
- Top-10 accuracy: 28.81%
- MRR@10: 0.0943
- Intra cosine mean: 0.96414 (std 0.03141)
- Inter cosine mean: 0.96439 (std 0.03104)
- Separation Δ: -0.00026 (effect size -0.0083)

Label Cluster Stats:
- Num labels: 362
- Avg cluster size: 7.00 (median 6, p90 11, max 18)

Observations:
- Embedding space still exhibits angular crowding (global cosine ≈0.96–0.97 across pairs).
- Coarse grouping substantially improves ranking metrics but not separation—dot/diacritic variants collapse together.
- Indicates need for a contrastive or projection-head refinement phase; pure pretrained encoder features are not sufficiently discriminative at fine label granularity.

### 23.2 Contrastive Fine-Tune / Post-Processing Plan (Updated After Initial & Extended Projection Trials)

Goals:
1. Increase fine-label discriminability without harming coarse (joining_group) cohesion.
2. Achieve meaningful separation effect size (target ≥0.4) after projection.

Tracks:
A. Post-Processing Head (No model weight updates initially):
   - Fit PCA (optionally remove top PCs) + scalar feature augmentation (group_count, tokens_non_eos, width_em, height_em).
   - Learn a small 2-layer MLP projection (e.g. 256(+4 feats)→512→128) with supervised contrastive on (fine and/or coarse).
   - Compare label-only vs hybrid (label + joining_group positives) vs coarse-only warmup.
   - Adjust hybrid weighting (alpha) to emphasize fine label discrimination.

B. Light Contrastive Fine-Tune (Unfreeze Upper Layers):
   - Unfreeze last transformer block (stage2) + bottleneck.
   - Loss: InfoNCE with temperature τ, positives = same label (Latin/digits/punct) OR same joining_group (Arabic script), negatives sampled across batch.
   - Hard negative mining: within same joining_group but different fine label to encourage diacritic sensitivity (optional second phase).

C. Evaluation Loop:
   - Track: top-k accuracy, MRR, intra/inter separation, effect size (fine & coarse), and diacritic gap (planned future metric).
   - Early stopping on validation separation improvement plateau.

Milestones:
1. Implement projection module + training script (Phase 1.5).
2. Generate baseline (done: 23.1).
3. Run projection-only contrastive (frozen base) – report gains.
4. If insufficient (<0.25 effect size), proceed to partial unfreeze.
5. Integrate diacritic sensitivity metric inside joining_group clusters.
6. Lock “Baseline v2” embedding spec (documented transformation pipeline).

Risks & Mitigations:
- Overfitting small clusters → Use label frequency floor, temperature regularization.
- Collapse after whitening → Monitor variance retention of top principal components pre/post projection.
- Script imbalance → Weighted sampling across scripts (Arabic vs Latin vs others).

Next Actions (Queued - Updated):
- Hybrid alpha weighting (DONE: alpha=0.7 run) → further sweep (0.6–0.95).
- Extend training horizons (next: 500 epoch curriculum) with more gradient steps (batch 256).
- Multi-PC removal beyond 3 (test remove-top={1,3,5,8}).
- Log explained variance (top 10 PCs) pre/post removal & after 500 epochs.
- Temperature scheduling (high→low) to promote early spread then sharpening.
- Hard negative mining (within same joining_group but diff fine label) after initial curriculum separation ≥0.02.
- Diacritic sensitivity metric integration threshold lowered to effect size ≥0.05 (was 0.1) to allow earlier feedback.
- Cluster-size reweighting (inverse log or sqrt) to prevent large coarse groups dominating loss.
- Persist transformation config (PCA params + alpha + temp schedule + removal settings).


 
### 23.4 Projection Attempts Findings (3-Epoch & 100-Epoch)
Runs:
1) 3-Epoch (batch=512, hybrid alpha=0.5, remove-top=1):
   - Fine Top-10 acc: 2.53% (raw 2.45%)
   - Fine effect size: 0.0033
   - Coarse Top-10 acc: 29.08% (raw 28.81%)
   - Coarse effect size: 0.0144

2) 100-Epoch (batch=256, hybrid alpha=0.7, remove-top=3):
   - Fine Top-10 acc: 2.49% (unchanged)
   - Fine separation Δ: 0.00347 (effect size 0.0175) ↑ but still << target
   - Coarse Top-10 acc: 28.97% (stable)
   - Coarse effect size: 0.01895 (slight ↑)

Observations:
- Longer training + alpha weighting increased separation marginally (fine effect size 0.0033 → 0.0175) but remains far below the ≥0.1 interim goal.
- Loss decreased steadily (4.79 → 3.16) indicating optimization progress without meaningful geometrical discrimination.
- Multi-PC removal (1→3) helped expand variance (cosine distribution widened) but not label separation.
- Persistent angular crowding indicates projection-only adaptation is insufficient without curriculum + harder negative pressure.

Revised Near-Term Strategy:
- Introduce curriculum (coarse-only warmup → fine-heavy → hybrid with hard negatives).
- Temperature schedule to encourage early broad dispersion (e.g. start 0.18 → end 0.05).
- Increase PC removal experiments (remove-top=5,8).
- Add cluster-size reweighting and label frequency balancing.
- Integrate a diacritic-sensitive feature (tiny subpath count).
- If after curriculum (≤200 epochs) fine effect size <0.05, plan partial encoder unfreeze.

Updated Success Criteria:
- Short-term milestone (post-curriculum 200 epochs): fine effect size ≥0.05.
- Mid-term (500 epochs or partial unfreeze): fine effect size ≥0.15; coarse accuracy within ±2% of baseline.
- Stretch: fine effect size ≥0.30 with diacritic gap >0.02 (difference between same fine vs diff fine / same joining_group).

Next Step (Executing):
- Prepare 500-epoch script variants with curriculum and logging.
 
### 22.12 Masking, NaNs & Phase B Remediation Plan (New)
**Summary of Failure Mode**
- After extracting 1000 embeddings (hierarchical_ordered checkpoint, use_vae=False), diagnostic pass revealed 985/1000 rows contained NaNs (and effectively zero-norm before guard).
- NaNs originated from encoder pooling: `(memory * mask).sum / mask.sum` where `mask.sum == 0` for glyphs whose packed command sequence began with EOS tokens (i.e., empty effective token span).
- Our simplified builder pre-filled all (G,S) slots with EOS and only overwrote a subset for real subpaths; many groups (and sometimes the first flattened token) were EOS → zero valid token count.
 
**Key DeepSVG Mask Semantics (from utils.py)**
- PAD and EOS share the same token index (EOS acts as both sequence terminator and pad filler).
- `_get_padding_mask`: marks tokens before first EOS as valid (cumulative EOS count == 0).
- All-EOS groups (or a glyph whose first token is EOS) yield zero valid tokens → denominator 0 in pooling.
- Hierarchical visibility merges per-group EOS pattern; second-stage pooling can also produce zero denominators if all groups collapse.
 
**Immediate Fixes Applied**
- Added denominator clamp (`clamp_min(1e-6)`) in both encoder pooling stages to prevent NaNs (safety net).
- Added NaN guard in wrapper (`encoder_loader.py`) that replaces NaNs with zeros then renormalizes (temporary mitigation, not a semantic fix).
 
**Phase B (In Progress) – Structural Remediation**
1. Refactor `SVGTensorBuilder`:
   - Compact real subpaths contiguously from group 0; trailing groups remain EOS.
   - Ensure at least one non-EOS command at position 0 (inject synthetic move if necessary).
   - Track `real_token_count` per glyph; skip glyphs with zero after parsing + filtering.
2. Add batch stats:
   - real_token_count min / mean / max
   - percent skipped (parse + empty) 
   - count of glyphs falling below a configurable minimum token threshold (e.g. 2).
3. Remove dependence on NaN guard (it stays enabled but should not trigger post-fix).
4. Re-run 128 glyph smoke test (hierarchical_ordered; use_vae=False) → expect 0 NaNs.
5. Re-run 1000 glyph batch → regenerate similarity metrics (Section 23.1).
 
**Deferred (Optional) Enhancements**
- Introduce SOS/EOS framing per group if empirical results show instability.
- Synthetic centroid move command for empty glyphs instead of skipping (if skip rate > target).
- Authentic argument tokenization parity (relative vs absolute) to reduce OOD risk.
 
**Fonts Checkpoint Decision**
- `hierarchical_ordered_fonts.pth.tar` uses 62 label embeddings (A–Z, a–z, digits); our label set = 1216.
- For now we exclude that checkpoint to avoid label conditioning mismatch.
- Future path: expand label embedding matrix & reinitialize; not prioritized until baseline encoder embeddings stabilize.
 
**Acceptance to Conclude Phase B**
- Zero (or negligible <0.5%) NaN rows without relying on post-hoc guard.
- real_token_count > 0 for ≥99% of processed glyphs (excluding deliberate skips).
- Similarity metrics no longer NaN; top‑5 accuracy measurably above random baseline.
 
---
  

  These defaults are now fixed for all future embedding runs unless explicitly changed (recording here for reproducibility).

  Summary:
  - Coordinate orientation: Font outline coordinates are y-up; we apply a single y-axis flip during normalization (`flip_y=True`) so internal coordinates fed to the model follow a consistent y-down convention (avoids double inversion when exporting SVG).
  - Canvas / nominal resolution: 256×256 square is the reference visualization size matching our normalized coordinate assumption (not a raster target for the model itself, but a convenient debug scale).
  - Sizing strategy for embeddings (DEFAULT): 
    1. EM-normalize first: (x, y) ← (x/upem, y/upem)
    2. Center glyph by its bounding box midpoint at (0,0)
    3. Uniformly scale to fit into canonical range (approx [-1, 1]) while preserving aspect ratio (`scale_to_unit=True`)
    4. Apply y-flip once (no further inversion when drawing)
  - Visualization default (for qualitative review): Same normalization as embeddings (unit fit, not tight-fit). Tight-fit mode is available purely for ad‑hoc inspection of shape detail; it is NOT used for generating embeddings to avoid injecting per-glyph scale variance.
  - Relative size preservation: We chose to normalize scale for embeddings (size invariance) and will optionally log raw EM bbox metrics (width_em, height_em, area_em) later if size-aware experiments are needed. (Hybrid strategy postponed.)
  - Long command sequences: No truncation. Outliers are retained to preserve shape fidelity; any performance concerns will be handled by selective filtering rather than geometric modification.
  - Quadratic curves: Converted via midpoint inference (TrueType-like chaining) to cubic segments; expansion statistics captured (`qcurve_segments_to_cubic`, payload length histogram).
  - Default diagnostic export directory: `artifacts/vis` (random stratified sample including diacritics and ligatures).
  - Orientation verification: Browser rendering confirms correct upright shapes; macOS Preview’s initial appearance was a viewer quirk (not a data issue).

  Rationale:
  - Consistent normalization (size + orientation) improves embedding comparability and stabilizes similarity metrics.
  - Retaining full geometry (no truncation, faithful qCurve expansion) is critical for subtle distinctions between visually similar Arabic forms and diacritics.
  - Decoupling visualization-tight-fitting from embedding normalization avoids accidental leakage of per-glyph scale noise into the learned representation baseline.

  Change Control:
  - Any future deviation (e.g., enabling tight-fit for embeddings or adding raw-size feature channels) must be documented here with a new subsection and embedding version increment.

  Embedding Normalization Version: `norm_v1` (y-flip + EM normalize + unit scale)

  - Call `.simplify_heuristic()` only if necessary (avoid changing topology prematurely).
  - Convert to `SVGTensor` (check `deepsvg/svglib/svg.py` & dataset utilities).

### 6.4 Data Loader Abstraction
- Implement `GlyphDataset`:
  - Args: `db_path`, `font_limit`, `label_filter`, `random_seed`, `max_paths`, etc.
  - Implements iter or returns list of `(glyph_id, font_id, label, SVGTensor)`.
  - Optional caching layer for parsed contour → SVGTensor.

### 6.5 Batching & Collation
- DeepSVG expects padded sequence tensors (see `svgtensor_dataset.py`).
- Reuse pad logic or implement minimal one:
  - Determine `max_len` per batch.
  - Pad command tokens & coordinate arrays with mask.
- Optimize to keep batch small enough for CPU (e.g., 16–64 depending on sequence length).

### 6.6 Embedding Extraction
- Freeze encoder modules (disable grad).
- For `model` object (after load):
  - Identify encoder forward call boundary: likely returns hierarchical latent(s).
  - Decide which latent to adopt:
    - Option A: Final Transformer encoder hidden state pooled (mean).
    - Option B: Provided latent vector if model architecture already aggregates.
  - Store dimension metadata.

### 6.7 Persistence Layer
- Memory structure:
  - `glyph_index` (list of records): `[ (glyph_id, font_id, label, embedding_offset) ]`.
- Disk:
  - Directory: `artifacts/embeddings/`
  - Files:
    - `encoder_embeddings.float32.memmap`
    - `metadata.parquet` or JSON lines (glyph & label mapping).
  - Process:
    - Preallocate memmap size = `num_glyphs * emb_dim`.
    - Write sequentially.
    - Flush at intervals.
  - Reload:
    - Load metadata.
    - Open memmap with appropriate shape.
- Version header in a small `VERSION` or embedded in metadata.

### 6.8 Similarity Search Prototype
- Implement simple cosine similarity:
  - Normalize embeddings at write-time (L2).
  - For query: matrix multiply `query_vec @ embeddings.T`.
- Evaluation:
  - For each glyph:
    - Retrieve top-k neighbors excluding itself.
    - Compute if label matches (top-1, top-5).
  - Aggregate statistics (macro accuracy).
  - Qualitative dumps: store small JSON: `glyph_id`, `label`, `neighbors: [...]`.
- (Future) Replace with FAISS or vector DB (Milvus / Qdrant / Weaviate).

### 6.9 Logging & Monitoring
- Lightweight logging (standard library or `loguru`).
- Track:
  - Time per N glyphs.
  - Average sequence length distribution.
  - Embedding norm statistics (sanity check).

### 6.10 Future Phase Hooks
- Vector DB integration (Qdrant/FAISS).
- REST API endpoint:
  - Upload font file → extract glyph outlines → embed → similarity search → assign codepoint label candidates with confidence.
- Active learning / human-in-the-loop review pipeline.
- Fine-tuning encoder on domain-specific glyph similarity objective (triplet / contrastive loss).

## 7. Risk Register (Phase 1)
| Risk | Impact | Mitigation |
|------|--------|------------|
| Contours format incompatibility | Blocking | Early schema inspection; write robust parser |
| Missing pretrained small weights | Delay | Mirror from upstream / fallback to medium and shrink dims later |
| Torch 1.4 performance on CPU | Slower iteration | Accept initial slowness; plan upgrade branch |
| Sequence length variance causing OOM (CPU RAM) | Performance degradation | Cap paths or segments; optional simplification |
| Poor embedding separability | Misclassification | Add normalization, try different encoder latent layer, consider PCA/UMAP diagnostic |

## 8. Milestones & Timeline (Indicative)
1. M1 – Environment + Pretrained weights (Day 1)
2. M2 – Contour parsing + sample SVGTensor conversion (Day 2)
3. M3 – Encoder embedding extraction for 1 font subset (Day 3)
4. M4 – Full 10–20 font subset embeddings + memmap persistence (Day 4)
5. M5 – Similarity metrics + qualitative review (Day 5)
6. M6 – Adjustments / documentation / readiness for Phase 2 (Day 6)

## 9. Metrics (Initial)
- Coverage: number of glyphs embedded / total candidate glyphs in sampled fonts.
- Embedding shape: `N x D`.
- Top-1 / Top-5 same-label neighbor accuracy.
- Average intra-class similarity vs inter-class similarity (mean cosine).
- Embedding norm mean ± std.

## 10. Open Questions
- Exact contour string syntax? (Need inspection).
- Are ligatures (`..._liga`) to be included now or deferred?
- Do we unify presentation forms later (isol/final/init/medi) or treat them separately always?
- Are there multi-path glyphs? (Affects normalization & path merging strategy).
- Do we need directionality (stroke order) preserved?

## 11. Immediate Next Actions (Execution Order)
1. Query and inspect a handful of `glyphs.contours` values.
2. Locate & download `deepsvg-small` weights; confirm load path.
3. Prototype minimal script: load model → dummy SVGTensor from synthetic path → obtain encoder latent shape.
4. Implement contour parser + normalization pipeline (unit-test on 3–5 samples).
5. Build dataset loader and batch collator.
6. Embed 1 font’s glyphs; store to temp memmap; validate reload.
7. Expand to 10–20 fonts; build similarity report.
8. Document findings & refine plan for next iteration.

## 12. Directory Layout (Planned Additions)
```
src/
  config/
    embedding_config.yaml
  data/
    glyph_loader.py
    contour_parser.py
  model/
    encoder_loader.py
    embedding_pipeline.py
  index/
    memmap_store.py
    similarity.py
  eval/
    metrics.py
    reports.py
  scripts/
    run_embed.py
    run_eval.py
artifacts/
  embeddings/
    encoder_embeddings.float32.memmap
    metadata.json
    neighbors_sample.json
```

## 13. Future Enhancements (Backlog)
- Contrastive fine-tuning on glyph pairs (positive: same label; negative: different families).
- Integration with a vector DB (Qdrant / Milvus).
- Web service (FastAPI) for font upload & labeling.
- Confidence calibration (e.g., temperature scaling).
- Active learning loop: uncertain glyphs surfaced for manual verification.

## 14. Maintenance Notes
- Keep vendor `deepsvg` code isolated; do not modify unless patching for compatibility. If patched, maintain a patch log.
- Add reproducibility seed handling (Python, NumPy, torch).
- Ensure deterministic preprocessing (no random augmentation in Phase 1).

---

# TODO Checklist

### Environment & Model
- [ ] Verify system Python version and create virtual environment.
- [ ] Install DeepSVG dependencies (torch 1.4 path).
- [ ] Attempt modern PyTorch install in separate branch (optional).
- [ ] Download / place `deepsvg-small` weights.
- [ ] Implement `encoder_loader.py` that returns encoder callable.

### Data Inspection
- [ ] Write quick script to SELECT sample rows from `glyphs` table.
- [ ] Document contour string format with examples.
- [ ] Decide on handling of multi-path glyphs.

### Parsing & Preprocessing
- [ ] Implement contour parser → internal path structure.
- [ ] Normalize coordinates (translate & scale).
- [ ] Convert to DeepSVG `SVG` object and test `.draw()` (optional visual debug).
- [ ] Create function to produce `SVGTensor` suitable for encoder.

### Dataset Loader
- [ ] Implement `GlyphDataset`.
- [ ] Add font sampling logic (random or deterministic).
- [ ] Add filtering for invalid / unparsable contours.

### Embedding Pipeline
- [ ] Implement batch collation (padding + masks).
- [ ] Extract embeddings (verify shape & detachment).
- [ ] Choose pooling strategy (CLS token equivalent / mean pooling).

### Persistence
- [ ] Preallocate memmap for embeddings (determine size dynamically).
- [ ] Write metadata (JSON or Parquet).
- [ ] Implement reload path & verification test.

### Similarity & Evaluation
- [ ] Normalize embeddings (L2).
- [ ] Implement cosine similarity search.
- [ ] Compute top-1 / top-5 accuracy.
- [ ] Dump qualitative nearest neighbors for inspection.
- [ ] Generate summary metrics report.

### Diagnostics & Logging
- [ ] Add logging for sequence length distribution.
- [ ] Add timer metrics for embedding throughput (glyphs/sec).
- [ ] Validate no NaNs in embeddings.

### Documentation
- [ ] README snippet or `docs/` page for running Phase 1 pipeline.
- [ ] Update `progress.md` with outcomes & metric snapshots.
- [ ] Add design notes for future vector DB integration.

### Backlog (Defer Until Phase 2)
- [ ] Vector DB integration (FAISS / Qdrant).
- [ ] REST API scaffolding (FastAPI).
- [ ] Fine-tuning strategy doc.
- [ ] Active learning pipeline concept.

---

## 15. Status Summary (Initial Commit)
Status: Planning stage. No code yet for embedding pipeline.  
Next Action: Inspect `glyphs.contours` data to finalize parser design.

(Progress updates will be appended below.)

---

## 16. Update Log
- Added initial inspection of glyph contour serialization (structure confirmed as nested JSON of drawing commands).

## 17. Contour Data Format (Observed) & Parser Design

### 17.1 Raw Format (from `glyphs.contours`)
Example (truncated):
[[
  ["moveTo", [126, 47]],
  ["curveTo", [[131, 44], [139, 46], [150, 52]]],
  ...,
  ["closePath", null]
],
[
  ["moveTo", [177, -82]],
  ["curveTo", [[202, -80], [213, -70], [211, -52]]],
  ...,
  ["closePath", null]
]]

Interpretation:
- Top-level: List of subpaths (each glyph may have multiple contours; outer + holes).
- Each subpath: Ordered list of drawing operations.
- Operations observed so far:
  - "moveTo", [x, y]                  (single point)
  - "curveTo", [[x1, y1],[x2, y2],[x3, y3]] (cubic Bézier control1, control2, endpoint)
  - "lineTo", [x, y] (not yet seen in sampled rows, but anticipate)
  - "closePath", null
- Coordinates: Integer (font units). Negative values appear (baseline / descender areas).
- Multiple subpaths imply: Fill + potential holes (see `hole_count` / `orientation` metadata columns).

### 17.2 Canonical Internal Representation
We will normalize each subpath into a sequence of tuples:
(cmd, points)

Where:
- cmd ∈ { 'm', 'l', 'c', 'z' } (matching DeepSVG minimal subset)
- points:
  - m: [(x, y)]
  - l: [(x, y)]
  - c: [(x1, y1), (x2, y2), (x3, y3)]
  - z: [] (or None)

Assume all coordinates are absolute (no relative commands present). If later relative commands appear, convert to absolute during parse.

### 17.3 Parsing Steps
1. JSON Decode:
   - `paths = json.loads(contours_text)`
   - Validate: list; each element list; each command is [commandName, payload].
2. For each subpath:
   - Initialize empty list of segments.
   - Track: min_x, min_y, max_x, max_y (for bounding box).
   - For each command:
     - Map "moveTo" → 'm'
     - Map "lineTo" → 'l'
     - Map "curveTo" → 'c'
     - Map "closePath" → 'z'
     - Validate payload shape.
     - Append canonical segment.
3. Bounding Box & Normalization:
   - After collecting all coordinates across subpaths:
     - width = max_x - min_x (if 0 → skip / discard degenerate)
     - height = max_y - min_y
     - Choose scale:
       - Option A (unit square): s = 1 / max(width, height); translate so min_x/min_y maps to 0, then center to (0.5,0.5) or center origin.
       - Proposed: Center at origin for symmetry:
         cx = (min_x + max_x)/2, cy = (min_y + max_y)/2
         s = 1 / max(width, height)
         x' = (x - cx) * s
         y' = (y - cy) * s
     - Optionally flip Y if font coordinates are y-up and DeepSVG expects y-down (verify by rendering a test glyph).
4. Orientation / Holes:
   - If `orientation` / `hole_count` is meaningful, we may preserve winding:
     - Compute signed area per subpath; if necessary, reverse point order for hole logic.
   - Phase 1: Keep raw order; log statistics; adjust later if rendering anomalies appear.
5. Degenerate Curves:
   - If cubic control points collapse to straight line (collinear + control points on segment), optionally downgrade to 'l' for efficiency (Phase 1: skip).
6. Output Structure:
   - List[Subpath], each Subpath = list[(cmd, points)] after normalization & scaling.

### 17.4 Conversion to DeepSVG Structures
Approach A (Direct):
- Use `deepsvg.svglib.svg.SVG` and construct `SVGPath` objects by feeding commands.
Approach B (Tokenization Pipeline Reuse):
- Mirror logic in `deepsvg/svglib` & `svgtensor_dataset.py` to produce a `SVGTensor`.
Plan:
- Implement utility `build_svg_from_contours(parsed_subpaths)` returning a `SVG` instance.
- Then call existing methods to convert to tensor (e.g., something akin to `svg.to_tensor(...)` if provided; else replicate minimal tokenization: command tokens array + coordinates arrays + mask).

### 17.5 Embedding Input Requirements (Assumptions to Validate)
- DeepSVG expects:
  - Fixed max number of commands per path (truncate or pad).
  - Command vocabulary indices: {m,l,c,z,...} — confirm by inspecting `model/config.py` & dataset code.
  - Coordinate normalization typically in [-1,1]. Our centering + scaling satisfies this.
Validation tasks:
- Feed a simple synthetic square path and ensure encoder forward pass succeeds (no shape mismatch).
- Log resulting latent shape.

### 17.6 Error Handling & Logging
- If JSON parsing fails → skip glyph (record in a `parse_errors.log`).
- If empty or degenerate bounding box → skip.
- Add counters:
  - total_glyphs
  - parsed_ok
  - skipped_parse
  - skipped_degenerate
- Maintain a sample of first N parsed canonical sequences for debugging.

### 17.7 Proposed Module Skeleton (`src/data/contour_parser.py`)
Functions:
- `parse_contours(raw_text: str) -> List[List[Tuple[str, List[Tuple[float,float]]]]]`
- `normalize_contours(parsed, center_origin=True, unit_scale=True, flip_y=False) -> parsed_normalized`
- `contours_to_svg(parsed_normalized) -> SVG`
- `svg_to_svgtensor(svg) -> SVGTensor` (wrap vendor code / replicate minimal logic)
- `contours_to_svgtensor(raw_text) -> SVGTensor` (pipeline convenience)

### 17.8 Edge Cases / Future Enhancements
- Quadratic curves (if appear later) → upscale to cubic.
- Arcs (unlikely here) → approximate with cubic segments.
- Stroke vs fill: Not encoded; assume fills only.
- Holes ordering / winding enforcement for better visual correctness (Phase 2).
- Caching: Hash raw_text → serialized tensor to avoid re-parsing (optional optimization).

### 17.9 Immediate Actions Derived
- Implement parser & normalization (no orientation logic yet).
- Run on a small batch (e.g., 100 glyphs) to collect statistics: avg commands, max commands.
- Decide on truncation length for batching (log distribution before deciding).
- Add unit test examples (synthetic + one real sample).

### 17.10 Added TODO Items
(Will be migrated into main checklist on next update)
- [ ] Implement `contour_parser.py` with pipeline functions.
- [ ] Collect command count distribution over sample (n=1000).
- [ ] Verify coordinate range post-normalization (assert max abs ≤ 1.05).
- [ ] Confirm DeepSVG tokenizer compatibility (adjust command mapping if needed).
- [ ] Synthetic glyph round-trip test (construct → parse → normalize → tensor → encode).

## 18. Quadratic Curves (`qCurveTo`) Handling

### 18.1 Updated Empirical Findings
Sampling 1,500 glyphs showed:
- 24,866 `qCurveTo` commands vs 2,452 `curveTo` (cubic) and 12,960 `lineTo`.
- Payload lengths observed for `qCurveTo`: 2,3,4,5,6,7,8,9,10,11,12 (11 distinct lengths).
- Dominant lengths: 2 (16,382), 3 (5,544), 5 (2,286) → ~97% of all qCurveTo commands.
- Very long payloads (≥8 points) are extremely rare (handful of cases).

### 18.2 Interpretation
The diversity of payload lengths indicates TrueType-style chained quadratic segments with implied on‑curve points between consecutive off‑curve control points. The raw serialization does not explicitly label on‑curve vs off‑curve; we infer structure heuristically.

### 18.3 Implemented Midpoint Inference Strategy
For a `qCurveTo` payload points list P = [p1, p2, ..., pn]:
1. Assume the final point pn is an explicit on‑curve endpoint.
2. Treat all intermediate points p1..p(n-1) as off‑curve controls.
3. Iterate through intermediates:
   - For consecutive off‑curve controls c1, c2 we insert an implied on‑curve midpoint m = (c1 + c2)/2, forming a quadratic segment (prev_on_curve, c1, m), then advance prev_on_curve = m and continue (leaving c2 to pair with the next point).
   - The last remaining control before pn forms the final quadratic (prev_on_curve, control, pn).
4. Each quadratic (P0, Q, P2) is converted to cubic (C1, C2, P2) via:
   C1 = P0 + 2/3 * (Q - P0)
   C2 = P2 + 2/3 * (Q - P2)

This approach preserves geometry closely for typical TrueType outlines while remaining deterministic and fast. The legacy “naive” pairing mode is still available via `qcurve_mode="naive"` for debugging.

### 18.4 Stats Hooks
`parse_contours` now accepts `qcurve_stats` dict to accumulate:
- qcurve_payload_len_{N}: count per payload length
- qcurve_segments_to_cubic: total cubic segments emitted from quadratic expansion
This enables downstream analysis (e.g., average expansion factor).

### 18.5 Outlier Handling Policy
We will NOT truncate very long command sequences. Instead:
- Keep complete geometry for maximum fidelity.
- (Future) Optionally filter out extreme outliers if they cause performance issues rather than truncating (to avoid shape corruption).
- Log any glyph whose total command count exceeds a configurable threshold (e.g., >1024) for inspection.

### 18.6 Risks & Mitigations (Revised)
| Risk | Impact | Mitigation |
|------|--------|-----------|
| Misclassification of on/off-curve for unusual payload patterns | Slight geometric drift | Rare; visualize a small sampled set containing long payloads |
| Performance impact from very long glyphs | Slower batch throughput | Pre-batch sort by length; process longest separately |
| Accumulated numeric error in midpoint inference | Minimal | Coordinates remain in font unit precision; normalization rescales early |

### 18.7 Follow-up Tasks
- [ ] Add visualization utility to export a few decoded contours (pre/post midpoint conversion) to SVG for manual QA.
- [ ] Collect percentile stats of total command counts (P90, P95, P99) on a larger sample (e.g. 20k glyphs).
- [ ] Add assertion tests for specific synthetic chained qCurveTo patterns.
- [ ] Record average expansion factor (quadratic segments → cubic segments) in logs.

### 18.8 Toggle & Reproducibility
- Parser parameter: `qcurve_mode` in `parse_contours` (default "midpoint").
- For reproducibility, log the mode and counts into embedding metadata artifacts.

(Previous subsection content replaced after implementing the improved midpoint inference.)

## 19. Font Metrics & Normalization Strategy

### 19.1 Observed `fonts` Table Schema
Fields of interest: `file_hash` (PK), `upem` (Units Per EM), `num_glyphs`.
Sample `upem` values: 500 (min) → 4096 (max), 22 distinct values. Dominant: 2048, then 1000.

### 19.2 Need for Font-Aware Normalization
Glyph coordinate magnitudes differ proportionally to `upem`. Normalizing only by glyph bounding box can:
- Remove relative stroke scale relationships across fonts.
- Inflate very small diacritics (changing shape density semantics).

Proposed hybrid normalization:
1. Scale raw coordinates to EM-relative float: x' = x / upem, y' = y / upem.
2. Compute glyph bbox in EM space.
3. Optional global scale to fit into target range (e.g., ensure max(|x'|,|y'|) ≤ 1 after centering).
4. Center at glyph bbox midpoint (post-EM scaling).
5. Preserve aspect ratio (no anisotropic scale).
6. Track original bbox size (w_em, h_em) as auxiliary features (possible future conditioning).

### 19.3 Handling Outliers
- If bbox extremely small (e.g., combining marks): Instead of scaling to fill range, apply minimum scale threshold to avoid over-amplifying noise. Possibly keep a global scale factor S = 1 / max_global_extent (computed over sample) for consistent embedding geometry.
Phase 1 simplification: per-glyph normalization (center + scale by its max dimension) BUT store:
- `scale_factor_em`
- `bbox_area_em`
for later model refinement.

### 19.4 Orientation & Baseline
Without baseline metrics (ascender/descender), vertical alignment differences might persist. If later needed, we can:
- Infer baseline from median y of large letters per font subset.
For now: rely on per-glyph centering.

### 19.5 Additional Metadata Capture
When reading fonts:
- Map `f_id` → (`upem`, `family_name`, `style_name`).
- Attach `upem` to each glyph record before normalization.

### 19.6 TODO Additions (Fonts & Normalization)
- [ ] Build `FontMetricsCache` (lazy load upem per font hash).
- [ ] Modify parser pipeline to accept `upem`.
- [ ] Implement dual normalization modes: `bbox_relative` vs `em_relative`.
- [ ] Record stats: distribution of (w_em, h_em).
- [ ] Evaluate effect on cosine similarity (compare both modes on subset).

## 20. Updated Overall TODO (New Items)

## 24. Hierarchical Faithful Embedding Pipeline Plan (New)

Objectives
- Eliminate ad-hoc one-stage & improvised builders.
- Reproduce DeepSVG’s original hierarchical (two-stage) encoder input format (groups × sequence) with authentic command / argument layout, padding, masking, and (optional) relative arguments.
- Achieve stable, NaN-free embeddings with meaningful intra vs inter label separation.

24.1 Scope
- Use checkpoint: hierarchical_ordered.pth.tar (bottleneck, no label conditioning, encode_stages=2).
- Implement canonical per-subpath SVGTensor generation (m, l, c, z; arcs deferred).
- Integrate grouping, packing, and encoder invocation mirroring repository utilities (no full-model fallback).
- Support (future toggle) relative argument encoding if `cfg.rel_targets` is ever True.

24.2 Components to Implement
1. svgtensor_builder (faithful)
   - Contours → list[SVGTensor] (one per subpath).
   - Ensure first command is ‘m’; synthesize if absent.
   - Append ‘z’ if ensure_close and not present.
   - Truncate before exceeding max_seq_len - 1 (reserve EOS), then .add_eos().pad().
2. packer
   - Pack up to max_num_groups → grouped (G,S) tensors (commands, args) with EOS / PAD.
   - Skip empty subpaths (no group rows with zero valid tokens).
3. encoder_wrapper refactor
   - Remove generic forward fallback (no TypeError path).
   - Always call model.encoder(seq_first_cmds, seq_first_args, label=None).
   - Apply bottleneck (hierarchical_ordered uses bottleneck; ensure use_vae=False).
   - Validate returned z shape: expect pooled latent per glyph after second stage.
4. dataset / batch iterator
   - Stream glyph rows, build grouped tensors, collate to (N,G,S) → permute to (S,G,N).
   - Early skip glyphs producing zero tokens.
5. normalization integration
   - Reuse norm_v2 pipeline (center + EM scale + uniform scale) exactly as already established.
6. evaluation adjustments
   - For similarity: enforce min_cluster >= 3 (or configurable) to avoid near-random top-k due to 2-member clusters.
   - Record baseline metrics: top-5/top-10 accuracy, MRR, intra/inter cos, separation.

24.3 Algorithmic Fidelity Checks
- Command indices must be in [0, n_commands-1]; EOS exclusively for padding/tail.

---

## 25. Zero-Embedding Failure (Legacy Builder) & Migration to Hierarchical Faithful (New)

### 25.1 Summary of Issue
During large-batch extraction with the legacy (non-hierarchical) builder + hierarchical checkpoint (`hierarchical_ordered.pth.tar`), nearly all embeddings were zero after pooling:
- Pre-L2 norms: mean=0.0, zero_rows=64/64 in most batches.
- Occasional single non-zero row indicated sporadic valid first token placement.
- tokens_non_eos counts were healthy (≈30–40), so geometry existed; collapse was due to mask semantics, not empty contours.

### 25.2 Root Cause
The hierarchical model expects grouped (G,S) sequences with at least one non-EOS command at the start of each glyph’s first used group. The legacy builder:
- Prefills with EOS tokens (shared PAD/EOS index) at position 0.
- Causes `_get_padding_mask` (first-EOS = sequence termination) to mark all tokens invalid.
- Pooling then divides by (effective_length=0) → protected from NaN by clamp but yields zero vectors.

### 25.3 Validation Fix
Switching to the hierarchical faithful builder (`--faithful-hier`) restored non-zero embeddings:
- Buckets grouped by actual group_count (g_used).
- Encode stats: row_norm_mean ≈ 18–19 before normalization; zero_rows=0.
- Utilization improved within buckets (no full (8×30) padding wastage).
This confirms the collapse was a structural input mismatch, not a numerical instability.

### 25.4 Actionable Guidelines
1. Always use `--faithful-hier` (or future canonical hierarchical builder) with hierarchical checkpoints (encode_stages=2).
2. Ensure first command of first group is a valid move (synthetic insert if parser omits).
3. Reject or repair glyphs whose first usable group starts with EOS.
4. Avoid mixing legacy flat builder and hierarchical checkpoints in production metrics.

### 25.5 Diagnostic Checklist (Smoke Test of N=128)
- Print encode stats per bucket: confirm `zero_rows=0`.
- Verify distribution of g_used (expect 1–6+ not all identical).
- Check sample command sequences: index 0 should be a non-EOS command token (e.g., MOVE).
- Confirm post-L2 norms are ~1.0 with negligible zero-norm rows.

### 25.6 30K Hierarchical Faithful Extraction Instructions
Command (CPU example):
```
python -m src.scripts.run_embed \
  --db dataset/glyphs.db \
  --pretrained deepsvg/pretrained/hierarchical_ordered.pth.tar \
  --faithful-hier \
  --limit 30000 \
  --batch-size 32 \
  --strategy norm_v2 \
  --out artifacts/hier_faithful30000_embeds.pt \
  --meta artifacts/hier_faithful30000_meta.jsonl \
  --device cpu \
  --progress-every 1000
```
Post-run sanity:
```
python - <<'PY'
import torch
E=torch.load('artifacts/hier_faithful30000_embeds.pt')
import numpy as np
norms=E.norm(dim=1)
print("Shape:", E.shape, "Zero-norm fraction:", float((norms<1e-8).sum())/len(norms))
PY
```
Expected zero-norm fraction ≈ 0.0.

### 25.7 PCA (Optional) for Projection
```
python -m src.scripts.pca_postprocess \
  --embeds artifacts/hier_faithful30000_embeds.pt \
  --fit-pca --pca-dim 128 \
  --out-dir artifacts/pca/hier_faithful30000
```

### 25.8 500-Epoch Projection Training (30K)
Baseline curriculum with hard negatives:
```
python -m src.scripts.train_projection_head \
  --embeds artifacts/hier_faithful30000_embeds.pt \
  --meta artifacts/hier_faithful30000_meta.jsonl \
  --pca-model artifacts/pca/hier_faithful30000 \
  --remove-top 5 \
  --augment-features \
  --epochs 500 \
  --curriculum coarse:50,fine:100,hybrid:350 \
  --hybrid-alpha-start 0.88 --hybrid-alpha-end 0.60 \
  --temp-start 0.20 --temp-end 0.05 --temp-mode cosine \
  --cluster-weighting inv_sqrt \
  --hard-neg --hard-neg-scale 1.7 \
  --batch-size 256 \
  --log-var-every 50 --var-topk 10 \
  --out-proj artifacts/projection/head_faithful30k_500ep.pt \
  --out-embeds artifacts/projection/hier_faithful30000_proj_500ep.pt \
  --log-json artifacts/projection/train_log_faithful30k_500ep.json \
  --device cpu
```

### 25.9 Post-Training Evaluation
```
python -m src.scripts.similarity_eval \
  --embeds artifacts/hier_faithful30000_embeds.pt \
  --meta artifacts/hier_faithful30000_meta.jsonl \
  --k 10 --dual-metrics --min-cluster 3 \
  --json-out artifacts/reports/base_faithful30k_similarity.json

python -m src.scripts.similarity_eval \
  --embeds artifacts/projection/hier_faithful30000_proj_500ep.pt \
  --meta artifacts/hier_faithful30000_meta.jsonl \
  --k 10 --dual-metrics --min-cluster 3 \
  --json-out artifacts/reports/proj_faithful30k_500ep_similarity.json
```

### 25.10 Next Optimization Levers (If Fine Metrics Still Flat)
- Try `--remove-top 3` variant.
- Increase hard negative scale (1.7 → 1.9) if coarse improves but fine stagnates.
- Adjust alpha decay (hold 0.88 for first 150 hybrid epochs before decaying).
- Introduce periodic embedding snapshots (future script patch).

### 25.11 Acceptance for “Healthy Extraction” Milestone
- Zero-norm fraction < 0.001.
- Encode per-bucket zero_rows=0.
- Similarity eval runs without warnings; no NaNs.
- Coarse effect size > 0 (positive separation).
- Fine effect size shows measurable upward trend when using hard negatives (target interim > 0.03 on 30K).

(End Section 25)
- Args: raw indices in valid bin range; PAD_VAL=-1; rely on embedding layer’s (args+1) shift.
- No negative indices < -1; no values ≥ args_dim.
- Mask sanity:
  - padding_mask.sum() per sample > 0.
  - No sample returns denominator 0 in pooling.

24.4 Validation Steps
1. Smoke Test (N=32)
   - Confirm: no NaNs; pre-L2 norm distribution (report min/mean/max).
   - Log token utilization (% of (G*S)).
2. Mid Test (N=512)
   - Confirm stable norms; capture cluster distribution.
3. Similarity Eval (N≥1000)
   - Compute: top-5, top-10 accuracy, MRR, intra/inter cosine, separation.
   - Target improvement: separation > 0.05 (initial goal) vs current ~0.015.
4. Regression Guard
   - Add assertion: if any encoder output row norm < 1e-9 or NaN → raise & log failing glyph id.

24.5 Risks & Mitigations
| Risk | Impact | Mitigation |
|------|--------|------------|
| Positional encoding mismatch | NaNs / wrong shapes | Load exact hierarchical config before state dict; never force one-stage for hierarchical ckpt. |
| Truncation of long subpaths | Lost detail | Log truncated token counts; future dynamic splitting if needed. |
| Small label clusters | Depressed top-k accuracy | Filter by min_cluster or upsample fonts per label for eval. |
| Ignoring arcs | Slight distribution shift | Keep 'a' slot reserved; optionally implement arc handling later. |
| Relative args absence (if config toggled) | Lower geometric invariance | Gate relative encoding behind config flag and add later if required. |

24.6 Implementation Order (Actionable)
1. Add faithful svgtensor builder module.
2. Add packer and batch collation for (N,G,S).
3. Refactor encoder wrapper (remove fallback, enforce encode-only path, disable VAE).
4. Integrate into run_embed via new flag --faithful-hier (default ON once stable).
5. Smoke test (N=32) log: token utilization, norms.
6. Scale test (N=512) collect metrics.
7. Full run (N≥1000) + similarity_eval (min_cluster=3).
8. Update progress.md with baseline metrics table.
9. Remove interim one-stage & legacy simplified builder from default path (leave behind flag).

24.7 Success Criteria (Baseline)
- 0 NaN embeddings across >1000 glyphs.
- Intra − Inter cosine separation ≥ 0.05 (stretch goal ≥ 0.08).
- Top-5 accuracy > random baseline (measure baseline random ~ 1/(avg cluster size -1)).
- Stable encode throughput (≤ 2s per 512 glyphs CPU baseline acceptable).

24.8 Post-Baseline Enhancements (Deferred)
- Implement arcs (‘a’): parameter extraction, quantization, mask.
- Relative argument encoding path.
- Contrastive fine-tuning head; downstream retrieval improvement.
- Projection head + metric learning (triplet / InfoNCE).
- Batch-level caching (reuse parsed SVGTensors).
- [ ] Analyze `qCurveTo` payload variants (sample 500).
- [ ] Implement quadratic → cubic conversion utility.
- [ ] Add conversion flag for debugging (`--no-qcubic`).
- [ ] Implement `FontMetricsCache`.
- [ ] Extend embedding script to store normalization metadata beside embeddings.
- [ ] Compare embedding neighborhood stability between normalization modes.


