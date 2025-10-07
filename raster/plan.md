# Raster Glyph Embedding Project – Plan (LeViT Integration)

## 1. Purpose & Rationale

The vector-only DeepSVG encoder approach produced near-isotropic embeddings with negligible fine-label discrimination. We pivot to a raster-based pipeline to:
- Capture subtle shape, stroke thickness, contour topology, hinting differences that the pretrained vector encoder flattened.
- Rapidly experiment with CPU-efficient backbones (starting with LeViT) for discriminative glyph retrieval.
- Produce stable 128‑D (or 256‑D) embeddings suitable for similarity search, clustering, and downstream indexing.

We will keep this project intentionally **compact**, avoiding a sprawling CLI or over-generalized abstractions. Initial target: a clean baseline that demonstrates >3–5× improvement in fine-label effect size over the current vector baseline (which hovers around ~0 effect size).

---

## 2. High-Level Goals (Phase 1)

| Goal | Description | Acceptance |
|------|-------------|------------|
| G1 | Rasterize glyph contours to normalized grayscale images | Deterministic 128×128 outputs; reproducible hashing |
| G2 | Train LeViT_128S (or 128) backbone + embedding head | Converges in <40 epochs on CPU/GPU |
| G3 | Produce 128-D normalized embeddings | L2 norms ≈1.0 |
| G4 | Achieve fine-label effect size >= 0.03 (vs ~0.00 baseline) | similarity_eval adaptation passes |
| G5 | Implement minimal indexing (optional) | Embedding export + (optional) FAISS/Annoy index script |
| G6 | Clear documentation & deterministic config | Single `plan.md` (this file) + concise README section |

Stretch (Phase 2+):
- ArcFace / CosFace margin variants
- Hard negative mining
- Quantization (INT8) for CPU acceleration
- Multi-resolution fusion (64 + 128)

---

## 3. Scope (In vs Out)

IN (Phase 1):
- Rasterization from existing glyph contour DB (`glyphs.db`)
- LeViT integration (pretrained or random init)
- Simple training script (`train.py`) with minimal arguments
- Embedding extraction (`embed.py`)
- Basic evaluation adapter calling existing similarity evaluator (or a trimmed internal version)
- Deterministic seed control

OUT (Phase 1):
- Augmentation hyper-parameter sweeps
- Distributed training
- Advanced curriculum schedules (keep simple classification + optional metric head)
- Full-blown experiment tracking (just JSON logs)

---

## 4. Directory Layout (Proposed Minimal)

```
raster/
  plan.md                # This file
  README.md              # Short quick-start (generated after baseline)
  rasterize.py           # Convert contour rows -> PNG tensors
  dataset.py             # PyTorch Dataset wrapper
  transforms.py          # Domain-safe augmentations
  model.py               # LeViT backbone wrapper + embed head
  train.py               # Minimal training loop (classification + embedding)
  embed.py               # Load trained model -> produce embeddings
  eval_similarity.py     # Adapter to existing similarity evaluation
  utils.py               # Seed, logging helpers
  checkpoints/           # Saved model weights
  artifacts/             # Embeddings, logs
```

We will NOT mirror the complexity of the prior `src/scripts` tree.

---

## 5. LeViT Integration Strategy

This section is now grounded directly in the inspected `LeViT/levit.py`, `engine.py`, and `losses.py` sources. We avoid undocumented assumptions.

### 5.1 Canonical Architecture Facts (From Source)
- Specification entry for `LeViT_128S` (in `levit.py`):
  - `C = "128_256_384"` → stage embedding dims = [128, 256, 384]
  - `D = 16` → key (per-head) dimension
  - `N = "4_6_8"` → number of attention/MLP residual *pairs* per stage (see depth handling)
  - `X = "2_3_4"` → depth list actually consumed as `depth=[2,3,4]` (count of residual blocks expansions per stage; each depth unit expands into Attention + optional MLP residual sequences)
  - `drop_path = 0` (no stochastic depth in 128S)
  - Distillation default = True in factory; we will disable for simplicity.
- Hybrid convolutional patch embedding (`b16(embed_dim[0], ...)`) is a 4-layer conv stack with strides (2,2,2,2). Effective patch size = 16 (input spatial resolution / 16 becomes initial token resolution).
- Token resolution progression for an input of size `img_size`:
  - After hybrid patch embed: `(img_size / 16) x (img_size / 16)` tokens.
  - Two downsampling stages via `AttentionSubsample` (stride=2 each) → resolution halves twice.
  - For original 224: 224/16 = 14 → 14² tokens → subsample → 7² → subsample → 4² (integer ceiling logic matches code).
  - For our planned 128: 128/16 = 8 → 8² → 4² → 2² tokens (final 4 tokens averaged).
- Each stage sequence:
  - Repeated blocks: `Residual(Attention)` then (if `mlp_ratio > 0`) `Residual(MLP)` where MLP implemented with two `Linear_BN` layers and activation (Hardswish).
  - Attention bias: Learned relative position biases constructed by enumerating spatial offsets (`attention_biases` + `attention_bias_idxs`).
  - Downsampling block: `AttentionSubsample` reduces spatial tokens and projects dimension to next stage embedding dim.

### 5.2 Distillation Mechanics (Source-Based)
- If `distillation=True`, model forward returns a tuple `(head_logits, dist_head_logits)` and in eval mode averaged.
- Our usage: set `distillation=False` to simplify forward & avoid DistillationLoss wrapper.
- The distillation head is another `BN_Linear` of identical dimensionality; disabling it removes unnecessary parameters & branching.

### 5.3 Input Size Adaptation (Critical Detail)
- `model_factory` in `levit.py` hardcodes `img_size` implicitly via default `LeViT.__init__(img_size=224)`.
- Attention bias tables depend on `resolution = img_size // patch_size`.
- To use 128×128 glyph rasters **correctly**, we must instantiate `LeViT` directly (NOT call `model_factory`) with `img_size=128` so relative bias indices match our token grid (8×8 initial tokens).
- Therefore: we will implement a small local constructor:
  ```
  from LeViT.levit import LeViT, specification
  # replicate logic but override img_size
  ```
  or adapt `model_factory` logic inline with `img_size` parameter.

### 5.4 Grayscale Handling (Explicit Options)
| Approach | Change | Pros | Cons |
|----------|--------|------|------|
| Replicate 1→3 channels | Stack same tensor thrice | Zero code changes; can load ImageNet weights later | Redundant compute |
| Reinitialize first conv for 1 channel | Replace weight shape (3 → 1) averaging RGB filters if loading pretrained | Slightly faster; memory lower | Extra step when loading weights |
| Dual-channel (glyph + distance transform) (Phase 2) | Extend conv to 2 channels & init heuristically | Potential additional shape signal | Requires modification of state dict |

Phase 1: replicate to 3 channels (lowest risk). Document the adaptation so we can later average weights if we enable pretrained initialization.

### 5.5 Embedding Feature Dim (Verified)
- Final stage embedding dimension for 128S is 384 (the last element of `C` list).
- We will take pre-classifier features by intercepting after global mean over tokens (i.e., after `x = x.mean(1)` and BEFORE `self.head`).
- Classification head (BN_Linear) applies BatchNorm1d over 384 then Linear → `num_classes`.
- We will attach a separate embedding head producing a 128-D (configurable) L2-normalized vector; classification head can remain for supervised training.

### 5.6 Proposed Embedding Head (Refined)
```
Feature (B, 384)
 -> (optional) BatchNorm1d or LayerNorm (decide empirically; start None to avoid norm stacking with BN_Linear)
 -> Linear(384 -> 256)
 -> Hardswish (match backbone activation) OR GELU (we choose GELU for smoother tails)
 -> Dropout(0.1)
 -> Linear(256 -> 128)
 -> L2 normalize
```
Justification:
- Maintain small parameter footprint.
- 128-D is adequate for retrieval + efficient indexing; can scale later if under-separating.
- Avoid second BN to reduce train/inference variance complexity with glyph domain.

### 5.7 Loss Strategy (Phase 1 – No Assumptions)
- Primary: CrossEntropy on classification head (fine labels).
- No distillation, no mixup, no cutmix (not appropriate to distort small glyph strokes initially).
- Metric enhancement deferred until baseline separation confirmed.
- Early stopping heuristic (optional): monitor retrieval effect size every N epochs (N=5).

### 5.8 Removal / Avoidance of Unused Components
- Not using: `DistillationLoss`, mixup, cutmix, random erase, EMA, distributed sampler complexity.
- Not using: `drop_path` (stays at 0 for 128S per spec).
- Keep attention bias exactly as implemented (no pruning / fusion modifications).

### 5.9 Exact Token Count Path (For img_size=128)
| Stage | Spatial Tokens | Emb Dim | Operation |
|-------|----------------|---------|-----------|
| After hybrid conv (patch embed) | 8×8 = 64 | 128 | Conv stack stride (2,2,2,2) |
| Stage 1 residual blocks (depth=2) | 64 | 128 | Attention + MLP pairs |
| Downsample 1 (AttentionSubsample stride=2) | 4×4 = 16 | 256 | Dimension increases |
| Stage 2 residual blocks (depth=3) | 16 | 256 | Repeated |
| Downsample 2 (stride=2) | 2×2 = 4 | 384 | Final stage dim |
| Stage 3 residual blocks (depth=4) | 4 | 384 | Repeated |
| Global mean | 1 | 384 | Pooled embedding |
| Heads | 1 | #classes / 128 | Class logits + embedding |

### 5.10 FLOPs & Parameter Notes
- FLOPs counter logic embedded in construction uses original `img_size` to accumulate into `model.FLOPS`. For correctness with 128 size, ensuring instantiation with `img_size=128` also recalculates FLOPs for reporting (informational).
- We will log `model.FLOPS` after build for transparency.

### 5.11 Pretrained Weights (Phase 2 Considerations)
- If later enabling pretrained ImageNet weights with different `img_size`, we can still run at 128 — relative bias tables sized for original 14×14 initial token grid will mismatch if `img_size` differs. Thus: if using pretrained weights we should preserve `img_size=224` OR reconstruct attention bias mapping for new resolution (non-trivial). Decision: Phase 1 uses random init to avoid this complexity.

### 5.12 Implementation Steps (Updated)
1. Implement `build_levit_128s(img_size=128, distillation=False)` replicating `model_factory` logic with override.
2. Add feature extraction hook returning pooled 384-d tensor.
3. Add embedding head module.
4. Provide forward returning dict:
   - `{'embedding': emb_128, 'logits': class_logits}`.
5. Provide utility to save both classifier + embedding head weights.

### 5.13 Validation Checks
- Assert tensor shapes at each stage (optional debug mode).
- Verify attention bias index buffer shapes match expected token count (no runtime broadcast errors).
- Confirm final pooled feature mean variance not degenerate (std > 0.01 across batch early training).

### 5.14 Non-Assumptions Declaration
We explicitly DO NOT assume:
- Any hidden intermediate feature maps beyond those defined in the sequential `blocks`.
- Any CLS token (LeViT uses mean pooling, not a special token).
- Any positional embedding parameter (LeViT replaces with relative attention biases).
- Drop path being active (it is zero here).
- Distillation required for convergence (omitted).

### 5.15 Summary of Adjustments vs Upstream
| Aspect | Upstream Default | Our Baseline |
|--------|------------------|--------------|
| img_size | 224 | 128 |
| distillation | True | False |
| pretrained | Optional | Not used (Phase 1) |
| data aug | rich (mixup, cutmix, rand-erase) | minimal geometric/intensity jitter |
| loss | CE (+ distillation) | CE only |
| head(s) | BN_Linear (+ dist head) | BN_Linear + embedding head |
| patch embed | b16 conv stack | unchanged |
| token pooling | mean | unchanged |
| attention relative bias | enabled | unchanged |

This grounding ensures we modify only what is necessary for glyph retrieval while preserving architectural integrity.


---

## 6. Rasterization Pipeline

### 6.1 Input Source
- Use `glyphs.db` (already in repository under `dataset/`).
- Each glyph row contains contours (previously parsed for vector pipeline).

### 6.2 Normalization
1. Extract all contour points (after point interpolation for curves if necessary).
2. Compute bounding box (xmin, xmax, ymin, ymax).
3. Scale uniformly to fit into S×S (default S=128) with a margin (e.g., 4 px).
4. Translate to centered coordinates.
5. Y-axis orientation: Ensure consistent (flip if needed) so glyph baseline orientation is stable.

### 6.3 Rendering
- Approximate curves into polylines (reuse existing midpoint inference for quadratics).
- Use Pillow:
  - Create `L` mode image (grayscale).
  - Draw filled polygons for closed contours; strokes for open paths.
- Anti-aliasing:
  - Option A: Render at 256×256 then downsample to 128×128 (improves stroke smoothness).
- Save OR keep purely in-memory (prefer in-memory to avoid I/O overhead; optionally cache as `.pt` tensor dataset for reproducibility).

### 6.4 Augmentations (Phase 1 Minimal)
- Random translate (±2 px)
- Slight scale jitter (±5%)
- Light contrast/gamma jitter
- (Optional) very light Gaussian blur (prob 0.1)
- Avoid horizontal flip, rotation (affects semantic identity for many scripts)

---

## 7. Training Loop (Minimal)

Pseudo-flow:

```
for epoch in range(E):
  for batch in loader:
    imgs, labels = batch
    logits = model(imgs)['logits']
    loss = CE(logits, labels)
    if arcface_enabled:
        arc_logits = arc_head(embeddings, labels)
        loss = loss + arc_weight * arcface_loss(arc_logits, labels)
    backprop + optimizer.step()
  validate()
  save best checkpoint
```

Evaluation (epoch end):
- Sample subset (e.g., 5k glyph images).
- Extract embeddings.
- Run similarity evaluation (kNN accuracy + effect size).
- Log JSON: `artifacts/train_log.json`.

---

## 8. Minimal Configuration (Hard-Coded Defaults)

| Parameter | Default |
|-----------|---------|
| Image size | 128 |
| Batch size | 128 |
| Epochs | 30 |
| Optimizer | AdamW (lr=2e-3 head, 1e-3 backbone) |
| Weight decay | 0.05 |
| Scheduler | Cosine decay + 5% warmup |
| Embedding dim | 128 |
| Model variant | LeViT_128S |
| Loss | CE (+ optional ArcFace) |
| ArcFace margin/scale | 0.35 / 30 (off by default) |
| Augmentations | Basic jitter (enabled) |
| Seed | 42 |

Command interface:
```
python raster/train.py  # uses defaults
(optional flags: --epochs, --arcface, --out-dir)
```

---

## 9. Embedding Extraction

Script: `embed.py`
Process:
1. Load trained checkpoint.
2. Iterate dataset (optionally all glyph IDs).
3. Accumulate embeddings → `raster/artifacts/embeddings.pt`
4. Emit metadata JSONL: glyph_id, label, raster_norm parameters (scale, offset).
5. (Optional) Build FAISS/Annoy index (Phase 2).

---

## 10. Evaluation Integration

We can:
- Reuse existing `similarity_eval` script by pointing to new `embeddings.pt` + metadata (converted to expected format).
OR
- Provide `eval_similarity.py` wrapper that:
  - Loads embeddings + labels
  - Computes top-k retrieval & effect size (coarse/fine if coarse mapping available)

Acceptance (Phase 1):
- Run after training epoch final → store `raster/artifacts/similarity_report.json`.

---

## 11. Milestones & Timeline

| Milestone | Description | ETA |
|-----------|-------------|-----|
| M1 | Rasterization prototype (single glyph → PNG) | Day 1 |
| M2 | Batch raster dataset (in-memory or cached tensors) | Day 1 |
| M3 | LeViT wrapper + embedding head | Day 2 |
| M4 | Baseline training run (30 epochs) | Day 2 |
| M5 | Embedding extraction + similarity eval | Day 3 |
| M6 | ArcFace optional run (if baseline < target) | Day 4 |
| M7 | Documentation & cleanup | Day 4 |

---

## 12. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Poor fine-label separation persists | Project pivot needed | Add margin loss + augment strategy; escalate to deeper variant |
| Slow CPU inference | Delays indexing | Quantize (INT8) or prune layering (later) |
| Rasterization artifacts (aliasing) | Noise in embeddings | 2× supersample downscale |
| Label noise / near-duplicates | Artificial ceiling | Cluster-based relabel (Phase 2) |
| Memory (storing full raster set) | OOM in large sets | On-the-fly rendering with small cache window |

---

## 13. Future Enhancements (Phase 2+)

- Multi-resolution dual-stream (64 + 128 merge)
- Contrastive pretraining (augment pairs consistency loss)
- INT8 quantization benchmarking (Torch AO / ONNX Runtime CPU EP)
- Hard negative mining across batch embedding cache
- Coarse/fine multi-task heads
- Automated glyph style normalization (stroke thickness equalization)

---

## 14. Success Criteria (Exit Phase 1)

| Metric | Target |
|--------|--------|
| Training stability | No NaNs, steady loss decrease |
| Fine-label effect size | ≥ 0.03 |
| Top-10 Fine accuracy | ≥ 2× vector baseline relative improvement |
| Embedding norm consistency | Mean norm ≈ 1.0 ± 1e-3 |
| Inference latency (CPU single image) | < 5 ms (target; optimize later if >10 ms) |

---

## 15. Implementation Order (Actionable Checklist)

1. `raster/rasterize.py`
   - Load glyph contours from `glyphs.db`
   - Render single & batch to tensors (torch.float32, [0,1])
   - Supersample optional flag (hard-coded True first pass)
2. `raster/dataset.py`
   - Dataset class wrapping rasterize-on-access with small LRU cache
3. `raster/model.py`
   - Import LeViT (from `../LeViT/levit.py`)
   - Wrap & add embedding head (distillation=False)
4. `raster/train.py`
   - Seed, dataset splits, training loop, CE loss
   - Basic JSON logging
   - Save `checkpoints/best.pt`
5. `raster/embed.py`
   - Load best checkpoint → full dataset embeddings
6. `raster/eval_similarity.py`
   - Minimal internal retrieval (reuse existing structure if convenient)
7. Verify metrics & adjust (optionally enable ArcFace)
8. Write `README.md` quick-start

---

## 16. Minimal Usage Examples (Planned)

(After implementation)

```
# Train (defaults)
python raster/train.py

# Extract embeddings
python raster/embed.py

# Evaluate retrieval
python raster/eval_similarity.py
```

---

## 17. Open Questions (To Revisit)

| Question | Decision Pending |
|----------|------------------|
| Use cached raster .pt vs on-the-fly for full 30K? | Start on-the-fly; measure |
| Include ArcFace in baseline? | Only if CE baseline < target |
| Quantization early? | Defer until Phase 2 |
| Multi-channel enhancements (distance transforms)? | Possibly augment with distance map channel later |

---

## 18. Immediate Next Actions

1. Implement rasterization utility with supersampling + normalization.
2. Prototype dataset + single forward through LeViT_128S (random weights) to confirm shapes.
3. Add embedding head & confirm output dim=128, L2-normalized.
4. Launch baseline training run (CE only).
5. Evaluate early epoch (5) embeddings for quick effect size sanity; adjust if totally flat.

---

## 19. Notes on LeViT Repo Integration

- We will treat `LeViT/levit.py` as a vendored dependency (no modifications unless strictly necessary).
- Access via relative import: `from LeViT.levit import LeViT_128S`
- Set `distillation=False` in our wrapper to simplify forward path.
- Pretrained weights (optional) can be fetched using their URLs (Phase 2; Phase 1 uses random init to avoid bias from ImageNet color semantics).

---

## 20. Exit Statement

This plan defines a contained raster embedding subsystem: focused, minimal, and metrics-driven. Once we confirm improved separation vs the vector baseline, we can iterate into margin losses, quantization, and multi-resolution expansions.

## 21. Implementation Status (Updated)

| Item | Path / Component | Status | Notes |
|------|------------------|--------|-------|
| 1 | `raster/rasterize.py` | DONE | On-the-fly rendering, supersampling, CLI export implemented. |
| 2 | `raster/dataset.py` | DONE | Dataset + light augmentations + LRU raster cache + split utility. |
| 3 | `raster/model.py` | DONE | Custom LeViT_128S (img_size=128) wrapper + embedding head + checkpoint I/O. |
| 4 | `raster/train.py` | DONE | Minimal training loop (CE), cosine LR, retrieval metrics, JSONL logging. |
| 5 | `raster/embed.py` | DONE | Batch embedding extraction + metadata JSONL + optional label vocab. |
| 6 | `raster/eval_similarity.py` | DONE | Lightweight retrieval/effect size evaluation (full or chunked). |
| 7 | `raster/visualize.py` | DONE | Grid visualization with optional contour polyline overlay. |
| 8 | Metrics verification pass | PENDING | To run after first baseline training completes. |
| 9 | `README.md` quick-start | PENDING | Will summarize end‑to‑end usage (train → embed → eval → visualize). |
| 10 | ArcFace / margin extension | BACKLOG | Activate only if baseline effect size < target. |

### Immediate Next Steps

#### Large-Scale Training Progress & Next Actions (Update)
We have:
- Implemented heuristic font metric fallback (fixed tiny + shifted glyph sizing).
- Added bottom-left font hash annotation in visualization.
- Added label frequency filtering (min_label_count / drop_singletons).
- Implemented pre-rasterization (in-memory + memmap) with reuse + parallel thread workers.
- Added full resume (model + optimizer state), epoch offset display.
- Added retrieval debug metrics (effect, diff, gap).
- Implemented ArcFace-style angular margin (optional).
- Added warning suppression, gradient clipping, configurable retrieval cadence.
- Implemented memmap reuse + clone/adopt datasets to avoid duplicate pre-raster builds.

Readiness for full 528k run:
- Memory: uint8 preraster ≈ 8.7 GB fits easily in 384 GB RAM (E48ds_v4).
- Parallelism: Use modest DataLoader workers (I/O light after pre-raster). Most CPU goes to model matmuls (set OMP/MKL threads).
- Retrieval: Cap to 8k samples to avoid excessive NxN similarity cost.
- Resume: Safe to recover from interruptions (best + optimizer state).
- ArcFace: Optional; start with margin=0.15; disable if early instability.

Recommended environment (shell):
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export PYTHONWARNINGS=ignore

Full 528k command (baseline + margin):
python -m raster.train \
  --db dataset/glyphs.db \
  --limit 528000 \
  --epochs 45 \
  --batch-size 256 \
  --val-frac 0.1 \
  --num-workers 4 \
  --pre-rasterize \
  --pre-raster-mmap \
  --pre-raster-workers 16 \
  --pre-raster-mmap-path preraster_full_528k_u8.dat \
  --min-label-count 2 \
  --drop-singletons \
  --warmup-frac 0.05 \
  --retrieval-cap 8000 \
  --retrieval-every 5 \
  --grad-clip 1.0 \
  --arcface-margin 0.15 \
  --arcface-scale 32 \
  --save-every 5 \
  --suppress-warnings

If you want a “no-margin” baseline first (recommended to quantify uplift), run the same command with:
  --arcface-margin 0.0

Early epoch monitoring checklist (first 5 epochs):
- train_loss: steady decrease (no spikes after warmup).
- val_loss: slight wobble then downward trend.
- val_acc: rising above (1 / num_classes) * 10 within first 8–10 epochs.
- effect_size: should become >0.25 by mid-run; if flat + val_acc rising, consider increasing margin later.
- grad_norm: stable (not exploding; ~3–8 typical early, then taper).

Next possible enhancements (post 528k baseline):
1. ANN retrieval (FAISS) for larger validation sampling.
2. Hard negative mining (adaptive retrieval pairs for margin head).
3. Dual-resolution (128 + 64) stacked channels.
4. True open-set rejection threshold calibration (distribution modeling on inter-class cosine).
5. Persist preraster metadata signature file (skip accidental rebuild).
6. Batch-level adaptive LR or cosine restart if plateau.

Success criteria for large-scale run:
- Font-disjoint val_acc > baseline small-run ratio scaled (target: >0.22 if 16–17% at 16k).
- Effect size ≥ 0.45 without margin, ≥ 0.52 with margin (targets).
- Stable training (no divergence, no NaNs) through epoch 45.
- Retrieval diff (mu_intra - mu_inter) trend plateauing late but not decreasing.

(Existing task list below retained.)
NOTE: For ephemeral Python diagnostics or quick inspections, invoke inline scripts as:
python -c '...entire script here...'
Ensure the whole code (including newlines if any) is inside a single pair of single quotes and avoid using unescaped single quotes inside the body. This is required for the current shell environment and replaces prior here-doc style attempts.
1. Run a baseline training epoch set (e.g. 30 epochs) and record:
   - `val_accuracy`, `topk_accuracy`, `effect_size` from `train.py`.
2. Extract embeddings (`embed.py`) and re-run `raster/eval_similarity.py` on full (or capped) set to confirm metrics match in-training retrieval sample.
3. Visual sanity check:
   - `visualize.py --db dataset/glyphs.db --limit 48 --out raster/artifacts/vis_grid.png --overlay`
4. If effect size < 0.03:
   - Increase epochs modestly (e.g. +10) or enable margin head (future patch).
5. Add concise `README.md` with quick-start commands referencing implemented scripts.
6. Implement even-odd / nesting-aware hole algorithm (`hole_strategy="even-odd"`) to correctly render multi-level counters (holes within holes).
7. Extend embedding extraction metadata: include original glyph bounding box (pre-normalization) and applied scale factor per glyph in metadata JSONL.

### Deviation Log
- Added `visualize.py` (was not in original checklist) to derisk raster quality early.
- Retrieval evaluation integrated directly into `train.py` (initially planned as post-process only).
- Kept augmentation scope conservative; no random erase / mixup introduced as originally (correctly) scoped out.

## 22. Rasterization Remediation (New)

Recent Issues Identified:
1. Holes not preserved: Inner counters (e.g., ‘o’, ‘a’, Arabic enclosed shapes) were filled.
2. Size normalization obliterated relative EM scale: Small diacritics / marks were scaled up to fill the square, losing semantic size cues.
3. Some glyphs drawn outside grid: Occasional coordinates exceeded [-1,1] mapping or post-normalization expansion caused clipping / spill.

Root Causes:
- (1) Legacy fill pipeline treated every closed polyline as a solid polygon (no winding / orientation analysis).
- (2) Tight unit scaling (`scale_to_unit=True`) applied uniformly; all glyphs forced to target_range, eliminating natural size relationships.
- (3) Extreme aspect / sparse shapes plus tight scaling and lack of clipping led to coordinates mapping outside image bounds.

Implemented Fixes (Code Updates):
- Hole Preservation: Added orientation-based solid/hole classification (`hole_strategy="orientation"`). Dominant cumulative signed-area orientation taken as solid; opposite orientation polygons filled with background to carve holes.
- Scale Policy Toggle: Introduced `RasterizerConfig.fit_mode`:
  - `"tight"` (default legacy) → scales to target_range.
  - `"preserve"` → bypasses scaling (retains original relative size; only centers).
  Internally maps to conditional `scale_to_unit` flag.
- Out-of-Bounds Control: Added `clip_out_of_bounds=True` to clamp post-transform pixel coordinates into raster frame.
- Extended `RasterizerConfig` fields:
  - `hole_strategy`
  - `fit_mode`
  - `clip_out_of_bounds`
- Updated rendering path to:
  - Pre-classify closed vs open polylines.
  - Compute signed area per closed polygon.
  - Separate filling (solids) from hole carving.
  - Optionally stroke open / closed paths (unchanged logic for strokes).
  
Remaining Work (Targeted) (Updated):
- Even-odd / depth parity hole algorithm (DONE) via `hole_strategy="even-odd"` (XOR parity fill) – supports arbitrary nesting depth.
- Embedding metadata enrichment (DONE) in `embed.py`: per-glyph `bbox_orig`, `scale_factor`, `fit_mode`.
- Regression glyph set (PENDING): include multi‑nested counters, thin strokes, large EM extents, near-open contours to guard against parity / closure regressions.
- Preserve Mode Refinement (UPDATED):
  * Phase 1: conditional down-scale only (retain small glyph size). Phase 2 (current): DB font metrics + adaptive compensation for short baseline glyphs.
  * DB font metrics integration (DONE): using `typo_ascent/typo_descent` (fallback `ascent/descent`) + UPEM normalization for cross-font comparability.
  * Added ratio-based compensation: glyph height ratio (glyph_h / (ascent+descent)) drives a gentle up-scale for mid-height baseline glyphs (e.g. Arabic ز) while preserving very small diacritics (<0.20 ratio) and tall glyphs.
  * Vertical centering tweak for short baseline glyphs: partial downward shift to reduce apparent “floating” when metric span is much larger than glyph bbox.
- Path Closure Artifacts (OBSERVED, FIX APPLIED): small epsilon auto-closing; pending regression set to ensure no over-closing.
  * Future: make `closure_eps` configurable in `RasterizerConfig`.
- Default Consistency Audit (UPDATED):
  * `DatasetConfig.fit_mode` default = "tight"; preserve mode now uses DB metrics + compensation logic automatically.
  * `RasterizerConfig.fit_mode` default previously diverged; unify behavior in future cleanup (TODO).
  * `RasterizerConfig.hole_strategy` default = "even-odd"; document in README.
- Scaling Logic Traceability (PENDING):
  * Record `overflow_downscale_applied: bool` alongside `scale_factor` to disambiguate neutral (1.0) scale due to small glyph vs preserved size after explicit disable.
- README / Usage Update (PENDING): Document new flags / semantics for hole strategy and preserve mode, plus metadata fields in embeddings.

Validation / Immediate Observations (Updated):
- Orientation vs even-odd parity matches for simple single-hole glyphs; parity succeeds on synthetic triple-nested ring test where orientation heuristic fails.
- Preserve mode now leaves small diacritics visually smaller; large glyphs no longer clip after conditional downscale.
- Ratio compensation improved size perception for short Arabic baseline glyphs (e.g. ز) without inflating diacritics; still monitoring extreme outliers (very tall ornamental ligatures).
- Grid visualization now shows counters and improved vertical positioning; gray wireframe outlines removed for cleaner inspection.
- Diacritic marks remain small (<0.20 ratio threshold) as intended (no compensation applied).
- Overflow clamp (vertical + horizontal) prevents clipping; metadata now records `comp_scale`, `ratio_compensation`, and `center_shift`.
- Overlays align with raster fills (polyline sampling unaffected).

Remaining / Open Edge Cases:
- Multi-level nesting (hole inside hole → solid island) not explicitly reconstituted; current orientation heuristic handles simple two-level but may invert if fonts mix winding conventions inconsistently.
- Glyph sets mixing clockwise / counter-clockwise conventions inconsistently (e.g., some fonts export all contours same winding + explicit hole markers omitted) could still yield false positives/negatives in hole carving.
- Very small diacritics under preserve mode may become near-single-pixel artifacts → optional minimum scale safety not yet implemented.

Next Steps (Prioritized):
1. Add nesting-aware hole resolution:
   - Sort closed polygons by absolute area descending.
   - Build containment tree (point-in-polygon or bbox prefilter + winding).
   - Alternate fill/background by depth parity (classic even-odd).
2. Implement auto-minimum visual size option:
   - If preserve-mode glyph’s bbox < X% of grid, scale up with scale factor logged (metadata: scaled_preserve_factor).
3. Performance Profiling:
   - Benchmark orientation + potential nesting logic on full 30K (cache parsed contours).
4. Contour Cache Layer:
   - Hash raw JSON → parsed & normalized (fit-mode–aware) to avoid re-parsing in repeated training epochs.
5. Metadata Extension:
   - Record original bbox (pre- + post-normalization) and applied scale factor.
6. Add CLI flags:
   - `--fit-mode {tight,preserve}`
   - `--min-preserve-scale` (future)
   - `--hole-strategy {orientation,even-odd,none}`
7. Regression Test Suite:
   - Curate small set (e.g., glyphs with: single hole, nested holes, diacritic, wide glyph, tall glyph) → golden PNG comparison.

Updated Success Criteria Additions:
| Criterion | Target |
|-----------|--------|
| Hole fidelity | 0 false fills on curated hole regression set |
| Relative size preservation (preserve mode) | Height ratio variance matches source bbox variance (±5%) |
| Out-of-bounds incidents | 0 clipped strokes in regression sample (visual diff) |
| Optional minimum scale logic | Not applied unless explicitly enabled (no silent scaling) |

Plan Adjustments:
- Section 5 (integration) implicitly assumes tight scaling; update doc after nesting logic (future).
- Section 6 normalization steps to include alternative flow for preserve mode (defer until nesting done).
- Section 21 status updated implicitly by this section; future table row to mark “Hole/Nesting Enhancements” once parity-based algorithm lands.

Immediate Action Items:
- [ ] Implement even-odd / depth parity fallback for multi-nested contours.
- [ ] Add regression glyph set + script (`raster/tests/render_regression.py`).
- [ ] Extend `embed.py` to emit per-glyph original bbox + applied scale metadata for downstream analysis.
- [ ] Update README after confirming regression stability.

(End of Plan)