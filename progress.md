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
- [ ] Analyze `qCurveTo` payload variants (sample 500).
- [ ] Implement quadratic → cubic conversion utility.
- [ ] Add conversion flag for debugging (`--no-qcubic`).
- [ ] Implement `FontMetricsCache`.
- [ ] Extend embedding script to store normalization metadata beside embeddings.
- [ ] Compare embedding neighborhood stability between normalization modes.


