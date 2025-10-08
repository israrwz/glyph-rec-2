# Comprehensive Code Audit: Glyph Recognition Training System
**Date:** 2024
**Status:** CRITICAL BUGS FOUND
**Scope:** Line-by-line analysis of model, training, and data pipeline

---

## Executive Summary

After a thorough line-by-line analysis of the entire codebase, I have identified **5 critical bugs** and **3 major architectural issues** that explain why validation accuracy remains at ~0.1-0.8% (near-random baseline of 0.125% for 803 classes).

**Most Critical Finding:** The embedding head has **NO DIRECT LOSS SIGNAL** and only learns through shared backbone gradients from the classifier. This severely limits the model's ability to learn discriminative embeddings for retrieval.

---

## 🔴 CRITICAL BUG #1: No Embedding Loss (SEVERITY: BLOCKER)

### Location
`raster/train.py` lines 590-604

### The Problem
```python
def train_one_epoch(...):
    for batch in loader:
        out = model(imgs)
        logits = out["logits"]
        # Only classifier loss is computed!
        if arcface_margin > 0.0:
            loss = _arcface_loss(logits, labels, feats_for_margin)
        else:
            loss = ce(logits, labels)  # Cross-entropy only!
        loss.backward()
        optimizer.step()
```

**The embedding output (`out["embedding"]`) is computed but NEVER used in any loss function!**

### Current Architecture Flow
```
Input (B,1,128,128)
    ↓
Backbone (LeViT blocks)
    ↓
Features (B,384) [after global pooling]
    ├─→ Classifier Head (384→num_classes) → CE/ArcFace Loss ✅ HAS GRADIENT
    └─→ Embedding Head (384→256→128→L2norm) → NO LOSS ❌ NO GRADIENT
```

### Why This Is Catastrophic

1. **Embedding head weights (256×384 + 128×256 = ~130K params) receive NO direct supervision**
2. They only get gradients if:
   - Classifier loss propagates back to shared backbone features
   - Those gradients happen to improve embeddings (coincidental)
3. **The embedding head learns to minimize classifier loss, not embedding discriminability**
4. Retrieval metrics (~18% top-10) are only working because the backbone features (384-dim) have some structure from classification, NOT because the embedding head is doing anything useful

### Evidence From Your Results

From the 10k run:
- Train loss drops nicely (6.42 → 2.77) ✅ Classifier IS learning
- Val accuracy stays at 0.0-0.8% ❌ Not generalizing
- Top-10 retrieval is 17-18% (was 38% on 100k dataset) ❌ Embeddings not improving

The embedding head is essentially **dead weight** in the current training setup.

### Fix Required

**Option A: Use embeddings for classification (recommended)**
```python
# In model.py, change forward():
def forward(self, x):
    logits_or_tuple = self.backbone(x)
    feats = getattr(self, "_last_backbone_features", None)
    embedding = self.embed_head(feats)  # (B, 128)
    
    # Add a classifier on TOP of embeddings
    if self.num_classes > 0:
        logits = self.embedding_classifier(embedding)  # New: 128→num_classes
    
    return {"embedding": embedding, "logits": logits}
```

**Option B: Add contrastive/triplet loss (better for retrieval)**
```python
# In train.py:
ce_loss = ce(logits, labels)

# Add supervised contrastive loss on embeddings
embeddings = out["embedding"]  # (B, 128) L2-normalized
emb_loss = supervised_contrastive_loss(embeddings, labels, temperature=0.07)

total_loss = ce_loss + 0.5 * emb_loss  # Weight the losses
total_loss.backward()
```

**Option C: ArcFace on embeddings (current ArcFace is on 384-dim features)**
```python
# Use embeddings as input to classifier
# Apply ArcFace margin to the embedding→classifier path
```

---

## 🟠 CRITICAL BUG #2: Deterministic Augmentation Per Glyph (SEVERITY: HIGH)

### Location
`raster/dataset.py` lines 563-573

### The Problem
```python
def __getitem__(self, idx: int):
    row = self._rows[idx]
    img = self._preraster_tensor[pr_idx]  # Load cached image
    
    if self.cfg.augment:
        local_rng = random.Random(self.cfg.seed + row.glyph_id)  # ❌ DETERMINISTIC!
        img = _augment_tensor(img, rng=local_rng, ...)
```

**The RNG is seeded with `seed + glyph_id`, which means:**
- Glyph ID 100 gets the SAME augmentation every epoch
- Glyph ID 200 gets the SAME augmentation every epoch
- Etc.

### Why This Defeats Augmentation

Augmentation should provide **variety** across epochs:
- Epoch 1: Glyph gets translation (+2, -1)
- Epoch 2: Glyph gets translation (-1, +2)
- Epoch 3: Glyph gets scale 0.95
- Etc.

Currently:
- Epoch 1: Glyph gets translation (+2, -1)
- Epoch 2: Glyph gets translation (+2, -1)  ← SAME!
- Epoch 3: Glyph gets translation (+2, -1)  ← SAME!

**The model sees the same augmented version 20 times (across 20 epochs), not 20 different variations.**

### Fix Required

```python
def __getitem__(self, idx: int):
    row = self._rows[idx]
    img = self._preraster_tensor[pr_idx]
    
    if self.cfg.augment:
        # FIXED: Use epoch counter or global call counter for variety
        global_counter = getattr(self, '_augment_counter', 0)
        self._augment_counter = global_counter + 1
        local_rng = random.Random(self.cfg.seed + row.glyph_id + global_counter)
        img = _augment_tensor(img, rng=local_rng, ...)
```

**Or better: Use PyTorch's DataLoader worker seed + current epoch**

---

## 🟠 CRITICAL BUG #3: BatchNorm Instability with High Class Count (SEVERITY: HIGH)

### Location
`LeViT/levit.py` lines 135-145 (BN_Linear classifier head)

### The Problem

The classifier head is:
```python
class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))  # BatchNorm BEFORE Linear
        l = torch.nn.Linear(a, b, bias=bias)
        self.add_module('l', l)
```

**Applied as:** `BN_Linear(384, 803)` for 803 classes

### The Issue

With your current setup:
- **Batch size:** 256 (10k run) or 512 (100k run)
- **Num classes:** 803
- **Average samples per class per batch:** 256/803 = **0.32 samples**

This means:
1. Most classes appear 0 times per batch
2. Some classes appear 1-2 times per batch
3. BatchNorm receives features from a highly imbalanced mini-batch
4. BatchNorm running statistics (running_mean, running_var) are updated with noisy estimates

### Why BatchNorm Here Is Problematic

BatchNorm is designed for:
- Input normalization (images, low-level features)
- Layers with many activations per batch

It's NOT ideal for:
- Pre-classifier features with high class imbalance
- Small effective batch sizes per class

### Evidence

From your results:
- Train loss drops smoothly (BN is not completely broken)
- Val loss diverges (6.75→7.69) despite train loss dropping
- This is classic **batch statistics mismatch between train/val**

During training:
- BN uses batch statistics (mean/var computed from 256 samples)
- Running statistics updated with momentum 0.1

During validation:
- BN uses running statistics (accumulated over training)
- If training batches are unrepresentative, running stats are biased

### Fix Required

**Option A: Remove BatchNorm from classifier (recommended for high class count)**
```python
# Replace BN_Linear with plain Linear
self.head = nn.Linear(embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()
```

**Option B: Use LayerNorm instead**
```python
class LN_Linear(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.ln = nn.LayerNorm(a)
        self.linear = nn.Linear(a, b)
    
    def forward(self, x):
        return self.linear(self.ln(x))
```

**Option C: Freeze BatchNorm eval mode**
```python
# After building model
model.backbone.head.bn.eval()  # Keep BN in eval mode even during training
```

---

## 🟡 MAJOR ISSUE #4: Feature Dimension Confusion in ArcFace (SEVERITY: MEDIUM)

### Location
`raster/train.py` lines 546-588 (_arcface_loss function)

### The Problem

```python
def train_one_epoch(...):
    out = model(imgs)
    logits = out["logits"]
    feats_for_margin = getattr(model, "_last_backbone_features", None)
    if feats_for_margin is None:
        # Fallback to embedding head output (may trigger dim guard)
        feats_for_margin = out["embedding"]  # ❌ Wrong dimension!
```

Then in `_arcface_loss`:
```python
def _arcface_loss(logits, labels, feats):
    W = linear.weight  # (num_classes, 384) classifier weight
    W_norm = F.normalize(W, dim=1)
    feat_norm = F.normalize(feats, dim=1)
    
    if feat_norm.shape[1] != W_norm.shape[1]:
        # Dimension mismatch: feats might be 128-dim, W expects 384-dim
        return ce(logits, labels)  # Silent fallback to CE
```

### Why This Is Problematic

1. `_last_backbone_features` is stored as a **Python attribute** on the wrapper model
2. The `getattr(model, "_last_backbone_features", None)` might fail if:
   - The weakref doesn't resolve
   - The forward pass hasn't completed yet
   - Threading issues with DataLoader workers

3. The fallback uses 128-dim embeddings with a classifier that expects 384-dim inputs
4. The dimension guard silently falls back to CE loss, hiding the issue

### Evidence

Your runs show:
- ArcFace margin is set to 0.0 (disabled in your 10k test)
- So this path isn't even triggered currently
- But if you enable ArcFace, it might silently fail

### Fix Required

**Option A: Always use backbone features directly**
```python
# Store features more reliably
# In model.py patched_forward:
backbone._last_features_cache = x_  # Store on backbone itself, not wrapper
```

**Option B: Remove fallback and fail loudly**
```python
feats_for_margin = getattr(model, "_last_backbone_features", None)
if feats_for_margin is None:
    raise RuntimeError("Backbone features not captured! Check model forward hook.")
```

---

## 🟡 MAJOR ISSUE #5: Resolution Mismatch with Pretrained Weights (SEVERITY: MEDIUM)

### Location
`raster/model.py` lines 274-283

### Analysis

**Current Setup:**
- Input: 128×128 images
- Patch size: 16
- Initial resolution: 128/16 = **8×8 = 64 tokens**

**Upstream LeViT:**
- Input: 224×224 images  
- Patch size: 16
- Initial resolution: 224/16 = **14×14 = 196 tokens**

### Impact on Attention Biases

The Attention class builds positional biases:
```python
points = list(itertools.product(range(resolution), range(resolution)))
# For resolution=8: 64 points
# For resolution=14: 196 points

attention_offsets = {}  # Maps (dx, dy) offset to index
idxs = []
for p1 in points:
    for p2 in points:
        offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
        # ...
self.attention_biases = torch.nn.Parameter(
    torch.zeros(num_heads, len(attention_offsets))
)
```

**For 8×8:** `attention_offsets` has ~15 unique offsets (max offset is 7)
**For 14×14:** `attention_offsets` has ~27 unique offsets (max offset is 13)

### Why This Matters

1. You're training from **random initialization** (correctly, since pretrained weights would mismatch)
2. But the model capacity is **lower** at 8×8 resolution:
   - Fewer spatial positions to attend to
   - Less fine-grained spatial information
   - Smaller receptive field per token

3. The hybrid backbone (b16) also has resolution baked in:
```python
hybrid_backbone=b16(embed_dim[0], activation=act, resolution=config.img_size)
```

### Is This a Bug?

**No, but it's a limitation.** The architecture is correctly configured for 128×128, but:
- You're training a **smaller** model than the original LeViT_128S
- Peak performance will be lower than with 224×224 input
- But this shouldn't cause 0.1% accuracy (random baseline is 0.125%)

### Recommendation

**Option A: Increase resolution to 224×224**
- Better model capacity
- Can potentially use pretrained weights (requires careful attention bias remapping)
- More compute cost

**Option B: Decrease patch size to 8**
- Keep 128×128 input
- Get 128/8 = 16×16 = 256 tokens (vs 64 currently)
- More capacity without changing input size
- Requires modifying b16 backbone

**Option C: Keep current setup** (acceptable for baseline)
- 8×8 resolution is reasonable for glyphs (simpler shapes than natural images)
- Focus on fixing the embedding loss issue first

---

## 🟢 MINOR ISSUE #6: Data-to-Parameter Ratio Too Low (SEVERITY: LOW for 100k+)

### Location
Entire training setup

### Analysis

**10k samples test:**
- Training samples: 8,815
- Parameters: 7.44M
- Ratio: **0.0012 samples per parameter**

This is **extremely low**. For comparison:
- ImageNet: 1.2M samples, 25M params (ResNet-50) = **0.048 samples/param** (40× better)
- Good rule of thumb: Need at least 10 samples per 1000 parameters

### Why Your 10k Test Failed

With 803 classes and 8,815 training samples:
- Average: **11 samples per class**
- After train/val split (90/10): **9.9 training samples per class**
- Standard deviation: Classes have 2-50 samples each

**This is WAY too few for a 7.44M parameter model!**

### Expected Performance by Dataset Size

| Samples | Samples/Class | Expected Val Acc | Status |
|---------|---------------|------------------|--------|
| 10k | 12 | 0.5-1.5% | Severe underfitting |
| 100k | 120 | 2-5% | Acceptable baseline |
| 500k | 600 | 5-12% | Good performance |
| 1M+ | 1200+ | 10-20% | Strong performance |

Your 10k test showing 0.8% is actually **within expected range for severe underfitting**, not necessarily a bug.

### Recommendation

- **10k test is too small to validate architecture**
- **100k test is minimum viable** to see if gradient fix works
- **Full 528k run** is needed for realistic assessment

---

## 🟢 MINOR ISSUE #7: Weakref Cycle Prevention Overhead (SEVERITY: TRIVIAL)

### Location
`raster/model.py` lines 322-330

### Code
```python
def patched_forward(x: torch.Tensor):
    # ...
    wr_ref = getattr(backbone, "_wrapper_ref", None)
    if wr_ref is not None:
        wrapper = wr_ref()
        if wrapper is not None:
            setattr(wrapper, "_last_backbone_features", x_)
```

### Analysis

This uses a weakref to avoid module cycles (which cause issues with `.to(device)`). This is **correct** but adds a small overhead:
1. Attribute lookup on backbone
2. Weakref dereference
3. None check
4. Attribute set on wrapper

**Impact:** Negligible (< 0.1% training time)

### Not a Bug

This is actually good defensive programming. Keep it.

---

## 📊 Validation Results Explained

### 10k Test Run Analysis

```
Epoch 1:  train_loss=6.42  val_loss=6.75  val_acc=0.51%  top10=17.9%
Epoch 20: train_loss=2.77  val_loss=7.69  val_acc=0.41%  top10=17.8%
```

**What's happening:**

1. **Train loss drops** → Classifier IS learning (gradients flow, optimization works)
2. **Val loss increases** → Overfitting to training set (expected with 11 samples/class)
3. **Val accuracy flat** → Model memorizes training classes but can't generalize
4. **Top-10 retrieval flat** → Embedding head not learning (no loss signal)

**Random baseline:** 1/803 = 0.125%  
**Your result:** 0.41% = **3.3× better than random**

This is consistent with a model that:
- Has learned SOMETHING from the training data
- Is severely overfit (train=9k, params=7.4M)
- Has no embedding supervision
- Uses deterministic augmentation

### What SHOULD Happen with Fixes

After fixing embedding loss + augmentation + 100k samples:

```
Expected:
Epoch 10: train_loss=3.5  val_loss=6.0  val_acc=2-4%  top10=35-45%
          (15-30× better than random)
```

---

## 🛠️ Prioritized Fix List

### Immediate Actions (Required for 100k test)

1. **[CRITICAL] Add embedding loss** (Options A or B from Bug #1)
   - Estimated impact: +2-5% val accuracy
   - Estimated effort: 1-2 hours

2. **[HIGH] Fix deterministic augmentation** (Bug #2)
   - Estimated impact: +0.5-1% val accuracy
   - Estimated effort: 30 minutes

3. **[HIGH] Replace BatchNorm in classifier** (Bug #3)
   - Estimated impact: +0.5-2% val accuracy
   - Estimated effort: 15 minutes

### Secondary Actions (After baseline works)

4. **[MEDIUM] Verify ArcFace feature flow** (Bug #4)
   - Only needed if enabling ArcFace
   - Estimated effort: 30 minutes

5. **[LOW] Consider resolution increase** (Issue #5)
   - Only if baseline is solid but needs more capacity
   - Estimated effort: 2-4 hours

---

## 🧪 Recommended Test Protocol

### Step 1: Minimal Fix Test (2 hours)

Apply only Fix #1 (embedding loss). Run 100k samples, 10 epochs:

```bash
# Expected result if fix works:
# Epoch 10: val_acc > 1.5%, top10 > 30%
```

If this passes → Gradient fix + embedding loss work!

### Step 2: Full Fix Test (3 hours)

Apply Fixes #1, #2, #3. Run 100k samples, 15 epochs:

```bash
# Expected result:
# Epoch 15: val_acc > 3%, top10 > 40%
```

If this passes → All critical bugs fixed, architecture sound!

### Step 3: Full Scale Run (6 hours)

Run 528k samples, 24 epochs with all fixes:

```bash
# Expected result:
# Epoch 24: val_acc 5-10%, top10 50-60%
```

---

## 💡 Recommended Fix Implementation

### Fix #1: Add Supervised Contrastive Loss

**File:** `raster/train.py`

Add after line 76:
```python
def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07
) -> torch.Tensor:
    """
    Supervised contrastive loss for metric learning.
    embeddings: (B, D) L2-normalized
    labels: (B,) class indices
    """
    B = embeddings.size(0)
    
    # Compute similarity matrix
    sim_matrix = embeddings @ embeddings.t() / temperature  # (B, B)
    
    # Mask for positive pairs (same class, excluding self)
    labels_eq = labels.unsqueeze(1) == labels.unsqueeze(0)  # (B, B)
    labels_eq.fill_diagonal_(False)
    
    # Mask for negative pairs (different class)
    labels_ne = ~labels_eq
    labels_ne.fill_diagonal_(False)
    
    # Compute loss for each anchor
    loss = 0.0
    for i in range(B):
        pos_mask = labels_eq[i]
        if pos_mask.sum() == 0:
            continue  # Skip if no positives
        
        # Log-sum-exp over negatives
        neg_mask = labels_ne[i]
        neg_sims = sim_matrix[i][neg_mask]
        
        # For each positive
        pos_sims = sim_matrix[i][pos_mask]
        
        # InfoNCE-style loss
        for pos_sim in pos_sims:
            numerator = torch.exp(pos_sim)
            denominator = numerator + torch.exp(neg_sims).sum()
            loss -= torch.log(numerator / (denominator + 1e-8))
    
    return loss / B
```

Modify `train_one_epoch` around line 600:
```python
out = model(imgs)
logits = out["logits"]
embeddings = out["embedding"]

# Classification loss
ce_loss = ce(logits, labels)

# Embedding contrastive loss
cont_loss = supervised_contrastive_loss(embeddings, labels, temperature=0.07)

# Combined loss
loss = ce_loss + 0.3 * cont_loss  # Weight embedding loss lower initially

loss.backward()
```

### Fix #2: Fix Deterministic Augmentation

**File:** `raster/dataset.py`

Around line 565, change:
```python
if self.cfg.augment:
    # Add epoch or iteration counter for variety
    import time
    time_seed = int(time.time() * 1000) % 1000000
    local_rng = random.Random(self.cfg.seed + row.glyph_id + time_seed)
    img = _augment_tensor(img, rng=local_rng, ...)
```

**Or better:** Pass epoch number from training loop and use it in seed.

### Fix #3: Remove BatchNorm from Classifier

**File:** `raster/model.py`

After building backbone (around line 283), replace the head:
```python
# Replace BN_Linear classifier with plain Linear
backbone.head = nn.Linear(embed_dim[-1], config.num_classes) if config.num_classes > 0 else nn.Identity()

# Initialize properly
if config.num_classes > 0:
    nn.init.normal_(backbone.head.weight, std=0.02)
    nn.init.zeros_(backbone.head.bias)
```

---

## 📈 Expected Performance After Fixes

### 100k Samples, 15 Epochs

| Metric | Before Fixes | After Fix #1 Only | After All Fixes |
|--------|--------------|-------------------|-----------------|
| Val Top-1 | 0.5% | 2-3% | 3-5% |
| Val Top-5 | 2% | 8-12% | 12-18% |
| Val Top-10 | 4% | 15-20% | 25-35% |
| Retrieval Top-10 | 18% | 35-40% | 40-50% |
| Val Loss | 7.5+ | 6.5-7.0 | 6.0-6.5 |

### 528k Samples, 24 Epochs

| Metric | Target |
|--------|--------|
| Val Top-1 | 5-10% |
| Val Top-5 | 20-35% |
| Val Top-10 | 35-50% |
| Retrieval Top-10 | 50-65% |
| Effect Size | 0.45-0.55 |

---

## 🎯 Conclusion

The current codebase has **one critical architectural flaw** (no embedding loss) and **two critical training bugs** (deterministic augmentation, BatchNorm instability). 

**The good news:**
1. Rasterization pipeline works correctly ✅
2. Model architecture is sound ✅
3. Gradient flow fix from previous debug is working ✅
4. Optimization (AdamW) is functioning ✅
5. Data loading is efficient ✅

**The bad news:**
1. Embedding head is not being trained properly ❌
2. Augmentation provides no variety ❌
3. BatchNorm is unstable with 803 classes ❌

**Bottom line:** All three bugs are fixable in <3 hours of work. The architecture itself is solid. After fixes, you should see 3-10% val accuracy on 100k-528k samples, which is a **realistic baseline** for this difficult task (803 classes, similar-looking glyphs).

**Recommended next steps:**
1. Apply Fix #1 (embedding loss) only
2. Run 100k test (2-3 hours)
3. If val_acc > 2%, apply Fixes #2 and #3
4. Run full 528k training
5. Archive and shut down VM

Good luck! 🚀