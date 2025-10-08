# Quick Fix Implementation Guide
**Target:** Fix critical bugs before next training run
**Time Required:** 2-3 hours total
**Priority:** Fix #1 is MANDATORY. Fixes #2 and #3 are highly recommended.

---

## Fix #1: Add Embedding Loss (CRITICAL - 1.5 hours)

### Option A: Supervised Contrastive Loss (Recommended)

**File:** `raster/train.py`

**Step 1:** Add the loss function after line 76 (after imports):

```python
def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Supervised contrastive loss for L2-normalized embeddings.
    
    For each sample i:
    - Positives: all j with labels[j] == labels[i], j != i
    - Negatives: all j with labels[j] != labels[i]
    
    Loss: -log(sum(exp(sim_pos)) / sum(exp(sim_all_except_self)))
    
    embeddings: (B, D) L2-normalized vectors
    labels: (B,) integer class labels
    temperature: scaling factor for similarities
    """
    device = embeddings.device
    B = embeddings.size(0)
    
    # Compute pairwise cosine similarities (already L2-normalized)
    sim_matrix = embeddings @ embeddings.t()  # (B, B)
    sim_matrix = sim_matrix / temperature
    
    # Create masks
    labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B) same class
    labels_eq = labels_eq.float()
    labels_eq.fill_diagonal_(0)  # Exclude self-similarity
    
    # For numerical stability, subtract max
    sim_max, _ = sim_matrix.max(dim=1, keepdim=True)
    sim_matrix = sim_matrix - sim_max.detach()
    
    # Compute exp(sim)
    exp_sim = torch.exp(sim_matrix)
    
    # Sum over positives
    pos_sum = (exp_sim * labels_eq).sum(dim=1)  # (B,)
    
    # Sum over all except self
    exp_sim.fill_diagonal_(0)
    all_sum = exp_sim.sum(dim=1)  # (B,)
    
    # Loss for samples that have at least one positive
    mask = labels_eq.sum(dim=1) > 0  # (B,) samples with positives
    
    if mask.sum() == 0:
        return torch.tensor(0.0, device=device)
    
    # -log(pos / all)
    log_prob = torch.log(pos_sum[mask] / (all_sum[mask] + 1e-8) + 1e-8)
    loss = -log_prob.mean()
    
    return loss
```

**Step 2:** Modify `train_one_epoch` function around line 595-605:

Find this block:
```python
        out = model(imgs)
        logits = out["logits"]
        # Prefer backbone raw features for ArcFace...
        feats_for_margin = getattr(model, "_last_backbone_features", None)
        if feats_for_margin is None:
            feats_for_margin = out["embedding"]
        if arcface_margin > 0.0:
            loss = _arcface_loss(logits, labels, feats_for_margin)
        else:
            loss = ce(logits, labels)
        loss.backward()
```

Replace with:
```python
        out = model(imgs)
        logits = out["logits"]
        embeddings = out["embedding"]  # (B, 128) L2-normalized
        
        # Classification loss
        feats_for_margin = getattr(model, "_last_backbone_features", None)
        if feats_for_margin is None:
            feats_for_margin = embeddings
        if arcface_margin > 0.0:
            ce_loss = _arcface_loss(logits, labels, feats_for_margin)
        else:
            ce_loss = ce(logits, labels)
        
        # Embedding contrastive loss
        emb_loss = supervised_contrastive_loss(embeddings, labels, temperature=0.07)
        
        # Combined loss (start with lower weight on embedding loss)
        emb_weight = 0.3  # Can tune this (0.1-0.5 range)
        loss = ce_loss + emb_weight * emb_loss
        
        loss.backward()
```

**Step 3:** Update logging to track both losses. Around line 635, modify the return statement:

Find:
```python
    return {
        "train_loss": total_loss / max(1, total_samples),
        "grad_norm": last_grad_norm,
        "grad_norm_unclipped": last_unclipped_grad_norm,
    }
```

Change to track separate losses:
```python
    # Add tracking variables at start of function
    total_ce_loss = 0.0
    total_emb_loss = 0.0
    
    # In the loop, accumulate:
    total_ce_loss += ce_loss.item() * bs
    total_emb_loss += emb_loss.item() * bs
    
    # Return:
    return {
        "train_loss": total_loss / max(1, total_samples),
        "ce_loss": total_ce_loss / max(1, total_samples),
        "emb_loss": total_emb_loss / max(1, total_samples),
        "grad_norm": last_grad_norm,
        "grad_norm_unclipped": last_unclipped_grad_norm,
    }
```

---

## Fix #2: Non-Deterministic Augmentation (HIGH - 30 minutes)

**File:** `raster/dataset.py`

**Step 1:** Add epoch tracking to dataset class.

Around line 247 in `__init__`, add:
```python
        self._rng = random.Random(config.seed)
        self._epoch = 0  # ADD THIS LINE
        self._global_call_counter = 0  # ADD THIS LINE
```

**Step 2:** Add method to update epoch:
```python
    def set_epoch(self, epoch: int):
        """Update epoch number for augmentation variety."""
        self._epoch = epoch
```

**Step 3:** Modify `__getitem__` around line 563:

Find:
```python
        if self.cfg.augment:
            local_rng = random.Random(self.cfg.seed + row.glyph_id)
            img = _augment_tensor(
                img,
                rng=local_rng,
                ...
            )
```

Replace with:
```python
        if self.cfg.augment:
            # Use epoch + counter for variety across epochs
            self._global_call_counter += 1
            seed = (
                self.cfg.seed 
                + row.glyph_id * 10000 
                + self._epoch * 1000000 
                + self._global_call_counter
            )
            local_rng = random.Random(seed)
            img = _augment_tensor(
                img,
                rng=local_rng,
                ...
            )
```

**Step 4:** Update training loop to set epoch. In `train.py`, find the main loop around line 850:

Find:
```python
    for epoch in range(start_epoch, args.epochs):
        # Learning rate schedule
        adjust_lrs(...)
```

Add after `adjust_lrs`:
```python
        # Update dataset epoch for augmentation variety
        if hasattr(train_ds, 'set_epoch'):
            train_ds.set_epoch(epoch)
```

---

## Fix #3: Remove BatchNorm from Classifier (HIGH - 15 minutes)

**File:** `raster/model.py`

**Step 1:** After building the backbone (around line 310, after the LeViT constructor), add:

Find:
```python
    backbone = LeViT(
        img_size=config.img_size,
        ...
        drop_path=drop_path,
    )
    
    # Feature hook: capture pooled features...
```

Add BEFORE the "Feature hook" comment:
```python
    # Replace BN_Linear classifier with plain Linear for stability with many classes
    if config.num_classes > 0:
        # Remove the BN_Linear head and replace with plain Linear
        in_features = embed_dim[-1]  # 384 for LeViT_128S
        backbone.head = nn.Linear(in_features, config.num_classes)
        # Initialize with same scheme as BN_Linear
        nn.init.normal_(backbone.head.weight, std=0.02)
        nn.init.zeros_(backbone.head.bias)
```

---

## Verification Steps

### After Applying Fixes

**Test 1: Gradient Flow**
```bash
python -m raster.debug.test_gradient_flow
```

Expected output:
```
✅ PASS: Embedding loss updates embedding head
✅ PASS: Embedding loss updates backbone
✅ ALL TESTS PASSED
```

**Test 2: Quick Training Test (10 epochs, 10k samples)**
```bash
python -m raster.train \
  --db dataset/glyphs.db \
  --limit 10000 \
  --epochs 10 \
  --batch-size 256 \
  --val-frac 0.1 \
  --num-workers 2 \
  --pre-rasterize \
  --min-label-count 2 \
  --drop-singletons \
  --lr-backbone 0.0015 \
  --lr-head 0.0030 \
  --weight-decay 0.01 \
  --warmup-frac 0.05 \
  --lr-schedule constant \
  --retrieval-cap 1000 \
  --retrieval-every 2 \
  --log-interval 50 \
  --suppress-warnings
```

**Expected results after 10 epochs:**
- Train loss: 4.0-5.0 (down from 6.4)
- Val loss: 6.5-7.5 (not exploding)
- Val acc: **0.8-2.0%** (up from 0.1-0.8%)
- Top-10 retrieval: **25-35%** (up from 17-18%)

If you see these improvements → Fixes are working! ✅

---

## 100k Validation Run (After Quick Test Passes)

```bash
export OMP_NUM_THREADS=40
export MKL_NUM_THREADS=40
export OPENBLAS_NUM_THREADS=40
export NUMEXPR_NUM_THREADS=40
export PYTHONUNBUFFERED=1

python -m raster.train \
  --db dataset/glyphs.db \
  --limit 100000 \
  --epochs 15 \
  --batch-size 512 \
  --val-frac 0.1 \
  --num-workers 2 \
  --pre-rasterize \
  --min-label-count 2 \
  --drop-singletons \
  --lr-backbone 0.0015 \
  --lr-head 0.0030 \
  --weight-decay 0.01 \
  --warmup-frac 0.05 \
  --lr-schedule constant \
  --grad-clip 0.0 \
  --arcface-margin 0.0 \
  --retrieval-cap 3000 \
  --retrieval-every 2 \
  --log-interval 100 \
  --suppress-warnings
```

**Expected results after 15 epochs:**
- Train loss: 3.0-4.0
- Val loss: 6.0-6.5
- Val acc: **3-5%** (30-40× better than random)
- Top-10 retrieval: **40-50%**
- Effect size: **0.40-0.50**

If these targets are hit → Full 528k run will succeed! 🚀

---

## Hyperparameter Tuning (Optional)

If the 100k run works but you want to squeeze out more performance:

### Embedding Loss Weight

Try different values:
```python
emb_weight = 0.1  # Less focus on embeddings, more on classification
emb_weight = 0.3  # Balanced (default)
emb_weight = 0.5  # More focus on embeddings
```

### Contrastive Temperature

```python
temperature = 0.05  # Sharper, more aggressive
temperature = 0.07  # Default
temperature = 0.10  # Softer, more forgiving
```

### Learning Rates

```bash
# Higher LR for heads (embeddings learning from scratch)
--lr-head 0.005  # Up from 0.003

# Lower weight decay (less regularization, more capacity)
--weight-decay 0.005  # Down from 0.01
```

---

## Common Issues After Fixes

### Issue: "RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation"

**Cause:** In-place operations breaking autograd.

**Fix:** In `supervised_contrastive_loss`, ensure no in-place ops:
```python
# BAD:
labels_eq.fill_diagonal_(0)

# GOOD:
mask = torch.eye(B, device=device, dtype=torch.bool)
labels_eq = labels_eq.masked_fill(mask, 0)
```

### Issue: "CUDA out of memory" (if using GPU)

**Fix:** Reduce batch size:
```bash
--batch-size 256  # Down from 512
```

### Issue: Val accuracy still stuck at 0.5%

**Check:**
1. Is embedding loss being computed? (print emb_loss.item())
2. Is emb_loss > 0.1? (Should be 2-5 initially)
3. Are embeddings normalized? (print embeddings.norm(dim=1).mean())
4. Are there positives in each batch? (print labels_eq.sum())

---

## Debug Checklist

Before running 100k test:

- [ ] Fix #1 applied: embedding loss added to training loop
- [ ] Fix #2 applied: augmentation uses epoch counter
- [ ] Fix #3 applied: BN_Linear replaced with Linear
- [ ] Gradient flow test passes
- [ ] 10k quick test shows improvement (val_acc > 1%, top-10 > 25%)
- [ ] Environment variables set (OMP_NUM_THREADS, etc.)
- [ ] Pre-raster memmap exists or will be regenerated
- [ ] Enough disk space for checkpoints (~500 MB)

---

## Expected Timeline

| Task | Duration | When to Run |
|------|----------|-------------|
| Apply fixes | 2-3 hours | Now |
| Quick test (10k) | 1 hour | After fixes |
| 100k validation | 2.5 hours | If quick test passes |
| Full 528k run | 5-6 hours | If 100k test passes |

**Total time to validated baseline:** ~5-7 hours  
**Total time to full training:** ~10-12 hours

---

## Success Criteria

### Minimum (100k, 15 epochs):
- ✅ Val acc > 2.5%
- ✅ Top-10 > 35%
- ✅ Effect size > 0.35

### Target (528k, 24 epochs):
- ✅ Val acc > 5%
- ✅ Top-10 > 50%
- ✅ Effect size > 0.45

### Stretch (528k, 24 epochs, with tuning):
- 🏆 Val acc > 8%
- 🏆 Top-10 > 60%
- 🏆 Effect size > 0.55

---

**Good luck!** 🚀

Remember: Fix #1 (embedding loss) is **non-negotiable**. The other two are highly recommended but the model might still improve without them.