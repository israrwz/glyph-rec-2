#!/bin/bash
# Quick validation test with fixes + optimized settings
# Based on analysis: 793 classes, batch_size 256 has too few positive pairs

export OMP_NUM_THREADS=40
export MKL_NUM_THREADS=40
export OPENBLAS_NUM_THREADS=40
export NUMEXPR_NUM_THREADS=40
export PYTHONUNBUFFERED=1

echo "========================================="
echo "Running 10k validation test - OPTIMIZED"
echo "Key changes from previous run:"
echo "  - Batch size: 512 (was 256)"
echo "  - Emb loss weight: 0.15 (was 0.3)"
echo ""
echo "With 793 classes:"
echo "  - batch_size 256: ~24 samples with positives (9%)"
echo "  - batch_size 512: ~48 samples with positives (18%)"
echo ""
echo "Expected improvements:"
echo "  - Emb loss should decrease below 3.0"
echo "  - Val acc: 1-2% by epoch 10"
echo "  - Top-10: 20-30%"
echo "========================================="
echo ""

python -m raster.train \
  --db dataset/glyphs.db \
  --limit 10000 \
  --epochs 10 \
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
  --emb-loss-weight 0.15 \
  --retrieval-cap 1000 \
  --retrieval-every 2 \
  --log-interval 50 \
  --suppress-warnings

echo ""
echo "========================================="
echo "Alternative: Disable embedding loss entirely"
echo "If above still fails, try:"
echo ""
echo "  --batch-size 256 --emb-loss-weight 0.0"
echo ""
echo "This trains only the classifier (no contrastive loss)"
echo "Should achieve 1-2% val acc from classifier alone"
echo "========================================="
