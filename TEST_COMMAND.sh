#!/bin/bash
# Quick validation test with all fixes applied
# Expected: val_acc > 1%, top-10 > 25% by epoch 10

export OMP_NUM_THREADS=40
export MKL_NUM_THREADS=40
export OPENBLAS_NUM_THREADS=40
export NUMEXPR_NUM_THREADS=40
export PYTHONUNBUFFERED=1

echo "========================================="
echo "Running 10k validation test with fixes"
echo "Expected improvements:"
echo "  - Val acc: 0.8-2.0% (vs 0.1-0.8% before)"
echo "  - Top-10: 25-35% (vs 17-18% before)"
echo "  - Both CE and Emb losses display"
echo "========================================="
echo ""

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
