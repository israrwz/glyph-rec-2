#!/bin/bash
# Full training with HYBRID contrastive grouping
# Arabic letters: grouped by joining_group (BEH, HAH, YEH, etc.)
# NO_JOIN glyphs: grouped by char_class (latin, diacritic, punctuation, etc.)

export OMP_NUM_THREADS=40
export MKL_NUM_THREADS=40
export OPENBLAS_NUM_THREADS=40
export NUMEXPR_NUM_THREADS=40
export PYTHONUNBUFFERED=1

echo "========================================="
echo "HYBRID CONTRASTIVE TRAINING"
echo ""
echo "Smart grouping prevents mixing:"
echo "  ❌ BEFORE: Latin 'A' + diacritic ◌̃ (both NO_JOIN)"
echo "  ✅ AFTER:  Latin 'A' → latin group"
echo "             Diacritic ◌̃ → diacritic group"
echo ""
echo "Grouping strategy:"
echo "  • Arabic letters → joining_group (248 groups)"
echo "  • Latin letters → cc_latin"
echo "  • Diacritics → cc_diacritic"
echo "  • Punctuation → cc_punctuation"
echo "  • Digits → cc_digit"
echo "  • Symbols → cc_symbol"
echo ""
echo "Total: ~260 contrastive groups (vs 1588 labels)"
echo "Result: Better embeddings for retrieval!"
echo "========================================="
echo ""

python -m raster.train \
  --db dataset/glyphs.db \
  --limit 531000 \
  --epochs 20 \
  --batch-size 1024 \
  --val-frac 0.1 \
  --num-workers 2 \
  --device cuda \
  --pre-rasterize \
  --pre-raster-mmap \
  --pre-raster-mmap-path preraster_full_531k_u8.dat \
  --min-label-count 2 \
  --drop-singletons \
  --lr-backbone 0.0015 \
  --lr-head 0.0030 \
  --weight-decay 0.008 \
  --warmup-frac 0.02 \
  --lr-schedule constant \
  --emb-loss-weight 0.25 \
  --retrieval-cap 5000 \
  --retrieval-every 5 \
  --log-interval 200 \
  --suppress-warnings

echo ""
echo "========================================="
echo "Training complete!"
echo "Check raster/checkpoints/best.pt"
echo "========================================="
