# Google Colab Notebook Cells
Copy and paste these cells into a new Google Colab notebook.

---

## Cell 1: Mount Drive & Setup

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install dependencies
!pip install timm -q

# Navigate to project (adjust path as needed)
# Option 1: If uploaded to Drive
# %cd /content/drive/MyDrive/glyph-rec-2

# Option 2: If cloning from GitHub
# !git clone https://github.com/your-username/glyph-rec-2.git
# %cd glyph-rec-2

# Verify GPU
import torch
print(f"✅ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT AVAILABLE'}")
print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB" if torch.cuda.is_available() else "")
```

---

## Cell 2: Full Training (531k samples, 20-40 minutes)

```python
import subprocess
import sys

# Configuration
config = {
    'db': 'dataset/glyphs.db',
    'limit': 531000,
    'epochs': 20,
    'batch_size': 1024,
    'val_frac': 0.1,
    'num_workers': 2,
    'device': 'cuda',
    'pre_rasterize': True,
    'pre_raster_mmap': True,
    'pre_raster_mmap_path': 'preraster_full_531k_u8.dat',
    'min_label_count': 2,
    'drop_singletons': True,
    'lr_backbone': 0.0015,
    'lr_head': 0.0030,
    'weight_decay': 0.008,
    'warmup_frac': 0.02,
    'lr_schedule': 'constant',
    'emb_loss_weight': 0.25,
    'retrieval_cap': 5000,
    'retrieval_every': 5,
    'log_interval': 200,
    'suppress_warnings': True,
}

# Build command
cmd = [sys.executable, '-m', 'raster.train']
for key, value in config.items():
    arg_name = f"--{key.replace('_', '-')}"
    if isinstance(value, bool):
        if value:
            cmd.append(arg_name)
    else:
        cmd.extend([arg_name, str(value)])

print("🚀 Starting training...")
print(f"Expected: 5-10% val acc, 55-75% top-10 retrieval")
print(f"Runtime: 20-40 minutes on T4/A100\n")

# Run training
result = subprocess.run(cmd)

if result.returncode == 0:
    print("\n✅ Training complete! Check raster/checkpoints/best.pt")
else:
    print(f"\n❌ Training failed with code {result.returncode}")
```

---

## Cell 3: Quick Test (100k samples, 1-2 hours)

```python
import subprocess
import sys

# Smaller config for quick validation
config = {
    'db': 'dataset/glyphs.db',
    'limit': 100000,
    'epochs': 15,
    'batch_size': 1024,
    'device': 'cuda',
    'pre_rasterize': True,
    'emb_loss_weight': 0.25,
    'retrieval_cap': 3000,
    'retrieval_every': 3,
    'suppress_warnings': True,
}

# Build and run command
cmd = [sys.executable, '-m', 'raster.train']
for key, value in config.items():
    arg_name = f"--{key.replace('_', '-')}"
    if isinstance(value, bool):
        if value:
            cmd.append(arg_name)
    else:
        cmd.extend([arg_name, str(value)])

print("🧪 Quick test: 100k samples")
print("Expected: 2-4% val acc\n")

subprocess.run(cmd)
```

---

## Cell 4: Monitor Training (run in parallel)

```python
# Watch training log in real-time
!tail -f raster/artifacts/train_log.jsonl
```

---

## Cell 5: Check Results

```python
import json
from pathlib import Path

# Load training log
log_file = Path('raster/artifacts/train_log.jsonl')
if log_file.exists():
    logs = [json.loads(line) for line in log_file.read_text().strip().split('\n')]
    
    print("📊 Training Summary\n")
    print(f"Total epochs: {len(logs)}")
    
    if logs:
        last = logs[-1]
        print(f"\nFinal Results (Epoch {last['epoch']}):")
        print(f"  Train loss: {last.get('train_loss', 'N/A'):.4f}")
        print(f"  CE loss: {last.get('ce_loss', 'N/A'):.4f}")
        print(f"  Emb loss: {last.get('emb_loss', 'N/A'):.4f}")
        print(f"  Val loss: {last.get('val_loss', 'N/A'):.4f}")
        print(f"  Val acc: {last.get('val_accuracy', 'N/A'):.4f} ({last.get('val_accuracy', 0)*100:.2f}%)")
        
        if 'topk_accuracy' in last:
            print(f"  Top-10: {last['topk_accuracy']:.4f} ({last['topk_accuracy']*100:.2f}%)")
        if 'effect_size' in last:
            print(f"  Effect size: {last.get('effect_size', 'N/A'):.4f}")
    
    # Best epoch
    best_acc = max(logs, key=lambda x: x.get('val_accuracy', 0))
    print(f"\nBest Val Acc: {best_acc['val_accuracy']:.4f} @ Epoch {best_acc['epoch']}")
else:
    print("❌ No training log found")
```

---

## Cell 6: Download Results

```python
from google.colab import files
import subprocess

# Create archive
!tar -czf glyph_results.tar.gz \
    raster/checkpoints/best.pt \
    raster/checkpoints/last.pt \
    raster/artifacts/train_log.jsonl \
    raster/artifacts/label_to_index.json

# Download
print("📦 Downloading results...")
files.download('glyph_results.tar.gz')
print("✅ Done!")
```

---

## Cell 7: Visualize Training Progress

```python
import json
import matplotlib.pyplot as plt
from pathlib import Path

log_file = Path('raster/artifacts/train_log.jsonl')
if log_file.exists():
    logs = [json.loads(line) for line in log_file.read_text().strip().split('\n')]
    
    epochs = [log['epoch'] for log in logs]
    train_loss = [log.get('train_loss', 0) for log in logs]
    val_loss = [log.get('val_loss', 0) for log in logs]
    val_acc = [log.get('val_accuracy', 0) * 100 for log in logs]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss curves
    axes[0].plot(epochs, train_loss, label='Train Loss', marker='o')
    axes[0].plot(epochs, val_loss, label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Val accuracy
    axes[1].plot(epochs, val_acc, label='Val Accuracy', marker='o', color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # CE vs Emb loss
    ce_loss = [log.get('ce_loss', 0) for log in logs]
    emb_loss = [log.get('emb_loss', 0) for log in logs]
    axes[2].plot(epochs, ce_loss, label='CE Loss', marker='o')
    axes[2].plot(epochs, emb_loss, label='Emb Loss', marker='s')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].set_title('CE vs Embedding Loss')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"✅ Training converged after {len(logs)} epochs")
else:
    print("❌ No training log found")
```

---

## Tips

### For T4 GPU (Free Tier)
- Use `batch_size: 1024` (default)
- Expected runtime: 30-40 minutes
- May hit 12-hour limit on very long runs

### For A100 GPU (Colab Pro)
- Can increase to `batch_size: 2048` or even `4096`
- Expected runtime: 15-20 minutes
- More positive pairs = better contrastive learning

### If You Hit Memory Issues
```python
# Reduce batch size
config['batch_size'] = 512

# Or reduce workers
config['num_workers'] = 0
```

### To Resume Interrupted Training
```python
# Add to config:
config['resume'] = 'raster/checkpoints/last.pt'
```

---

## Expected Results (531k samples)

| Metric | Target |
|--------|--------|
| Val Top-1 Accuracy | 5-10% |
| Val Top-10 Accuracy | 35-50% |
| Retrieval Top-10 | 55-75% |
| Effect Size | 0.50-0.65 |
| Training Time | 20-40 min (GPU) |

---

## Troubleshooting

### "No GPU available"
Enable GPU: Runtime → Change runtime type → Hardware accelerator → GPU

### "Out of memory"
Reduce batch size: `config['batch_size'] = 512`

### "ModuleNotFoundError: No module named 'timm'"
Run: `!pip install timm`

### "FileNotFoundError: dataset/glyphs.db"
Check path and make sure database is uploaded

### Training seems slow
Verify GPU is being used: `!nvidia-smi` should show python process