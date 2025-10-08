"""
COLAB_TRAINING.py
=================
Google Colab training script for glyph recognition with hybrid contrastive learning.

Usage in Colab:
1. Upload this file and the project to Colab
2. Run setup cells
3. Run training cell
4. Download results

Expected runtime: 20-40 minutes on T4/A100 GPU
Expected results: 5-10% val accuracy, 55-75% top-10 retrieval
"""

import subprocess
import sys
import os
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

TRAINING_CONFIG = {
    # Data
    "db": "dataset/glyphs.db",
    "limit": 531000,
    "val_frac": 0.1,
    # Training
    "epochs": 20,
    "batch_size": 1024,  # Can increase to 2048 on A100
    "device": "cuda",
    "num_workers": 2,
    # Optimization
    "lr_backbone": 0.0015,
    "lr_head": 0.0030,
    "weight_decay": 0.008,
    "warmup_frac": 0.02,
    "lr_schedule": "constant",
    # Loss weights
    "emb_loss_weight": 0.25,  # Hybrid contrastive loss weight
    # Data processing
    "pre_rasterize": True,
    "pre_raster_mmap": True,
    "pre_raster_mmap_path": "preraster_full_531k_u8.dat",
    "min_label_count": 2,
    "drop_singletons": True,
    # Evaluation
    "retrieval_cap": 5000,
    "retrieval_every": 5,
    "log_interval": 200,
    # Other
    "suppress_warnings": True,
}


# =============================================================================
# SETUP FUNCTIONS
# =============================================================================


def setup_colab():
    """Mount Drive and install dependencies."""
    print("=" * 60)
    print("GOOGLE COLAB SETUP")
    print("=" * 60)

    # Mount Drive
    try:
        from google.colab import drive

        drive.mount("/content/drive")
        print("✅ Google Drive mounted")
    except Exception as e:
        print(f"⚠️  Drive mount failed (may already be mounted): {e}")

    # Install dependencies
    print("\nInstalling dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "timm", "-q"], check=True)
    print("✅ Dependencies installed (timm)")

    # Check GPU
    import torch

    print(f"\n{'=' * 60}")
    print("GPU CHECK")
    print("=" * 60)
    if torch.cuda.is_available():
        print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
        print(
            f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
        print(f"   CUDA Version: {torch.version.cuda}")
    else:
        print("❌ No GPU detected! Training will be VERY slow.")
        print(
            "   Enable GPU: Runtime → Change runtime type → Hardware accelerator → GPU"
        )
        return False

    return True


def verify_project_structure():
    """Check that required files exist."""
    print(f"\n{'=' * 60}")
    print("PROJECT STRUCTURE CHECK")
    print("=" * 60)

    required_files = [
        "raster/train.py",
        "raster/model.py",
        "raster/dataset.py",
        "raster/rasterize.py",
        "dataset/glyphs.db",
    ]

    missing = []
    for f in required_files:
        if Path(f).exists():
            print(f"✅ {f}")
        else:
            print(f"❌ {f} - MISSING!")
            missing.append(f)

    if missing:
        print(f"\n❌ Missing {len(missing)} required files!")
        print("Please upload the complete project structure.")
        return False

    print("\n✅ All required files present")
    return True


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================


def build_command(config):
    """Build training command from config dict."""
    cmd = [sys.executable, "-m", "raster.train"]

    for key, value in config.items():
        arg_name = f"--{key.replace('_', '-')}"

        if isinstance(value, bool):
            if value:
                cmd.append(arg_name)
        else:
            cmd.extend([arg_name, str(value)])

    return cmd


def run_training(config):
    """Execute training with given configuration."""
    print(f"\n{'=' * 60}")
    print("STARTING TRAINING")
    print("=" * 60)

    print("\nKey Configuration:")
    print(f"  Dataset: {config['limit']:,} samples")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Embedding loss weight: {config['emb_loss_weight']}")
    print(f"  Device: {config['device']}")

    print("\nHybrid Contrastive Grouping:")
    print("  ✓ Arabic letters → joining_group (BEH, HAH, etc.)")
    print("  ✓ NO_JOIN glyphs → char_class (latin, diacritic, etc.)")
    print("  ✓ Prevents mixing Latin/diacritics/punctuation")

    print("\nExpected Results (531k samples):")
    print("  Target: Val acc 5-10%, Top-10 retrieval 55-75%")
    print("  Runtime: 20-40 minutes on T4/A100")

    cmd = build_command(config)
    print(f"\nCommand: {' '.join(cmd)}")
    print(f"\n{'=' * 60}\n")

    # Run training
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with return code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        return False


def download_results():
    """Create archive and download results."""
    print(f"\n{'=' * 60}")
    print("DOWNLOADING RESULTS")
    print("=" * 60)

    try:
        from google.colab import files
    except ImportError:
        print("⚠️  Not running in Colab, skipping download")
        return

    archive_name = "glyph_training_results.tar.gz"

    # Create archive
    print("\nCreating archive...")
    files_to_archive = [
        "raster/checkpoints/best.pt",
        "raster/checkpoints/last.pt",
        "raster/artifacts/train_log.jsonl",
        "raster/artifacts/label_to_index.json",
    ]

    # Check what exists
    existing = [f for f in files_to_archive if Path(f).exists()]
    if not existing:
        print("❌ No checkpoint files found!")
        return

    print(f"Archiving {len(existing)} files...")
    cmd = ["tar", "-czf", archive_name] + existing
    subprocess.run(cmd, check=True)

    # Download
    print(f"Downloading {archive_name}...")
    files.download(archive_name)
    print("✅ Download complete!")


# =============================================================================
# QUICK TEST CONFIGURATION (100k samples)
# =============================================================================

QUICK_TEST_CONFIG = {
    "db": "dataset/glyphs.db",
    "limit": 100000,
    "epochs": 15,
    "batch_size": 1024,
    "device": "cuda",
    "pre_rasterize": True,
    "emb_loss_weight": 0.25,
    "retrieval_cap": 3000,
    "retrieval_every": 3,
    "log_interval": 100,
    "suppress_warnings": True,
}


# =============================================================================
# MAIN EXECUTION
# =============================================================================


def main(mode="full"):
    """
    Run complete training pipeline.

    Args:
        mode: 'full' (531k samples) or 'quick' (100k samples)
    """
    print("=" * 60)
    print("GLYPH RECOGNITION TRAINING")
    print("Hybrid Contrastive Learning with LeViT")
    print("=" * 60)

    # Setup
    if not setup_colab():
        print("\n❌ Setup failed! Please enable GPU and try again.")
        return False

    if not verify_project_structure():
        print("\n❌ Project structure check failed!")
        return False

    # Select config
    config = TRAINING_CONFIG if mode == "full" else QUICK_TEST_CONFIG
    print(f"\nMode: {'FULL TRAINING' if mode == 'full' else 'QUICK TEST'}")

    # Train
    success = run_training(config)

    if success:
        print("\n✅ Training completed successfully!")
        print("\nResults saved to:")
        print("  - raster/checkpoints/best.pt (best validation accuracy)")
        print("  - raster/checkpoints/last.pt (latest epoch)")
        print("  - raster/artifacts/train_log.jsonl (metrics)")

        # Offer to download
        response = input("\nDownload results? (y/n): ").strip().lower()
        if response == "y":
            download_results()
    else:
        print("\n❌ Training failed!")
        print("Check logs above for error details.")

    return success


# =============================================================================
# COLAB CELL FUNCTIONS (import these in notebook)
# =============================================================================


def train_full():
    """Run full 531k training (20-40 min)."""
    return main(mode="full")


def train_quick():
    """Run quick 100k test (1-2 hours)."""
    return main(mode="quick")


def train_custom(**kwargs):
    """
    Run training with custom configuration.

    Example:
        train_custom(limit=200000, epochs=15, batch_size=2048)
    """
    config = TRAINING_CONFIG.copy()
    config.update(kwargs)
    return run_training(config)


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    """
    USAGE IN GOOGLE COLAB:

    # Cell 1: Setup and import
    !pip install timm -q
    exec(open('COLAB_TRAINING.py').read())

    # Cell 2: Run full training (recommended)
    train_full()

    # Alternative: Quick test first
    # train_quick()

    # Alternative: Custom config
    # train_custom(limit=200000, epochs=15, emb_loss_weight=0.3)

    # Cell 3: Download results (optional)
    # download_results()
    """

    # If run directly (not in Colab), use command line
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "quick"], default="full")
    args = parser.parse_args()

    success = main(mode=args.mode)
    sys.exit(0 if success else 1)
