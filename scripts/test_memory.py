#!/usr/bin/env python
"""
Memory Test Script

Checks if your system can handle the AmEx 50GB dataset.
Provides recommendations for chunk size and configuration.
"""

import psutil
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))


def bytes_to_gb(bytes_val):
    """Convert bytes to GB."""
    return bytes_val / (1024 ** 3)


def check_system_memory():
    """Check system memory and provide recommendations."""
    print("=" * 70)
    print("AMEX DATASET MEMORY CHECK")
    print("=" * 70)

    # Get system memory
    mem = psutil.virtual_memory()
    total_ram = bytes_to_gb(mem.total)
    available_ram = bytes_to_gb(mem.available)
    used_ram = bytes_to_gb(mem.used)

    print(f"\n📊 System Memory:")
    print(f"  Total RAM: {total_ram:.1f} GB")
    print(f"  Used RAM: {used_ram:.1f} GB")
    print(f"  Available RAM: {available_ram:.1f} GB")
    print(f"  Usage: {mem.percent:.1f}%")

    # Check swap
    swap = psutil.swap_memory()
    if swap.total > 0:
        swap_gb = bytes_to_gb(swap.total)
        print(f"  Swap Space: {swap_gb:.1f} GB")
    else:
        print(f"  Swap Space: None (consider adding swap)")

    # Check disk space
    disk = psutil.disk_usage('.')
    free_disk = bytes_to_gb(disk.free)
    print(f"\n💾 Disk Space:")
    print(f"  Free Disk: {free_disk:.1f} GB")

    # Provide recommendations
    print(f"\n🎯 Recommendations:")

    # RAM check
    if total_ram < 16:
        print("  ❌ RAM Status: INSUFFICIENT")
        print("  ⚠️  Minimum 32GB RAM required for full pipeline")
        print("  💡 Consider: Cloud instance or smaller chunk size")
        chunk_size = 100_000
        can_train = False
    elif total_ram < 32:
        print("  ⚠️  RAM Status: MARGINAL")
        print("  💡 Should work with optimizations")
        print("  💡 Close all other applications")
        chunk_size = 250_000
        can_train = True
    elif total_ram < 64:
        print("  ✅ RAM Status: GOOD")
        print("  💡 Recommended for full pipeline")
        chunk_size = 500_000
        can_train = True
    else:
        print("  ✅ RAM Status: EXCELLENT")
        print("  💡 Can handle all models efficiently")
        chunk_size = 1_000_000
        can_train = True

    # Disk check
    if free_disk < 100:
        print("  ❌ Disk Status: INSUFFICIENT")
        print("  ⚠️  Need at least 100GB free space")
        print("  💡 Clean up disk or use external drive")
        can_train = False
    elif free_disk < 200:
        print("  ⚠️  Disk Status: MARGINAL")
        print("  💡 Monitor disk usage during training")
    else:
        print("  ✅ Disk Status: GOOD")

    # GPU check
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"\n🎮 GPU:")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_mem = torch.cuda.get_device_properties(i).total_memory
                gpu_mem_gb = bytes_to_gb(gpu_mem)
                print(f"  ✅ GPU {i}: {gpu_name} ({gpu_mem_gb:.1f} GB)")

            if gpu_mem_gb < 8:
                print("  💡 GPU memory < 8GB: Use smaller batch sizes")
        else:
            print(f"\n🎮 GPU: None detected (CPU training only)")
    except ImportError:
        print(f"\n🎮 GPU: PyTorch not installed")

    # Configuration recommendations
    print(f"\n⚙️  Recommended Configuration:")
    print(f"  chunk_size: {chunk_size:,}")

    if total_ram < 32:
        print(f"  batch_size: 128")
        print(f"  num_workers: 2")
        print(f"  models: lightgbm only (fastest)")
    elif total_ram < 64:
        print(f"  batch_size: 256")
        print(f"  num_workers: 4")
        print(f"  models: lightgbm, xgboost, catboost")
    else:
        print(f"  batch_size: 512")
        print(f"  num_workers: 8")
        print(f"  models: all models")

    # Estimated memory usage
    print(f"\n📈 Estimated Memory Usage:")

    if chunk_size == 100_000:
        peak_usage = 8
    elif chunk_size == 250_000:
        peak_usage = 12
    elif chunk_size == 500_000:
        peak_usage = 18
    else:
        peak_usage = 24

    print(f"  Peak RAM during training: ~{peak_usage} GB")
    print(f"  Buffer available: {available_ram - peak_usage:.1f} GB")

    if available_ram < peak_usage:
        print(f"  ⚠️  WARNING: May need to close other apps")

    # Final verdict
    print(f"\n" + "=" * 70)
    if can_train:
        print("✅ STATUS: READY FOR TRAINING")
        print("\nNext steps:")
        print("  1. Update configs/config.yaml with recommended settings")
        print("  2. Run: python train.py --config configs/config.yaml")
    else:
        print("❌ STATUS: NOT READY")
        print("\nOptions:")
        print("  1. Upgrade RAM to 32GB+")
        print("  2. Use cloud instance (AWS, GCP, Kaggle)")
        print("  3. Free up disk space")

    print("=" * 70)

    return can_train


def test_chunk_loading():
    """Test loading a small chunk to verify setup."""
    print("\n🧪 Testing chunk loading...")

    try:
        import pandas as pd
        import numpy as np
        from pathlib import Path

        # Check if data exists
        data_file = Path('./input/train_data.csv')

        if not data_file.exists():
            print("  ⚠️  Data file not found: ./input/train_data.csv")
            print("  💡 Download data and place in ./input/ directory")
            return False

        # Try loading a small chunk
        print("  Loading 10,000 rows as test...")
        chunk = pd.read_csv(data_file, nrows=10_000)

        print(f"  ✅ Successfully loaded {len(chunk):,} rows")
        print(f"  ✅ Columns: {len(chunk.columns)}")
        print(f"  ✅ Memory: {chunk.memory_usage().sum() / 1024**2:.2f} MB")

        # Test memory optimization
        from src.preprocessing import reduce_mem_usage

        print("  Testing memory optimization...")
        chunk_opt = reduce_mem_usage(chunk, verbose=False)

        original_mb = chunk.memory_usage().sum() / 1024**2
        optimized_mb = chunk_opt.memory_usage().sum() / 1024**2
        reduction = 100 * (1 - optimized_mb / original_mb)

        print(f"  ✅ Memory reduced: {original_mb:.2f} MB → {optimized_mb:.2f} MB")
        print(f"  ✅ Reduction: {reduction:.1f}%")

        return True

    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        return False


if __name__ == "__main__":
    # Check system
    can_train = check_system_memory()

    # Test loading (if data available)
    if can_train:
        print("\n")
        test_chunk_loading()

    # Exit code
    sys.exit(0 if can_train else 1)
