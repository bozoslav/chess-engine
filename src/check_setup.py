#!/usr/bin/env python3
"""
Quick setup check before training
"""

import os
import sys
import torch

print("\n" + "="*70)
print("🔍 TRAINING SETUP CHECK")
print("="*70 + "\n")

# Check Python version
print(f"✅ Python version: {sys.version.split()[0]}")

# Check PyTorch
print(f"✅ PyTorch version: {torch.__version__}")

# Check device
if torch.backends.mps.is_available():
    print("✅ Device: MPS (Apple Silicon) available")
elif torch.cuda.is_available():
    print(f"✅ Device: CUDA available (GPU: {torch.cuda.get_device_name(0)})")
else:
    print("⚠️  Device: Only CPU available (training will be slow)")

# Check data file
data_path = '../data/lichess_db_eval.jsonl.zst'
if os.path.exists(data_path):
    size_mb = os.path.getsize(data_path) / (1024 * 1024)
    print(f"✅ Data file found: {size_mb:.1f} MB")
else:
    print(f"❌ Data file NOT found: {data_path}")
    print("   Please ensure your Lichess database is in the data/ folder")

# Check model file
model_path = 'model.py'
if os.path.exists(model_path):
    print(f"✅ Model file found: {model_path}")
else:
    print(f"❌ Model file NOT found: {model_path}")

# Check config
config_path = 'config.json'
if os.path.exists(config_path):
    print(f"✅ Config file found: {config_path}")
else:
    print(f"⚠️  Config file not found (will use defaults)")

# Check for existing checkpoint
checkpoint_path = 'training_data/checkpoint.pth'
if os.path.exists(checkpoint_path):
    print(f"✅ Existing checkpoint found (will resume)")
else:
    print(f"ℹ️  No checkpoint found (will start fresh)")

# Check dependencies
try:
    import chess
    print(f"✅ python-chess installed")
except ImportError:
    print(f"❌ python-chess NOT installed")
    print("   Install with: pip install python-chess")

try:
    import zstandard
    print(f"✅ zstandard installed")
except ImportError:
    print(f"❌ zstandard NOT installed")
    print("   Install with: pip install zstandard")

print("\n" + "="*70)

# Final verdict
if os.path.exists(data_path) and os.path.exists(model_path):
    print("\n✅ Ready to train! Run: python train.py")
else:
    print("\n❌ Please fix the issues above before training")

print()
