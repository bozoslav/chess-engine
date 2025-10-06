#!/usr/bin/env python3
"""
View training progress and statistics
"""

import csv
import os
import sys

def analyze_training_log(log_path='training_data/training_log.csv'):
    """Analyze and display training progress"""
    
    if not os.path.exists(log_path):
        print(f"❌ Log file not found: {log_path}")
        print("   Start training first with: python train.py")
        return
    
    print("\n" + "="*70)
    print("📊 TRAINING PROGRESS ANALYSIS")
    print("="*70 + "\n")
    
    rows = []
    with open(log_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    
    if not rows:
        print("No training data yet.")
        return
    
    # Latest metrics
    latest = rows[-1]
    print("📈 Latest Epoch:")
    print(f"   Epoch:      {latest['epoch']}")
    print(f"   Train Loss: {float(latest['train_loss']):.6f}")
    print(f"   Train MAE:  {float(latest['train_mae']):.4f}")
    print(f"   Val Loss:   {float(latest['val_loss']):.6f}")
    print(f"   Val MAE:    {float(latest['val_mae']):.4f}")
    print(f"   LR:         {float(latest['lr']):.2e}")
    print(f"   Time:       {float(latest['epoch_time']):.1f}s")
    
    # Best validation
    best_val_idx = min(range(len(rows)), key=lambda i: float(rows[i]['val_loss']))
    best = rows[best_val_idx]
    print(f"\n🌟 Best Validation:")
    print(f"   Epoch:      {best['epoch']}")
    print(f"   Val Loss:   {float(best['val_loss']):.6f}")
    print(f"   Val MAE:    {float(best['val_mae']):.4f}")
    
    # Overall statistics
    train_losses = [float(r['train_loss']) for r in rows]
    val_losses = [float(r['val_loss']) for r in rows]
    val_maes = [float(r['val_mae']) for r in rows]
    
    print(f"\n📊 Overall Statistics:")
    print(f"   Total Epochs:     {len(rows)}")
    print(f"   Train Loss Range: {min(train_losses):.4f} - {max(train_losses):.4f}")
    print(f"   Val Loss Range:   {min(val_losses):.4f} - {max(val_losses):.4f}")
    print(f"   Val MAE Range:    {min(val_maes):.4f} - {max(val_maes):.4f}")
    
    # Recent trend (last 10 epochs)
    if len(rows) >= 10:
        recent = rows[-10:]
        recent_val = [float(r['val_loss']) for r in recent]
        trend = "📉 Improving" if recent_val[-1] < recent_val[0] else "📈 Not improving"
        print(f"\n🔄 Recent Trend (last 10 epochs): {trend}")
    
    print("\n" + "="*70 + "\n")
    
    # Show last 5 epochs
    print("Recent epochs:")
    print(f"{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>12} {'Val MAE':>10} {'LR':>12}")
    print("-" * 70)
    for row in rows[-5:]:
        print(f"{row['epoch']:>6} {float(row['train_loss']):>12.6f} "
              f"{float(row['val_loss']):>12.6f} {float(row['val_mae']):>10.4f} "
              f"{float(row['lr']):>12.2e}")
    
    print()

if __name__ == "__main__":
    analyze_training_log()
