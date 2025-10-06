#!/usr/bin/env python3
import chess
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import zstandard as zstd
import io
import os
import math
import time
import signal
from model import ChessPositionNet, board_to_tensor

class LichessEvalDataset(Dataset):
    def __init__(self, jsonl_path, total_samples=500000, chunk_size=150000):
        """
        Load a large pool of positions once, then sample from it each epoch.
        This avoids distribution shift from sequential chunks.
        
        Args:
            total_samples: Total positions to load into memory (500k default)
            chunk_size: Positions to use per "epoch cycle" (150k default)
        """
        self.all_positions = []  # Large pool loaded once
        self.data = []  # Current active chunk
        self.jsonl_path = jsonl_path
        self.total_samples = total_samples
        self.chunk_size = chunk_size
        
        print(f"Loading {total_samples} random positions from database...")
        self._load_all_positions()
        print(f"Loaded {len(self.all_positions)} total positions")
        print("Shuffling initial chunk...")
        self.resample_chunk()
        
    def _load_all_positions(self):
        """Load a large random sample from the entire database"""
        self.all_positions = []
        loaded = 0
        skip_prob = 0.0  # Start by loading everything
        
        with open(self.jsonl_path, 'rb') as f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                text_stream = io.TextIOWrapper(reader, encoding='utf-8')
                for i, line in enumerate(text_stream):
                    # Random skip to sample from entire file
                    if random.random() < skip_prob:
                        continue
                    
                    if loaded >= self.total_samples:
                        break
                        
                    try:
                        entry = json.loads(line.strip())
                        fen = entry['fen']
                        evals = entry.get('evals', [])
                        if not evals:
                            continue
                        deepest = max(evals, key=lambda e: e.get('depth', 0))
                        pvs = deepest.get('pvs', [])
                        if not pvs or 'cp' not in pvs[0]:
                            continue
                        cp = pvs[0]['cp']
                        ev = max(-10.0, min(10.0, cp / 100.0))
                        self.all_positions.append((fen, ev))
                        loaded += 1
                        
                        if loaded % 100000 == 0:
                            print(f"  Loaded {loaded}/{self.total_samples}")
                    except:
                        continue
    
    def resample_chunk(self):
        """Resample a fresh chunk from the full pool"""
        self.data = random.sample(
            self.all_positions, 
            min(self.chunk_size, len(self.all_positions))
        )
        random.shuffle(self.data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        fen, ev = self.data[idx]
        board = chess.Board(fen)
        indices, features = board_to_tensor(board)
        return indices, features.squeeze(0), torch.tensor([ev], dtype=torch.float32)

def train():
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    print(f'Using {device}')
    
    os.makedirs('training_data', exist_ok=True)
    
    model = ChessPositionNet(dropout=0.1, square_dropout=0.05, use_attention=True, aux_material_head=True).to(device)
    
    # Compile model for speed boost (PyTorch 2.0+)
    # Note: torch.compile() on MPS has issues, only compile on CUDA
    if hasattr(torch, 'compile') and device.type == 'cuda':
        print("Compiling model for faster training...")
        model = torch.compile(model)
        print("✓ Model compiled!")
    elif device.type == 'mps':
        print("⚠️  Skipping torch.compile() - not fully supported on MPS yet")
    
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=5e-4)  # Increased LR, balanced weight decay
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
    criterion = nn.MSELoss()
    
    ckpt_file = 'training_data/checkpoint.pth'
    epoch = 0
    best_val = float('inf')
    chunk = 0
    chunk_reload_epoch = -10  # Track when we last reloaded
    
    if os.path.exists(ckpt_file):
        print("Loading checkpoint...")
        try:
            ckpt = torch.load(ckpt_file, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'], strict=False)
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            epoch = ckpt.get('epoch', 0)
            best_val = ckpt.get('best_val', float('inf'))
            chunk = ckpt.get('chunk', 0)
            chunk_reload_epoch = ckpt.get('chunk_reload_epoch', -10)
            # Restore scheduler state
            if 'scheduler_state_dict' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            print(f"Resumed from epoch {epoch}")
        except Exception as e:
            print(f"Load failed: {e}")
    
    dataset = LichessEvalDataset('../data/lichess_db_eval.jsonl.zst', 
                                  total_samples=500000,  # Load 500k positions once
                                  chunk_size=150000)     # Use 150k per cycle
    
    # Load or create persistent validation set
    val_set_file = 'training_data/val_set.json'
    if os.path.exists(val_set_file):
        print("Loading persistent validation set...")
        with open(val_set_file, 'r') as f:
            val_data = json.load(f)
        print(f"Loaded {len(val_data)} validation positions")
    else:
        print("Creating new validation set...")
        val_data = random.sample(dataset.data, min(8000, len(dataset.data) // 10))
        with open(val_set_file, 'w') as f:
            json.dump(val_data, f)
        print(f"Saved {len(val_data)} validation positions to {val_set_file}")
    
    log_file = 'training_data/log.csv'
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            f.write('epoch,train_loss,val_loss,val_mae\n')
    
    def evaluate():
        model.eval()
        loss = 0
        mae = 0
        with torch.no_grad():
            for fen, tgt in val_data[:2000]:
                board = chess.Board(fen)
                idx, feat = board_to_tensor(board)
                out, _ = model(idx.unsqueeze(0).to(device), feat.to(device))
                t = torch.tensor([[tgt]], dtype=torch.float32, device=device)
                loss += criterion(out, t).item()
                mae += abs(out.item() - tgt)
        return loss / 2000, mae / 2000
    
    stop = [False]
    def handler(s, f):
        print("\nStopping...")
        stop[0] = True
    signal.signal(signal.SIGINT, handler)
    
    print(f"\nTraining from epoch {epoch+1}")
    print("Press Ctrl+C to stop\n")
    
    try:
        while not stop[0]:
            # LR warmup after chunk reload (set before training)
            if epoch == chunk_reload_epoch:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 1e-4  # Warmup LR
                print(f"  🔥 LR warmup: reset to {1e-4:.1e}")
            
            current_lr = optimizer.param_groups[0]['lr']
            
            # Start timing the epoch
            epoch_start_time = time.time()
            samples_processed = 0
            
            model.train()
            # Increased batch size for M4 Max (36GB unified memory)
            # Reduced workers to avoid file descriptor leak on macOS
            loader = DataLoader(
                dataset, 
                batch_size=1024,           # Increased from 256 (2x larger)
                shuffle=True,
                num_workers=2,            # Reduced from 4 to prevent file leak
                pin_memory=(device.type == 'cuda'),  # Only on CUDA, not MPS
                persistent_workers=True   # Keep workers alive between epochs
            )
            ep_loss = 0
            batches = 0
            
            for idx, feat, tgt in loader:
                idx, feat, tgt = idx.to(device), feat.to(device), tgt.to(device)
                optimizer.zero_grad()
                out, aux = model(idx, feat)
                loss = criterion(out, tgt)
                if aux is not None and feat.size(1) >= 6:
                    loss += 0.1 * nn.functional.mse_loss(aux, feat[:, 5])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                ep_loss += loss.item()
                batches += 1
                samples_processed += idx.size(0)  # Track samples processed
                if stop[0]:
                    break
            
            # Calculate epoch timing and throughput
            epoch_time = time.time() - epoch_start_time
            samples_per_sec = samples_processed / epoch_time if epoch_time > 0 else 0
            
            avg_loss = ep_loss / max(1, batches)
            val_loss, val_mae = evaluate()
            
            is_best = val_loss < best_val
            if is_best:
                best_val = val_loss
            
            # Format time nicely
            if epoch_time < 60:
                time_str = f"{epoch_time:.1f}s"
            else:
                mins = int(epoch_time // 60)
                secs = int(epoch_time % 60)
                time_str = f"{mins}m{secs:02d}s"
            
            # Print epoch summary with timing info
            print(f"Epoch {epoch+1:4d} | Train: {avg_loss:.4f} | Val: {val_loss:.4f} (MAE: {val_mae:.3f}) | "
                  f"LR: {current_lr:.1e} | Time: {time_str} | {samples_per_sec:.0f} samples/s | Chunk {chunk}" 
                  + (" 🌟" if is_best else ""))
            
            with open(log_file, 'a') as f:
                f.write(f"{epoch+1},{avg_loss:.6f},{val_loss:.6f},{val_mae:.6f}\n")
            
            ckpt = {
                'epoch': epoch+1, 
                'model_state_dict': model.state_dict(), 
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val': best_val,
                'chunk': chunk,
                'chunk_reload_epoch': chunk_reload_epoch
            }
            torch.save(ckpt, ckpt_file)
            if is_best:
                torch.save(ckpt, 'training_data/best_model.pth')
            
            # Step scheduler after epoch completes (unless we just did LR warmup)
            if epoch != chunk_reload_epoch:
                scheduler.step()
            
            # Resample from the large pool every 5 epochs
            if (epoch+1) % 5 == 0:
                print(f"  🔄 Resampling fresh batch from {len(dataset.all_positions)} position pool...")
                dataset.resample_chunk()
                chunk += 1
                chunk_reload_epoch = epoch + 1
                print(f"  📦 Chunk {chunk} ready with {len(dataset.data)} positions, will use LR warmup next epoch")
            
            epoch += 1
    finally:
        print(f"\nSaved! Best val: {best_val:.6f}\n")

if __name__ == "__main__":
    train()
