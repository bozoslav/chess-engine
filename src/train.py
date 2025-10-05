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
from main import ChessPositionNet, board_to_tensor
import time


class LichessEvalDataset(Dataset):
    def __init__(self, jsonl_path, max_samples=100000, validation=False, val_fraction=0.05):
        self.data = []
        self.jsonl_path = jsonl_path
        self.max_samples = max_samples
        self.offset = 0
        self.validation = validation
        self.val_fraction = val_fraction
        self._val_cache = []
        print(f"Loading data from {jsonl_path}...")
        self._load_data()
        
    def _load_data(self):
        self.data = []
        loaded_count = 0
        
        with open(self.jsonl_path, 'rb') as f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                text_stream = io.TextIOWrapper(reader, encoding='utf-8')
                
                for i, line in enumerate(text_stream):
                    # Skip lines before offset
                    if i < self.offset:
                        continue
                    
                    if loaded_count >= self.max_samples:
                        self.offset = i  # Save position for next load
                        break
                    
                    try:
                        entry = json.loads(line.strip())
                        fen = entry['fen']
                        evals = entry.get('evals', [])
                        
                        if not evals:
                            continue
                        
                        # Find the deepest evaluation
                        deepest_eval = max(evals, key=lambda e: e.get('depth', 0))
                        pvs = deepest_eval.get('pvs', [])
                        
                        if not pvs or 'cp' not in pvs[0]:
                            continue
                        
                        # Get centipawn evaluation
                        cp = pvs[0]['cp']
                        
                        # Normalize centipawn to a reasonable range
                        # Divide by 100 to convert centipawns to pawns
                        normalized_eval = cp / 100.0
                        
                        # Clip to [-10, 10] to avoid extreme values
                        normalized_eval = max(-10.0, min(10.0, normalized_eval))
                        
                        if self.validation and len(self._val_cache) < int(self.max_samples * self.val_fraction):
                            self._val_cache.append((fen, normalized_eval))
                        else:
                            self.data.append((fen, normalized_eval))
                        loaded_count += 1
                        
                        if loaded_count % 50000 == 0:
                            print(f"Loaded {loaded_count} positions...")
                    
                    except Exception as e:
                        continue
        
        print(f"Dataset loaded: {len(self.data)} positions (offset now at {self.offset})")
    
    def reload_next_batch(self):
        print(f"\nLoading next {self.max_samples} positions from offset {self.offset}...")
        self._load_data()

    def validation_slice(self):
        if self._val_cache:
            return self._val_cache
        # fallback simple slice
        return self.data[: max(1, int(0.05 * len(self.data)))]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        fen, eval_score = self.data[idx]
        board = chess.Board(fen)
        indices, side_to_move = board_to_tensor(board)
        return indices, side_to_move.squeeze(0), torch.tensor([eval_score], dtype=torch.float32)

class RunningStats:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
    def update(self, x: float):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2
    def variance(self):
        return self.M2 / (self.n - 1) if self.n > 1 else 1.0
    def std(self):
        return math.sqrt(self.variance())


def train_model(model, dataset, batch_size=256, base_lr=3e-4, device='cpu', checkpoint_path='training_data/chess_model_checkpoint.pth', start_epoch=0, epochs_per_reload=5, replay_ratio=0.3, weight_decay=1e-4, use_standardization=True, max_epochs=None):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    stats = RunningStats()  # raw target stats (pawns)
    # We'll store mean/std used for standardization (lagged so we don't leak batch info)
    standardize = use_standardization
    
    print(f"Training on device: {device}")
    print(f"Training (starting epoch {start_epoch}) with reload every {epochs_per_reload} epochs.")
    print("Press Ctrl+C to stop training after the current epoch completes.")
    wall_start = time.time()
    
    epoch = start_epoch
    stop_training = False
    
    def signal_handler(sig, frame):
        nonlocal stop_training
        print("\n\nCtrl+C detected. Will stop after current epoch completes...")
        stop_training = True
    
    import signal
    signal.signal(signal.SIGINT, signal_handler)
    
    replay_buffer = []  # stores (fen, eval) tuples from previous chunks
    BEST_VAL = None
    LOG_PATH = 'training_data/training_log.csv'
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, 'w') as lf:
            lf.write('epoch,train_loss,val_loss,val_mae,lr,mean,std,epoch_sec,samples_per_sec,elapsed_sec\n')

    val_slice = dataset.validation_slice()

    def evaluate():
        model.eval()
        with torch.no_grad():
            losses = []  # loss in standardized space if enabled else raw
            abs_err = []  # MAE in raw pawn units
            for fen, target in val_slice[:2000]:  # cap for speed
                board = chess.Board(fen)
                pidx, stm = board_to_tensor(board)
                # stm already shape [1,1]; avoid unsqueeze duplication
                out = model(pidx.unsqueeze(0).to(device), stm.to(device))
                # Convert output to raw scale (already raw because model predicts raw). If standardizing, apply same transform for loss.
                tgt_raw = torch.tensor([[target]], dtype=torch.float32, device=device)
                if standardize and stats.n > 10:
                    cur_mean = stats.mean
                    cur_std = stats.std()
                    if cur_std < 1e-6:
                        cur_std = 1.0
                    tgt_std = (tgt_raw - cur_mean) / cur_std
                    out_std = (out - cur_mean) / cur_std
                    loss = criterion(out_std, tgt_std).item()
                else:
                    loss = criterion(out, tgt_raw).item()
                losses.append(loss)
                abs_err.append(abs(out.item() - target))
            return (sum(losses)/len(losses) if losses else 0.0), (sum(abs_err)/len(abs_err) if abs_err else 0.0)

    try:
        while not stop_training:
            model.train()
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            total_loss = 0
            batches = 0
            epoch_start = time.time()
            samples_this_epoch = 0
            for inputs, side_to_move, targets in train_loader:
                inputs = inputs.to(device)
                side_to_move = side_to_move.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs, side_to_move)  # raw predictions (pawns)
                # Update running stats with raw targets BEFORE using them for this batch standardization (online estimate)
                for v in targets.view(-1).tolist():
                    stats.update(v)
                if standardize and stats.n > 10:
                    cur_mean = stats.mean
                    cur_std = stats.std()
                    if cur_std < 1e-6:
                        cur_std = 1.0
                    targets_std = (targets - cur_mean) / cur_std
                    outputs_std = (outputs - cur_mean) / cur_std
                    loss = criterion(outputs_std, targets_std)
                else:
                    loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step(epoch + batches/ max(1,len(train_loader)))
                batch_size_eff = inputs.size(0)
                total_loss += loss.item()
                samples_this_epoch += batch_size_eff
                batches += 1
                if stop_training:
                    break
            avg_loss = total_loss / max(1, batches)
            val_loss, val_mae = evaluate()
            cur_lr = optimizer.param_groups[0]['lr']
            loss_space = 'std' if (standardize and stats.n > 10) else 'raw'
            epoch_time = time.time() - epoch_start
            elapsed = time.time() - wall_start
            sps = samples_this_epoch / epoch_time if epoch_time > 0 else 0.0
            print(f"Epoch {epoch+1} | train {avg_loss:.4f} ({loss_space}) | val {val_loss:.4f} ({loss_space}) | MAE {val_mae:.3f} pawns | lr {cur_lr:.2e} | mean {stats.mean:.2f} std {stats.std():.2f} | {epoch_time:.1f}s | {sps:.1f} samp/s | total {elapsed/60:.1f}m")
            with open(LOG_PATH, 'a') as lf:
                lf.write(f"{epoch+1},{avg_loss:.6f},{val_loss:.6f},{val_mae:.6f},{cur_lr:.6e},{stats.mean:.6f},{stats.std():.6f},{epoch_time:.3f},{sps:.2f},{elapsed:.2f}\n")
            if BEST_VAL is None or val_loss < BEST_VAL:
                BEST_VAL = val_loss
                torch.save({'epoch': epoch+1,'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),'loss': val_loss,'target_mean': stats.mean,'target_std': stats.std(),'standardized_loss': (standardize and stats.n > 10), 'epoch_time': epoch_time, 'samples_per_sec': sps, 'elapsed_sec': elapsed}, 'training_data/best_model.pth')
                print("New best model saved.")
            if (epoch+1) % epochs_per_reload == 0:
                # Preserve a portion of current batch into replay buffer
                if replay_ratio > 0 and len(dataset.data) > 0:
                    sample_size = int(replay_ratio * len(dataset.data))
                    if sample_size > 0:
                        replay_samples = random.sample(dataset.data, sample_size)
                        replay_buffer.extend(replay_samples)
                        # Keep buffer from growing unbounded: cap at 3x current dataset size
                        max_buffer = 3 * len(dataset.data)
                        if len(replay_buffer) > max_buffer:
                            replay_buffer = replay_buffer[-max_buffer:]
                        print(f"Replay buffer updated: {len(replay_buffer)} positions (added {sample_size}).")
                dataset.reload_next_batch()
                # Mix in replay samples
                if replay_buffer:
                    mix_size = min(len(replay_buffer), int(replay_ratio * len(dataset.data))) if replay_ratio > 0 else 0
                    if mix_size > 0:
                        mix_samples = random.sample(replay_buffer, mix_size)
                        dataset.data.extend(mix_samples)
                        print(f"Mixed {mix_size} replay samples into current dataset (total now {len(dataset.data)}).")
                val_slice = dataset.validation_slice()
            if (epoch+1) % 5 == 0:
                torch.save({'epoch': epoch+1,'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),'loss': val_loss,'target_mean': stats.mean,'target_std': stats.std(),'standardized_loss': (standardize and stats.n > 10), 'replay_buffer_size': len(replay_buffer), 'replay_ratio': replay_ratio, 'epoch_time': epoch_time, 'samples_per_sec': sps, 'elapsed_sec': elapsed}, checkpoint_path)
            epoch += 1
            # Early stop if max_epochs reached
            if max_epochs is not None and epoch >= max_epochs:
                print(f"Reached max_epochs={max_epochs}; stopping training loop.")
                stop_training = True
            
            
    except Exception as e:
        print(f"\nError during training: {e}")
        print("Saving checkpoint...")
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': BEST_VAL if BEST_VAL is not None else 0,
            'target_mean': stats.mean,
            'target_std': stats.std(),
            'standardized_loss': (standardize and stats.n > 10),
            'replay_buffer_size': len(replay_buffer),
            'replay_ratio': replay_ratio
        }
        torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")
    
    # Save final checkpoint when stopping
    print("Saving final checkpoint...")
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': BEST_VAL if BEST_VAL is not None else 0,
        'target_mean': stats.mean,
        'target_std': stats.std(),
        'standardized_loss': (standardize and stats.n > 10),
        'replay_buffer_size': len(replay_buffer),
        'replay_ratio': replay_ratio
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Final checkpoint saved to {checkpoint_path}")
    
    print("Training stopped!")
    return model

if __name__ == "__main__":
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print('Using MPS')
    else:
        device = torch.device('cpu')
        print('Using CPU')
    os.makedirs('training_data', exist_ok=True)
    dataset = LichessEvalDataset('../data/lichess_db_eval.jsonl.zst', max_samples=100000)
    model = ChessPositionNet()
    trained = train_model(model, dataset, device=device)
    torch.save(trained.state_dict(), 'training_data/chess_model_final.pth')
    print('Final model saved to training_data/chess_model_final.pth')
