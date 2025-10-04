import chess
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import zstandard as zstd
import io
import os
from main import ChessPositionNet, board_to_tensor

class LichessEvalDataset(Dataset):
    def __init__(self, jsonl_path, max_samples=100000):
        self.data = []
        self.jsonl_path = jsonl_path
        self.max_samples = max_samples
        self.offset = 0
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
                        
                        self.data.append((fen, normalized_eval))
                        loaded_count += 1
                        
                        if loaded_count % 50000 == 0:
                            print(f"Loaded {loaded_count} positions...")
                    
                    except Exception as e:
                        continue
        
        print(f"Dataset loaded: {len(self.data)} positions (offset now at {self.offset})")
    
    def reload_next_batch(self):
        """Load the next batch of data from where we left off"""
        print(f"\nLoading next {self.max_samples} positions from offset {self.offset}...")
        self._load_data()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        fen, eval_score = self.data[idx]
        board = chess.Board(fen)
        tensor = board_to_tensor(board)
        return tensor, torch.tensor([eval_score], dtype=torch.float32)

def train_model(model, dataset, batch_size=64, learning_rate=0.001, device='cpu', checkpoint_path='checkpoints/chess_model_checkpoint.pth', start_epoch=0, epochs_per_reload=10):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"Training on device: {device}")
    print(f"Training indefinitely (starting from epoch {start_epoch})...")
    print(f"Will reload new data every {epochs_per_reload} epochs.")
    print("Press Ctrl+C to stop training after the current epoch completes.")
    
    epoch = start_epoch
    stop_training = False
    
    def signal_handler(sig, frame):
        nonlocal stop_training
        print("\n\nCtrl+C detected. Will stop after current epoch completes...")
        stop_training = True
    
    import signal
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        while not stop_training:
            # Create new data loader for current dataset
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            
            model.train()
            total_loss = 0
            batch_count = 0
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
                
                if stop_training:
                    print("Stopping training in the middle of epoch...")
                    break
            
            avg_epoch_loss = total_loss / batch_count if batch_count > 0 else 0
            print(f"Epoch [{epoch+1}] completed. Average Loss: {avg_epoch_loss:.4f}")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_epoch_loss,
                }
                torch.save(checkpoint, checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")
            
            epoch += 1
            
            # Load new data every epochs_per_reload epochs
            if not stop_training and (epoch % epochs_per_reload == 0):
                dataset.reload_next_batch()
                print(f"Loaded fresh data for next {epochs_per_reload} epochs...")
            
    except Exception as e:
        print(f"\nError during training: {e}")
        print("Saving checkpoint...")
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_epoch_loss if 'avg_epoch_loss' in locals() else 0,
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
    
    # Save final checkpoint when stopping
    print("Saving final checkpoint...")
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_epoch_loss if 'avg_epoch_loss' in locals() else 0,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Final checkpoint saved to {checkpoint_path}")
    
    print("Training stopped!")
    return model

if __name__ == "__main__":
    # Set device (use MPS for Apple Silicon, or CPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Create checkpoints directory
    os.makedirs('checkpoints', exist_ok=True)
    checkpoint_path = 'checkpoints/chess_model_checkpoint.pth'
    
    # Load dataset (limit to 100k samples for testing)
    dataset = LichessEvalDataset('../data/lichess_db_eval.jsonl.zst', max_samples=100000)
    
    # Initialize model
    model = ChessPositionNet()
    start_epoch = 0
    
    # Load checkpoint if it exists
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming from epoch {start_epoch}")
    else:
        print("No checkpoint found. Starting training from scratch.")
    
    # Train model
    trained_model = train_model(model, dataset, batch_size=64, learning_rate=0.001, 
                                device=device, checkpoint_path=checkpoint_path, start_epoch=start_epoch)
    
    # Save final model
    torch.save(trained_model.state_dict(), 'chess_model_final.pth')
    print("Final model saved to chess_model_final.pth")
