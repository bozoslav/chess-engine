#!/usr/bin/env python3
"""
Test your model on specific positions

Usage:
    python test_model.py [--model path/to/model.pth]
"""

import chess
import torch
import argparse
from model import ChessPositionNet, board_to_tensor

def load_model(model_path='training_data/best_model.pth'):
    """Load the trained model"""
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    print(f"Loading model from {model_path}...")
    model = ChessPositionNet(
        dropout=0.0,
        square_dropout=0.0,
        use_attention=True,
        aux_material_head=True
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    epoch = checkpoint.get('epoch', '?')
    best_val = checkpoint.get('best_val', '?')
    print(f"✓ Model loaded (epoch {epoch}, val loss: {best_val})")
    print(f"✓ Running on {device}\n")
    
    return model, device

def evaluate_position(model, device, fen):
    """Evaluate a position from FEN"""
    board = chess.Board(fen)
    
    with torch.no_grad():
        indices, features = board_to_tensor(board)
        indices = indices.unsqueeze(0).to(device)
        features = features.to(device)
        output, _ = model(indices, features)
        score = output.item()
    
    return score

def test_famous_positions(model, device):
    """Test model on some famous chess positions"""
    
    test_cases = [
        {
            "name": "Starting Position",
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "expected": "~0.0 (equal)",
        },
        {
            "name": "Scholar's Mate (White winning)",
            "fen": "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4",
            "expected": "+10.0 (White wins)",
        },
        {
            "name": "White up a Queen",
            "fen": "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "expected": "~+9.0 (White advantage)",
        },
        {
            "name": "Black up a Queen",
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1",
            "expected": "~-9.0 (Black advantage)",
        },
        {
            "name": "Endgame: King + Pawn vs King",
            "fen": "8/8/8/4k3/8/8/4P3/4K3 w - - 0 1",
            "expected": "~+1.0 (White advantage)",
        },
        {
            "name": "Italian Game (balanced)",
            "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
            "expected": "~0.0 to +0.3",
        },
    ]
    
    print("🧪 Testing model on famous positions:\n")
    print("="*70)
    
    for i, test in enumerate(test_cases, 1):
        board = chess.Board(test["fen"])
        score = evaluate_position(model, device, test["fen"])
        
        print(f"\n{i}. {test['name']}")
        print(f"   Expected: {test['expected']}")
        print(f"   Model: {score:+.2f} pawns")
        print(f"\n{board}\n")
        print("-"*70)

def interactive_mode(model, device):
    """Interactive mode - evaluate positions from FEN"""
    print("\n🎯 Interactive Evaluation Mode")
    print("="*70)
    print("Enter FEN positions to evaluate (or 'quit' to exit)")
    print("Example: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    print("="*70 + "\n")
    
    while True:
        fen = input("Enter FEN (or 'quit'): ").strip()
        
        if fen.lower() == 'quit':
            break
        
        if fen.lower() == 'start':
            fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        
        try:
            board = chess.Board(fen)
            score = evaluate_position(model, device, fen)
            
            print(f"\n{board}")
            print(f"\nEvaluation: {score:+.2f} pawns")
            
            if score > 2:
                print("💪 White has a strong advantage")
            elif score > 0.5:
                print("👍 White is slightly better")
            elif score > -0.5:
                print("🤝 Position is roughly equal")
            elif score > -2:
                print("👍 Black is slightly better")
            else:
                print("💪 Black has a strong advantage")
            
            print()
        except ValueError as e:
            print(f"❌ Invalid FEN: {e}\n")

def main():
    parser = argparse.ArgumentParser(description='Test your chess model')
    parser.add_argument('--model', type=str, default='training_data/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode')
    
    args = parser.parse_args()
    
    try:
        model, device = load_model(args.model)
    except FileNotFoundError:
        print(f"❌ Error: Model file not found at {args.model}")
        print("Train your model first or specify a different path with --model")
        return
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    if args.interactive:
        interactive_mode(model, device)
    else:
        test_famous_positions(model, device)
        print("\n💡 Tip: Use --interactive flag to evaluate custom positions!")

if __name__ == "__main__":
    main()
