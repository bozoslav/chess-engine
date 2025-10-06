#!/usr/bin/env python3
"""
Play chess against your trained model!

Usage:
    python play_vs_model.py [--model path/to/model.pth] [--depth 3] [--color white]

Controls:
    - Enter moves in UCI format (e.g., e2e4, g1f3)
    - Type 'quit' to exit
    - Type 'undo' to take back last move
    - Type 'show' to display the board
"""

import chess
import chess.svg
import torch
import argparse
import time
from model import ChessPositionNet, board_to_tensor

class ChessEngine:
    def __init__(self, model_path='training_data/best_model.pth', device=None):
        """Initialize the chess engine with trained model"""
        if device is None:
            self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        else:
            self.device = device
            
        print(f"Loading model from {model_path}...")
        self.model = ChessPositionNet(
            dropout=0.0,  # No dropout for inference
            square_dropout=0.0,
            use_attention=True,
            aux_material_head=True
        ).to(self.device)
        
        # Load model weights
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.model.eval()
        
        epoch = checkpoint.get('epoch', '?')
        best_val = checkpoint.get('best_val', '?')
        print(f"✓ Model loaded (epoch {epoch}, val loss: {best_val})")
        print(f"✓ Running on {self.device}")
    
    def evaluate_position(self, board):
        """Evaluate a chess position, returns score in pawns (from white's perspective)"""
        with torch.no_grad():
            indices, features = board_to_tensor(board)
            indices = indices.unsqueeze(0).to(self.device)
            features = features.to(self.device)
            output, _ = self.model(indices, features)
            score = output.item()
            
            # Flip score if black to move (model always evaluates from white's perspective)
            if not board.turn:  # Black to move
                score = -score
                
            return score
    
    def get_best_move(self, board, depth=3):
        """
        Find best move using minimax search with neural network evaluation
        
        Args:
            board: chess.Board position
            depth: search depth (1-5 recommended)
        
        Returns:
            best_move: chess.Move
            best_score: evaluation score
        """
        legal_moves = list(board.legal_moves)
        
        if not legal_moves:
            return None, 0.0
        
        if depth <= 0:
            return None, self.evaluate_position(board)
        
        best_move = None
        best_score = float('-inf') if board.turn else float('inf')
        
        # Track thinking time (no verbose output)
        start_time = time.time()
        
        for move in legal_moves:
            board.push(move)
            
            if depth == 1:
                # Leaf node - just evaluate
                score = self.evaluate_position(board)
            else:
                # Recursive search
                _, score, _ = self.get_best_move(board, depth - 1)
            
            board.pop()
            
            # Minimax logic
            if board.turn:  # White to move (maximizing)
                if score > best_score:
                    best_score = score
                    best_move = move
            else:  # Black to move (minimizing)
                if score < best_score:
                    best_score = score
                    best_move = move
        
        # Return thinking time along with best move
        elapsed = time.time() - start_time
        
        return best_move, best_score, elapsed


def print_board(board):
    """Print the chess board in a nice ASCII format"""
    print("\n" + "="*33)
    print(board)
    print("="*33)
    print(f"Turn: {'White' if board.turn else 'Black'}")
    print(f"FEN: {board.fen()}")
    print()


def get_player_move(board):
    """Get a valid move from the player"""
    while True:
        try:
            move_str = input("Your move (e.g., e2e4, or 'quit'/'undo'/'show'): ").strip().lower()
            
            if move_str == 'quit':
                return 'quit'
            elif move_str == 'undo':
                return 'undo'
            elif move_str == 'show':
                print_board(board)
                continue
            
            # Try to parse the move
            move = chess.Move.from_uci(move_str)
            
            if move in board.legal_moves:
                return move
            else:
                print(f"❌ Illegal move! Legal moves: {', '.join([m.uci() for m in list(board.legal_moves)[:10]])}...")
        except ValueError:
            print("❌ Invalid move format! Use UCI notation (e.g., e2e4, g1f3)")


def play_game(engine, player_color='white', depth=3):
    """Main game loop"""
    board = chess.Board()
    move_history = []
    
    print("\n" + "🎮 " + "="*50)
    print(f"  Playing as {player_color.upper()} against the model")
    print(f"  Model search depth: {depth}")
    print("="*52)
    
    print_board(board)
    
    player_is_white = (player_color.lower() == 'white')
    
    while not board.is_game_over():
        is_player_turn = (board.turn == player_is_white)
        
        if is_player_turn:
            # Player's turn
            print(f"\n{'White' if board.turn else 'Black'} to move (YOU)")
            move = get_player_move(board)
            
            if move == 'quit':
                print("\n👋 Thanks for playing!")
                return
            elif move == 'undo':
                if len(move_history) >= 2:
                    # Undo both player and engine moves
                    board.pop()
                    board.pop()
                    move_history = move_history[:-2]
                    print("↩️  Undone last 2 moves")
                    print_board(board)
                else:
                    print("❌ Nothing to undo!")
                continue
            
            board.push(move)
            move_history.append(move)
            print(f"✓ You played: {move.uci()}")
            
        else:
            # Engine's turn
            print(f"\n{'White' if board.turn else 'Black'} to move (ENGINE)")
            print("  🤔 Thinking...", end='', flush=True)
            
            move, score, think_time = engine.get_best_move(board, depth=depth)
            
            if move is None:
                break
            
            board.push(move)
            move_history.append(move)
            
            # Format thinking time
            if think_time < 1:
                time_str = f"{think_time*1000:.0f}ms"
            elif think_time < 60:
                time_str = f"{think_time:.1f}s"
            else:
                mins = int(think_time // 60)
                secs = int(think_time % 60)
                time_str = f"{mins}m{secs:02d}s"
            
            print(f"\r🤖 Engine plays: {move.uci()} (eval: {score:+.2f} pawns, time: {time_str})       ")
        
        print_board(board)
        
        # Show game status
        if board.is_check():
            print("⚠️  CHECK!")
    
    # Game over
    print("\n" + "="*52)
    print("🏁 GAME OVER")
    print("="*52)
    
    if board.is_checkmate():
        winner = "White" if not board.turn else "Black"
        print(f"🎉 Checkmate! {winner} wins!")
    elif board.is_stalemate():
        print("🤝 Stalemate - Draw!")
    elif board.is_insufficient_material():
        print("🤝 Draw by insufficient material")
    elif board.is_fifty_moves():
        print("🤝 Draw by fifty-move rule")
    elif board.is_repetition():
        print("🤝 Draw by threefold repetition")
    
    print(f"\nFinal position: {board.fen()}")
    print(f"Moves played: {len(move_history)}")


def main():
    parser = argparse.ArgumentParser(description='Play chess against your trained model')
    parser.add_argument('--model', type=str, default='training_data/best_model.pth',
                        help='Path to model checkpoint (default: training_data/best_model.pth)')
    parser.add_argument('--depth', type=int, default=3,
                        help='Search depth for engine (1-5, default: 3)')
    parser.add_argument('--color', type=str, default='white', choices=['white', 'black'],
                        help='Your color (default: white)')
    
    args = parser.parse_args()
    
    print("\n♟️  Welcome to Chess vs Model! ♟️\n")
    
    # Initialize engine
    try:
        engine = ChessEngine(model_path=args.model)
    except FileNotFoundError:
        print(f"❌ Error: Model file not found at {args.model}")
        print("Train your model first or specify a different path with --model")
        return
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Play game
    play_game(engine, player_color=args.color, depth=args.depth)


if __name__ == "__main__":
    main()
