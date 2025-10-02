import chess.engine
import chess
import torch

torch.tensor([1, 2, 3])

board = chess.Board("1k1r4/pp1b1R2/3q2pp/4p3/2B5/4Q3/PPP2B2/2K5 b - - 0 1")

def board_to_tensor(board):
    piece_map = board.piece_map()
    tensor = torch.zeros((12, 8, 8), dtype=torch.float32)

    piece_to_index = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5
    }

    for square, piece in piece_map.items():
        row = 7 - (square // 8)
        col = square % 8
        index = piece_to_index[piece.piece_type]
        if piece.color == chess.BLACK:
            index += 6
        tensor[index, row, col] = 1.0

    return tensor

tensor = board_to_tensor(board)
print(tensor)

