torch.tensor([1, 2, 3])
def board_to_tensor(board):
import chess.engine
import chess
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChessPositionNet(nn.Module):
    def __init__(self):
        super(ChessPositionNet, self).__init__()
        # Input: 12x8x8, flatten to 768
        self.fc1 = nn.Linear(12*8*8, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # Output: position score

    def forward(self, x):
        x = x.view(-1, 12*8*8)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

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

if __name__ == "__main__":
    board = chess.Board("1k1r4/pp1b1R2/3q2pp/4p3/2B5/4Q3/PPP2B2/2K5 b - - 0 1")
    net = ChessPositionNet()
    tensor = board_to_tensor(board)
    tensor = tensor.unsqueeze(0)  # Add batch dimension
    score = net(tensor)
    print("Position score:", score.item())

tensor = board_to_tensor(board)
print(tensor)

