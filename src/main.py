import chess
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChessPositionNet(nn.Module):
    def __init__(self):
        super(ChessPositionNet, self).__init__()
        self.emb = nn.Embedding(13*8*8, 1024)
        self.fc1 = nn.Linear(1024 + 1, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, side_to_move):
        x = self.emb(x)
        x = x.sum(axis=1)

        if side_to_move.dim() == 0:
            side_to_move = side_to_move.view(1,1)
        elif side_to_move.dim() == 1:
            side_to_move = side_to_move.unsqueeze(1)
        elif side_to_move.dim() > 2:
            side_to_move = side_to_move.view(side_to_move.size(0), -1)

        if side_to_move.size(1) != 1:
            side_to_move = side_to_move[:, :1]

        x = torch.cat([x, side_to_move], dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def board_to_tensor(board):
    piece_map = board.piece_map()
    indices = torch.zeros(8*8, dtype=torch.long)

    piece_to_index = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5
    }

    for square in range(64):
        row = 7 - (square // 8)
        col = square % 8
        piece = board.piece_at(square)
        if piece:
            index = piece_to_index[piece.piece_type]
            if piece.color == chess.BLACK:
                index += 6
            indices[square] = row + 8*col + 8*8*(index+1)
        else:
            indices[square] = 0

    side_to_move = torch.tensor([[1.0 if board.turn == chess.WHITE else -1.0]], dtype=torch.float32)
    
    return indices, side_to_move

if __name__ == "__main__":
    #board = chess.Board("2bq1rk1/pr3ppn/1p2p3/7P/2pP1B1P/2P5/PPQ2PB1/R3R1K1 w - -")
    board = chess.Board("1n3rk1/1pq1bp1p/r2pb3/4p1p1/4n3/1N6/PPP2PPP/2KR3R w - - 0 15") 
    net = ChessPositionNet()
    indices, side_to_move = board_to_tensor(board)
    indices = indices.unsqueeze(0)  # Add batch dimension
    score = net(indices, side_to_move)
    print("Position score:", score.item())