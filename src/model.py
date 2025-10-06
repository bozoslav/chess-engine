import chess
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChessPositionNet(nn.Module):
    def __init__(self, feature_count=1+4+1+1, use_attention=True, attn_dim=512, attn_heads=8, dropout=0.0, square_dropout=0.0, aux_material_head=True, activation='relu'):
        """Chess evaluation network.
        Args:
            feature_count: number of scalar features appended after pooling.
            use_attention: whether to apply a single self-attention block before pooling.
            attn_dim: internal attention embedding size (projected from 1024 embedding).
            attn_heads: number of attention heads.
        """
        super(ChessPositionNet, self).__init__()
        self.emb = nn.Embedding(13*8*8, 1024)
        self.square_dropout_p = square_dropout
        self.aux_material_head = aux_material_head
        act = activation.lower()
        if act == 'gelu':
            self.act = nn.GELU()
        else:
            self.act = nn.ReLU()
        self.feature_count = feature_count
        self.use_attention = use_attention
        self.attn_dim = attn_dim
        if use_attention:
            self.proj = nn.Linear(1024, attn_dim)
            self.attn = nn.MultiheadAttention(attn_dim, attn_heads, batch_first=True)
            self.ff = nn.Sequential(
                nn.Linear(attn_dim, attn_dim*2),
                nn.ReLU(),
                nn.Linear(attn_dim*2, attn_dim)
            )
            pooled_dim = attn_dim * 2
        else:
            self.proj = None
            pooled_dim = 1024 * 2
        self.norm = nn.LayerNorm(pooled_dim + feature_count)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.fc1 = nn.Linear(pooled_dim + feature_count, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        if self.aux_material_head:
            # Predict normalized material diff (already scaled to ~[-1,1])
            self.aux_head = nn.Linear(256, 1)
        else:
            self.aux_head = None
        # Parameter count summary
        total_params = sum(p.numel() for p in self.parameters())
        print(f"ChessPositionNet params: {total_params/1e6:.2f}M | attention={use_attention} | aux_material={self.aux_material_head} | sq_dropout={self.square_dropout_p}")

    def forward(self, x, features):
        x = self.emb(x)
        # Square dropout: randomly drop entire square embeddings pre-attention/pooling
        if self.training and self.square_dropout_p > 0:
            mask = torch.rand(x.size(0), x.size(1), device=x.device) < self.square_dropout_p
            x = x.masked_fill(mask.unsqueeze(-1), 0.0)
        if self.use_attention:
            y = self.proj(x)
            attn_out, _ = self.attn(y, y, y)
            y = y + attn_out
            ff_out = self.ff(y)
            y = y + ff_out
            mean_pool = y.mean(dim=1)
            max_pool, _ = y.max(dim=1)
            x = torch.cat([mean_pool, max_pool], dim=1)
        else:
            mean_pool = x.mean(dim=1)
            max_pool, _ = x.max(dim=1)
            x = torch.cat([mean_pool, max_pool], dim=1)
        if features.dim() == 1:
            features = features.unsqueeze(0)
        if features.dim() > 2:
            features = features.view(features.size(0), -1)
        if features.size(0) != x.size(0):
            if features.size(0) == 1:
                features = features.expand(x.size(0), -1)
        if features.size(1) != self.feature_count:
            if features.size(1) < self.feature_count:
                pad = self.feature_count - features.size(1)
                features = torch.cat([features, features.new_zeros(features.size(0), pad)], dim=1)
            else:
                features = features[:, :self.feature_count]
        x = torch.cat([x, features], dim=1)
        x = self.norm(x)
        x = self.dropout(x)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        out = self.fc3(x)
        aux = None
        if self.aux_head is not None:
            aux = self.aux_head(x).squeeze(-1)  # shape [B]
        return out, aux

def board_to_tensor(board, add_features=True):
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

    side_to_move = 1.0 if board.turn == chess.WHITE else -1.0
    if not add_features:
        return indices, torch.tensor([[side_to_move]], dtype=torch.float32)

    cr = [
        1.0 if board.has_kingside_castling_rights(chess.WHITE) else -1.0,
        1.0 if board.has_queenside_castling_rights(chess.WHITE) else -1.0,
        1.0 if board.has_kingside_castling_rights(chess.BLACK) else -1.0,
        1.0 if board.has_queenside_castling_rights(chess.BLACK) else -1.0,
    ]
    values = {chess.PAWN:1, chess.KNIGHT:3, chess.BISHOP:3, chess.ROOK:5, chess.QUEEN:9, chess.KING:0}
    material_white = 0
    material_black = 0
    non_pawn_total = 0
    for sq, piece in piece_map.items():
        v = values[piece.piece_type]
        if piece.color == chess.WHITE:
            material_white += v
        else:
            material_black += v
        if piece.piece_type not in (chess.PAWN, chess.KING):
            non_pawn_total += v
    material_diff = (material_white - material_black) / 39.0
    phase = non_pawn_total / 50.0
    features = torch.tensor([[side_to_move] + cr + [material_diff] + [phase]], dtype=torch.float32)
    return indices, features
