#include "movegen.h"

#include <vector>

std::vector<Move> genLegalMoves(Board& board) {
  const Color side = board.sideToMove();

  std::vector<Move> ret;
  for (int i = 0; i < Board::kBoardSize; ++i) {
    for (int j = 0; j < Board::kBoardSize; ++j) {
      const Piece p = board.at(i, j);

      if (p == Piece::None) continue;
      if (!matchesColor(p, side)) continue;

      for (int x = 0; x < Board::kBoardSize; ++x) {
        for (int y = 0; y < Board::kBoardSize; ++y) {
          const bool isPromoSquare =
              pieceType(p) == PieceType::Pawn &&
              (x == 0 || x == Board::kBoardSize - 1);

          if (isPromoSquare) {
            const PieceType promos[4] = {
                PieceType::Queen,
                PieceType::Rook,
                PieceType::Bishop,
                PieceType::Knight,
            };

            for (PieceType promo : promos) {
              const Move m{i, j, x, y, promo};

              if (board.makeMove(m)) {
                ret.emplace_back(m);
                board.undoMove();
              }
            }
          } else {
            const Move m{i, j, x, y};

            if (board.makeMove(m)) {
              ret.emplace_back(m);
              board.undoMove();
            }
          }
        }
      }
    }
  }

  return ret;
}

int perft(Board& board, int depth) {
  if (depth == 0) return 1;

  int nodes = 0;

  std::vector<Move> moves = genLegalMoves(board);
  for (Move move : moves) {
    board.makeMove(move);
    nodes += perft(board, depth - 1);
    board.undoMove();
  }

  return nodes;
}
