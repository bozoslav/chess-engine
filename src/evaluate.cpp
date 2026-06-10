#include "evaluate.h"

namespace {

constexpr int kPieceSquare[7][bitboard::kSquareCount] = {
    {},
    {
        0,   0,   0,   0,   0,   0,   0,   0,
        50,  50,  50,  50,  50,  50,  50,  50,
        10,  10,  20,  30,  30,  20,  10,  10,
        5,   5,   10,  25,  25,  10,  5,   5,
        0,   0,   0,   20,  20,  0,   0,   0,
        5,   -5,  -10, 0,   0,   -10, -5,  5,
        5,   10,  10,  -20, -20, 10,  10,  5,
        0,   0,   0,   0,   0,   0,   0,   0,
    },
    {
        -50, -40, -30, -30, -30, -30, -40, -50,
        -40, -20, 0,   5,   5,   0,   -20, -40,
        -30, 5,   10,  15,  15,  10,  5,   -30,
        -30, 0,   15,  20,  20,  15,  0,   -30,
        -30, 5,   15,  20,  20,  15,  5,   -30,
        -30, 0,   10,  15,  15,  10,  0,   -30,
        -40, -20, 0,   0,   0,   0,   -20, -40,
        -50, -40, -30, -30, -30, -30, -40, -50,
    },
    {
        -20, -10, -10, -10, -10, -10, -10, -20,
        -10, 5,   0,   0,   0,   0,   5,   -10,
        -10, 10,  10,  10,  10,  10,  10,  -10,
        -10, 0,   10,  10,  10,  10,  0,   -10,
        -10, 5,   5,   10,  10,  5,   5,   -10,
        -10, 0,   5,   10,  10,  5,   0,   -10,
        -10, 0,   0,   0,   0,   0,   0,   -10,
        -20, -10, -10, -10, -10, -10, -10, -20,
    },
    {
        0,  0,  0,  5,  5,  0,  0,  0,
        -5, 0,  0,  0,  0,  0,  0,  -5,
        -5, 0,  0,  0,  0,  0,  0,  -5,
        -5, 0,  0,  0,  0,  0,  0,  -5,
        -5, 0,  0,  0,  0,  0,  0,  -5,
        -5, 0,  0,  0,  0,  0,  0,  -5,
        5,  10, 10, 10, 10, 10, 10, 5,
        0,  0,  0,  0,  0,  0,  0,  0,
    },
    {
        -20, -10, -10, -5,  -5,  -10, -10, -20,
        -10, 0,   5,   0,   0,   0,   0,   -10,
        -10, 5,   5,   5,   5,   5,   0,   -10,
        0,   0,   5,   5,   5,   5,   0,   -5,
        -5,  0,   5,   5,   5,   5,   0,   -5,
        -10, 0,   5,   5,   5,   5,   0,   -10,
        -10, 0,   0,   0,   0,   0,   0,   -10,
        -20, -10, -10, -5,  -5,  -10, -10, -20,
    },
    {
        20,  30,  10,  0,   0,   10,  30,  20,
        20,  20,  0,   0,   0,   0,   20,  20,
        -10, -20, -20, -20, -20, -20, -20, -10,
        -20, -30, -30, -40, -40, -30, -30, -20,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
    },
};

constexpr Square mirrorForBlack(Square square) { return square ^ 56; }

int evaluateColor(const Board& board, Color color) {
  int score = 0;

  for (int typeValue = static_cast<int>(PieceType::Pawn);
       typeValue <= static_cast<int>(PieceType::King); ++typeValue) {
    const PieceType type = static_cast<PieceType>(typeValue);
    Bitboard pieces = board.pieces(color, type);
    while (pieces != 0) {
      const Square square = bitboard::popLsb(pieces);
      const Square tableSquare =
          color == Color::White ? square : mirrorForBlack(square);
      score += pieceValue(type) + kPieceSquare[typeValue][tableSquare];
    }
  }

  if (bitboard::popcount(board.pieces(color, PieceType::Bishop)) >= 2) {
    score += 30;
  }

  return score;
}

}  // namespace

int pieceValue(PieceType type) {
  switch (type) {
    case PieceType::Pawn:
      return 100;
    case PieceType::Knight:
      return 320;
    case PieceType::Bishop:
      return 330;
    case PieceType::Rook:
      return 500;
    case PieceType::Queen:
      return 900;
    case PieceType::King:
    case PieceType::None:
    default:
      return 0;
  }
}

int evaluate(const Board& board) {
  const int whiteScore = evaluateColor(board, Color::White);
  const int blackScore = evaluateColor(board, Color::Black);
  const int score = whiteScore - blackScore;
  return board.sideToMove() == Color::White ? score : -score;
}
