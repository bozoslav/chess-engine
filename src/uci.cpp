#include "uci.h"

#include "movegen.h"

namespace {

constexpr Square kNoSquare = -1;

Square parseSquare(char file, char rank) {
  if (file < 'a' || file > 'h') return kNoSquare;
  if (rank < '1' || rank > '8') return kNoSquare;
  return (rank - '1') * bitboard::kFileCount + (file - 'a');
}

PieceType parsePromotion(char promotion) {
  switch (promotion) {
    case 'n':
      return PieceType::Knight;
    case 'b':
      return PieceType::Bishop;
    case 'r':
      return PieceType::Rook;
    case 'q':
      return PieceType::Queen;
    default:
      return PieceType::None;
  }
}

}

bool moveFromUci(const Board& board, std::string_view text, Move& move) {
  if (text.size() != 4 && text.size() != 5) return false;

  const Square from = parseSquare(text[0], text[1]);
  const Square to = parseSquare(text[2], text[3]);
  if (from == kNoSquare || to == kNoSquare) return false;

  const PieceType promotion =
      text.size() == 5 ? parsePromotion(text[4]) : PieceType::None;
  if (text.size() == 5 && promotion == PieceType::None) return false;

  MoveList moves;
  genLegalMoves(board, moves);
  if (moves.overflowed()) return false;

  for (const Move candidate : moves) {
    if (candidate.fromSquare() == from && candidate.toSquare() == to &&
        candidate.promo() == promotion) {
      move = candidate;
      return true;
    }
  }

  return false;
}
