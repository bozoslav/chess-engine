#include "movegen.h"

#include <cstdint>

#include "attacks.h"

namespace {

constexpr int kPieceTypeCount = 7;

struct PositionBitboards {
  Bitboard pieces[2][kPieceTypeCount] = {};
  Bitboard occupancy[2] = {};
  Bitboard all = 0;
};

constexpr int colorIndex(Color color) { return static_cast<int>(color); }

constexpr bool isPromotionRank(Color side, Square square) {
  const int rank = bitboard::rankOf(square);
  return (side == Color::White && rank == 7) ||
         (side == Color::Black && rank == 0);
}

constexpr bool isPawnStartRank(Color side, Square square) {
  const int rank = bitboard::rankOf(square);
  return (side == Color::White && rank == 1) ||
         (side == Color::Black && rank == 6);
}

PositionBitboards buildBitboards(const Board& board) {
  PositionBitboards bitboards;

  for (int x = 0; x < Board::kBoardSize; ++x) {
    for (int y = 0; y < Board::kBoardSize; ++y) {
      const Piece piece = board.at(x, y);
      if (piece == Piece::None) continue;

      const Color color = isWhitePiece(piece) ? Color::White : Color::Black;
      const PieceType type = pieceType(piece);
      const Bitboard pieceBit = bitboard::bit(bitboard::squareFromCoords(x, y));
      const int sideIndex = colorIndex(color);
      const int typeIndex = static_cast<int>(type);

      bitboards.pieces[sideIndex][typeIndex] |= pieceBit;
      bitboards.occupancy[sideIndex] |= pieceBit;
      bitboards.all |= pieceBit;
    }
  }

  return bitboards;
}

void pushLegalMove(Board& board, MoveList& moves, Square from, Square to,
                   PieceType promotion = PieceType::None) {
  const Move move{bitboard::coordX(from), bitboard::coordY(from),
                  bitboard::coordX(to), bitboard::coordY(to), promotion};

  if (board.makeMove(move)) {
    moves.push(move);
    board.undoMove();
  }
}

void pushPawnMove(Board& board, MoveList& moves, Color side, Square from,
                  Square to) {
  if (!isPromotionRank(side, to)) {
    pushLegalMove(board, moves, from, to);
    return;
  }

  static constexpr PieceType kPromotions[4] = {
      PieceType::Queen,
      PieceType::Rook,
      PieceType::Bishop,
      PieceType::Knight,
  };

  for (const PieceType promotion : kPromotions) {
    pushLegalMove(board, moves, from, to, promotion);
  }
}

void genPawnMoves(Board& board, MoveList& moves,
                  const PositionBitboards& bitboards, Color side) {
  const int sideIndex = colorIndex(side);
  const Bitboard own = bitboards.occupancy[sideIndex];
  Bitboard pawns =
      bitboards.pieces[sideIndex][static_cast<int>(PieceType::Pawn)];
  const int forward = side == Color::White ? 8 : -8;

  while (pawns != 0) {
    const Square from = bitboard::popLsb(pawns);
    const Square oneForward = from + forward;

    if (oneForward >= 0 && oneForward < bitboard::kSquareCount &&
        (bitboards.all & bitboard::bit(oneForward)) == 0) {
      pushPawnMove(board, moves, side, from, oneForward);

      const Square twoForward = from + 2 * forward;
      if (isPawnStartRank(side, from) &&
          (bitboards.all & bitboard::bit(twoForward)) == 0) {
        pushLegalMove(board, moves, from, twoForward);
      }
    }

    Bitboard captures = AttackTables::pawnAttacks(side, from) & ~own;
    while (captures != 0) {
      const Square to = bitboard::popLsb(captures);
      pushPawnMove(board, moves, side, from, to);
    }
  }
}

void genAttackMoves(Board& board, MoveList& moves, Bitboard pieces,
                    Bitboard own, Bitboard all, PieceType type) {
  while (pieces != 0) {
    const Square from = bitboard::popLsb(pieces);
    Bitboard attacks = 0;

    switch (type) {
      case PieceType::Knight:
        attacks = AttackTables::knightAttacks(from);
        break;
      case PieceType::Bishop:
        attacks = AttackTables::bishopAttacks(from, all);
        break;
      case PieceType::Rook:
        attacks = AttackTables::rookAttacks(from, all);
        break;
      case PieceType::Queen:
        attacks = AttackTables::queenAttacks(from, all);
        break;
      case PieceType::King:
        attacks = AttackTables::kingAttacks(from);
        break;
      case PieceType::Pawn:
      case PieceType::None:
      default:
        break;
    }

    attacks &= ~own;
    while (attacks != 0) {
      pushLegalMove(board, moves, from, bitboard::popLsb(attacks));
    }

    if (type == PieceType::King && (from == 4 || from == 60)) {
      pushLegalMove(board, moves, from, from + 2);
      pushLegalMove(board, moves, from, from - 2);
    }
  }
}

void ensureAttackTablesInitialized() {
  if (!AttackTables::initialized()) AttackTables::init();
}

}  // namespace

void genLegalMoves(Board& board, MoveList& moves) {
  ensureAttackTablesInitialized();

  const Color side = board.sideToMove();
  const int sideIndex = colorIndex(side);
  const PositionBitboards bitboards = buildBitboards(board);
  const Bitboard own = bitboards.occupancy[sideIndex];
  const Bitboard all = bitboards.all;

  moves.clear();

  genPawnMoves(board, moves, bitboards, side);
  genAttackMoves(
      board, moves,
      bitboards.pieces[sideIndex][static_cast<int>(PieceType::Knight)], own,
      all, PieceType::Knight);
  genAttackMoves(
      board, moves,
      bitboards.pieces[sideIndex][static_cast<int>(PieceType::Bishop)], own,
      all, PieceType::Bishop);
  genAttackMoves(board, moves,
                 bitboards.pieces[sideIndex][static_cast<int>(PieceType::Rook)],
                 own, all, PieceType::Rook);
  genAttackMoves(
      board, moves,
      bitboards.pieces[sideIndex][static_cast<int>(PieceType::Queen)], own, all,
      PieceType::Queen);
  genAttackMoves(board, moves,
                 bitboards.pieces[sideIndex][static_cast<int>(PieceType::King)],
                 own, all, PieceType::King);
}

std::uint64_t perft(Board& board, int depth) {
  if (depth == 0) return 1;

  std::uint64_t nodes = 0;

  MoveList moves;
  genLegalMoves(board, moves);
  if (moves.overflowed()) return 0;

  for (const Move move : moves) {
    board.makeMove(move);
    nodes += perft(board, depth - 1);
    board.undoMove();
  }

  return nodes;
}
