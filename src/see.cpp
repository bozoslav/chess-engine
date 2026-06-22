#include "see.h"

#include <algorithm>

#include "attacks.h"
#include "evaluate.h"

namespace {

constexpr int kColorCount = 2;
constexpr int kPieceTypeCount = 7;
constexpr Square kNoSquare = -1;
constexpr int kMaxExchangePly = 32;

int colorIndex(Color color) {
  return color == Color::White ? static_cast<int>(Color::White)
                               : static_cast<int>(Color::Black);
}

int pieceIndex(PieceType type) { return static_cast<int>(type); }

Bitboard squareBit(Square square) { return bitboard::bit(square); }

Piece capturedPiece(const Board& board, Move move) {
  if (move.isEnPassant()) {
    return board.sideToMove() == Color::White ? Piece::BlackPawn
                                              : Piece::WhitePawn;
  }

  return board.at(move.toSquare());
}

Square enPassantCapturedSquare(Color side, Square target) {
  return target + (side == Color::White ? -8 : 8);
}

Bitboard attackersTo(const Bitboard pieces[kColorCount][kPieceTypeCount],
                     Color side, Square target, Bitboard occupancy) {
  const int sideIndex = colorIndex(side);
  const Bitboard pawns =
      AttackTables::pawnAttacks(oppositeColor(side), target) &
      pieces[sideIndex][pieceIndex(PieceType::Pawn)];
  const Bitboard knights = AttackTables::knightAttacks(target) &
                           pieces[sideIndex][pieceIndex(PieceType::Knight)];
  const Bitboard bishops = AttackTables::bishopAttacks(target, occupancy) &
                           (pieces[sideIndex][pieceIndex(PieceType::Bishop)] |
                            pieces[sideIndex][pieceIndex(PieceType::Queen)]);
  const Bitboard rooks = AttackTables::rookAttacks(target, occupancy) &
                         (pieces[sideIndex][pieceIndex(PieceType::Rook)] |
                          pieces[sideIndex][pieceIndex(PieceType::Queen)]);

  return pawns | knights | bishops | rooks;
}

struct Attacker {
  Square square = kNoSquare;
  PieceType type = PieceType::None;
};

Attacker leastValuableAttacker(
    const Bitboard pieces[kColorCount][kPieceTypeCount], Color side,
    Square target, Bitboard occupancy) {
  const int sideIndex = colorIndex(side);
  const Bitboard attackers = attackersTo(pieces, side, target, occupancy);
  if (attackers == 0) return {};

  constexpr PieceType kOrder[] = {PieceType::Pawn, PieceType::Knight,
                                  PieceType::Bishop, PieceType::Rook,
                                  PieceType::Queen};
  for (const PieceType type : kOrder) {
    Bitboard candidates = attackers & pieces[sideIndex][pieceIndex(type)];
    if (candidates != 0) {
      return {bitboard::lsb(candidates), type};
    }
  }

  return {};
}

void removePiece(Bitboard pieces[kColorCount][kPieceTypeCount], Color color,
                 PieceType type, Square square) {
  pieces[colorIndex(color)][pieceIndex(type)] &= ~squareBit(square);
}

void addPiece(Bitboard pieces[kColorCount][kPieceTypeCount], Color color,
              PieceType type, Square square) {
  pieces[colorIndex(color)][pieceIndex(type)] |= squareBit(square);
}

}  // namespace

int staticExchangeEvaluation(const Board& board, Move move) {
  if (!AttackTables::initialized()) AttackTables::init();

  const Piece movingPiece = board.at(move.fromSquare());
  if (movingPiece == Piece::None) return 0;

  const Color movingSide = board.sideToMove();
  if (!matchesColor(movingPiece, movingSide)) return 0;

  const PieceType movingType = pieceType(movingPiece);
  const Piece captured = capturedPiece(board, move);
  const PieceType capturedType = pieceType(captured);
  const PieceType placedType = move.isPromotion() ? move.promo() : movingType;
  const int promotionGain =
      move.isPromotion() ? pieceValue(placedType) - pieceValue(PieceType::Pawn)
                         : 0;

  if (!move.isCapture()) return promotionGain;
  if (captured == Piece::None) return promotionGain;

  Bitboard pieces[kColorCount][kPieceTypeCount] = {};
  constexpr Color kColors[] = {Color::White, Color::Black};
  for (const Color color : kColors) {
    for (int type = static_cast<int>(PieceType::Pawn);
         type <= static_cast<int>(PieceType::King); ++type) {
      pieces[colorIndex(color)][type] =
          board.pieces(color, static_cast<PieceType>(type));
    }
  }

  const Square from = move.fromSquare();
  const Square target = move.toSquare();
  const Bitboard fromBit = squareBit(from);
  const Bitboard targetBit = squareBit(target);
  Square capturedSquare = target;
  if (move.isEnPassant()) {
    capturedSquare = enPassantCapturedSquare(movingSide, target);
  }

  Bitboard occupancy = board.allPieces();
  removePiece(pieces, movingSide, movingType, from);
  occupancy &= ~fromBit;
  removePiece(pieces, oppositeColor(movingSide), capturedType, capturedSquare);
  if (move.isEnPassant()) occupancy &= ~squareBit(capturedSquare);
  addPiece(pieces, movingSide, placedType, target);
  occupancy |= targetBit;

  int gains[kMaxExchangePly] = {};
  gains[0] = pieceValue(capturedType) + promotionGain;

  Color attackerSide = oppositeColor(movingSide);
  Color targetOwner = movingSide;
  PieceType targetType = placedType;
  int depth = 0;

  while (depth + 1 < kMaxExchangePly) {
    const Attacker attacker =
        leastValuableAttacker(pieces, attackerSide, target, occupancy);
    if (attacker.square == kNoSquare) break;

    ++depth;
    gains[depth] = pieceValue(targetType) - gains[depth - 1];

    removePiece(pieces, targetOwner, targetType, target);
    removePiece(pieces, attackerSide, attacker.type, attacker.square);
    addPiece(pieces, attackerSide, attacker.type, target);
    occupancy &= ~squareBit(attacker.square);
    occupancy |= targetBit;

    targetOwner = attackerSide;
    targetType = attacker.type;
    attackerSide = oppositeColor(attackerSide);
  }

  while (depth > 0) {
    --depth;
    gains[depth] = -std::max(-gains[depth], gains[depth + 1]);
  }

  return gains[0];
}

bool staticExchangeNonLosing(const Board& board, Move move, int threshold) {
  if (!AttackTables::initialized()) AttackTables::init();

  const Piece movingPiece = board.at(move.fromSquare());
  if (movingPiece == Piece::None) return 0 >= threshold;

  const Color movingSide = board.sideToMove();
  if (!matchesColor(movingPiece, movingSide)) return 0 >= threshold;

  const PieceType movingType = pieceType(movingPiece);
  const Piece captured = capturedPiece(board, move);
  const PieceType capturedType = pieceType(captured);
  const PieceType placedType = move.isPromotion() ? move.promo() : movingType;
  const int promotionGain =
      move.isPromotion() ? pieceValue(placedType) - pieceValue(PieceType::Pawn)
                         : 0;

  if (!move.isCapture() || captured == Piece::None) {
    return promotionGain >= threshold;
  }

  const int initialGain = pieceValue(capturedType) + promotionGain - threshold;
  if (initialGain < 0) return false;

  Bitboard pieces[kColorCount][kPieceTypeCount] = {};
  constexpr Color kColors[] = {Color::White, Color::Black};
  for (const Color color : kColors) {
    for (int type = static_cast<int>(PieceType::Pawn);
         type <= static_cast<int>(PieceType::King); ++type) {
      pieces[colorIndex(color)][type] =
          board.pieces(color, static_cast<PieceType>(type));
    }
  }

  const Square from = move.fromSquare();
  const Square target = move.toSquare();
  Square capturedSquare = target;
  if (move.isEnPassant()) {
    capturedSquare = enPassantCapturedSquare(movingSide, target);
  }

  Bitboard occupancy = board.allPieces();
  removePiece(pieces, movingSide, movingType, from);
  occupancy &= ~squareBit(from);
  removePiece(pieces, oppositeColor(movingSide), capturedType, capturedSquare);
  if (move.isEnPassant()) occupancy &= ~squareBit(capturedSquare);
  addPiece(pieces, movingSide, placedType, target);
  occupancy |= squareBit(target);

  int gains[kMaxExchangePly] = {};
  gains[0] = initialGain;
  Color attackerSide = oppositeColor(movingSide);
  Color targetOwner = movingSide;
  PieceType targetType = placedType;
  int depth = 0;

  while (depth + 1 < kMaxExchangePly) {
    const Attacker attacker =
        leastValuableAttacker(pieces, attackerSide, target, occupancy);
    if (attacker.square == kNoSquare) break;

    ++depth;
    gains[depth] = pieceValue(targetType) - gains[depth - 1];
    if (std::max(-gains[depth - 1], gains[depth]) < 0) break;

    removePiece(pieces, targetOwner, targetType, target);
    removePiece(pieces, attackerSide, attacker.type, attacker.square);
    addPiece(pieces, attackerSide, attacker.type, target);
    occupancy &= ~squareBit(attacker.square);

    targetOwner = attackerSide;
    targetType = attacker.type;
    attackerSide = oppositeColor(attackerSide);
  }

  while (depth > 0) {
    --depth;
    gains[depth] = -std::max(-gains[depth], gains[depth + 1]);
  }
  return gains[0] >= 0;
}
