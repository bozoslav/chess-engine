#include "movegen.h"

#include <cstdint>

#include "attacks.h"

namespace {

constexpr Square kNoSquare = -1;

constexpr Bitboard squareBit(Square square) { return bitboard::bit(square); }

constexpr bool isValidSquare(Square square) {
  return square >= 0 && square < bitboard::kSquareCount;
}

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

Piece pieceAt(const Board& board, Square square) {
  return board.at(bitboard::coordX(square), bitboard::coordY(square));
}

void pushMove(MoveList& moves, Square from, Square to,
              MoveFlag flag = MoveFlag::Quiet) {
  moves.push(Move::fromSquares(from, to, flag));
}

void pushPawnMove(MoveList& moves, Color side, Square from, Square to,
                  bool isCapture) {
  if (!isPromotionRank(side, to)) {
    pushMove(moves, from, to, isCapture ? MoveFlag::Capture : MoveFlag::Quiet);
    return;
  }

  static constexpr PieceType kPromotions[4] = {
      PieceType::Queen,
      PieceType::Rook,
      PieceType::Bishop,
      PieceType::Knight,
  };

  for (const PieceType promotion : kPromotions) {
    pushMove(moves, from, to, Move::promotionFlag(promotion, isCapture));
  }
}

Bitboard attackersTo(const Board& board, Square square, Color attackingColor,
                     Bitboard occupancy) {
  const Bitboard pawns = board.pieces(attackingColor, PieceType::Pawn);
  const Bitboard knights = board.pieces(attackingColor, PieceType::Knight);
  const Bitboard bishops = board.pieces(attackingColor, PieceType::Bishop);
  const Bitboard rooks = board.pieces(attackingColor, PieceType::Rook);
  const Bitboard queens = board.pieces(attackingColor, PieceType::Queen);
  const Bitboard king = board.pieces(attackingColor, PieceType::King);

  return (AttackTables::pawnAttacks(oppositeColor(attackingColor), square) &
          pawns) |
         (AttackTables::knightAttacks(square) & knights) |
         (AttackTables::bishopAttacks(square, occupancy) & (bishops | queens)) |
         (AttackTables::rookAttacks(square, occupancy) & (rooks | queens)) |
         (AttackTables::kingAttacks(square) & king);
}

bool isSquareAttacked(const Board& board, Square square, Color attackingColor,
                      Bitboard occupancy) {
  return attackersTo(board, square, attackingColor, occupancy) != 0;
}

bool areAligned(Square first, Square second) {
  const int firstRank = bitboard::rankOf(first);
  const int firstFile = bitboard::fileOf(first);
  const int secondRank = bitboard::rankOf(second);
  const int secondFile = bitboard::fileOf(second);
  const int dr = secondRank - firstRank;
  const int df = secondFile - firstFile;

  return dr == 0 || df == 0 || dr == df || dr == -df;
}

Bitboard squaresBetween(Square first, Square second) {
  if (!areAligned(first, second)) return 0;

  const int firstRank = bitboard::rankOf(first);
  const int firstFile = bitboard::fileOf(first);
  const int secondRank = bitboard::rankOf(second);
  const int secondFile = bitboard::fileOf(second);
  const int stepRank = (secondRank > firstRank) - (secondRank < firstRank);
  const int stepFile = (secondFile > firstFile) - (secondFile < firstFile);

  Bitboard mask = 0;
  int rank = firstRank + stepRank;
  int file = firstFile + stepFile;
  while (rank != secondRank || file != secondFile) {
    mask |= squareBit(rank * bitboard::kFileCount + file);
    rank += stepRank;
    file += stepFile;
  }

  return mask;
}

bool isOrthogonalDirection(int dr, int df) { return dr == 0 || df == 0; }

bool isDiagonalDirection(int dr, int df) { return dr != 0 && df != 0; }

bool sliderMatchesDirection(PieceType type, int dr, int df) {
  if (type == PieceType::Queen) return true;
  if (isOrthogonalDirection(dr, df)) return type == PieceType::Rook;
  if (isDiagonalDirection(dr, df)) return type == PieceType::Bishop;
  return false;
}

struct PinInfo {
  Bitboard pinned = 0;
  Bitboard mask[bitboard::kSquareCount] = {};
};

PinInfo computePins(const Board& board, Color side) {
  static constexpr int kDirections[8][2] = {
      {1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1},
  };

  PinInfo pins;
  const Color enemy = oppositeColor(side);
  const Square kingSquare = board.kingSquare(side);
  const int kingRank = bitboard::rankOf(kingSquare);
  const int kingFile = bitboard::fileOf(kingSquare);

  for (const auto& direction : kDirections) {
    const int dr = direction[0];
    const int df = direction[1];
    Square blocker = kNoSquare;
    Bitboard ray = 0;

    int rank = kingRank + dr;
    int file = kingFile + df;
    while (rank >= 0 && rank < bitboard::kRankCount && file >= 0 &&
           file < bitboard::kFileCount) {
      const Square square = rank * bitboard::kFileCount + file;
      const Piece piece = pieceAt(board, square);

      if (piece != Piece::None) {
        if (matchesColor(piece, side)) {
          if (blocker != kNoSquare) break;
          blocker = square;
        } else {
          if (blocker != kNoSquare && matchesColor(piece, enemy) &&
              sliderMatchesDirection(pieceType(piece), dr, df)) {
            pins.pinned |= squareBit(blocker);
            pins.mask[blocker] = ray | squareBit(square);
          }
          break;
        }
      }

      ray |= squareBit(square);
      rank += dr;
      file += df;
    }
  }

  return pins;
}

Bitboard evasionMaskForSingleCheck(const Board& board, Square kingSquare,
                                   Square checkerSquare) {
  const Piece checker = pieceAt(board, checkerSquare);
  const PieceType checkerType = pieceType(checker);
  if (checkerType == PieceType::Bishop || checkerType == PieceType::Rook ||
      checkerType == PieceType::Queen) {
    return squaresBetween(kingSquare, checkerSquare) | squareBit(checkerSquare);
  }

  return squareBit(checkerSquare);
}

Bitboard legalTargetMaskForPiece(Square from, Bitboard targets,
                                 const PinInfo& pins, Bitboard evasionMask) {
  if ((pins.pinned & squareBit(from)) != 0) {
    targets &= pins.mask[from];
  }

  targets &= evasionMask;
  return targets;
}

void generateKingMoves(const Board& board, MoveList& moves, Color side,
                       Bitboard all, Bitboard own, Bitboard enemyKing,
                       Bitboard enemyPieces, bool noisyOnly) {
  const Color enemy = oppositeColor(side);
  const Square from = board.kingSquare(side);
  const Bitboard fromBit = squareBit(from);
  Bitboard targets = AttackTables::kingAttacks(from) & ~own & ~enemyKing;
  if (noisyOnly) targets &= enemyPieces;
  const Bitboard occupancyWithoutKing = all & ~fromBit;

  while (targets != 0) {
    const Square to = bitboard::popLsb(targets);
    if (!isSquareAttacked(board, to, enemy, occupancyWithoutKing)) {
      pushMove(moves, from, to,
               (enemyPieces & squareBit(to)) != 0 ? MoveFlag::Capture
                                                  : MoveFlag::Quiet);
    }
  }
}

void generateCastling(const Board& board, MoveList& moves, Color side,
                      Bitboard all, Bitboard checkers) {
  if (checkers != 0) return;

  const Color enemy = oppositeColor(side);
  const Square kingFrom = board.kingSquare(side);
  const Square expectedKingFrom = side == Color::White ? 4 : 60;
  if (kingFrom != expectedKingFrom) return;

  const Square kingSideRook = side == Color::White ? 7 : 63;
  const Square queenSideRook = side == Color::White ? 0 : 56;

  if (board.canCastleKingSide(side) &&
      (board.pieces(side, PieceType::Rook) & squareBit(kingSideRook)) != 0) {
    const Square transit = kingFrom + 1;
    const Square target = kingFrom + 2;
    const Bitboard emptyMask = squareBit(transit) | squareBit(target);
    if ((all & emptyMask) == 0 &&
        !isSquareAttacked(board, transit, enemy, all) &&
        !isSquareAttacked(board, target, enemy, all)) {
      pushMove(moves, kingFrom, target, MoveFlag::KingCastle);
    }
  }

  if (board.canCastleQueenSide(side) &&
      (board.pieces(side, PieceType::Rook) & squareBit(queenSideRook)) != 0) {
    const Square transit = kingFrom - 1;
    const Square target = kingFrom - 2;
    const Square gap = kingFrom - 3;
    const Bitboard emptyMask =
        squareBit(transit) | squareBit(target) | squareBit(gap);
    if ((all & emptyMask) == 0 &&
        !isSquareAttacked(board, transit, enemy, all) &&
        !isSquareAttacked(board, target, enemy, all)) {
      pushMove(moves, kingFrom, target, MoveFlag::QueenCastle);
    }
  }
}

void generatePawnMoves(const Board& board, MoveList& moves, Color side,
                       const PinInfo& pins, Bitboard evasionMask,
                       Bitboard checkers, bool noisyOnly) {
  const Color enemy = oppositeColor(side);
  const Bitboard all = board.allPieces();
  const Bitboard enemyKing = board.pieces(enemy, PieceType::King);
  const Bitboard enemyPieces = board.occupancy(enemy) & ~enemyKing;
  const Square kingSquare = board.kingSquare(side);
  const int forward = side == Color::White ? 8 : -8;
  Bitboard pawns = board.pieces(side, PieceType::Pawn);

  while (pawns != 0) {
    const Square from = bitboard::popLsb(pawns);
    const Bitboard fromBit = squareBit(from);
    const bool pinned = (pins.pinned & fromBit) != 0;

    const Square oneForward = from + forward;
    if (isValidSquare(oneForward) && (all & squareBit(oneForward)) == 0) {
      Bitboard quietTargets = squareBit(oneForward);
      quietTargets =
          legalTargetMaskForPiece(from, quietTargets, pins, evasionMask);
      if (quietTargets != 0 &&
          (!noisyOnly || isPromotionRank(side, oneForward))) {
        pushPawnMove(moves, side, from, oneForward, false);
      }

      const Square twoForward = from + 2 * forward;
      if (!noisyOnly && isPawnStartRank(side, from) &&
          (all & squareBit(twoForward)) == 0) {
        Bitboard doubleTarget = squareBit(twoForward);
        doubleTarget =
            legalTargetMaskForPiece(from, doubleTarget, pins, evasionMask);
        if (doubleTarget != 0) {
          pushMove(moves, from, twoForward, MoveFlag::DoublePawnPush);
        }
      }
    }

    Bitboard captures = AttackTables::pawnAttacks(side, from) & enemyPieces;
    captures = legalTargetMaskForPiece(from, captures, pins, evasionMask);
    while (captures != 0) {
      pushPawnMove(moves, side, from, bitboard::popLsb(captures), true);
    }

    if (!board.hasEnPassant()) continue;
    const Square epTarget = board.enPassantSquare();
    if ((AttackTables::pawnAttacks(side, from) & squareBit(epTarget)) == 0) {
      continue;
    }

    const Square capturedPawnSquare =
        epTarget + (side == Color::White ? -8 : 8);
    if (!isValidSquare(capturedPawnSquare)) continue;
    const Piece capturedPawn = pieceAt(board, capturedPawnSquare);
    if (capturedPawn == Piece::None || !matchesColor(capturedPawn, enemy) ||
        pieceType(capturedPawn) != PieceType::Pawn) {
      continue;
    }

    const Bitboard epTargetBit = squareBit(epTarget);
    const Bitboard capturedPawnBit = squareBit(capturedPawnSquare);
    const bool resolvesCheck = checkers == 0 ||
                               (checkers & capturedPawnBit) != 0 ||
                               (evasionMask & epTargetBit) != 0;
    if (!resolvesCheck) continue;

    if (pinned && (pins.mask[from] & epTargetBit) == 0) continue;

    Bitboard occupancyAfter = all;
    occupancyAfter &= ~fromBit;
    occupancyAfter &= ~capturedPawnBit;
    occupancyAfter |= epTargetBit;
    if (!isSquareAttacked(board, kingSquare, enemy, occupancyAfter)) {
      pushMove(moves, from, epTarget, MoveFlag::EnPassant);
    }
  }
}

void generatePieceMoves(const Board& board, MoveList& moves, Color side,
                        PieceType type, const PinInfo& pins,
                        Bitboard evasionMask, bool noisyOnly) {
  const Color enemy = oppositeColor(side);
  const Bitboard own = board.occupancy(side);
  const Bitboard enemyKing = board.pieces(enemy, PieceType::King);
  const Bitboard all = board.allPieces();
  Bitboard pieces = board.pieces(side, type);

  while (pieces != 0) {
    const Square from = bitboard::popLsb(pieces);
    Bitboard targets = 0;

    switch (type) {
      case PieceType::Knight:
        targets = AttackTables::knightAttacks(from);
        break;
      case PieceType::Bishop:
        targets = AttackTables::bishopAttacks(from, all);
        break;
      case PieceType::Rook:
        targets = AttackTables::rookAttacks(from, all);
        break;
      case PieceType::Queen:
        targets = AttackTables::queenAttacks(from, all);
        break;
      case PieceType::Pawn:
      case PieceType::King:
      case PieceType::None:
      default:
        break;
    }

    targets &= ~own;
    targets &= ~enemyKing;
    if (noisyOnly) targets &= board.occupancy(enemy);
    targets = legalTargetMaskForPiece(from, targets, pins, evasionMask);

    while (targets != 0) {
      const Square to = bitboard::popLsb(targets);
      pushMove(moves, from, to,
               (board.occupancy(enemy) & squareBit(to)) != 0 ? MoveFlag::Capture
                                                             : MoveFlag::Quiet);
    }
  }
}

void ensureAttackTablesInitialized() {
  if (!AttackTables::initialized()) AttackTables::init();
}

void generateLegalMoves(const Board& board, MoveList& moves, bool noisyOnly) {
  ensureAttackTablesInitialized();
  moves.clear();

  const Color side = board.sideToMove();
  const Color enemy = oppositeColor(side);
  const Bitboard all = board.allPieces();
  const Bitboard own = board.occupancy(side);
  const Bitboard enemyKing = board.pieces(enemy, PieceType::King);
  const Square kingSquare = board.kingSquare(side);
  const Bitboard checkers = attackersTo(board, kingSquare, enemy, all);
  // In check, every legal evasion is tactically relevant.
  if (checkers != 0) noisyOnly = false;

  generateKingMoves(board, moves, side, all, own, enemyKing,
                    board.occupancy(enemy) & ~enemyKing, noisyOnly);
  if (!noisyOnly) generateCastling(board, moves, side, all, checkers);

  if (bitboard::popcount(checkers) >= 2) return;

  const PinInfo pins = computePins(board, side);
  Bitboard evasionMask = ~Bitboard{0};
  if (checkers != 0) {
    const Square checker = bitboard::lsb(checkers);
    evasionMask = evasionMaskForSingleCheck(board, kingSquare, checker);
  }

  generatePawnMoves(board, moves, side, pins, evasionMask, checkers, noisyOnly);
  generatePieceMoves(board, moves, side, PieceType::Knight, pins, evasionMask,
                     noisyOnly);
  generatePieceMoves(board, moves, side, PieceType::Bishop, pins, evasionMask,
                     noisyOnly);
  generatePieceMoves(board, moves, side, PieceType::Rook, pins, evasionMask,
                     noisyOnly);
  generatePieceMoves(board, moves, side, PieceType::Queen, pins, evasionMask,
                     noisyOnly);
}

}  // namespace

void genLegalMoves(const Board& board, MoveList& moves) {
  generateLegalMoves(board, moves, false);
}

void genLegalNoisyMoves(const Board& board, MoveList& moves) {
  generateLegalMoves(board, moves, true);
}

std::uint64_t perft(Board& board, int depth) {
  if (depth == 0) return 1;

  std::uint64_t nodes = 0;

  MoveList moves;
  genLegalMoves(board, moves);
  if (moves.overflowed()) return 0;

  for (const Move move : moves) {
    if (!board.makeGeneratedMove(move)) return 0;
    nodes += perft(board, depth - 1);
    board.undoMove();
  }

  return nodes;
}
