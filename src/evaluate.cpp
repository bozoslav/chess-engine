#include "evaluate.h"

#include <array>

#include "attacks.h"

namespace {

constexpr int kPieceSquare[7][bitboard::kSquareCount] = {
    {},
    {
        0,  0,  0,  0,   0,   0,  0,  0,  50, 50, 50,  50, 50, 50,  50, 50,
        10, 10, 20, 30,  30,  20, 10, 10, 5,  5,  10,  25, 25, 10,  5,  5,
        0,  0,  0,  20,  20,  0,  0,  0,  5,  -5, -10, 0,  0,  -10, -5, 5,
        5,  10, 10, -20, -20, 10, 10, 5,  0,  0,  0,   0,  0,  0,   0,  0,
    },
    {
        -50, -40, -30, -30, -30, -30, -40, -50, -40, -20, 0,   5,   5,
        0,   -20, -40, -30, 5,   10,  15,  15,  10,  5,   -30, -30, 0,
        15,  20,  20,  15,  0,   -30, -30, 5,   15,  20,  20,  15,  5,
        -30, -30, 0,   10,  15,  15,  10,  0,   -30, -40, -20, 0,   0,
        0,   0,   -20, -40, -50, -40, -30, -30, -30, -30, -40, -50,
    },
    {
        -20, -10, -10, -10, -10, -10, -10, -20, -10, 5,   0,   0,   0,
        0,   5,   -10, -10, 10,  10,  10,  10,  10,  10,  -10, -10, 0,
        10,  10,  10,  10,  0,   -10, -10, 5,   5,   10,  10,  5,   5,
        -10, -10, 0,   5,   10,  10,  5,   0,   -10, -10, 0,   0,   0,
        0,   0,   0,   -10, -20, -10, -10, -10, -10, -10, -10, -20,
    },
    {
        0,  0,  0,  5,  5,  0,  0,  0,  -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0,  0,  0,  0,  0,  0,  -5, -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0,  0,  0,  0,  0,  0,  -5, -5, 0, 0, 0, 0, 0, 0, -5,
        5,  10, 10, 10, 10, 10, 10, 5,  0,  0, 0, 0, 0, 0, 0, 0,
    },
    {
        -20, -10, -10, -5,  -5,  -10, -10, -20, -10, 0,   5,   0,   0,
        0,   0,   -10, -10, 5,   5,   5,   5,   5,   0,   -10, 0,   0,
        5,   5,   5,   5,   0,   -5,  -5,  0,   5,   5,   5,   5,   0,
        -5,  -10, 0,   5,   5,   5,   5,   0,   -10, -10, 0,   0,   0,
        0,   0,   0,   -10, -20, -10, -10, -5,  -5,  -10, -10, -20,
    },
    {
        20,  30,  10,  0,   0,   10,  30,  20,  20,  20,  0,   0,   0,
        0,   20,  20,  -10, -20, -20, -20, -20, -20, -20, -10, -20, -30,
        -30, -40, -40, -30, -30, -20, -30, -40, -40, -50, -50, -40, -40,
        -30, -30, -40, -40, -50, -50, -40, -40, -30, -30, -40, -40, -50,
        -50, -40, -40, -30, -30, -40, -40, -50, -50, -40, -40, -30,
    },
};

constexpr Square mirrorForBlack(Square square) { return square ^ 56; }

constexpr int kDoubledPawnPenalty = 12;
constexpr int kIsolatedPawnPenalty = 10;
constexpr int kRookOpenFileBonus = 20;
constexpr int kRookSemiOpenFileBonus = 10;
constexpr int kKingShieldFrontBonus = 8;
constexpr int kKingShieldSecondBonus = 4;
constexpr int kKnightMobilityBonus = 4;
constexpr int kBishopMobilityBonus = 4;
constexpr int kRookMobilityBonus = 2;
constexpr int kQueenMobilityBonus = 1;

constexpr int kPassedPawnBonus[8] = {0, 5, 10, 20, 35, 60, 100, 0};

constexpr Bitboard fileMask(int file) noexcept {
  return 0x0101010101010101ULL << file;
}

constexpr Bitboard passedPawnMask(int colorIndex, Square square) noexcept {
  const int rank = bitboard::rankOf(square);
  const int file = bitboard::fileOf(square);
  Bitboard mask = 0;

  for (int targetFile = file - 1; targetFile <= file + 1; ++targetFile) {
    if (targetFile < 0 || targetFile >= bitboard::kFileCount) continue;

    if (colorIndex == static_cast<int>(Color::White)) {
      for (int targetRank = rank + 1; targetRank < bitboard::kRankCount;
           ++targetRank) {
        mask |= bitboard::bit(targetRank * bitboard::kFileCount + targetFile);
      }
    } else {
      for (int targetRank = rank - 1; targetRank >= 0; --targetRank) {
        mask |= bitboard::bit(targetRank * bitboard::kFileCount + targetFile);
      }
    }
  }

  return mask;
}

constexpr std::array<std::array<Bitboard, bitboard::kSquareCount>, 2>
makePassedPawnMasks() noexcept {
  std::array<std::array<Bitboard, bitboard::kSquareCount>, 2> masks = {};
  for (Square square = 0; square < bitboard::kSquareCount; ++square) {
    masks[static_cast<int>(Color::White)][square] =
        passedPawnMask(static_cast<int>(Color::White), square);
    masks[static_cast<int>(Color::Black)][square] =
        passedPawnMask(static_cast<int>(Color::Black), square);
  }
  return masks;
}

constexpr auto kPassedPawnMasks = makePassedPawnMasks();

int colorIndex(Color color) {
  return color == Color::White ? static_cast<int>(Color::White)
                               : static_cast<int>(Color::Black);
}

int relativeRank(Color color, Square square) {
  const int rank = bitboard::rankOf(square);
  return color == Color::White ? rank : bitboard::kRankCount - 1 - rank;
}

int evaluatePawnStructure(const Board& board, Color color) {
  int score = 0;
  const Bitboard pawns = board.pieces(color, PieceType::Pawn);
  const Bitboard enemyPawns =
      board.pieces(oppositeColor(color), PieceType::Pawn);

  for (int file = 0; file < bitboard::kFileCount; ++file) {
    const int count = bitboard::popcount(pawns & fileMask(file));
    if (count > 1) {
      score -= (count - 1) * kDoubledPawnPenalty;
    }
  }

  Bitboard remaining = pawns;
  while (remaining != 0) {
    const Square square = bitboard::popLsb(remaining);
    const int file = bitboard::fileOf(square);

    Bitboard adjacentFiles = 0;
    if (file > 0) adjacentFiles |= fileMask(file - 1);
    if (file + 1 < bitboard::kFileCount) adjacentFiles |= fileMask(file + 1);
    if ((pawns & adjacentFiles) == 0) {
      score -= kIsolatedPawnPenalty;
    }

    if ((enemyPawns & kPassedPawnMasks[colorIndex(color)][square]) == 0) {
      score += kPassedPawnBonus[relativeRank(color, square)];
    }
  }

  return score;
}

int evaluateMobility(const Board& board, Color color) {
  int score = 0;
  const Bitboard own = board.occupancy(color);
  const Bitboard occupancy = board.allPieces();

  Bitboard pieces = board.pieces(color, PieceType::Knight);
  while (pieces != 0) {
    const Square square = bitboard::popLsb(pieces);
    score += bitboard::popcount(AttackTables::knightAttacks(square) & ~own) *
             kKnightMobilityBonus;
  }

  pieces = board.pieces(color, PieceType::Bishop);
  while (pieces != 0) {
    const Square square = bitboard::popLsb(pieces);
    score += bitboard::popcount(AttackTables::bishopAttacks(square, occupancy) &
                                ~own) *
             kBishopMobilityBonus;
  }

  pieces = board.pieces(color, PieceType::Rook);
  while (pieces != 0) {
    const Square square = bitboard::popLsb(pieces);
    score += bitboard::popcount(AttackTables::rookAttacks(square, occupancy) &
                                ~own) *
             kRookMobilityBonus;
  }

  pieces = board.pieces(color, PieceType::Queen);
  while (pieces != 0) {
    const Square square = bitboard::popLsb(pieces);
    score += bitboard::popcount(AttackTables::queenAttacks(square, occupancy) &
                                ~own) *
             kQueenMobilityBonus;
  }

  return score;
}

int evaluateRookFiles(const Board& board, Color color) {
  int score = 0;
  const Bitboard ownPawns = board.pieces(color, PieceType::Pawn);
  const Bitboard allPawns =
      ownPawns | board.pieces(oppositeColor(color), PieceType::Pawn);

  Bitboard rooks = board.pieces(color, PieceType::Rook);
  while (rooks != 0) {
    const Square square = bitboard::popLsb(rooks);
    const Bitboard file = fileMask(bitboard::fileOf(square));
    if ((allPawns & file) == 0) {
      score += kRookOpenFileBonus;
    } else if ((ownPawns & file) == 0) {
      score += kRookSemiOpenFileBonus;
    }
  }

  return score;
}

int evaluateKingShield(const Board& board, Color color) {
  const Square king = board.kingSquare(color);
  if (king < 0) return 0;

  int score = 0;
  const Bitboard ownPawns = board.pieces(color, PieceType::Pawn);
  const int direction = color == Color::White ? 1 : -1;
  const int kingRank = bitboard::rankOf(king);
  const int kingFile = bitboard::fileOf(king);

  for (int file = kingFile - 1; file <= kingFile + 1; ++file) {
    if (file < 0 || file >= bitboard::kFileCount) continue;

    const int frontRank = kingRank + direction;
    if (frontRank >= 0 && frontRank < bitboard::kRankCount &&
        (ownPawns & bitboard::bit(frontRank * bitboard::kFileCount + file)) !=
            0) {
      score += kKingShieldFrontBonus;
    }

    const int secondRank = kingRank + 2 * direction;
    if (secondRank >= 0 && secondRank < bitboard::kRankCount &&
        (ownPawns & bitboard::bit(secondRank * bitboard::kFileCount + file)) !=
            0) {
      score += kKingShieldSecondBonus;
    }
  }

  return score;
}

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

  score += evaluatePawnStructure(board, color);
  score += evaluateMobility(board, color);
  score += evaluateRookFiles(board, color);
  score += evaluateKingShield(board, color);

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
  if (!AttackTables::initialized()) AttackTables::init();

  const int whiteScore = evaluateColor(board, Color::White);
  const int blackScore = evaluateColor(board, Color::Black);
  const int score = whiteScore - blackScore;
  return board.sideToMove() == Color::White ? score : -score;
}
