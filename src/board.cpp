#include "board.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <string_view>

#include "types.h"
#include "zobrist.h"

namespace {

constexpr int kBlackBackRank = 0;
constexpr int kBlackPawnRank = 1;
constexpr int kWhitePawnRank = Board::kBoardSize - 2;
constexpr int kWhiteBackRank = Board::kBoardSize - 1;

constexpr int kQueensideRookFile = 0;
constexpr int kQueensideKnightFile = 1;
constexpr int kQueensideBishopFile = 2;
constexpr int kQueenFile = 3;
constexpr int kKingFile = 4;
constexpr int kKingsideBishopFile = 5;
constexpr int kKingsideKnightFile = 6;
constexpr int kKingsideRookFile = Board::kBoardSize - 1;
constexpr Square kNoSquare = -1;

constexpr int colorIndex(Color color) { return static_cast<int>(color); }

Square squareFromCoords(int x, int y) {
  return bitboard::squareFromCoords(x, y);
}

Color colorOfPiece(Piece piece) {
  return isWhitePiece(piece) ? Color::White : Color::Black;
}

bool isPromoPiece(PieceType type) {
  return type == PieceType::Queen || type == PieceType::Rook ||
         type == PieceType::Bishop || type == PieceType::Knight;
}

Piece makePiece(Color side, PieceType type) {
  if (side == Color::White) {
    switch (type) {
      case PieceType::Pawn:
        return Piece::WhitePawn;
      case PieceType::Knight:
        return Piece::WhiteKnight;
      case PieceType::Bishop:
        return Piece::WhiteBishop;
      case PieceType::Rook:
        return Piece::WhiteRook;
      case PieceType::Queen:
        return Piece::WhiteQueen;
      case PieceType::King:
        return Piece::WhiteKing;
      case PieceType::None:
      default:
        return Piece::None;
    }
  }

  switch (type) {
    case PieceType::Pawn:
      return Piece::BlackPawn;
    case PieceType::Knight:
      return Piece::BlackKnight;
    case PieceType::Bishop:
      return Piece::BlackBishop;
    case PieceType::Rook:
      return Piece::BlackRook;
    case PieceType::Queen:
      return Piece::BlackQueen;
    case PieceType::King:
      return Piece::BlackKing;
    case PieceType::None:
    default:
      return Piece::None;
  }
}

Piece pieceFromFen(char c) {
  switch (c) {
    case 'P':
      return Piece::WhitePawn;
    case 'N':
      return Piece::WhiteKnight;
    case 'B':
      return Piece::WhiteBishop;
    case 'R':
      return Piece::WhiteRook;
    case 'Q':
      return Piece::WhiteQueen;
    case 'K':
      return Piece::WhiteKing;
    case 'p':
      return Piece::BlackPawn;
    case 'n':
      return Piece::BlackKnight;
    case 'b':
      return Piece::BlackBishop;
    case 'r':
      return Piece::BlackRook;
    case 'q':
      return Piece::BlackQueen;
    case 'k':
      return Piece::BlackKing;
    default:
      return Piece::None;
  }
}

void skipSpaces(std::string_view text, std::size_t& pos) {
  while (pos < text.size() && text[pos] == ' ') ++pos;
}

bool nextFenField(std::string_view fen, std::size_t& pos,
                  std::string_view& field) {
  skipSpaces(fen, pos);
  if (pos >= fen.size()) return false;

  const std::size_t start = pos;
  while (pos < fen.size() && fen[pos] != ' ') ++pos;
  field = fen.substr(start, pos - start);
  return !field.empty();
}

bool parseFenSquare(std::string_view square, int& x, int& y) {
  if (square.size() != 2) return false;

  const char file = square[0];
  const char rank = square[1];
  if (file < 'a' || file > 'h') return false;
  if (rank < '1' || rank > '8') return false;

  y = file - 'a';
  x = Board::kBoardSize - (rank - '0');
  return true;
}

bool pathIsClear(const Piece board[Board::kBoardSize][Board::kBoardSize],
                 const Move& move) {
  const int stepX = (move.toX() > move.fromX()) - (move.toX() < move.fromX());
  const int stepY = (move.toY() > move.fromY()) - (move.toY() < move.fromY());

  int x = move.fromX() + stepX;
  int y = move.fromY() + stepY;

  while (x != move.toX() || y != move.toY()) {
    if (board[x][y] != Piece::None) return false;
    x += stepX;
    y += stepY;
  }

  return true;
}

}  // namespace

Board::Board() {
  for (int x = 0; x < kBoardSize; ++x) {
    for (int y = 0; y < kBoardSize; ++y) board[x][y] = Piece::None;
  }

  for (int file = 0; file < kBoardSize; ++file) {
    board[kBlackPawnRank][file] = Piece::BlackPawn;
    board[kWhitePawnRank][file] = Piece::WhitePawn;
  }

  board[kBlackBackRank][kQueensideRookFile] = Piece::BlackRook;
  board[kBlackBackRank][kKingsideRookFile] = Piece::BlackRook;
  board[kWhiteBackRank][kQueensideRookFile] = Piece::WhiteRook;
  board[kWhiteBackRank][kKingsideRookFile] = Piece::WhiteRook;

  board[kBlackBackRank][kQueensideKnightFile] = Piece::BlackKnight;
  board[kBlackBackRank][kKingsideKnightFile] = Piece::BlackKnight;
  board[kWhiteBackRank][kQueensideKnightFile] = Piece::WhiteKnight;
  board[kWhiteBackRank][kKingsideKnightFile] = Piece::WhiteKnight;

  board[kBlackBackRank][kQueensideBishopFile] = Piece::BlackBishop;
  board[kBlackBackRank][kKingsideBishopFile] = Piece::BlackBishop;
  board[kWhiteBackRank][kQueensideBishopFile] = Piece::WhiteBishop;
  board[kWhiteBackRank][kKingsideBishopFile] = Piece::WhiteBishop;

  board[kBlackBackRank][kQueenFile] = Piece::BlackQueen;
  board[kWhiteBackRank][kQueenFile] = Piece::WhiteQueen;

  board[kBlackBackRank][kKingFile] = Piece::BlackKing;
  board[kWhiteBackRank][kKingFile] = Piece::WhiteKing;

  side = Color::White;
  wCastleK = true;
  wCastleQ = true;
  bCastleK = true;
  bCastleQ = true;
  hasEp = false;
  epX = -1;
  epY = -1;
  epSquare = kNoSquare;
  histSize = 0;

  rebuildBitboards();
  zobristKey = computeZobristKey();
  keyHistory[0] = zobristKey;
  keyHistorySize = 1;
  assert(bitboardsAreConsistent());
}

bool Board::setFromFen(std::string_view fen) {
  Piece parsedBoard[kBoardSize][kBoardSize] = {};

  std::size_t pos = 0;
  std::string_view field;
  if (!nextFenField(fen, pos, field)) return false;

  int x = 0;
  int y = 0;
  for (char c : field) {
    if (c == '/') {
      if (y != kBoardSize || x == kBoardSize - 1) return false;
      ++x;
      y = 0;
      continue;
    }

    if (c >= '1' && c <= '8') {
      y += c - '0';
      if (y > kBoardSize) return false;
      continue;
    }

    const Piece piece = pieceFromFen(c);
    if (piece == Piece::None || y >= kBoardSize) return false;
    parsedBoard[x][y++] = piece;
  }

  if (x != kBoardSize - 1 || y != kBoardSize) return false;

  if (!nextFenField(fen, pos, field) || field.size() != 1) return false;
  Color parsedSide;
  if (field[0] == 'w') {
    parsedSide = Color::White;
  } else if (field[0] == 'b') {
    parsedSide = Color::Black;
  } else {
    return false;
  }

  if (!nextFenField(fen, pos, field)) return false;
  bool parsedWCastleK = false;
  bool parsedWCastleQ = false;
  bool parsedBCastleK = false;
  bool parsedBCastleQ = false;
  if (field != "-") {
    for (char c : field) {
      switch (c) {
        case 'K':
          parsedWCastleK = true;
          break;
        case 'Q':
          parsedWCastleQ = true;
          break;
        case 'k':
          parsedBCastleK = true;
          break;
        case 'q':
          parsedBCastleQ = true;
          break;
        default:
          return false;
      }
    }
  }

  if (!nextFenField(fen, pos, field)) return false;
  bool parsedHasEp = false;
  int parsedEpX = -1;
  int parsedEpY = -1;
  if (field != "-") {
    if (!parseFenSquare(field, parsedEpX, parsedEpY)) return false;
    parsedHasEp = true;
  }

  for (int row = 0; row < kBoardSize; ++row) {
    for (int file = 0; file < kBoardSize; ++file) {
      board[row][file] = parsedBoard[row][file];
    }
  }

  side = parsedSide;
  wCastleK = parsedWCastleK;
  wCastleQ = parsedWCastleQ;
  bCastleK = parsedBCastleK;
  bCastleQ = parsedBCastleQ;
  hasEp = parsedHasEp;
  epX = parsedEpX;
  epY = parsedEpY;
  epSquare = parsedHasEp ? squareFromCoords(parsedEpX, parsedEpY) : kNoSquare;
  histSize = 0;
  rebuildBitboards();
  zobristKey = computeZobristKey();
  keyHistory[0] = zobristKey;
  keyHistorySize = 1;
  assert(bitboardsAreConsistent());
  return true;
}

bool Board::isInsideBoard(int x, int y) const {
  return x >= 0 && x < kBoardSize && y >= 0 && y < kBoardSize;
}

bool Board::isCorrectSideToMove(Piece piece) const {
  return matchesColor(piece, side);
}

bool Board::isValidPawnMove(const Move& move, Piece movingPiece,
                            Piece targetPiece) const {
  const int direction = isWhitePiece(movingPiece) ? -1 : 1;
  const int startRank =
      isWhitePiece(movingPiece) ? kWhitePawnRank : kBlackPawnRank;
  const int promoRank =
      isWhitePiece(movingPiece) ? kBlackBackRank : kWhiteBackRank;
  const int dx = move.toX() - move.fromX();
  const int dy = move.toY() - move.fromY();
  bool ok = false;

  if (dy == 0) {
    if (dx == direction && targetPiece == Piece::None) ok = true;
    if (move.fromX() == startRank && dx == 2 * direction &&
        targetPiece == Piece::None &&
        board[move.fromX() + direction][move.fromY()] == Piece::None)
      ok = true;
  } else if (std::abs(dy) == 1 && dx == direction) {
    if (targetPiece != Piece::None) {
      ok = true;
    } else if (hasEp && move.toX() == epX && move.toY() == epY) {
      const int capX = move.toX() + (isWhitePiece(movingPiece) ? 1 : -1);
      const Piece epPawn =
          isWhitePiece(movingPiece) ? Piece::BlackPawn : Piece::WhitePawn;
      ok = isInsideBoard(capX, move.toY()) && board[capX][move.toY()] == epPawn;
    }
  }

  if (!ok) return false;

  if (move.toX() == promoRank) return isPromoPiece(move.promo());

  return move.promo() == PieceType::None;
}

bool Board::isValidKnightMove(const Move& move) const {
  const int dx = std::abs(move.toX() - move.fromX());
  const int dy = std::abs(move.toY() - move.fromY());
  return (dx == 1 && dy == 2) || (dx == 2 && dy == 1);
}

bool Board::isValidBishopMove(const Move& move) const {
  const int dx = std::abs(move.toX() - move.fromX());
  const int dy = std::abs(move.toY() - move.fromY());
  return dx == dy && dx != 0 && pathIsClear(board, move);
}

bool Board::isValidRookMove(const Move& move) const {
  const bool movesStraight =
      ((move.fromX() == move.toX()) != (move.fromY() == move.toY()));
  return movesStraight && pathIsClear(board, move);
}

bool Board::isValidQueenMove(const Move& move) const {
  return isValidBishopMove(move) || isValidRookMove(move);
}

bool Board::isValidKingMove(const Move& move, Piece movingPiece) const {
  const int dx = std::abs(move.toX() - move.fromX());
  const int dy = std::abs(move.toY() - move.fromY());
  if (dx <= 1 && dy <= 1 && (dx != 0 || dy != 0)) return true;

  if (dx != 0 || dy != 2) return false;

  const bool white = movingPiece == Piece::WhiteKing;
  const bool kingSide = move.toY() > move.fromY();
  const bool hasRight = white ? (kingSide ? wCastleK : wCastleQ)
                              : (kingSide ? bCastleK : bCastleQ);
  if (!hasRight) return false;
  if (isKingInCheckForSide(white ? Color::White : Color::Black)) return false;

  const int rookY = kingSide ? kKingsideRookFile : kQueensideRookFile;
  const Piece rook = white ? Piece::WhiteRook : Piece::BlackRook;
  if (board[move.fromX()][rookY] != rook) return false;

  const int step = kingSide ? 1 : -1;
  for (int y = move.fromY() + step; y != rookY; y += step) {
    if (board[move.fromX()][y] != Piece::None) return false;
  }

  if (isSquareUnderAttack(move.fromX(), move.fromY() + step,
                          oppositeColor(white ? Color::White : Color::Black)))
    return false;
  if (isSquareUnderAttack(move.toX(), move.toY(),
                          oppositeColor(white ? Color::White : Color::Black)))
    return false;

  return true;
}

bool Board::isValidPieceMove(const Move& move, Piece movingPiece,
                             Piece targetPiece) const {
  switch (pieceType(movingPiece)) {
    case PieceType::Pawn:
      return isValidPawnMove(move, movingPiece, targetPiece);
    case PieceType::Knight:
      return isValidKnightMove(move);
    case PieceType::Bishop:
      return isValidBishopMove(move);
    case PieceType::Rook:
      return isValidRookMove(move);
    case PieceType::Queen:
      return isValidQueenMove(move);
    case PieceType::King:
      return isValidKingMove(move, movingPiece);
    case PieceType::None:
    default:
      return false;
  }
}

bool Board::isSquareUnderAttack(int x, int y, Color attackingColor) const {
  const Piece attackingPawn =
      attackingColor == Color::White ? Piece::WhitePawn : Piece::BlackPawn;
  const Piece attackingKing =
      attackingColor == Color::White ? Piece::WhiteKing : Piece::BlackKing;

  const int pawnRow = x + (attackingColor == Color::White ? 1 : -1);
  if (isInsideBoard(pawnRow, y - 1) && board[pawnRow][y - 1] == attackingPawn)
    return true;
  if (isInsideBoard(pawnRow, y + 1) && board[pawnRow][y + 1] == attackingPawn)
    return true;

  static const int knightOffsets[8][2] = {
      {-2, -1}, {-2, 1}, {-1, -2}, {-1, 2}, {1, -2}, {1, 2}, {2, -1}, {2, 1},
  };

  for (const auto& offset : knightOffsets) {
    const int nx = x + offset[0];
    const int ny = y + offset[1];
    if (!isInsideBoard(nx, ny)) continue;

    const Piece piece = board[nx][ny];
    if (matchesColor(piece, attackingColor) &&
        pieceType(piece) == PieceType::Knight)
      return true;
  }

  static const int bishopDirections[4][2] = {
      {-1, -1},
      {-1, 1},
      {1, -1},
      {1, 1},
  };

  for (const auto& direction : bishopDirections) {
    int nx = x + direction[0];
    int ny = y + direction[1];

    while (isInsideBoard(nx, ny)) {
      const Piece piece = board[nx][ny];
      if (piece != Piece::None) {
        if (matchesColor(piece, attackingColor) &&
            (pieceType(piece) == PieceType::Bishop ||
             pieceType(piece) == PieceType::Queen))
          return true;
        break;
      }
      nx += direction[0];
      ny += direction[1];
    }
  }

  static const int rookDirections[4][2] = {
      {-1, 0},
      {1, 0},
      {0, -1},
      {0, 1},
  };

  for (const auto& direction : rookDirections) {
    int nx = x + direction[0];
    int ny = y + direction[1];

    while (isInsideBoard(nx, ny)) {
      const Piece piece = board[nx][ny];
      if (piece != Piece::None) {
        if (matchesColor(piece, attackingColor) &&
            (pieceType(piece) == PieceType::Rook ||
             pieceType(piece) == PieceType::Queen))
          return true;
        break;
      }
      nx += direction[0];
      ny += direction[1];
    }
  }

  for (int dx = -1; dx <= 1; ++dx) {
    for (int dy = -1; dy <= 1; ++dy) {
      if (dx == 0 && dy == 0) continue;

      const int nx = x + dx;
      const int ny = y + dy;
      if (isInsideBoard(nx, ny) && board[nx][ny] == attackingKing) return true;
    }
  }

  return false;
}

bool Board::isKingInCheckForSide(Color kingColor) const {
  const Piece king =
      kingColor == Color::White ? Piece::WhiteKing : Piece::BlackKing;

  for (int x = 0; x < kBoardSize; ++x) {
    for (int y = 0; y < kBoardSize; ++y) {
      if (board[x][y] == king) {
        return isSquareUnderAttack(x, y, oppositeColor(kingColor));
      }
    }
  }

  return false;
}

bool Board::isKingInCheck() const { return isKingInCheckForSide(side); }

void Board::updateCastlingRights(const Move& move, Piece movingPiece,
                                 Piece capturedPiece) {
  if (movingPiece == Piece::WhiteKing) {
    wCastleK = false;
    wCastleQ = false;
  } else if (movingPiece == Piece::BlackKing) {
    bCastleK = false;
    bCastleQ = false;
  } else if (movingPiece == Piece::WhiteRook) {
    if (move.fromX() == kWhiteBackRank && move.fromY() == kQueensideRookFile) {
      wCastleQ = false;
    }
    if (move.fromX() == kWhiteBackRank && move.fromY() == kKingsideRookFile) {
      wCastleK = false;
    }
  } else if (movingPiece == Piece::BlackRook) {
    if (move.fromX() == kBlackBackRank && move.fromY() == kQueensideRookFile) {
      bCastleQ = false;
    }
    if (move.fromX() == kBlackBackRank && move.fromY() == kKingsideRookFile) {
      bCastleK = false;
    }
  }

  if (capturedPiece == Piece::WhiteRook) {
    if (move.toX() == kWhiteBackRank && move.toY() == kQueensideRookFile) {
      wCastleQ = false;
    }
    if (move.toX() == kWhiteBackRank && move.toY() == kKingsideRookFile) {
      wCastleK = false;
    }
  } else if (capturedPiece == Piece::BlackRook) {
    if (move.toX() == kBlackBackRank && move.toY() == kQueensideRookFile) {
      bCastleQ = false;
    }
    if (move.toX() == kBlackBackRank && move.toY() == kKingsideRookFile) {
      bCastleK = false;
    }
  }
}

void Board::clearBitboards() {
  for (int color = 0; color < kColorCount; ++color) {
    occupancyBB[color] = 0;
    for (int type = 0; type < kPieceTypeCount; ++type) {
      pieceBB[color][type] = 0;
    }
    kingSq[color] = kNoSquare;
  }

  allBB = 0;
}

void Board::addPieceToBitboards(Piece piece, int x, int y) {
  if (piece == Piece::None) return;

  const Color color = colorOfPiece(piece);
  const PieceType type = pieceType(piece);
  const int sideIndex = colorIndex(color);
  const int typeIndex = static_cast<int>(type);
  const Square square = squareFromCoords(x, y);
  const Bitboard bit = bitboard::bit(square);

  pieceBB[sideIndex][typeIndex] |= bit;
  occupancyBB[sideIndex] |= bit;
  allBB |= bit;

  if (type == PieceType::King) {
    kingSq[sideIndex] = square;
  }
}

void Board::removePieceFromBitboards(Piece piece, int x, int y) {
  if (piece == Piece::None) return;

  const Color color = colorOfPiece(piece);
  const PieceType type = pieceType(piece);
  const int sideIndex = colorIndex(color);
  const int typeIndex = static_cast<int>(type);
  const Square square = squareFromCoords(x, y);
  const Bitboard bit = bitboard::bit(square);

  pieceBB[sideIndex][typeIndex] &= ~bit;
  occupancyBB[sideIndex] &= ~bit;
  allBB &= ~bit;

  if (type == PieceType::King && kingSq[sideIndex] == square) {
    kingSq[sideIndex] = kNoSquare;
  }
}

void Board::rebuildBitboards() {
  clearBitboards();

  for (int x = 0; x < kBoardSize; ++x) {
    for (int y = 0; y < kBoardSize; ++y) {
      addPieceToBitboards(board[x][y], x, y);
    }
  }
}

void Board::putPiece(int x, int y, Piece piece) {
  const Piece previous = board[x][y];
  if (previous == piece) return;

  const Square square = squareFromCoords(x, y);
  if (previous != Piece::None) {
    zobristKey ^= zobrist::piece(previous, square);
  }

  removePieceFromBitboards(previous, x, y);
  board[x][y] = piece;
  addPieceToBitboards(piece, x, y);

  if (piece != Piece::None) {
    zobristKey ^= zobrist::piece(piece, square);
  }
}

int Board::castlingRightsMask() const {
  int mask = 0;
  if (wCastleK) mask |= 1;
  if (wCastleQ) mask |= 2;
  if (bCastleK) mask |= 4;
  if (bCastleQ) mask |= 8;
  return mask;
}

std::uint64_t Board::computeZobristKey() const {
  std::uint64_t key = 0;

  for (int x = 0; x < kBoardSize; ++x) {
    for (int y = 0; y < kBoardSize; ++y) {
      const Piece piece = board[x][y];
      if (piece == Piece::None) continue;
      key ^= zobrist::piece(piece, squareFromCoords(x, y));
    }
  }

  key ^= zobrist::castling(castlingRightsMask());
  if (hasEp) key ^= zobrist::enPassant(epSquare);
  if (side == Color::Black) key ^= zobrist::sideToMove();

  return key;
}

bool Board::bitboardsAreConsistent() const {
  Bitboard expectedPieces[kColorCount][kPieceTypeCount] = {};
  Bitboard expectedOccupancy[kColorCount] = {};
  Bitboard expectedAll = 0;
  Square expectedKings[kColorCount] = {kNoSquare, kNoSquare};

  for (int x = 0; x < kBoardSize; ++x) {
    for (int y = 0; y < kBoardSize; ++y) {
      const Piece piece = board[x][y];
      if (piece == Piece::None) continue;

      const Color color = colorOfPiece(piece);
      const PieceType type = pieceType(piece);
      const int sideIndex = colorIndex(color);
      const int typeIndex = static_cast<int>(type);
      const Square square = squareFromCoords(x, y);
      const Bitboard bit = bitboard::bit(square);

      expectedPieces[sideIndex][typeIndex] |= bit;
      expectedOccupancy[sideIndex] |= bit;
      expectedAll |= bit;

      if (type == PieceType::King) {
        if (expectedKings[sideIndex] != kNoSquare) return false;
        expectedKings[sideIndex] = square;
      }
    }
  }

  if ((occupancyBB[colorIndex(Color::White)] &
       occupancyBB[colorIndex(Color::Black)]) != 0)
    return false;
  if ((occupancyBB[colorIndex(Color::White)] |
       occupancyBB[colorIndex(Color::Black)]) != allBB)
    return false;

  for (int color = 0; color < kColorCount; ++color) {
    if (occupancyBB[color] != expectedOccupancy[color]) return false;
    if (kingSq[color] != expectedKings[color]) return false;

    Bitboard mergedPieces = 0;
    for (int type = 0; type < kPieceTypeCount; ++type) {
      if (pieceBB[color][type] != expectedPieces[color][type]) return false;
      mergedPieces |= pieceBB[color][type];
    }
    if (mergedPieces != occupancyBB[color]) return false;
  }

  if (allBB != expectedAll) return false;

  const Square expectedEp =
      hasEp ? squareFromCoords(epX, epY) : static_cast<Square>(kNoSquare);
  if (epSquare != expectedEp) return false;

  if (keyHistorySize <= 0 || keyHistorySize > kMaxHistory + 1) return false;
  if (keyHistory[keyHistorySize - 1] != zobristKey) return false;

  return zobristKey == computeZobristKey();
}

bool Board::makeMove(const Move& move) {
  if (!isInsideBoard(move.fromX(), move.fromY()) ||
      !isInsideBoard(move.toX(), move.toY())) {
    return false;
  }
  if (move.fromX() == move.toX() && move.fromY() == move.toY()) return false;

  const Piece movingPiece = board[move.fromX()][move.fromY()];
  const Piece targetPiece = board[move.toX()][move.toY()];

  if (movingPiece == Piece::None) return false;
  if (pieceType(targetPiece) == PieceType::King) return false;
  if (targetPiece != Piece::None && isSameColor(movingPiece, targetPiece)) {
    return false;
  }
  if (!isCorrectSideToMove(movingPiece)) return false;
  if (!isValidPieceMove(move, movingPiece, targetPiece)) return false;
  if (histSize >= kMaxHistory) return false;

  const Color movingSide = side;

  MoveState& state = history[histSize++];
  state.move = move;
  state.movedPiece = movingPiece;
  state.placedPiece = movingPiece;
  state.capturedPiece = targetPiece;
  state.capX = move.toX();
  state.capY = move.toY();
  state.wasEp = false;
  state.wasCastle = false;
  state.rookFromX = -1;
  state.rookFromY = -1;
  state.rookToX = -1;
  state.rookToY = -1;
  state.prevSide = side;
  state.prevWCastleK = wCastleK;
  state.prevWCastleQ = wCastleQ;
  state.prevBCastleK = bCastleK;
  state.prevBCastleQ = bCastleQ;
  state.prevHasEp = hasEp;
  state.prevEpX = epX;
  state.prevEpY = epY;
  state.prevEpSquare = epSquare;
  state.prevZobristKey = zobristKey;
  state.prevKeyHistorySize = keyHistorySize;

  const PieceType movingType = pieceType(movingPiece);
  const bool white = movingSide == Color::White;
  const int previousCastlingRights = castlingRightsMask();
  if (hasEp) zobristKey ^= zobrist::enPassant(epSquare);
  zobristKey ^= zobrist::castling(previousCastlingRights);

  const bool isEnPassant =
      move.isEnPassant() ||
      (movingType == PieceType::Pawn && targetPiece == Piece::None &&
       move.fromY() != move.toY() && hasEp && move.toX() == epX &&
       move.toY() == epY);
  if (isEnPassant) {
    state.wasEp = true;
    state.capX = move.toX() + (white ? 1 : -1);
    state.capY = move.toY();
    state.capturedPiece = board[state.capX][state.capY];
    putPiece(state.capX, state.capY, Piece::None);
  }

  if (movingType == PieceType::Pawn && move.isPromotion()) {
    const int promoRank = white ? kBlackBackRank : kWhiteBackRank;
    if (move.toX() == promoRank) {
      state.placedPiece = makePiece(movingSide, move.promo());
    }
  }

  const bool isCastle =
      move.isCastle() ||
      (movingType == PieceType::King && move.fromX() == move.toX() &&
       std::abs(move.toY() - move.fromY()) == 2);
  if (isCastle) {
    state.wasCastle = true;
    state.rookFromX = move.fromX();
    state.rookToX = move.fromX();
    state.rookFromY =
        move.toY() > move.fromY() ? kKingsideRookFile : kQueensideRookFile;
    state.rookToY =
        move.toY() > move.fromY() ? kKingsideBishopFile : kQueenFile;
  }

  putPiece(move.toX(), move.toY(), state.placedPiece);
  putPiece(move.fromX(), move.fromY(), Piece::None);
  if (state.wasCastle) {
    const Piece rook = board[state.rookFromX][state.rookFromY];
    putPiece(state.rookToX, state.rookToY, rook);
    putPiece(state.rookFromX, state.rookFromY, Piece::None);
  }

  updateCastlingRights(move, movingPiece, targetPiece);
  hasEp = false;
  epX = -1;
  epY = -1;
  epSquare = kNoSquare;
  if (move.isDoublePawnPush() || (movingType == PieceType::Pawn &&
                                  std::abs(move.toX() - move.fromX()) == 2)) {
    hasEp = true;
    epX = (move.fromX() + move.toX()) / 2;
    epY = move.fromY();
    epSquare = squareFromCoords(epX, epY);
  }
  zobristKey ^= zobrist::castling(castlingRightsMask());
  if (hasEp) zobristKey ^= zobrist::enPassant(epSquare);
  side = oppositeColor(side);
  zobristKey ^= zobrist::sideToMove();

  if (isKingInCheckForSide(movingSide)) {
    undoMove();
    return false;
  }

  keyHistory[keyHistorySize++] = zobristKey;
  assert(bitboardsAreConsistent());
  return true;
}

bool Board::undoMove() {
  if (histSize == 0) return false;

  const MoveState& state = history[--histSize];

  if (state.wasCastle) {
    const Piece rook = board[state.rookToX][state.rookToY];
    putPiece(state.rookFromX, state.rookFromY, rook);
    putPiece(state.rookToX, state.rookToY, Piece::None);
  }

  putPiece(state.move.fromX(), state.move.fromY(), state.movedPiece);
  if (state.wasEp) {
    putPiece(state.move.toX(), state.move.toY(), Piece::None);
    putPiece(state.capX, state.capY, state.capturedPiece);
  } else {
    putPiece(state.move.toX(), state.move.toY(), state.capturedPiece);
  }
  side = state.prevSide;
  wCastleK = state.prevWCastleK;
  wCastleQ = state.prevWCastleQ;
  bCastleK = state.prevBCastleK;
  bCastleQ = state.prevBCastleQ;
  hasEp = state.prevHasEp;
  epX = state.prevEpX;
  epY = state.prevEpY;
  epSquare = state.prevEpSquare;
  zobristKey = state.prevZobristKey;
  keyHistorySize = state.prevKeyHistorySize;

  assert(bitboardsAreConsistent());
  return true;
}

namespace {

const char* pieceGlyph(Piece piece) {
  switch (piece) {
    case Piece::None:
      return "·";
    case Piece::WhitePawn:
      return "♟";
    case Piece::WhiteKnight:
      return "♞";
    case Piece::WhiteBishop:
      return "♝";
    case Piece::WhiteRook:
      return "♜";
    case Piece::WhiteQueen:
      return "♛";
    case Piece::WhiteKing:
      return "♚";
    case Piece::BlackPawn:
      return "♙";
    case Piece::BlackKnight:
      return "♘";
    case Piece::BlackBishop:
      return "♗";
    case Piece::BlackRook:
      return "♖";
    case Piece::BlackQueen:
      return "♕";
    case Piece::BlackKing:
      return "♔";
    default:
      return "?";
  }
}

}  // namespace

void Board::printBoard() const {
  for (int x = 0; x < kBoardSize; ++x) {
    for (int y = 0; y < kBoardSize; ++y) std::cout << pieceGlyph(board[x][y]);
    std::cout << '\n';
  }
  std::cout << "\n";
}

Color Board::sideToMove() const { return side; }

Piece Board::at(int x, int y) const { return board[x][y]; }

Bitboard Board::pieces(Color color, PieceType type) const {
  return pieceBB[colorIndex(color)][static_cast<int>(type)];
}

Bitboard Board::occupancy(Color color) const {
  return occupancyBB[colorIndex(color)];
}

Bitboard Board::allPieces() const { return allBB; }

Square Board::kingSquare(Color color) const {
  return kingSq[colorIndex(color)];
}

bool Board::hasEnPassant() const { return hasEp; }

Square Board::enPassantSquare() const { return epSquare; }

bool Board::canCastleKingSide(Color color) const {
  return color == Color::White ? wCastleK : bCastleK;
}

bool Board::canCastleQueenSide(Color color) const {
  return color == Color::White ? wCastleQ : bCastleQ;
}

std::uint64_t Board::key() const { return zobristKey; }

int Board::repetitionCount() const {
  int count = 0;
  for (int index = keyHistorySize - 1; index >= 0; --index) {
    if (keyHistory[index] == zobristKey) ++count;
  }
  return count;
}

bool Board::hasRepeatedPosition() const { return repetitionCount() >= 2; }

bool Board::isThreefoldRepetition() const { return repetitionCount() >= 3; }
