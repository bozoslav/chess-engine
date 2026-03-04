#include "board.h"

#include <cmath>
#include <iostream>

#include "types.h"

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

bool pathIsClear(const Piece board[Board::kBoardSize][Board::kBoardSize],
                 const Move& move) {
  const int stepX = (move.toX > move.fromX) - (move.toX < move.fromX);
  const int stepY = (move.toY > move.fromY) - (move.toY < move.fromY);

  int x = move.fromX + stepX;
  int y = move.fromY + stepY;

  while (x != move.toX || y != move.toY) {
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
  histSize = 0;
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
  const int dx = move.toX - move.fromX;
  const int dy = move.toY - move.fromY;
  bool ok = false;

  if (dy == 0) {
    if (dx == direction && targetPiece == Piece::None) ok = true;
    if (move.fromX == startRank && dx == 2 * direction &&
        targetPiece == Piece::None &&
        board[move.fromX + direction][move.fromY] == Piece::None)
      ok = true;
  } else if (std::abs(dy) == 1 && dx == direction) {
    if (targetPiece != Piece::None) {
      ok = true;
    } else if (hasEp && move.toX == epX && move.toY == epY) {
      const int capX = move.toX + (isWhitePiece(movingPiece) ? 1 : -1);
      const Piece epPawn =
          isWhitePiece(movingPiece) ? Piece::BlackPawn : Piece::WhitePawn;
      ok = isInsideBoard(capX, move.toY) && board[capX][move.toY] == epPawn;
    }
  }

  if (!ok) return false;

  if (move.toX == promoRank) return isPromoPiece(move.promo);

  return move.promo == PieceType::None;
}

bool Board::isValidKnightMove(const Move& move) const {
  const int dx = std::abs(move.toX - move.fromX);
  const int dy = std::abs(move.toY - move.fromY);
  return (dx == 1 && dy == 2) || (dx == 2 && dy == 1);
}

bool Board::isValidBishopMove(const Move& move) const {
  const int dx = std::abs(move.toX - move.fromX);
  const int dy = std::abs(move.toY - move.fromY);
  return dx == dy && dx != 0 && pathIsClear(board, move);
}

bool Board::isValidRookMove(const Move& move) const {
  const bool movesStraight =
      ((move.fromX == move.toX) != (move.fromY == move.toY));
  return movesStraight && pathIsClear(board, move);
}

bool Board::isValidQueenMove(const Move& move) const {
  return isValidBishopMove(move) || isValidRookMove(move);
}

bool Board::isValidKingMove(const Move& move, Piece movingPiece) const {
  const int dx = std::abs(move.toX - move.fromX);
  const int dy = std::abs(move.toY - move.fromY);
  if (dx <= 1 && dy <= 1 && (dx != 0 || dy != 0)) return true;

  if (dx != 0 || dy != 2) return false;

  const bool white = movingPiece == Piece::WhiteKing;
  const bool kingSide = move.toY > move.fromY;
  const bool hasRight = white ? (kingSide ? wCastleK : wCastleQ)
                              : (kingSide ? bCastleK : bCastleQ);
  if (!hasRight) return false;
  if (isKingInCheckForSide(white ? Color::White : Color::Black)) return false;

  const int rookY = kingSide ? kKingsideRookFile : kQueensideRookFile;
  const Piece rook = white ? Piece::WhiteRook : Piece::BlackRook;
  if (board[move.fromX][rookY] != rook) return false;

  const int step = kingSide ? 1 : -1;
  for (int y = move.fromY + step; y != rookY; y += step) {
    if (board[move.fromX][y] != Piece::None) return false;
  }

  if (isSquareUnderAttack(move.fromX, move.fromY + step,
                          oppositeColor(white ? Color::White : Color::Black)))
    return false;
  if (isSquareUnderAttack(move.toX, move.toY,
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
    if (move.fromX == kWhiteBackRank && move.fromY == kQueensideRookFile) {
      wCastleQ = false;
    }
    if (move.fromX == kWhiteBackRank && move.fromY == kKingsideRookFile) {
      wCastleK = false;
    }
  } else if (movingPiece == Piece::BlackRook) {
    if (move.fromX == kBlackBackRank && move.fromY == kQueensideRookFile) {
      bCastleQ = false;
    }
    if (move.fromX == kBlackBackRank && move.fromY == kKingsideRookFile) {
      bCastleK = false;
    }
  }

  if (capturedPiece == Piece::WhiteRook) {
    if (move.toX == kWhiteBackRank && move.toY == kQueensideRookFile) {
      wCastleQ = false;
    }
    if (move.toX == kWhiteBackRank && move.toY == kKingsideRookFile) {
      wCastleK = false;
    }
  } else if (capturedPiece == Piece::BlackRook) {
    if (move.toX == kBlackBackRank && move.toY == kQueensideRookFile) {
      bCastleQ = false;
    }
    if (move.toX == kBlackBackRank && move.toY == kKingsideRookFile) {
      bCastleK = false;
    }
  }
}

bool Board::makeMove(const Move& move) {
  if (!isInsideBoard(move.fromX, move.fromY) ||
      !isInsideBoard(move.toX, move.toY)) {
    return false;
  }
  if (move.fromX == move.toX && move.fromY == move.toY) return false;

  const Piece movingPiece = board[move.fromX][move.fromY];
  const Piece targetPiece = board[move.toX][move.toY];

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
  state.capX = move.toX;
  state.capY = move.toY;
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

  const PieceType movingType = pieceType(movingPiece);
  const bool white = movingSide == Color::White;

  if (movingType == PieceType::Pawn && targetPiece == Piece::None &&
      move.fromY != move.toY && hasEp && move.toX == epX && move.toY == epY) {
    state.wasEp = true;
    state.capX = move.toX + (white ? 1 : -1);
    state.capY = move.toY;
    state.capturedPiece = board[state.capX][state.capY];
    board[state.capX][state.capY] = Piece::None;
  }

  if (movingType == PieceType::Pawn) {
    const int promoRank = white ? kBlackBackRank : kWhiteBackRank;
    if (move.toX == promoRank) {
      state.placedPiece = makePiece(movingSide, move.promo);
    }
  }

  if (movingType == PieceType::King && move.fromX == move.toX &&
      std::abs(move.toY - move.fromY) == 2) {
    state.wasCastle = true;
    state.rookFromX = move.fromX;
    state.rookToX = move.fromX;
    state.rookFromY = move.toY > move.fromY ? kKingsideRookFile : kQueensideRookFile;
    state.rookToY = move.toY > move.fromY ? kKingsideBishopFile : kQueenFile;
  }

  board[move.toX][move.toY] = state.placedPiece;
  board[move.fromX][move.fromY] = Piece::None;
  if (state.wasCastle) {
    board[state.rookToX][state.rookToY] = board[state.rookFromX][state.rookFromY];
    board[state.rookFromX][state.rookFromY] = Piece::None;
  }

  updateCastlingRights(move, movingPiece, targetPiece);
  hasEp = false;
  epX = -1;
  epY = -1;
  if (movingType == PieceType::Pawn && std::abs(move.toX - move.fromX) == 2) {
    hasEp = true;
    epX = (move.fromX + move.toX) / 2;
    epY = move.fromY;
  }
  side = oppositeColor(side);

  if (isKingInCheckForSide(movingSide)) {
    undoMove();
    return false;
  }

  return true;
}

bool Board::undoMove() {
  if (histSize == 0) return false;

  const MoveState& state = history[--histSize];

  if (state.wasCastle) {
    board[state.rookFromX][state.rookFromY] = board[state.rookToX][state.rookToY];
    board[state.rookToX][state.rookToY] = Piece::None;
  }

  board[state.move.fromX][state.move.fromY] = state.movedPiece;
  if (state.wasEp) {
    board[state.move.toX][state.move.toY] = Piece::None;
    board[state.capX][state.capY] = state.capturedPiece;
  } else {
    board[state.move.toX][state.move.toY] = state.capturedPiece;
  }
  side = state.prevSide;
  wCastleK = state.prevWCastleK;
  wCastleQ = state.prevWCastleQ;
  bCastleK = state.prevBCastleK;
  bCastleQ = state.prevBCastleQ;
  hasEp = state.prevHasEp;
  epX = state.prevEpX;
  epY = state.prevEpY;

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
