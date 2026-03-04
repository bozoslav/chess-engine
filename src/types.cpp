#include "types.h"

bool isWhitePiece(Piece piece) {
  switch (piece) {
    case Piece::WhitePawn:
    case Piece::WhiteKnight:
    case Piece::WhiteBishop:
    case Piece::WhiteRook:
    case Piece::WhiteQueen:
    case Piece::WhiteKing:
      return true;
    default:
      return false;
  }
}

bool isBlackPiece(Piece piece) {
  switch (piece) {
    case Piece::BlackPawn:
    case Piece::BlackKnight:
    case Piece::BlackBishop:
    case Piece::BlackRook:
    case Piece::BlackQueen:
    case Piece::BlackKing:
      return true;
    default:
      return false;
  }
}

bool matchesColor(Piece piece, Color color) {
  return (color == Color::White && isWhitePiece(piece)) ||
         (color == Color::Black && isBlackPiece(piece));
}

bool isSameColor(Piece first, Piece second) {
  if (first == Piece::None || second == Piece::None) return false;
  return (isWhitePiece(first) && isWhitePiece(second)) ||
         (isBlackPiece(first) && isBlackPiece(second));
}

PieceType pieceType(Piece piece) {
  switch (piece) {
    case Piece::WhitePawn:
    case Piece::BlackPawn:
      return PieceType::Pawn;
    case Piece::WhiteKnight:
    case Piece::BlackKnight:
      return PieceType::Knight;
    case Piece::WhiteBishop:
    case Piece::BlackBishop:
      return PieceType::Bishop;
    case Piece::WhiteRook:
    case Piece::BlackRook:
      return PieceType::Rook;
    case Piece::WhiteQueen:
    case Piece::BlackQueen:
      return PieceType::Queen;
    case Piece::WhiteKing:
    case Piece::BlackKing:
      return PieceType::King;
    case Piece::None:
    default:
      return PieceType::None;
  }
}

Color oppositeColor(Color color) {
  return color == Color::White ? Color::Black : Color::White;
}