#pragma once

enum class Color : unsigned char {
  White,
  Black,
};

enum class PieceType : unsigned char {
  None = 0,
  Pawn = 1,
  Knight = 2,
  Bishop = 3,
  Rook = 4,
  Queen = 5,
  King = 6,
};

enum class Piece : unsigned char {
  None = 0,
  WhitePawn = 1,
  WhiteKnight = 2,
  WhiteBishop = 3,
  WhiteRook = 4,
  WhiteQueen = 5,
  WhiteKing = 6,
  BlackPawn = 11,
  BlackKnight = 12,
  BlackBishop = 13,
  BlackRook = 14,
  BlackQueen = 15,
  BlackKing = 16,
};
