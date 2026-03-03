#pragma once

#include "move.h"
#include "types.h"

class Board {
 public:
  static constexpr int kBoardSize = 8;

  Board();

  bool makeMove(const Move& move);
  bool undoMove();
  bool isKingInCheck() const;
  void printBoard() const;

 private:
  struct MoveState {
    Move move;
    Piece movedPiece;
    Piece capturedPiece;
    Color prevSide;
    bool prevWCastleK;
    bool prevWCastleQ;
    bool prevBCastleK;
    bool prevBCastleQ;
  };

  static constexpr int kMaxHistory = 1024;

  bool isInsideBoard(int x, int y) const;
  bool isCorrectSideToMove(Piece piece) const;
  bool isValidPieceMove(const Move& move, Piece movingPiece,
                        Piece targetPiece) const;
  bool isValidPawnMove(const Move& move, Piece movingPiece,
                       Piece targetPiece) const;
  bool isValidKnightMove(const Move& move) const;
  bool isValidBishopMove(const Move& move) const;
  bool isValidRookMove(const Move& move) const;
  bool isValidQueenMove(const Move& move) const;
  bool isValidKingMove(const Move& move) const;
  bool isSquareUnderAttack(int x, int y, Color attackingColor) const;
  bool isKingInCheckForSide(Color kingColor) const;
  void updateCastlingRights(const Move& move, Piece movingPiece,
                            Piece capturedPiece);

  Piece board[kBoardSize][kBoardSize];
  Color side;
  bool wCastleK;
  bool wCastleQ;
  bool bCastleK;
  bool bCastleQ;
  MoveState history[kMaxHistory];
  int histSize;
};
