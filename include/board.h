#pragma once

#include "move.h"

class Board {
 public:
  static constexpr int kBoardSize = 8;

  Board();

  bool undoMove();
  void printBoard() const;
  Color sideToMove() const;
  bool isKingInCheck() const;
  Piece at(int x, int y) const;
  bool makeMove(const Move& move);

 private:
  struct MoveState {
    Move move;
    Piece movedPiece;
    Piece placedPiece;
    Piece capturedPiece;
    int capX;
    int capY;
    bool wasEp;
    bool wasCastle;
    int rookFromX;
    int rookFromY;
    int rookToX;
    int rookToY;
    Color prevSide;
    bool prevWCastleK;
    bool prevWCastleQ;
    bool prevBCastleK;
    bool prevBCastleQ;
    bool prevHasEp;
    int prevEpX;
    int prevEpY;
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
  bool isValidKingMove(const Move& move, Piece movingPiece) const;
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
  bool hasEp;
  int epX;
  int epY;
  MoveState history[kMaxHistory];
  int histSize;
};
