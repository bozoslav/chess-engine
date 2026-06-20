#pragma once

#include <cstdint>
#include <string>
#include <string_view>

#include "bitboard.h"
#include "move.h"

class Board {
 public:
  static constexpr int kBoardSize = 8;
  static constexpr int kColorCount = 2;
  static constexpr int kPieceTypeCount = 7;
  static constexpr int kMaxHistory = 1024;

  struct PieceDelta {
    Piece piece = Piece::None;
    Square from = -1;
    Square to = -1;
  };

  Board();

  bool undoMove();
  void printBoard() const;
  Color sideToMove() const;
  bool isKingInCheck() const;
  Piece at(int x, int y) const;
  Bitboard pieces(Color color, PieceType type) const;
  Bitboard occupancy(Color color) const;
  Bitboard allPieces() const;
  Square kingSquare(Color color) const;
  bool hasEnPassant() const;
  Square enPassantSquare() const;
  bool canCastleKingSide(Color color) const;
  bool canCastleQueenSide(Color color) const;
  std::uint64_t key() const;
  int repetitionCount() const;
  bool hasRepeatedPosition() const;
  bool isThreefoldRepetition() const;
  int ply() const;
  std::uint64_t previousKey() const;
  bool lastMoveWasNull() const;
  bool lastMoveChangedKingSquare(Color& color) const;
  int lastMovePieceDeltas(PieceDelta* deltas, int maxDeltas) const;
  bool makeMove(const Move& move);
  // Fast path for moves returned by genLegalMoves()/genLegalNoisyMoves().
  // The caller guarantees pseudo-legal geometry and king safety.
  bool makeGeneratedMove(const Move& move);
  bool makeNullMove();
  bool undoNullMove();
  bool setFromFen(std::string_view fen);
  std::string toFen() const;

 private:
  struct MoveState {
    Move move;
    Piece movedPiece;
    Piece placedPiece;
    Piece capturedPiece;
    int capX;
    int capY;
    bool wasNull;
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
    Square prevEpSquare;
    std::uint64_t prevZobristKey;
    int prevKeyHistorySize;
  };

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
  void clearBitboards();
  void addPieceToBitboards(Piece piece, int x, int y);
  void removePieceFromBitboards(Piece piece, int x, int y);
  void rebuildBitboards();
  void putPieceNoHash(int x, int y, Piece piece);
  void putPiece(int x, int y, Piece piece);
  int castlingRightsMask() const;
  std::uint64_t computeZobristKey() const;
  bool bitboardsAreConsistent() const;
  bool makeMoveImpl(const Move& move, bool validate);

  Piece board[kBoardSize][kBoardSize];
  Bitboard pieceBB[kColorCount][kPieceTypeCount];
  Bitboard occupancyBB[kColorCount];
  Bitboard allBB;
  Square kingSq[kColorCount];
  Color side;
  bool wCastleK;
  bool wCastleQ;
  bool bCastleK;
  bool bCastleQ;
  bool hasEp;
  int epX;
  int epY;
  Square epSquare;
  std::uint64_t zobristKey;
  MoveState history[kMaxHistory];
  std::uint64_t keyHistory[kMaxHistory + 1];
  int histSize;
  int keyHistorySize;
};
