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
  bool undoGeneratedMove();
  void printBoard() const;
  Color sideToMove() const;
  bool isKingInCheck() const;
  Piece at(Square square) const noexcept;
  Piece at(int x, int y) const noexcept;
  Bitboard pieces(Color color, PieceType type) const;
  Bitboard occupancy(Color color) const;
  Bitboard allPieces() const;
  Square kingSquare(Color color) const;
  bool hasEnPassant() const;
  Square enPassantSquare() const;
  bool canCastleKingSide(Color color) const;
  bool canCastleQueenSide(Color color) const;
  std::uint64_t key() const;
  int halfmoveClock() const;
  int fullmoveNumber() const;
  bool isFiftyMoveDraw() const;
  bool isInsufficientMaterial() const;
  bool tracksGeneratedHalfmoves() const;
  void setTrackGeneratedHalfmoves(bool enabled);
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
    Square capturedSquare;
    bool wasNull;
    bool wasEp;
    bool wasCastle;
    Square rookFrom;
    Square rookTo;
    Color prevSide;
    bool prevWCastleK;
    bool prevWCastleQ;
    bool prevBCastleK;
    bool prevBCastleQ;
    bool prevHasEp;
    int prevEpX;
    int prevEpY;
    Square prevEpSquare;
    std::uint16_t prevHalfmoveState;
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
  void addPieceToBitboards(Piece piece, Square square);
  void removePieceFromBitboards(Piece piece, Square square);
  void rebuildBitboards();
  void putPieceNoHash(Square square, Piece piece);
  Piece removePieceNoHash(Square square);
  void putPiece(Square square, Piece piece);
  Piece removePiece(Square square);
  int castlingRightsMask() const;
  std::uint64_t computeZobristKey() const;
  bool bitboardsAreConsistent() const;
  template <bool Validate, bool UpdateHalfmove, bool UpdateFullmove>
  bool makeMoveImpl(const Move& move);
  template <bool RestoreMoveCounters>
  bool undoMoveImpl();

  Piece board[bitboard::kSquareCount];
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
  std::uint16_t halfmoveClockValue;
  std::uint16_t fullmoveNumberValue;
  bool trackGeneratedHalfmoveValue;
  std::uint64_t zobristKey;
  MoveState history[kMaxHistory];
  std::uint64_t keyHistory[kMaxHistory + 1];
  int histSize;
  int keyHistorySize;
};
