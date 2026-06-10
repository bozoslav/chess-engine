#include "search.h"

#include <algorithm>

#include "evaluate.h"
#include "movegen.h"

namespace {

constexpr int kInfinity = 32000;
constexpr int kMateScore = 30000;
constexpr int kMaxPly = 96;

struct SearchState {
  std::uint64_t nodes = 0;
};

Piece capturedPiece(const Board& board, const Move& move) {
  if (move.isEnPassant()) {
    return board.sideToMove() == Color::White ? Piece::BlackPawn
                                              : Piece::WhitePawn;
  }

  return board.at(move.toX(), move.toY());
}

int moveOrderScore(const Board& board, const Move& move) {
  int score = 0;

  if (move.isCapture()) {
    const Piece captured = capturedPiece(board, move);
    const Piece moving = board.at(move.fromX(), move.fromY());
    score += 10000;
    score += 10 * pieceValue(pieceType(captured));
    score -= pieceValue(pieceType(moving));
  }

  if (move.isPromotion()) {
    score += 8000 + pieceValue(move.promo());
  }

  if (move.isCastle()) {
    score += 50;
  }

  return score;
}

void selectBestMove(const Board& board, MoveList& moves, std::size_t first) {
  std::size_t best = first;
  int bestScore = moveOrderScore(board, moves[first]);

  for (std::size_t index = first + 1; index < moves.size(); ++index) {
    const int score = moveOrderScore(board, moves[index]);
    if (score > bestScore) {
      bestScore = score;
      best = index;
    }
  }

  if (best != first) {
    std::swap(moves[first], moves[best]);
  }
}

bool isNoisyMove(const Move& move) {
  return move.isCapture() || move.isPromotion();
}

int quiescence(Board& board, int alpha, int beta, int ply, SearchState& state);

int negamax(Board& board, int depth, int alpha, int beta, int ply,
            SearchState& state) {
  ++state.nodes;

  if (ply > 0 && board.hasRepeatedPosition()) return 0;
  if (ply >= kMaxPly) return evaluate(board);
  if (depth <= 0) return quiescence(board, alpha, beta, ply, state);

  MoveList moves;
  genLegalMoves(board, moves);
  if (moves.overflowed()) return evaluate(board);

  if (moves.empty()) {
    return board.isKingInCheck() ? -kMateScore + ply : 0;
  }

  for (std::size_t index = 0; index < moves.size(); ++index) {
    selectBestMove(board, moves, index);
    const Move move = moves[index];
    if (!board.makeMove(move)) continue;

    const int score =
        -negamax(board, depth - 1, -beta, -alpha, ply + 1, state);
    board.undoMove();

    if (score >= beta) return beta;
    if (score > alpha) alpha = score;
  }

  return alpha;
}

int quiescence(Board& board, int alpha, int beta, int ply,
               SearchState& state) {
  ++state.nodes;

  if (ply > 0 && board.hasRepeatedPosition()) return 0;
  if (ply >= kMaxPly) return evaluate(board);

  const bool inCheck = board.isKingInCheck();
  if (!inCheck) {
    const int standPat = evaluate(board);
    if (standPat >= beta) return beta;
    if (standPat > alpha) alpha = standPat;
  }

  MoveList moves;
  genLegalMoves(board, moves);
  if (moves.overflowed()) return evaluate(board);

  if (moves.empty()) {
    return inCheck ? -kMateScore + ply : alpha;
  }

  for (std::size_t index = 0; index < moves.size(); ++index) {
    selectBestMove(board, moves, index);
    const Move move = moves[index];
    if (!inCheck && !isNoisyMove(move)) continue;
    if (!board.makeMove(move)) continue;

    const int score = -quiescence(board, -beta, -alpha, ply + 1, state);
    board.undoMove();

    if (score >= beta) return beta;
    if (score > alpha) alpha = score;
  }

  return alpha;
}

}  // namespace

SearchResult searchBestMove(Board& board, SearchLimits limits) {
  if (limits.depth < 1) limits.depth = 1;

  SearchState state;
  SearchResult result;
  result.depth = limits.depth;

  MoveList moves;
  genLegalMoves(board, moves);
  if (moves.overflowed()) {
    result.score = evaluate(board);
    return result;
  }

  if (moves.empty()) {
    result.score = board.isKingInCheck() ? -kMateScore : 0;
    return result;
  }

  int alpha = -kInfinity;
  constexpr int beta = kInfinity;

  for (std::size_t index = 0; index < moves.size(); ++index) {
    selectBestMove(board, moves, index);
    const Move move = moves[index];
    if (!board.makeMove(move)) continue;

    const int score =
        -negamax(board, limits.depth - 1, -beta, -alpha, 1, state);
    board.undoMove();

    if (!result.hasBestMove || score > result.score) {
      result.bestMove = move;
      result.score = score;
      result.hasBestMove = true;
    }

    if (score > alpha) alpha = score;
  }

  result.nodes = state.nodes;
  return result;
}
