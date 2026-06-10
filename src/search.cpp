#include "search.h"

#include <algorithm>

#include "evaluate.h"
#include "movegen.h"
#include "transposition_table.h"

namespace {

constexpr int kInfinity = 32000;
constexpr int kMateScore = 30000;
constexpr int kMaxPly = 96;

struct SearchState {
  std::uint64_t nodes = 0;
  std::uint64_t ttHits = 0;
  std::uint64_t ttCutoffs = 0;
  std::uint64_t ttStores = 0;
  std::uint64_t ttMoveUses = 0;
  TranspositionTable* tt = nullptr;
};

Piece capturedPiece(const Board& board, const Move& move) {
  if (move.isEnPassant()) {
    return board.sideToMove() == Color::White ? Piece::BlackPawn
                                              : Piece::WhitePawn;
  }

  return board.at(move.toX(), move.toY());
}

int scoreToTT(int score, int ply) {
  if (score > kMateScore - kMaxPly) return score + ply;
  if (score < -kMateScore + kMaxPly) return score - ply;
  return score;
}

int scoreFromTT(int score, int ply) {
  if (score > kMateScore - kMaxPly) return score - ply;
  if (score < -kMateScore + kMaxPly) return score + ply;
  return score;
}

bool sameMove(const Move& first, const Move& second) {
  return first.raw() == second.raw();
}

int moveOrderScore(const Board& board, const Move& move,
                   const Move* ttMove) {
  if (ttMove != nullptr && sameMove(move, *ttMove)) return 1'000'000;

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

void selectBestMove(const Board& board, MoveList& moves, std::size_t first,
                    const Move* ttMove) {
  std::size_t best = first;
  int bestScore = moveOrderScore(board, moves[first], ttMove);

  for (std::size_t index = first + 1; index < moves.size(); ++index) {
    const int score = moveOrderScore(board, moves[index], ttMove);
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

void storeTT(SearchState& state, const Board& board, int depth, int score,
             int ply, TranspositionBound bound, Move bestMove) {
  if (state.tt == nullptr) return;
  state.tt->store(board.key(), depth, scoreToTT(score, ply), bound, bestMove);
  ++state.ttStores;
}

int quiescence(Board& board, int alpha, int beta, int ply, SearchState& state);

int negamax(Board& board, int depth, int alpha, int beta, int ply,
            SearchState& state) {
  ++state.nodes;

  const int repetitions = board.repetitionCount();
  if (ply > 0 && repetitions >= 2) return 0;
  if (ply >= kMaxPly) return evaluate(board);
  if (depth <= 0) return quiescence(board, alpha, beta, ply, state);

  const int originalAlpha = alpha;
  Move ttMove;
  bool hasTTMove = false;
  if (state.tt != nullptr && repetitions == 1) {
    TranspositionProbe probe;
    if (state.tt->probe(board.key(), probe)) {
      ++state.ttHits;
      hasTTMove = probe.hasBestMove;
      ttMove = probe.bestMove;

      if (probe.depth >= depth) {
        const int ttScore = scoreFromTT(probe.score, ply);
        if (probe.bound == TranspositionBound::Exact) {
          ++state.ttCutoffs;
          return ttScore;
        }
        if (probe.bound == TranspositionBound::Lower && ttScore >= beta) {
          ++state.ttCutoffs;
          return ttScore;
        }
        if (probe.bound == TranspositionBound::Upper && ttScore <= alpha) {
          ++state.ttCutoffs;
          return ttScore;
        }
      }
    }
  }

  MoveList moves;
  genLegalMoves(board, moves);
  if (moves.overflowed()) return evaluate(board);

  if (moves.empty()) {
    const int score = board.isKingInCheck() ? -kMateScore + ply : 0;
    storeTT(state, board, depth, score, ply, TranspositionBound::Exact, {});
    return score;
  }

  int bestScore = -kInfinity;
  Move bestMove;
  bool searchedMove = false;

  for (std::size_t index = 0; index < moves.size(); ++index) {
    selectBestMove(board, moves, index, hasTTMove ? &ttMove : nullptr);
    const Move move = moves[index];
    if (hasTTMove && index == 0 && sameMove(move, ttMove)) {
      ++state.ttMoveUses;
    }
    if (!board.makeMove(move)) continue;

    const int score =
        -negamax(board, depth - 1, -beta, -alpha, ply + 1, state);
    board.undoMove();

    searchedMove = true;
    if (score > bestScore) {
      bestScore = score;
      bestMove = move;
    }

    if (score >= beta) {
      storeTT(state, board, depth, score, ply, TranspositionBound::Lower,
              move);
      return score;
    }
    if (score > alpha) alpha = score;
  }

  if (!searchedMove) return evaluate(board);

  const TranspositionBound bound = bestScore <= originalAlpha
                                      ? TranspositionBound::Upper
                                      : TranspositionBound::Exact;
  storeTT(state, board, depth, bestScore, ply, bound, bestMove);
  return bestScore;
}

int quiescence(Board& board, int alpha, int beta, int ply,
               SearchState& state) {
  ++state.nodes;

  if (ply > 0 && board.hasRepeatedPosition()) return 0;
  if (ply >= kMaxPly) return evaluate(board);

  const bool inCheck = board.isKingInCheck();
  if (!inCheck) {
    const int standPat = evaluate(board);
    if (standPat >= beta) return standPat;
    if (standPat > alpha) alpha = standPat;
  }

  MoveList moves;
  genLegalMoves(board, moves);
  if (moves.overflowed()) return evaluate(board);

  if (moves.empty()) {
    return inCheck ? -kMateScore + ply : alpha;
  }

  for (std::size_t index = 0; index < moves.size(); ++index) {
    selectBestMove(board, moves, index, nullptr);
    const Move move = moves[index];
    if (!inCheck && !isNoisyMove(move)) continue;
    if (!board.makeMove(move)) continue;

    const int score = -quiescence(board, -beta, -alpha, ply + 1, state);
    board.undoMove();

    if (score >= beta) return score;
    if (score > alpha) alpha = score;
  }

  return alpha;
}

}  // namespace

void clearSearchState() { globalTranspositionTable().clear(); }

SearchResult searchBestMove(Board& board, SearchLimits limits) {
  if (limits.depth < 1) limits.depth = 1;

  SearchState state;
  state.tt = limits.useTranspositionTable ? &globalTranspositionTable()
                                          : nullptr;
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

  Move ttMove;
  bool hasTTMove = false;
  TranspositionProbe probe;
  if (state.tt != nullptr && state.tt->probe(board.key(), probe)) {
    ++state.ttHits;
    hasTTMove = probe.hasBestMove;
    ttMove = probe.bestMove;
  }

  for (std::size_t index = 0; index < moves.size(); ++index) {
    selectBestMove(board, moves, index, hasTTMove ? &ttMove : nullptr);
    const Move move = moves[index];
    if (hasTTMove && index == 0 && sameMove(move, ttMove)) {
      ++state.ttMoveUses;
    }
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

  if (result.hasBestMove) {
    storeTT(state, board, limits.depth, result.score, 0,
            TranspositionBound::Exact, result.bestMove);
  }

  result.nodes = state.nodes;
  result.ttHits = state.ttHits;
  result.ttCutoffs = state.ttCutoffs;
  result.ttStores = state.ttStores;
  result.ttMoveUses = state.ttMoveUses;
  return result;
}
