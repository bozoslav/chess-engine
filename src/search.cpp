#include "search.h"

#include <algorithm>
#include <chrono>

#include "evaluate.h"
#include "movegen.h"
#include "transposition_table.h"

namespace {

constexpr int kInfinity = 32000;
constexpr int kMateScore = 30000;
constexpr int kMaxPly = 96;
constexpr int kHistoryLimit = 16384;
constexpr int kTTMoveScore = 1'000'000;
constexpr int kCaptureScore = 500'000;
constexpr int kPromotionScore = 450'000;
constexpr int kFirstKillerScore = 300'000;
constexpr int kSecondKillerScore = 299'000;
constexpr std::uint64_t kTimeCheckMask = 2047;

using Clock = std::chrono::steady_clock;

struct SearchState {
  std::uint64_t nodes = 0;
  std::uint64_t ttHits = 0;
  std::uint64_t ttCutoffs = 0;
  std::uint64_t ttStores = 0;
  std::uint64_t ttMoveUses = 0;
  std::uint64_t killerMoveUses = 0;
  std::uint64_t historyMoveUses = 0;
  std::uint64_t quietCutoffs = 0;
  std::uint64_t pvsResearches = 0;
  std::uint64_t aspirationResearches = 0;
  Move killerMoves[kMaxPly][2];
  Move pvTable[kMaxPly][kMaxPly];
  int pvLength[kMaxPly] = {};
  int history[2][64][64] = {};
  TranspositionTable* tt = nullptr;
  Clock::time_point deadline;
  bool useQuietOrdering = true;
  bool usePVS = true;
  bool useAspirationWindows = true;
  bool hasDeadline = false;
  bool stopped = false;
  int aspirationWindow = 50;
};

int colorIndex(Color color) { return color == Color::White ? 0 : 1; }

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

bool deadlineExpired(SearchState& state) {
  if (!state.hasDeadline || state.stopped) return state.stopped;
  if ((state.nodes & kTimeCheckMask) != 0) return false;

  if (Clock::now() >= state.deadline) {
    state.stopped = true;
  }
  return state.stopped;
}

bool visitNode(SearchState& state) {
  ++state.nodes;
  return deadlineExpired(state);
}

bool isQuietMove(const Move& move) {
  return !move.isCapture() && !move.isPromotion();
}

bool isKillerMove(const SearchState& state, int ply, const Move& move) {
  if (ply < 0 || ply >= kMaxPly) return false;
  return sameMove(state.killerMoves[ply][0], move) ||
         sameMove(state.killerMoves[ply][1], move);
}

int historyScore(const SearchState& state, Color side, const Move& move) {
  return state.history[colorIndex(side)][move.fromSquare()][move.toSquare()];
}

void recordKillerMove(SearchState& state, int ply, const Move& move) {
  if (ply < 0 || ply >= kMaxPly) return;
  if (!isQuietMove(move)) return;
  if (sameMove(state.killerMoves[ply][0], move)) return;

  state.killerMoves[ply][1] = state.killerMoves[ply][0];
  state.killerMoves[ply][0] = move;
}

void updateHistory(SearchState& state, Color side, const Move& move,
                   int depth) {
  if (!isQuietMove(move)) return;

  int& value =
      state.history[colorIndex(side)][move.fromSquare()][move.toSquare()];
  value += depth * depth;
  if (value > kHistoryLimit) value = kHistoryLimit;
}

int moveOrderScore(const Board& board, const Move& move,
                   const SearchState* state, int ply, const Move* ttMove,
                   bool useQuietOrdering) {
  if (ttMove != nullptr && sameMove(move, *ttMove)) return kTTMoveScore;

  int score = 0;

  if (move.isCapture()) {
    const Piece captured = capturedPiece(board, move);
    const Piece moving = board.at(move.fromX(), move.fromY());
    score += kCaptureScore;
    score += 10 * pieceValue(pieceType(captured));
    score -= pieceValue(pieceType(moving));
  }

  if (move.isPromotion()) {
    score += kPromotionScore + pieceValue(move.promo());
  }

  if (move.isCastle()) {
    score += 50;
  }

  if (useQuietOrdering && state != nullptr && isQuietMove(move)) {
    if (ply >= 0 && ply < kMaxPly) {
      if (sameMove(state->killerMoves[ply][0], move)) {
        score += kFirstKillerScore;
      } else if (sameMove(state->killerMoves[ply][1], move)) {
        score += kSecondKillerScore;
      }
    }
    score += historyScore(*state, board.sideToMove(), move);
  }

  return score;
}

void selectBestMove(const Board& board, MoveList& moves, std::size_t first,
                    const SearchState* state, int ply, const Move* ttMove,
                    bool useQuietOrdering) {
  std::size_t best = first;
  int bestScore =
      moveOrderScore(board, moves[first], state, ply, ttMove, useQuietOrdering);

  for (std::size_t index = first + 1; index < moves.size(); ++index) {
    const int score = moveOrderScore(board, moves[index], state, ply, ttMove,
                                     useQuietOrdering);
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
  if (state.tt == nullptr || state.stopped) return;
  state.tt->store(board.key(), depth, scoreToTT(score, ply), bound, bestMove);
  ++state.ttStores;
}

void clearPv(SearchState& state) {
  for (int ply = 0; ply < kMaxPly; ++ply) {
    state.pvLength[ply] = 0;
  }
}

void updatePv(SearchState& state, int ply, const Move& move) {
  if (ply < 0 || ply >= kMaxPly) return;

  state.pvTable[ply][0] = move;
  int length = 1;

  if (ply + 1 < kMaxPly) {
    const int childLength = state.pvLength[ply + 1];
    while (length < kMaxPly && length <= childLength) {
      state.pvTable[ply][length] = state.pvTable[ply + 1][length - 1];
      ++length;
    }
  }

  state.pvLength[ply] = length;
}

int quiescence(Board& board, int alpha, int beta, int ply, SearchState& state);

int negamax(Board& board, int depth, int alpha, int beta, int ply,
            SearchState& state) {
  if (visitNode(state)) return evaluate(board);
  if (ply >= 0 && ply < kMaxPly) state.pvLength[ply] = 0;

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
    selectBestMove(board, moves, index, &state, ply,
                   hasTTMove ? &ttMove : nullptr, state.useQuietOrdering);
    const Move move = moves[index];
    const Color movingSide = board.sideToMove();
    const bool quiet = isQuietMove(move);
    if (hasTTMove && index == 0 && sameMove(move, ttMove)) {
      ++state.ttMoveUses;
    } else if (state.useQuietOrdering && quiet &&
               isKillerMove(state, ply, move)) {
      ++state.killerMoveUses;
    } else if (state.useQuietOrdering && quiet &&
               historyScore(state, movingSide, move) > 0) {
      ++state.historyMoveUses;
    }
    if (!board.makeMove(move)) continue;

    int score = 0;
    if (state.usePVS && searchedMove && depth > 1 && alpha + 1 < beta) {
      score = -negamax(board, depth - 1, -alpha - 1, -alpha, ply + 1, state);
      if (!state.stopped && score > alpha && score < beta) {
        ++state.pvsResearches;
        score = -negamax(board, depth - 1, -beta, -alpha, ply + 1, state);
      }
    } else {
      score = -negamax(board, depth - 1, -beta, -alpha, ply + 1, state);
    }
    board.undoMove();
    if (state.stopped) return alpha;

    searchedMove = true;
    if (score > bestScore) {
      bestScore = score;
      bestMove = move;
    }

    if (score >= beta) {
      if (state.useQuietOrdering && quiet) {
        recordKillerMove(state, ply, move);
        updateHistory(state, movingSide, move, depth);
        ++state.quietCutoffs;
      }
      storeTT(state, board, depth, score, ply, TranspositionBound::Lower, move);
      return score;
    }
    if (score > alpha) {
      alpha = score;
      updatePv(state, ply, move);
    }
  }

  if (!searchedMove) return evaluate(board);

  const TranspositionBound bound = bestScore <= originalAlpha
                                       ? TranspositionBound::Upper
                                       : TranspositionBound::Exact;
  storeTT(state, board, depth, bestScore, ply, bound, bestMove);
  return bestScore;
}

int quiescence(Board& board, int alpha, int beta, int ply, SearchState& state) {
  if (visitNode(state)) return evaluate(board);

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
    selectBestMove(board, moves, index, nullptr, ply, nullptr, false);
    const Move move = moves[index];
    if (!inCheck && !isNoisyMove(move)) continue;
    if (!board.makeMove(move)) continue;

    const int score = -quiescence(board, -beta, -alpha, ply + 1, state);
    board.undoMove();
    if (state.stopped) return alpha;

    if (score >= beta) return score;
    if (score > alpha) alpha = score;
  }

  return alpha;
}

SearchResult copyStats(const SearchState& state, SearchResult result) {
  result.nodes = state.nodes;
  result.ttHits = state.ttHits;
  result.ttCutoffs = state.ttCutoffs;
  result.ttStores = state.ttStores;
  result.ttMoveUses = state.ttMoveUses;
  result.killerMoveUses = state.killerMoveUses;
  result.historyMoveUses = state.historyMoveUses;
  result.quietCutoffs = state.quietCutoffs;
  result.pvsResearches = state.pvsResearches;
  result.aspirationResearches = state.aspirationResearches;
  result.stopped = state.stopped;
  return result;
}

SearchResult makeCompleteResult(const SearchState& state, SearchResult result) {
  result = copyStats(state, result);
  result.pvLength = std::min(state.pvLength[0], kSearchMaxPvLength);
  for (int index = 0; index < result.pvLength; ++index) {
    result.principalVariation[index] = state.pvTable[0][index];
  }
  if (result.pvLength == 0 && result.hasBestMove) {
    result.principalVariation[0] = result.bestMove;
    result.pvLength = 1;
  }
  return result;
}

SearchResult searchRoot(Board& board, int depth, int alpha, int beta,
                        SearchState& state) {
  SearchResult result;
  result.depth = depth;
  clearPv(state);

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

  const int originalAlpha = alpha;

  Move ttMove;
  bool hasTTMove = false;
  TranspositionProbe probe;
  if (state.tt != nullptr && state.tt->probe(board.key(), probe)) {
    ++state.ttHits;
    hasTTMove = probe.hasBestMove;
    ttMove = probe.bestMove;
  }

  for (std::size_t index = 0; index < moves.size(); ++index) {
    selectBestMove(board, moves, index, &state, 0,
                   hasTTMove ? &ttMove : nullptr, state.useQuietOrdering);
    const Move move = moves[index];
    const Color movingSide = board.sideToMove();
    const bool quiet = isQuietMove(move);
    if (hasTTMove && index == 0 && sameMove(move, ttMove)) {
      ++state.ttMoveUses;
    } else if (state.useQuietOrdering && quiet &&
               isKillerMove(state, 0, move)) {
      ++state.killerMoveUses;
    } else if (state.useQuietOrdering && quiet &&
               historyScore(state, movingSide, move) > 0) {
      ++state.historyMoveUses;
    }
    if (!board.makeMove(move)) continue;

    int score = 0;
    if (state.usePVS && result.hasBestMove && depth > 1 && alpha + 1 < beta) {
      score = -negamax(board, depth - 1, -alpha - 1, -alpha, 1, state);
      if (!state.stopped && score > alpha && score < beta) {
        ++state.pvsResearches;
        score = -negamax(board, depth - 1, -beta, -alpha, 1, state);
      }
    } else {
      score = -negamax(board, depth - 1, -beta, -alpha, 1, state);
    }
    board.undoMove();
    if (state.stopped) return copyStats(state, result);

    if (!result.hasBestMove || score > result.score) {
      result.bestMove = move;
      result.score = score;
      result.hasBestMove = true;
      updatePv(state, 0, move);
    }

    if (score >= beta) {
      if (state.useQuietOrdering && quiet) {
        recordKillerMove(state, 0, move);
        updateHistory(state, movingSide, move, depth);
        ++state.quietCutoffs;
      }
      storeTT(state, board, depth, score, 0, TranspositionBound::Lower, move);
      return makeCompleteResult(state, result);
    }
    if (score > alpha) alpha = score;
  }

  if (result.hasBestMove) {
    const TranspositionBound bound = result.score <= originalAlpha
                                         ? TranspositionBound::Upper
                                         : TranspositionBound::Exact;
    storeTT(state, board, depth, result.score, 0, bound, result.bestMove);
  }

  return makeCompleteResult(state, result);
}

SearchResult searchDepth(Board& board, int depth, SearchState& state,
                         const SearchResult& previousResult) {
  int alpha = -kInfinity;
  int beta = kInfinity;
  int window = std::max(1, state.aspirationWindow);
  bool aspirating = state.useAspirationWindows && previousResult.hasBestMove &&
                    depth > 1 && window < kInfinity / 2;

  if (aspirating) {
    alpha = std::max(-kInfinity, previousResult.score - window);
    beta = std::min(kInfinity, previousResult.score + window);
  }

  SearchResult result;
  while (true) {
    result = searchRoot(board, depth, alpha, beta, state);
    if (state.stopped || !result.hasBestMove || !aspirating) return result;
    if (result.score > alpha && result.score < beta) return result;

    ++state.aspirationResearches;
    window = std::min(window * 2, kInfinity);
    alpha = std::max(-kInfinity, result.score - window);
    beta = std::min(kInfinity, result.score + window);

    if (alpha == -kInfinity && beta == kInfinity) {
      aspirating = false;
    }
  }
}

}  // namespace

void clearSearchState() { globalTranspositionTable().clear(); }

SearchResult searchBestMove(Board& board, SearchLimits limits) {
  if (limits.depth < 1) limits.depth = 1;

  SearchState state;
  state.tt =
      limits.useTranspositionTable ? &globalTranspositionTable() : nullptr;
  state.useQuietOrdering = limits.useQuietOrdering;
  state.usePVS = limits.usePVS;
  state.useAspirationWindows = limits.useAspirationWindows;
  state.aspirationWindow = std::max(1, limits.aspirationWindow);
  if (limits.timeLimitMs > 0) {
    state.hasDeadline = true;
    state.deadline = Clock::now() + std::chrono::milliseconds(
                                        static_cast<int>(limits.timeLimitMs));
  }

  SearchResult result;
  const int firstDepth = limits.iterativeDeepening ? 1 : limits.depth;
  for (int depth = firstDepth; depth <= limits.depth; ++depth) {
    if (deadlineExpired(state)) break;

    const SearchResult current = searchDepth(board, depth, state, result);
    if (state.stopped) {
      if (!result.hasBestMove && current.hasBestMove) result = current;
      result = copyStats(state, result);
      result.stopped = true;
      break;
    }

    result = current;
    if (limits.onDepthComplete != nullptr) {
      limits.onDepthComplete(result, limits.infoContext);
    }
    if (!result.hasBestMove) break;
  }

  return result;
}
