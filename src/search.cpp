#include "search.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

#include "evaluate.h"
#include "movegen.h"
#include "see.h"
#include "transposition_table.h"

namespace {

constexpr int kInfinity = 32000;
constexpr int kMateScore = 30000;
constexpr int kMaxPly = 96;
constexpr int kHistoryLimit = 16384;
constexpr int kTTMoveScore = 1'000'000;
constexpr int kCaptureScore = 500'000;
constexpr int kBadCaptureScore = 100'000;
constexpr int kPromotionScore = 450'000;
constexpr int kFirstKillerScore = 300'000;
constexpr int kSecondKillerScore = 299'000;
constexpr int kCounterMoveScore = 250'000;
constexpr std::uint64_t kTimeCheckMask = 2047;
constexpr int kNullMoveMinDepth = 4;
constexpr int kMaxLmrReduction = 3;

using Clock = std::chrono::steady_clock;

struct SearchState {
  std::uint64_t nodes = 0;
  std::uint64_t publishedNodes = 0;
  std::uint64_t ttHits = 0;
  std::uint64_t ttCutoffs = 0;
  std::uint64_t ttStores = 0;
  std::uint64_t ttMoveUses = 0;
  std::uint64_t killerMoveUses = 0;
  std::uint64_t counterMoveUses = 0;
  std::uint64_t historyMoveUses = 0;
  std::uint64_t continuationHistoryUses = 0;
  std::uint64_t captureHistoryUses = 0;
  std::uint64_t quietCutoffs = 0;
  std::uint64_t pvsResearches = 0;
  std::uint64_t aspirationResearches = 0;
  std::uint64_t nullMoveAttempts = 0;
  std::uint64_t nullMovePrunes = 0;
  std::uint64_t lmrAttempts = 0;
  std::uint64_t lmrResearches = 0;
  std::uint64_t seePrunes = 0;
  std::uint64_t futilityPrunes = 0;
  std::uint64_t lateMovePrunes = 0;
  Move killerMoves[kMaxPly][2];
  Move counterMoves[2][64][64];
  Move previousMoves[kMaxPly];
  bool hasPreviousMove[kMaxPly] = {};
  Move pvTable[kMaxPly][kMaxPly];
  int pvLength[kMaxPly] = {};
  int history[2][64][64] = {};
  int continuationHistory[2][64][64] = {};
  int captureHistory[2][7][7][64] = {};
  TranspositionTable* tt = nullptr;
  std::atomic_bool* stopSignal = nullptr;
  std::atomic<std::uint64_t>* sharedNodeCounter = nullptr;
  Clock::time_point startTime;
  Clock::time_point deadline;
  bool useQuietOrdering = true;
  bool usePVS = true;
  bool useAspirationWindows = true;
  bool useNullMovePruning = true;
  bool useLateMoveReductions = true;
  bool useStaticExchangeEvaluation = true;
  bool useFutilityPruning = true;
  bool useLateMovePruning = true;
  bool hasDeadline = false;
  bool stopped = false;
  int aspirationWindow = 50;
  int nullMoveReduction = 2;
  int lmrMinDepth = 3;
  int lmrMinMoveNumber = 4;
  int futilityMargin = 120;
  int lateMovePruningMaxDepth = 3;
  int lateMovePruningBaseMoveCount = 8;
  int rootMoveSeed = 0;
};

int colorIndex(Color color) { return color == Color::White ? 0 : 1; }

int pieceTypeIndex(PieceType type) { return static_cast<int>(type); }

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

void publishNodeProgress(SearchState& state) {
  if (state.sharedNodeCounter == nullptr ||
      state.publishedNodes == state.nodes)
    return;
  state.sharedNodeCounter->fetch_add(state.nodes - state.publishedNodes,
                                     std::memory_order_relaxed);
  state.publishedNodes = state.nodes;
}

std::uint64_t aggregateNodeCount(const SearchState& state) {
  if (state.sharedNodeCounter == nullptr) return state.nodes;
  return state.sharedNodeCounter->load(std::memory_order_relaxed) +
         (state.nodes - state.publishedNodes);
}

bool deadlineExpired(SearchState& state) {
  if (state.stopSignal != nullptr &&
      state.stopSignal->load(std::memory_order_relaxed)) {
    state.stopped = true;
    return true;
  }
  if (state.stopped) return true;
  if (!state.hasDeadline && state.sharedNodeCounter == nullptr) return false;
  if ((state.nodes & kTimeCheckMask) != 0) return false;
  publishNodeProgress(state);
  if (!state.hasDeadline) return false;

  if (Clock::now() >= state.deadline) {
    state.stopped = true;
  }
  return state.stopped;
}

bool visitNode(SearchState& state) {
  ++state.nodes;
  return deadlineExpired(state);
}

std::uint64_t elapsedMilliseconds(const SearchState& state) {
  const std::chrono::duration<double, std::milli> elapsed =
      Clock::now() - state.startTime;
  return static_cast<std::uint64_t>(std::max(0.0, elapsed.count()));
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

bool hasPreviousMove(const SearchState& state, int ply) {
  return ply > 0 && ply <= kMaxPly && state.hasPreviousMove[ply - 1];
}

Move previousMove(const SearchState& state, int ply) {
  return state.previousMoves[ply - 1];
}

void setPreviousMove(SearchState& state, int ply, const Move& move) {
  if (ply < 0 || ply >= kMaxPly) return;
  state.previousMoves[ply] = move;
  state.hasPreviousMove[ply] = true;
}

void clearPreviousMove(SearchState& state, int ply) {
  if (ply < 0 || ply >= kMaxPly) return;
  state.hasPreviousMove[ply] = false;
}

bool isCounterMove(const SearchState& state, Color side, int ply,
                   const Move& move) {
  if (!hasPreviousMove(state, ply)) return false;
  const Move previous = previousMove(state, ply);
  return sameMove(state.counterMoves[colorIndex(side)][previous.fromSquare()]
                                    [previous.toSquare()],
                  move);
}

int continuationHistoryScore(const SearchState& state, Color side, int ply,
                             const Move& move) {
  if (!hasPreviousMove(state, ply)) return 0;
  const Move previous = previousMove(state, ply);
  return state.continuationHistory[colorIndex(side)][previous.toSquare()]
                                  [move.toSquare()];
}

int quietHistoryScore(const SearchState& state, Color side, int ply,
                      const Move& move) {
  return historyScore(state, side, move) +
         continuationHistoryScore(state, side, ply, move);
}

int captureHistoryScore(const SearchState& state, const Board& board,
                        const Move& move) {
  if (!move.isCapture()) return 0;
  const Piece moving = board.at(move.fromX(), move.fromY());
  const Piece captured = capturedPiece(board, move);
  return state.captureHistory[colorIndex(board.sideToMove())][pieceTypeIndex(
      pieceType(moving))][pieceTypeIndex(pieceType(captured))][move.toSquare()];
}

bool isQuietCandidateForPruning(const SearchState& state, int ply, Color side,
                                const Move& move) {
  return isQuietMove(move) && !isKillerMove(state, ply, move) &&
         !isCounterMove(state, side, ply, move) &&
         quietHistoryScore(state, side, ply, move) <= 0;
}

void recordKillerMove(SearchState& state, int ply, const Move& move) {
  if (ply < 0 || ply >= kMaxPly) return;
  if (!isQuietMove(move)) return;
  if (sameMove(state.killerMoves[ply][0], move)) return;

  state.killerMoves[ply][1] = state.killerMoves[ply][0];
  state.killerMoves[ply][0] = move;
}

int historyBonusForDepth(int depth) {
  return std::min(kHistoryLimit, depth * depth + depth - 1);
}

void applyHistoryBonus(int& value, int bonus) {
  bonus = std::clamp(bonus, -kHistoryLimit, kHistoryLimit);
  const int absBonus = bonus < 0 ? -bonus : bonus;
  value += bonus - value * absBonus / kHistoryLimit;
  value = std::clamp(value, -kHistoryLimit, kHistoryLimit);
}

void updateHistory(SearchState& state, Color side, const Move& move,
                   int bonus) {
  if (!isQuietMove(move)) return;

  int& value =
      state.history[colorIndex(side)][move.fromSquare()][move.toSquare()];
  applyHistoryBonus(value, bonus);
}

void updateContinuationHistory(SearchState& state, Color side, int ply,
                               const Move& move, int bonus) {
  if (!isQuietMove(move) || !hasPreviousMove(state, ply)) return;

  const Move previous = previousMove(state, ply);
  int& value = state.continuationHistory[colorIndex(side)][previous.toSquare()]
                                        [move.toSquare()];
  applyHistoryBonus(value, bonus);
}

void updateCounterMove(SearchState& state, Color side, int ply,
                       const Move& move) {
  if (!isQuietMove(move) || !hasPreviousMove(state, ply)) return;

  const Move previous = previousMove(state, ply);
  state.counterMoves[colorIndex(side)][previous.fromSquare()]
                    [previous.toSquare()] = move;
}

void updateCaptureHistory(SearchState& state, const Board& board,
                          const Move& move, int bonus) {
  if (!move.isCapture()) return;

  const Piece moving = board.at(move.fromX(), move.fromY());
  const Piece captured = capturedPiece(board, move);
  int& value =
      state.captureHistory[colorIndex(board.sideToMove())][pieceTypeIndex(
          pieceType(moving))][pieceTypeIndex(pieceType(captured))]
                          [move.toSquare()];
  applyHistoryBonus(value, bonus);
}

void updateQuietOrdering(SearchState& state, Color side, int ply,
                         const Move& move, const Move* failedQuietMoves,
                         std::size_t failedQuietCount, int depth) {
  const int bonus = historyBonusForDepth(depth);
  recordKillerMove(state, ply, move);
  updateHistory(state, side, move, bonus);
  updateContinuationHistory(state, side, ply, move, bonus);
  updateCounterMove(state, side, ply, move);

  const int malus = -bonus;
  for (std::size_t index = 0; index < failedQuietCount; ++index) {
    const Move failed = failedQuietMoves[index];
    if (sameMove(failed, move)) continue;
    updateHistory(state, side, failed, malus);
    updateContinuationHistory(state, side, ply, failed, malus);
  }
}

void updateCaptureOrdering(SearchState& state, const Board& board,
                           const Move& move, const Move* failedCaptures,
                           std::size_t failedCaptureCount, int depth) {
  const int bonus = historyBonusForDepth(depth);
  updateCaptureHistory(state, board, move, bonus);

  const int malus = -bonus;
  for (std::size_t index = 0; index < failedCaptureCount; ++index) {
    const Move failed = failedCaptures[index];
    if (sameMove(failed, move)) continue;
    updateCaptureHistory(state, board, failed, malus);
  }
}

int rootDiversificationScore(int seed, const Move& move) {
  std::uint32_t value = static_cast<std::uint32_t>(move.raw());
  value ^= static_cast<std::uint32_t>(seed) + 0x9e3779b9U + (value << 6) +
           (value >> 2);
  value ^= value >> 16;
  value *= 0x7feb352dU;
  value ^= value >> 15;
  return static_cast<int>(value & 4095U);
}

int moveOrderScore(const Board& board, const Move& move,
                   const SearchState* state, int ply, const Move* ttMove,
                   bool useQuietOrdering, bool useSee, int* seeValue) {
  if (seeValue != nullptr) *seeValue = 0;
  if (ttMove != nullptr && sameMove(move, *ttMove)) return kTTMoveScore;

  int score = 0;

  if (move.isCapture()) {
    const Piece captured = capturedPiece(board, move);
    const Piece moving = board.at(move.fromX(), move.fromY());
    const int seeScore = useSee ? staticExchangeEvaluation(board, move)
                                : pieceValue(pieceType(captured)) -
                                      pieceValue(pieceType(moving));
    if (seeValue != nullptr) *seeValue = seeScore;
    score += seeScore >= 0 ? kCaptureScore : kBadCaptureScore;
    score += seeScore * 16;
    score += 10 * pieceValue(pieceType(captured));
    score -= pieceValue(pieceType(moving));
    if (state != nullptr) {
      score += captureHistoryScore(*state, board, move);
    }
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
    if (isCounterMove(*state, board.sideToMove(), ply, move)) {
      score += kCounterMoveScore;
    }
    score += quietHistoryScore(*state, board.sideToMove(), ply, move);
  }

  if (state != nullptr && ply == 0 && state->rootMoveSeed != 0 &&
      (ttMove == nullptr || !sameMove(move, *ttMove))) {
    score += rootDiversificationScore(state->rootMoveSeed, move);
  }

  return score;
}

void scoreMoves(const Board& board, const MoveList& moves, int* scores,
                const SearchState* state, int ply, const Move* ttMove,
                bool useQuietOrdering, bool useSee, int* seeValues = nullptr) {
  for (std::size_t index = 0; index < moves.size(); ++index) {
    scores[index] = moveOrderScore(
        board, moves[index], state, ply, ttMove, useQuietOrdering, useSee,
        seeValues != nullptr ? &seeValues[index] : nullptr);
  }
}

void selectBestMove(MoveList& moves, int* scores, std::size_t first,
                    int* auxiliary = nullptr) {
  std::size_t best = first;
  int bestScore = scores[first];

  for (std::size_t index = first + 1; index < moves.size(); ++index) {
    const int score = scores[index];
    if (score > bestScore) {
      bestScore = score;
      best = index;
    }
  }

  if (best != first) {
    std::swap(moves[first], moves[best]);
    std::swap(scores[first], scores[best]);
    if (auxiliary != nullptr) std::swap(auxiliary[first], auxiliary[best]);
  }
}

bool hasNonPawnMaterial(const Board& board, Color side) {
  return (board.pieces(side, PieceType::Knight) |
          board.pieces(side, PieceType::Bishop) |
          board.pieces(side, PieceType::Rook) |
          board.pieces(side, PieceType::Queen)) != 0;
}

int lmrReduction(int depth, std::size_t moveIndex, int quietHistory) {
  int reduction = 1;
  if (depth >= 6 && moveIndex >= 7) ++reduction;
  if (depth >= 10 && moveIndex >= 15) ++reduction;
  if (depth >= 5 && quietHistory < -kHistoryLimit / 4) ++reduction;
  if (quietHistory > kHistoryLimit / 4) --reduction;
  return std::clamp(reduction, 1, kMaxLmrReduction);
}

bool canFutilityPrune(const SearchState& state, int depth, int alpha,
                      int staticEval, bool inCheck, bool cutNode,
                      bool searchedMove, bool quietCandidate) {
  if (!state.useFutilityPruning || inCheck || !cutNode || !searchedMove ||
      !quietCandidate || depth > 2 || alpha <= -kMateScore + kMaxPly) {
    return false;
  }

  return staticEval + state.futilityMargin * depth <= alpha;
}

bool canLateMovePrune(const SearchState& state, int depth,
                      std::size_t moveIndex, bool inCheck, bool cutNode,
                      bool searchedMove, bool quietCandidate) {
  if (!state.useLateMovePruning || inCheck || !cutNode || !searchedMove ||
      !quietCandidate || depth > state.lateMovePruningMaxDepth) {
    return false;
  }

  const std::size_t threshold = static_cast<std::size_t>(
      state.lateMovePruningBaseMoveCount + depth * depth);
  return moveIndex >= threshold;
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
            SearchState& state, bool allowNullMove) {
  if (visitNode(state)) return evaluate(board);
  if (ply >= 0 && ply < kMaxPly) state.pvLength[ply] = 0;

  const bool repeated = board.hasRepeatedPosition();
  if (ply > 0 && repeated) return 0;
  if (ply >= kMaxPly) return evaluate(board);
  if (depth <= 0) return quiescence(board, alpha, beta, ply, state);

  const int originalAlpha = alpha;
  Move ttMove;
  bool hasTTMove = false;
  if (state.tt != nullptr && !repeated) {
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

  const bool inCheck = board.isKingInCheck();
  const bool cutNode = alpha + 1 == beta;
  int staticEval = 0;
  bool hasStaticEval = false;
  const bool canTryNullMove =
      state.useNullMovePruning && allowNullMove && cutNode &&
      depth >= kNullMoveMinDepth && ply > 0 && !repeated &&
      beta > -kMateScore + kMaxPly && beta < kMateScore - kMaxPly && !inCheck &&
      hasNonPawnMaterial(board, board.sideToMove());
  if (canTryNullMove) {
    staticEval = evaluate(board);
    hasStaticEval = true;
  }
  if (canTryNullMove && staticEval >= beta) {
    ++state.nullMoveAttempts;
    if (board.makeNullMove()) {
      const int reduction = std::max(1, state.nullMoveReduction + depth / 6);
      const int nullDepth = std::max(0, depth - 1 - reduction);
      clearPreviousMove(state, ply);
      const int score =
          -negamax(board, nullDepth, -beta, -beta + 1, ply + 1, state, false);
      board.undoNullMove();

      if (state.stopped) return alpha;
      if (score >= beta) {
        ++state.nullMovePrunes;
        storeTT(state, board, depth, score, ply, TranspositionBound::Lower, {});
        return score;
      }
    }
  }

  MoveList moves;
  genLegalMoves(board, moves);
  if (moves.overflowed()) return evaluate(board);

  if (moves.empty()) {
    const int score = inCheck ? -kMateScore + ply : 0;
    storeTT(state, board, depth, score, ply, TranspositionBound::Exact, {});
    return score;
  }

  int bestScore = -kInfinity;
  Move bestMove;
  bool searchedMove = false;
  Move failedQuietMoves[MoveList::kCapacity];
  std::size_t failedQuietCount = 0;
  Move failedCaptures[MoveList::kCapacity];
  std::size_t failedCaptureCount = 0;
  int moveScores[MoveList::kCapacity] = {};
  scoreMoves(board, moves, moveScores, &state, ply,
             hasTTMove ? &ttMove : nullptr, state.useQuietOrdering,
             state.useStaticExchangeEvaluation);

  for (std::size_t index = 0; index < moves.size(); ++index) {
    selectBestMove(moves, moveScores, index);
    const Move move = moves[index];
    const Color movingSide = board.sideToMove();
    const bool quiet = isQuietMove(move);
    if (hasTTMove && index == 0 && sameMove(move, ttMove)) {
      ++state.ttMoveUses;
    } else if (state.useQuietOrdering && quiet &&
               isKillerMove(state, ply, move)) {
      ++state.killerMoveUses;
    } else if (state.useQuietOrdering && quiet &&
               isCounterMove(state, movingSide, ply, move)) {
      ++state.counterMoveUses;
    } else if (!quiet && captureHistoryScore(state, board, move) > 0) {
      ++state.captureHistoryUses;
    } else if (state.useQuietOrdering && quiet &&
               continuationHistoryScore(state, movingSide, ply, move) > 0) {
      ++state.continuationHistoryUses;
    } else if (state.useQuietOrdering && quiet &&
               historyScore(state, movingSide, move) > 0) {
      ++state.historyMoveUses;
    }
    const bool quietCandidate =
        isQuietCandidateForPruning(state, ply, movingSide, move);
    const int quietScore = quietHistoryScore(state, movingSide, ply, move);
    if ((state.useFutilityPruning || state.useLateMovePruning) &&
        !hasStaticEval && !inCheck) {
      staticEval = evaluate(board);
      hasStaticEval = true;
    }
    if (canFutilityPrune(state, depth, alpha, staticEval, inCheck, cutNode,
                         searchedMove, quietCandidate)) {
      ++state.futilityPrunes;
      continue;
    }
    if (canLateMovePrune(state, depth, index, inCheck, cutNode, searchedMove,
                         quietCandidate)) {
      ++state.lateMovePrunes;
      continue;
    }
    const bool canReduce =
        state.useLateMoveReductions && searchedMove && quiet && cutNode &&
        depth >= state.lmrMinDepth &&
        index >= static_cast<std::size_t>(state.lmrMinMoveNumber - 1) &&
        !inCheck && !isKillerMove(state, ply, move) &&
        !isCounterMove(state, movingSide, ply, move) && quietScore <= 0;
    if (!board.makeGeneratedMove(move)) continue;

    int score = 0;
    setPreviousMove(state, ply, move);
    if (canReduce) {
      ++state.lmrAttempts;
      const int reduction = lmrReduction(depth, index, quietScore);
      const int reducedDepth = std::max(0, depth - 1 - reduction);
      score = -negamax(board, reducedDepth, -alpha - 1, -alpha, ply + 1, state,
                       true);
      if (!state.stopped && score > alpha) {
        ++state.lmrResearches;
        score = -negamax(board, depth - 1, -beta, -alpha, ply + 1, state, true);
      }
    } else if (state.usePVS && searchedMove && depth > 1 && alpha + 1 < beta) {
      score =
          -negamax(board, depth - 1, -alpha - 1, -alpha, ply + 1, state, true);
      if (!state.stopped && score > alpha && score < beta) {
        ++state.pvsResearches;
        score = -negamax(board, depth - 1, -beta, -alpha, ply + 1, state, true);
      }
    } else {
      score = -negamax(board, depth - 1, -beta, -alpha, ply + 1, state, true);
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
        updateQuietOrdering(state, movingSide, ply, move, failedQuietMoves,
                            failedQuietCount, depth);
        ++state.quietCutoffs;
      } else if (move.isCapture()) {
        updateCaptureOrdering(state, board, move, failedCaptures,
                              failedCaptureCount, depth);
      }
      storeTT(state, board, depth, score, ply, TranspositionBound::Lower, move);
      return score;
    }
    if (state.useQuietOrdering && quiet) {
      failedQuietMoves[failedQuietCount++] = move;
    } else if (move.isCapture()) {
      failedCaptures[failedCaptureCount++] = move;
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
  if (inCheck) {
    genLegalMoves(board, moves);
  } else {
    genLegalNoisyMoves(board, moves);
  }
  if (moves.overflowed()) return evaluate(board);

  if (moves.empty()) {
    return inCheck ? -kMateScore + ply : alpha;
  }

  int moveScores[MoveList::kCapacity] = {};
  int seeValues[MoveList::kCapacity] = {};
  scoreMoves(board, moves, moveScores, &state, ply, nullptr, false,
             state.useStaticExchangeEvaluation, seeValues);

  for (std::size_t index = 0; index < moves.size(); ++index) {
    selectBestMove(moves, moveScores, index, seeValues);
    const Move move = moves[index];
    if (!inCheck && state.useStaticExchangeEvaluation && move.isCapture() &&
        !move.isPromotion() && seeValues[index] < 0) {
      ++state.seePrunes;
      continue;
    }
    if (!board.makeGeneratedMove(move)) continue;

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
  result.counterMoveUses = state.counterMoveUses;
  result.historyMoveUses = state.historyMoveUses;
  result.continuationHistoryUses = state.continuationHistoryUses;
  result.captureHistoryUses = state.captureHistoryUses;
  result.quietCutoffs = state.quietCutoffs;
  result.pvsResearches = state.pvsResearches;
  result.aspirationResearches = state.aspirationResearches;
  result.nullMoveAttempts = state.nullMoveAttempts;
  result.nullMovePrunes = state.nullMovePrunes;
  result.lmrAttempts = state.lmrAttempts;
  result.lmrResearches = state.lmrResearches;
  result.seePrunes = state.seePrunes;
  result.futilityPrunes = state.futilityPrunes;
  result.lateMovePrunes = state.lateMovePrunes;
  result.timeMs = elapsedMilliseconds(state);
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

  const bool useRootTTMove = hasTTMove && state.rootMoveSeed == 0;
  int moveScores[MoveList::kCapacity] = {};
  scoreMoves(board, moves, moveScores, &state, 0,
             useRootTTMove ? &ttMove : nullptr, state.useQuietOrdering,
             state.useStaticExchangeEvaluation);

  Move failedQuietMoves[MoveList::kCapacity];
  std::size_t failedQuietCount = 0;
  Move failedCaptures[MoveList::kCapacity];
  std::size_t failedCaptureCount = 0;
  for (std::size_t index = 0; index < moves.size(); ++index) {
    selectBestMove(moves, moveScores, index);
    const Move move = moves[index];
    const Color movingSide = board.sideToMove();
    const bool quiet = isQuietMove(move);
    if (useRootTTMove && index == 0 && sameMove(move, ttMove)) {
      ++state.ttMoveUses;
    } else if (state.useQuietOrdering && quiet &&
               isKillerMove(state, 0, move)) {
      ++state.killerMoveUses;
    } else if (state.useQuietOrdering && quiet &&
               isCounterMove(state, movingSide, 0, move)) {
      ++state.counterMoveUses;
    } else if (!quiet && captureHistoryScore(state, board, move) > 0) {
      ++state.captureHistoryUses;
    } else if (state.useQuietOrdering && quiet &&
               continuationHistoryScore(state, movingSide, 0, move) > 0) {
      ++state.continuationHistoryUses;
    } else if (state.useQuietOrdering && quiet &&
               historyScore(state, movingSide, move) > 0) {
      ++state.historyMoveUses;
    }
    if (!board.makeGeneratedMove(move)) continue;

    int score = 0;
    setPreviousMove(state, 0, move);
    if (state.usePVS && result.hasBestMove && depth > 1 && alpha + 1 < beta) {
      score = -negamax(board, depth - 1, -alpha - 1, -alpha, 1, state, true);
      if (!state.stopped && score > alpha && score < beta) {
        ++state.pvsResearches;
        score = -negamax(board, depth - 1, -beta, -alpha, 1, state, true);
      }
    } else {
      score = -negamax(board, depth - 1, -beta, -alpha, 1, state, true);
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
        updateQuietOrdering(state, movingSide, 0, move, failedQuietMoves,
                            failedQuietCount, depth);
        ++state.quietCutoffs;
      } else if (move.isCapture()) {
        updateCaptureOrdering(state, board, move, failedCaptures,
                              failedCaptureCount, depth);
      }
      storeTT(state, board, depth, score, 0, TranspositionBound::Lower, move);
      return makeCompleteResult(state, result);
    }
    if (state.useQuietOrdering && quiet) {
      failedQuietMoves[failedQuietCount++] = move;
    } else if (move.isCapture()) {
      failedCaptures[failedCaptureCount++] = move;
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

SearchResult searchBestMoveSingle(Board& board, SearchLimits limits) {
  if (limits.depth < 1) limits.depth = 1;

  SearchState state;
  state.startTime = Clock::now();
  state.tt =
      limits.useTranspositionTable ? &globalTranspositionTable() : nullptr;
  state.stopSignal = limits.stopSignal;
  state.sharedNodeCounter = limits.sharedNodeCounter;
  state.useQuietOrdering = limits.useQuietOrdering;
  state.usePVS = limits.usePVS;
  state.useAspirationWindows = limits.useAspirationWindows;
  state.useNullMovePruning = limits.useNullMovePruning;
  state.useLateMoveReductions = limits.useLateMoveReductions;
  state.useStaticExchangeEvaluation = limits.useStaticExchangeEvaluation;
  state.useFutilityPruning = limits.useFutilityPruning;
  state.useLateMovePruning = limits.useLateMovePruning;
  state.aspirationWindow = std::max(1, limits.aspirationWindow);
  state.nullMoveReduction = std::max(1, limits.nullMoveReduction);
  state.lmrMinDepth = std::max(1, limits.lmrMinDepth);
  state.lmrMinMoveNumber = std::max(1, limits.lmrMinMoveNumber);
  state.futilityMargin = std::max(1, limits.futilityMargin);
  state.lateMovePruningMaxDepth = std::max(1, limits.lateMovePruningMaxDepth);
  state.lateMovePruningBaseMoveCount =
      std::max(1, limits.lateMovePruningBaseMoveCount);
  state.rootMoveSeed = limits.rootMoveSeed;
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
      SearchResult info = result;
      info.nodes = aggregateNodeCount(state);
      limits.onDepthComplete(info, limits.infoContext);
    }
    if (!result.hasBestMove) break;
  }

  publishNodeProgress(state);
  return result;
}

void addStats(SearchResult& target, const SearchResult& source) {
  target.nodes += source.nodes;
  target.ttHits += source.ttHits;
  target.ttCutoffs += source.ttCutoffs;
  target.ttStores += source.ttStores;
  target.ttMoveUses += source.ttMoveUses;
  target.killerMoveUses += source.killerMoveUses;
  target.counterMoveUses += source.counterMoveUses;
  target.historyMoveUses += source.historyMoveUses;
  target.continuationHistoryUses += source.continuationHistoryUses;
  target.captureHistoryUses += source.captureHistoryUses;
  target.quietCutoffs += source.quietCutoffs;
  target.pvsResearches += source.pvsResearches;
  target.aspirationResearches += source.aspirationResearches;
  target.nullMoveAttempts += source.nullMoveAttempts;
  target.nullMovePrunes += source.nullMovePrunes;
  target.lmrAttempts += source.lmrAttempts;
  target.lmrResearches += source.lmrResearches;
  target.seePrunes += source.seePrunes;
  target.futilityPrunes += source.futilityPrunes;
  target.lateMovePrunes += source.lateMovePrunes;
  target.timeMs = std::max(target.timeMs, source.timeMs);
  target.stopped = target.stopped || source.stopped;
}

bool betterResult(const SearchResult& candidate, const SearchResult& current) {
  if (!candidate.hasBestMove) return false;
  if (!current.hasBestMove) return true;
  if (candidate.depth != current.depth) return candidate.depth > current.depth;
  if (candidate.stopped != current.stopped) return !candidate.stopped;
  return candidate.score > current.score;
}

void diversifyWorkerLimits(SearchLimits& limits, int workerId) {
  limits.onDepthComplete = nullptr;
  limits.infoContext = nullptr;
  limits.aspirationWindow += 8 * (workerId & 3);
  limits.rootMoveSeed =
      static_cast<int>(0x9e3779b9U * static_cast<std::uint32_t>(workerId + 1));
  if ((workerId & 1) != 0) {
    limits.lmrMinMoveNumber += 1;
  }
  if ((workerId & 2) != 0) {
    limits.nullMoveReduction += 1;
  }
  if ((workerId & 4) != 0) {
    limits.useAspirationWindows = false;
  }
}

SearchResult combineThreadResults(const SearchResult* results, int count) {
  SearchResult best;
  for (int index = 0; index < count; ++index) {
    if (betterResult(results[index], best)) best = results[index];
  }

  SearchResult stats;
  for (int index = 0; index < count; ++index) {
    addStats(stats, results[index]);
  }

  const std::uint64_t nodes = stats.nodes;
  const std::uint64_t ttHits = stats.ttHits;
  const std::uint64_t ttCutoffs = stats.ttCutoffs;
  const std::uint64_t ttStores = stats.ttStores;
  const std::uint64_t ttMoveUses = stats.ttMoveUses;
  const std::uint64_t killerMoveUses = stats.killerMoveUses;
  const std::uint64_t counterMoveUses = stats.counterMoveUses;
  const std::uint64_t historyMoveUses = stats.historyMoveUses;
  const std::uint64_t continuationHistoryUses = stats.continuationHistoryUses;
  const std::uint64_t captureHistoryUses = stats.captureHistoryUses;
  const std::uint64_t quietCutoffs = stats.quietCutoffs;
  const std::uint64_t pvsResearches = stats.pvsResearches;
  const std::uint64_t aspirationResearches = stats.aspirationResearches;
  const std::uint64_t nullMoveAttempts = stats.nullMoveAttempts;
  const std::uint64_t nullMovePrunes = stats.nullMovePrunes;
  const std::uint64_t lmrAttempts = stats.lmrAttempts;
  const std::uint64_t lmrResearches = stats.lmrResearches;
  const std::uint64_t seePrunes = stats.seePrunes;
  const std::uint64_t futilityPrunes = stats.futilityPrunes;
  const std::uint64_t lateMovePrunes = stats.lateMovePrunes;
  best.nodes = nodes;
  best.ttHits = ttHits;
  best.ttCutoffs = ttCutoffs;
  best.ttStores = ttStores;
  best.ttMoveUses = ttMoveUses;
  best.killerMoveUses = killerMoveUses;
  best.counterMoveUses = counterMoveUses;
  best.historyMoveUses = historyMoveUses;
  best.continuationHistoryUses = continuationHistoryUses;
  best.captureHistoryUses = captureHistoryUses;
  best.quietCutoffs = quietCutoffs;
  best.pvsResearches = pvsResearches;
  best.aspirationResearches = aspirationResearches;
  best.nullMoveAttempts = nullMoveAttempts;
  best.nullMovePrunes = nullMovePrunes;
  best.lmrAttempts = lmrAttempts;
  best.lmrResearches = lmrResearches;
  best.seePrunes = seePrunes;
  best.futilityPrunes = futilityPrunes;
  best.lateMovePrunes = lateMovePrunes;
  best.timeMs = stats.timeMs;
  best.stopped = best.stopped || stats.stopped;
  return best;
}

SearchResult searchBestMove(Board& board, SearchLimits limits) {
  if (limits.useTranspositionTable) globalTranspositionTable().newSearch();
  const int threadCount = std::clamp(limits.threads, 1, 128);
  if (threadCount == 1) {
    return searchBestMoveSingle(board, limits);
  }

  std::atomic_bool localStopSignal{false};
  std::atomic<std::uint64_t> sharedNodeCounter{0};
  std::atomic_bool* sharedStopSignal =
      limits.stopSignal != nullptr ? limits.stopSignal : &localStopSignal;
  limits.stopSignal = sharedStopSignal;
  limits.sharedNodeCounter = &sharedNodeCounter;

  std::vector<SearchResult> results(static_cast<std::size_t>(threadCount));
  std::vector<std::thread> workers;
  workers.reserve(static_cast<std::size_t>(threadCount - 1));

  for (int workerId = 1; workerId < threadCount; ++workerId) {
    workers.emplace_back([&, workerId]() {
      Board workerBoard = board;
      SearchLimits workerLimits = limits;
      workerLimits.threads = 1;
      diversifyWorkerLimits(workerLimits, workerId);
      results[static_cast<std::size_t>(workerId)] =
          searchBestMoveSingle(workerBoard, workerLimits);
    });
  }

  Board mainBoard = board;
  SearchLimits mainLimits = limits;
  mainLimits.threads = 1;
  results[0] = searchBestMoveSingle(mainBoard, mainLimits);

  for (std::thread& worker : workers) {
    worker.join();
  }

  return combineThreadResults(results.data(), threadCount);
}
