#include "search.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>

#if defined(__APPLE__)
#include <pthread/qos.h>
#endif

#include "evaluate.h"
#include "movegen.h"
#include "nnue.h"
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
constexpr int kSingularTTDepthMargin = 3;
constexpr int kCorrectionHistorySize = 4096;
constexpr int kCorrectionHistoryLimit = 8192;
constexpr std::size_t kMaxHistoryMalusMoves = 32;

constexpr int kLmrDepthCount = kMaxPly + 1;
constexpr int kLmrMoveCount = static_cast<int>(MoveList::kCapacity) + 1;

auto computeLmrTable() {
  std::array<std::array<std::uint8_t, kLmrMoveCount>, kLmrDepthCount> table{};
  for (int depth = 0; depth < kLmrDepthCount; ++depth) {
    for (int moveNumber = 0; moveNumber < kLmrMoveCount; ++moveNumber) {
      if (depth < 2 || moveNumber < 2) {
        table[depth][moveNumber] = 1;
        continue;
      }

      // Late quiet moves become progressively less likely to refute alpha as
      // both depth and move number grow. Unlike the old three-level table,
      // this logarithmic schedule continues scaling at deep nodes.
      const double scaled =
          0.75 + std::log(static_cast<double>(depth)) *
                     std::log(static_cast<double>(moveNumber)) / 2.25;
      table[depth][moveNumber] = static_cast<std::uint8_t>(
          std::max(1, static_cast<int>(scaled)));
    }
  }
  return table;
}
const auto kLmrTable = computeLmrTable();

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
  std::uint64_t singularSearches = 0;
  std::uint64_t singularExtensions = 0;
  std::uint64_t probCutAttempts = 0;
  std::uint64_t probCutPrunes = 0;
  std::uint64_t reverseFutilityPrunes = 0;
  std::uint64_t razorAttempts = 0;
  std::uint64_t razorPrunes = 0;
  std::uint64_t internalIterativeReductions = 0;
  Move killerMoves[kMaxPly][2] = {};
  Move counterMoves[2][64][64] = {};
  Move previousMoves[kMaxPly];
  bool hasPreviousMove[kMaxPly] = {};
  Move pvTable[kMaxPly][kMaxPly];
  int pvLength[kMaxPly] = {};
  int staticEvalAtPly[kMaxPly] = {};
  bool hasStaticEvalAtPly[kMaxPly] = {};
  int history[2][64][64] = {};
  int continuationHistory[2][64][64] = {};
  int captureHistory[2][7][7][64] = {};
  std::int16_t correctionHistory[2][kCorrectionHistorySize] = {};
  TranspositionTable* tt = nullptr;
  std::atomic_bool* stopSignal = nullptr;
  std::atomic<std::uint64_t>* sharedNodeCounter = nullptr;
  std::atomic<std::uint32_t>* sharedRootHint = nullptr;
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
  bool useSingularExtensions = true;
  bool useProbCut = true;
  bool useReverseFutilityPruning = true;
  bool useRazoring = true;
  bool useInternalIterativeReduction = true;
  bool useCorrectionHistory = false;
  bool hasDeadline = false;
  bool stopped = false;
  bool preferSharedRootMove = true;
  int aspirationWindow = 50;
  int nullMoveReduction = 2;
  int lmrMinDepth = 3;
  int lmrMinMoveNumber = 4;
  int futilityMargin = 120;
  int lateMovePruningMaxDepth = 3;
  int lateMovePruningBaseMoveCount = 8;
  int singularMinDepth = 6;
  int singularMarginPerDepth = 2;
  int probCutMinDepth = 5;
  int probCutReduction = 3;
  int probCutMargin = 100;
  int reverseFutilityMargin = 85;
  int razorMargin = 250;
  int rootMoveSeed = 0;
};

int colorIndex(Color color) { return color == Color::White ? 0 : 1; }

void prioritizeSearchThread() {
#if defined(__APPLE__)
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
}

int pieceTypeIndex(PieceType type) { return static_cast<int>(type); }

Piece capturedPiece(const Board& board, const Move& move) {
  if (move.isEnPassant()) {
    return board.sideToMove() == Color::White ? Piece::BlackPawn
                                              : Piece::WhitePawn;
  }

  return board.at(move.toSquare());
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
  if (state.sharedNodeCounter == nullptr || state.publishedNodes == state.nodes)
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
  const Piece moving = board.at(move.fromSquare());
  const Piece captured = capturedPiece(board, move);
  return state.captureHistory[colorIndex(board.sideToMove())][pieceTypeIndex(
      pieceType(moving))][pieceTypeIndex(pieceType(captured))][move.toSquare()];
}

std::size_t correctionHistoryIndex(const Board& board) {
  std::uint64_t key =
      board.pieces(Color::White, PieceType::Pawn) * 0x9e3779b97f4a7c15ULL;
  key ^= board.pieces(Color::Black, PieceType::Pawn) * 0xbf58476d1ce4e5b9ULL;
  key ^= key >> 32U;
  return static_cast<std::size_t>(key) & (kCorrectionHistorySize - 1);
}

int correctedStaticEval(const SearchState& state, const Board& board,
                        int rawEval) {
  if (!state.useCorrectionHistory) return rawEval;
  const int correction =
      state.correctionHistory[colorIndex(board.sideToMove())]
                             [correctionHistoryIndex(board)];
  return std::clamp(rawEval + correction / 16, -kMateScore + kMaxPly,
                    kMateScore - kMaxPly);
}

void updateCorrectionHistory(SearchState& state, const Board& board,
                             int rawEval, int searchScore, int depth) {
  if (!state.useCorrectionHistory || depth <= 0 ||
      searchScore <= -kMateScore + kMaxPly ||
      searchScore >= kMateScore - kMaxPly) {
    return;
  }

  std::int16_t& entry =
      state.correctionHistory[colorIndex(board.sideToMove())]
                             [correctionHistoryIndex(board)];
  const int bonus =
      std::clamp((searchScore - rawEval) * std::min(depth, 16),
                 -kCorrectionHistoryLimit, kCorrectionHistoryLimit);
  int value = entry;
  const int absoluteBonus = bonus < 0 ? -bonus : bonus;
  value += bonus - value * absoluteBonus / kCorrectionHistoryLimit;
  entry = static_cast<std::int16_t>(
      std::clamp(value, -kCorrectionHistoryLimit, kCorrectionHistoryLimit));
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

  const Piece moving = board.at(move.fromSquare());
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
    const Piece moving = board.at(move.fromSquare());
    const int materialDelta = pieceValue(pieceType(captured)) -
                              pieceValue(pieceType(moving));
    const bool seeNonLosing =
        useSee ? staticExchangeNonLosing(board, move) : materialDelta >= 0;
    const int seeScore = seeNonLosing ? 0 : -1;
    if (seeValue != nullptr) *seeValue = seeScore;
    score += seeNonLosing ? kCaptureScore : kBadCaptureScore;
    score += materialDelta * 16;
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

void selectBestMove(MoveList& moves, int* scores, std::size_t first,
                    std::size_t end, int* auxiliary = nullptr) {
  std::size_t best = first;
  int bestScore = scores[first];

  for (std::size_t index = first + 1; index < end; ++index) {
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

enum class MovePickerStage : std::uint8_t {
  TTMove,
  GenerateCaptures,
  GoodCaptures,
  GenerateQuiets,
  Quiets,
  BadCaptures,
  Done,
};

class MovePicker {
 public:
  MovePicker(Board& board, SearchState& state, int ply, Move ttMove,
             bool hasTTMove, bool allowQuiets = true, Move excludedMove = {},
             bool noisyOnly = false)
      : board_(board),
        state_(state),
        ply_(ply),
        ttMove_(ttMove),
        excludedMove_(excludedMove),
        hasTTMove_(hasTTMove),
        allowQuiets_(allowQuiets),
        noisyOnly_(noisyOnly) {}

  bool nextMove(Move& move) {
    while (true) {
      switch (stage_) {
        case MovePickerStage::TTMove:
          stage_ = MovePickerStage::GenerateCaptures;
          if (hasTTMove_ && !isExcluded(ttMove_) && validateTTMove()) {
            ttMoveReturned_ = true;
            lastStage_ = MovePickerStage::TTMove;
            lastSee_ = ttMove_.isCapture()
                           ? (staticExchangeNonLosing(board_, ttMove_) ? 0
                                                                      : -1)
                           : 0;
            move = ttMove_;
            return true;
          }
          hasTTMove_ = false;
          break;

        case MovePickerStage::GenerateCaptures:
          generateCaptures();
          stage_ = MovePickerStage::GoodCaptures;
          break;

        case MovePickerStage::GoodCaptures:
          if (nextFromRange(goodCaptureCursor_, goodCaptureEnd_, move,
                            MovePickerStage::GoodCaptures, 0)) {
            return true;
          }
          stage_ = allowQuiets_ ? MovePickerStage::GenerateQuiets
                                : MovePickerStage::BadCaptures;
          break;

        case MovePickerStage::GenerateQuiets:
          quietBegin_ = moves_.size();
          appendLegalQuiets(board_, moves_);
          for (std::size_t index = quietBegin_; index < moves_.size();
               ++index) {
            scores_[index] = moveOrderScore(
                board_, moves_[index], &state_, ply_, nullptr,
                state_.useQuietOrdering, false, nullptr);
          }
          quietCursor_ = quietBegin_;
          stage_ = MovePickerStage::Quiets;
          break;

        case MovePickerStage::Quiets:
          if (nextFromRange(quietCursor_, moves_.size(),
                            move, MovePickerStage::Quiets, 0)) {
            return true;
          }
          stage_ = MovePickerStage::BadCaptures;
          break;

        case MovePickerStage::BadCaptures:
          if (nextFromRange(badCaptureCursor_, captureEnd_, move,
                            MovePickerStage::BadCaptures, -1)) {
            return true;
          }
          stage_ = MovePickerStage::Done;
          break;

        case MovePickerStage::Done:
          return false;
      }
    }
  }

  bool overflowed() const { return moves_.overflowed(); }

  MovePickerStage lastStage() const { return lastStage_; }
  int lastSee() const { return lastSee_; }

 private:
  bool isExcluded(Move move) const {
    return excludedMove_.raw() != 0 && sameMove(move, excludedMove_);
  }

  bool validateTTMove() {
    if (ttMove_.raw() == 0 || ttMove_.fromSquare() == ttMove_.toSquare()) {
      return false;
    }
    const Piece moving = board_.at(ttMove_.fromSquare());
    const Piece target = board_.at(ttMove_.toSquare());
    return moving != Piece::None && matchesColor(moving, board_.sideToMove()) &&
           pieceType(target) != PieceType::King &&
           (target == Piece::None || !isSameColor(moving, target));
  }

  void generateCaptures() {
    if (noisyOnly_) {
      genLegalNoisyMoves(board_, moves_);
    } else {
      genLegalCaptures(board_, moves_);
    }
    captureEnd_ = moves_.size();
    goodCaptureEnd_ = 0;
    for (std::size_t index = 0; index < captureEnd_; ++index) {
      int see = 0;
      scores_[index] =
          moveOrderScore(board_, moves_[index], &state_, ply_, nullptr,
                         false, state_.useStaticExchangeEvaluation, &see);
      if (see < 0) continue;
      if (index != goodCaptureEnd_) {
        std::swap(moves_[index], moves_[goodCaptureEnd_]);
        std::swap(scores_[index], scores_[goodCaptureEnd_]);
      }
      ++goodCaptureEnd_;
    }
    goodCaptureCursor_ = 0;
    badCaptureCursor_ = goodCaptureEnd_;
  }

  bool nextFromRange(std::size_t& cursor, std::size_t end, Move& move,
                     MovePickerStage stage, int see) {
    while (cursor < end) {
      selectBestMove(moves_, scores_, cursor, end);
      const std::size_t selected = cursor++;
      const Move candidate = moves_[selected];
      if (isExcluded(candidate) ||
          (ttMoveReturned_ && sameMove(candidate, ttMove_))) {
        continue;
      }

      lastStage_ = stage;
      lastSee_ = see;
      move = candidate;
      return true;
    }
    return false;
  }

  Board& board_;
  SearchState& state_;
  int ply_;
  Move ttMove_;
  Move excludedMove_;
  bool hasTTMove_;
  bool allowQuiets_;
  bool noisyOnly_;
  bool ttMoveReturned_ = false;
  MovePickerStage stage_ = MovePickerStage::TTMove;
  MovePickerStage lastStage_ = MovePickerStage::Done;
  MoveList moves_;
  int scores_[MoveList::kCapacity];
  std::size_t goodCaptureCursor_ = 0;
  std::size_t goodCaptureEnd_ = 0;
  std::size_t badCaptureCursor_ = 0;
  std::size_t captureEnd_ = 0;
  std::size_t quietBegin_ = 0;
  std::size_t quietCursor_ = 0;
  int lastSee_ = 0;
};

bool hasNonPawnMaterial(const Board& board, Color side) {
  return (board.pieces(side, PieceType::Knight) |
          board.pieces(side, PieceType::Bishop) |
          board.pieces(side, PieceType::Rook) |
          board.pieces(side, PieceType::Queen)) != 0;
}

int lmrReduction(int depth, std::size_t moveIndex, int quietHistory,
                 bool improving) {
  const int d = std::clamp(depth, 0, kLmrDepthCount - 1);
  const int m =
      static_cast<int>(std::clamp<std::size_t>(moveIndex, 0, kLmrMoveCount - 1));
  int reduction = kLmrTable[d][m];
  if (depth >= 5 && quietHistory < -kHistoryLimit / 4) ++reduction;
  if (!improving && depth >= 5) ++reduction;
  if (quietHistory > kHistoryLimit / 4) --reduction;
  return std::max(1, reduction);
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
             int ply, TranspositionBound bound, Move bestMove,
             int staticEval = TranspositionTable::kNoStaticEval) {
  if (state.tt == nullptr || state.stopped) return;
  state.tt->store(board.key(), depth, scoreToTT(score, ply), bound, bestMove,
                  staticEval);
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

bool isRuleDraw(const Board& board) {
  if (board.isInsufficientMaterial()) return true;
  if (!board.isFiftyMoveDraw()) return false;
  if (!board.isKingInCheck()) return true;

  // Checkmate takes precedence over a 50-move claim. At the boundary this
  // small legal-move probe distinguishes mate from an ordinary checked draw.
  MoveList evasions;
  genLegalMoves(board, evasions);
  return !evasions.empty();
}

int negamax(Board& board, int depth, int alpha, int beta, int ply,
            SearchState& state, bool allowNullMove, Move excludedMove = {},
            bool allowProbCut = true) {
  if (visitNode(state)) return evaluate(board);
  if (ply >= 0 && ply < kMaxPly) state.pvLength[ply] = 0;
  if (ply >= 0 && ply < kMaxPly) state.hasStaticEvalAtPly[ply] = false;

  const bool repeated = board.hasRepeatedPosition();
  const bool hasExcludedMove = excludedMove.raw() != 0;
  if (ply > 0 && repeated) return 0;
  if (ply > 0 && isRuleDraw(board)) return 0;
  if (ply >= kMaxPly) return evaluate(board);
  if (depth <= 0) return quiescence(board, alpha, beta, ply, state);

  const int originalAlpha = alpha;
  Move ttMove{};
  bool hasTTMove = false;
  bool hasTTEntry = false;
  int ttScore = 0;
  TranspositionProbe ttProbe;
  if (state.tt != nullptr && !repeated && !hasExcludedMove) {
    if (state.tt->probe(board.key(), ttProbe)) {
      hasTTEntry = true;
      ++state.ttHits;
      hasTTMove = ttProbe.hasBestMove;
      ttMove = ttProbe.bestMove;
      ttScore = scoreFromTT(ttProbe.score, ply);

      if (ttProbe.depth >= depth) {
        if (ttProbe.bound == TranspositionBound::Exact) {
          ++state.ttCutoffs;
          return ttScore;
        }
        if (ttProbe.bound == TranspositionBound::Lower && ttScore >= beta) {
          ++state.ttCutoffs;
          return ttScore;
        }
        if (ttProbe.bound == TranspositionBound::Upper && ttScore <= alpha) {
          ++state.ttCutoffs;
          return ttScore;
        }
      }
    }
  }

  const bool inCheck = board.isKingInCheck();
  const bool pvNode = alpha + 1 < beta;
  const bool cutNode = !pvNode;
  int rawStaticEval = 0;
  int staticEval = 0;
  bool hasStaticEval = false;
  auto ensureStaticEval = [&]() {
    if (hasStaticEval) return;
    if (hasTTEntry && ttProbe.hasStaticEval) {
      rawStaticEval = ttProbe.staticEval;
    } else {
      rawStaticEval = evaluate(board);
    }
    staticEval = correctedStaticEval(state, board, rawStaticEval);
    hasStaticEval = true;
    if (ply >= 0 && ply < kMaxPly) {
      state.staticEvalAtPly[ply] = staticEval;
      state.hasStaticEvalAtPly[ply] = true;
    }
  };

  const bool canReverseFutilityPrune =
      !hasExcludedMove && state.useReverseFutilityPruning && cutNode &&
      depth <= 6 && ply > 0 && !repeated && !inCheck &&
      beta > -kMateScore + kMaxPly && beta < kMateScore - kMaxPly;
  if (canReverseFutilityPrune) {
    ensureStaticEval();
    if (staticEval - state.reverseFutilityMargin * depth >= beta) {
      ++state.reverseFutilityPrunes;
      return staticEval;
    }
  }

  const bool canRazor =
      !hasExcludedMove && state.useRazoring && cutNode && depth <= 3 &&
      ply > 0 && !repeated && !inCheck && !hasTTMove &&
      alpha > -kMateScore + kMaxPly && alpha < kMateScore - kMaxPly;
  if (canRazor) {
    ensureStaticEval();
    if (staticEval + state.razorMargin * depth <= alpha) {
      ++state.razorAttempts;
      const int razorScore = quiescence(board, alpha - 1, alpha, ply, state);
      if (state.stopped) return alpha;
      if (razorScore <= alpha) {
        ++state.razorPrunes;
        return razorScore;
      }
    }
  }

  if (!hasExcludedMove && state.useInternalIterativeReduction && !hasTTMove &&
      depth >= 5 && !inCheck) {
    --depth;
    ++state.internalIterativeReductions;
  }

  const bool canTryNullMove =
      !hasExcludedMove && state.useNullMovePruning && allowNullMove &&
      cutNode && depth >= kNullMoveMinDepth && ply > 0 && !repeated &&
      beta > -kMateScore + kMaxPly && beta < kMateScore - kMaxPly && !inCheck &&
      hasNonPawnMaterial(board, board.sideToMove());
  if (canTryNullMove) {
    ensureStaticEval();
  }
  if (canTryNullMove && staticEval >= beta) {
    ++state.nullMoveAttempts;
    if (board.makeNullMove()) {
      const int reduction = std::max(1, state.nullMoveReduction + depth / 6);
      const int nullDepth = std::max(0, depth - 1 - reduction);
      clearPreviousMove(state, ply);
      const int score =
          -negamax(board, nullDepth, -beta, -beta + 1, ply + 1, state, false);
      nnue::rewindAccumulator(board);
      board.undoNullMove();

      if (state.stopped) return alpha;
      if (score >= beta) {
        ++state.nullMovePrunes;
        storeTT(state, board, depth, score, ply, TranspositionBound::Lower, {},
                rawStaticEval);
        return score;
      }
    }
  }

  const bool canTryProbCut =
      !hasExcludedMove && allowProbCut && state.useProbCut && cutNode &&
      depth >= state.probCutMinDepth && ply > 0 && !repeated && !inCheck &&
      beta > -kMateScore + kMaxPly && beta < kMateScore - kMaxPly;
  if (canTryProbCut) {
    ensureStaticEval();
    if (staticEval >= beta - state.probCutMargin) {
      ++state.probCutAttempts;
      const int probCutBeta = std::min(kInfinity, beta + state.probCutMargin);
      const int probCutDepth = std::max(0, depth - 1 - state.probCutReduction);
      MovePicker probCutPicker(board, state, ply, {}, false, false);
      Move probCutMove;
      while (probCutPicker.nextMove(probCutMove)) {
        if (probCutPicker.lastSee() < 0) continue;
        if (!board.makeGeneratedMove(probCutMove)) continue;

        setPreviousMove(state, ply, probCutMove);
        const int score =
            -negamax(board, probCutDepth, -probCutBeta, -probCutBeta + 1,
                     ply + 1, state, false, {}, false);
        nnue::rewindAccumulator(board);
        board.undoGeneratedMove();
        if (state.stopped) return alpha;
        if (score >= probCutBeta) {
          ++state.probCutPrunes;
          storeTT(state, board, probCutDepth, score, ply,
                  TranspositionBound::Lower, probCutMove, rawStaticEval);
          return score;
        }
      }
    }
  }

  int bestScore = -kInfinity;
  Move bestMove;
  bool foundMove = false;
  bool searchedMove = false;
  Move failedQuietMoves[kMaxHistoryMalusMoves];
  std::size_t failedQuietCount = 0;
  Move failedCaptures[kMaxHistoryMalusMoves];
  std::size_t failedCaptureCount = 0;
  MovePicker movePicker(board, state, ply, ttMove, hasTTMove, true,
                        excludedMove);
  Move move;
  std::size_t index = 0;
  while (movePicker.nextMove(move)) {
    const std::size_t moveIndex = index++;
    foundMove = true;
    const Color movingSide = board.sideToMove();
    const bool quiet = isQuietMove(move);
    const bool isTTMove = movePicker.lastStage() == MovePickerStage::TTMove;
    if (isTTMove) {
      ++state.ttMoveUses;
    } else if (state.useQuietOrdering && quiet &&
               isKillerMove(state, ply, move)) {
      ++state.killerMoveUses;
    } else if (state.useQuietOrdering && quiet &&
               isCounterMove(state, movingSide, ply, move)) {
      ++state.counterMoveUses;
    } else if (move.isCapture() &&
               captureHistoryScore(state, board, move) > 0) {
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
      ensureStaticEval();
    }
    if (canFutilityPrune(state, depth, alpha, staticEval, inCheck, cutNode,
                         searchedMove, quietCandidate)) {
      ++state.futilityPrunes;
      continue;
    }
    if (canLateMovePrune(state, depth, moveIndex, inCheck, cutNode,
                         searchedMove, quietCandidate)) {
      ++state.lateMovePrunes;
      continue;
    }

    int extension = 0;
    const bool singularCandidate =
        isTTMove && state.useSingularExtensions && hasTTEntry &&
        depth >= state.singularMinDepth &&
        ttProbe.depth >= depth - kSingularTTDepthMargin &&
        (ttProbe.bound == TranspositionBound::Lower ||
         ttProbe.bound == TranspositionBound::Exact) &&
        ttScore > -kMateScore + kMaxPly && ttScore < kMateScore - kMaxPly;
    if (singularCandidate) {
      ++state.singularSearches;
      const int singularBeta = ttScore - state.singularMarginPerDepth * depth;
      const int singularDepth = std::max(1, (depth - 1) / 2);
      const int singularScore =
          negamax(board, singularDepth, singularBeta - 1, singularBeta, ply,
                  state, false, ttMove, false);
      if (state.stopped) return alpha;
      if (singularScore < singularBeta) {
        extension = 1;
        ++state.singularExtensions;
      }
    }

    const bool canReduce =
        state.useLateMoveReductions && searchedMove && quiet && cutNode &&
        depth >= state.lmrMinDepth &&
        moveIndex >= static_cast<std::size_t>(state.lmrMinMoveNumber - 1) &&
        !inCheck && !isKillerMove(state, ply, move) &&
        !isCounterMove(state, movingSide, ply, move) && quietScore <= 0;
    if (!board.makeGeneratedMove(move)) continue;

    int score = 0;
    const int childDepth = depth - 1 + extension;
    setPreviousMove(state, ply, move);
    if (canReduce) {
      ++state.lmrAttempts;
      const bool improving =
          !hasStaticEval || ply < 2 || !state.hasStaticEvalAtPly[ply - 2] ||
          staticEval > state.staticEvalAtPly[ply - 2];
      const int reduction =
          lmrReduction(depth, moveIndex, quietScore, improving);
      const int reducedDepth = std::max(0, childDepth - reduction);
      score = -negamax(board, reducedDepth, -alpha - 1, -alpha, ply + 1, state,
                       true);
      if (!state.stopped && score > alpha) {
        ++state.lmrResearches;
        if (state.usePVS && pvNode) {
          score = -negamax(board, childDepth, -alpha - 1, -alpha, ply + 1,
                           state, true);
          if (!state.stopped && score > alpha && score < beta) {
            ++state.pvsResearches;
            score = -negamax(board, childDepth, -beta, -alpha, ply + 1, state,
                             true);
          }
        } else {
          score =
              -negamax(board, childDepth, -beta, -alpha, ply + 1, state, true);
        }
      }
    } else if (state.usePVS && searchedMove && depth > 1 && alpha + 1 < beta) {
      score =
          -negamax(board, childDepth, -alpha - 1, -alpha, ply + 1, state, true);
      if (!state.stopped && score > alpha && score < beta) {
        ++state.pvsResearches;
        score =
            -negamax(board, childDepth, -beta, -alpha, ply + 1, state, true);
      }
    } else {
      score = -negamax(board, childDepth, -beta, -alpha, ply + 1, state, true);
    }
    nnue::rewindAccumulator(board);
    board.undoGeneratedMove();
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
      if (!hasExcludedMove) {
        if (hasStaticEval && !inCheck && quiet) {
          updateCorrectionHistory(state, board, rawStaticEval, score, depth);
        }
        storeTT(state, board, depth, score, ply, TranspositionBound::Lower,
                move, hasStaticEval ? rawStaticEval
                                    : TranspositionTable::kNoStaticEval);
      }
      return score;
    }
    if (state.useQuietOrdering && quiet) {
      if (failedQuietCount < kMaxHistoryMalusMoves) {
        failedQuietMoves[failedQuietCount++] = move;
      }
    } else if (move.isCapture()) {
      if (failedCaptureCount < kMaxHistoryMalusMoves) {
        failedCaptures[failedCaptureCount++] = move;
      }
    }
    if (score > alpha) {
      alpha = score;
      updatePv(state, ply, move);
    }
  }

  if (movePicker.overflowed()) return evaluate(board);
  if (!foundMove) {
    if (hasExcludedMove) return alpha;
    const int score = inCheck ? -kMateScore + ply : 0;
    storeTT(state, board, depth, score, ply, TranspositionBound::Exact, {});
    return score;
  }
  if (!searchedMove) return hasStaticEval ? staticEval : evaluate(board);

  const TranspositionBound bound = bestScore <= originalAlpha
                                       ? TranspositionBound::Upper
                                       : TranspositionBound::Exact;
  if (!hasExcludedMove) {
    if (hasStaticEval && !inCheck && isQuietMove(bestMove)) {
      updateCorrectionHistory(state, board, rawStaticEval, bestScore, depth);
    }
    storeTT(state, board, depth, bestScore, ply, bound, bestMove,
            hasStaticEval ? rawStaticEval : TranspositionTable::kNoStaticEval);
  }
  return bestScore;
}

int quiescence(Board& board, int alpha, int beta, int ply, SearchState& state) {
  if (visitNode(state)) return evaluate(board);

  if (ply > 0 && board.hasRepeatedPosition()) return 0;
  if (ply > 0 && isRuleDraw(board)) return 0;
  if (ply >= kMaxPly) return evaluate(board);

  const bool inCheck = board.isKingInCheck();
  if (!inCheck) {
    const int standPat = evaluate(board);
    if (standPat >= beta) return standPat;
    if (standPat > alpha) alpha = standPat;
  }

  MovePicker movePicker(board, state, ply, {}, false, false, {}, true);
  Move move;
  bool foundMove = false;
  while (movePicker.nextMove(move)) {
    foundMove = true;
    if (!inCheck && state.useStaticExchangeEvaluation && move.isCapture() &&
        !move.isPromotion() && movePicker.lastSee() < 0) {
      ++state.seePrunes;
      continue;
    }
    if (!board.makeGeneratedMove(move)) continue;

    const int score = -quiescence(board, -beta, -alpha, ply + 1, state);
    nnue::rewindAccumulator(board);
    board.undoGeneratedMove();
    if (state.stopped) return alpha;

    if (score >= beta) return score;
    if (score > alpha) alpha = score;
  }

  if (movePicker.overflowed()) return evaluate(board);
  if (!foundMove && inCheck) return -kMateScore + ply;

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
  result.singularSearches = state.singularSearches;
  result.singularExtensions = state.singularExtensions;
  result.probCutAttempts = state.probCutAttempts;
  result.probCutPrunes = state.probCutPrunes;
  result.reverseFutilityPrunes = state.reverseFutilityPrunes;
  result.razorAttempts = state.razorAttempts;
  result.razorPrunes = state.razorPrunes;
  result.internalIterativeReductions = state.internalIterativeReductions;
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

  const int originalAlpha = alpha;

  Move ttMove{};
  bool hasTTMove = false;
  TranspositionProbe probe;
  if (state.tt != nullptr && state.tt->probe(board.key(), probe)) {
    ++state.ttHits;
    hasTTMove = probe.hasBestMove;
    ttMove = probe.bestMove;
  }

  if (!hasTTMove && state.sharedRootHint != nullptr &&
      state.preferSharedRootMove) {
    const std::uint32_t hint =
        state.sharedRootHint->load(std::memory_order_relaxed);
    const Move hintedMove = Move::fromRaw(static_cast<std::uint16_t>(hint));
    if (hintedMove.raw() != 0) {
      ttMove = hintedMove;
      hasTTMove = true;
    }
  }

  const bool useRootTTMove = hasTTMove && state.preferSharedRootMove;
  Move failedQuietMoves[kMaxHistoryMalusMoves];
  std::size_t failedQuietCount = 0;
  Move failedCaptures[kMaxHistoryMalusMoves];
  std::size_t failedCaptureCount = 0;
  MovePicker movePicker(board, state, 0, ttMove, useRootTTMove);
  Move move;
  std::size_t index = 0;
  while (movePicker.nextMove(move)) {
    ++index;
    const Color movingSide = board.sideToMove();
    const bool quiet = isQuietMove(move);
    if (movePicker.lastStage() == MovePickerStage::TTMove) {
      ++state.ttMoveUses;
    } else if (state.useQuietOrdering && quiet &&
               isKillerMove(state, 0, move)) {
      ++state.killerMoveUses;
    } else if (state.useQuietOrdering && quiet &&
               isCounterMove(state, movingSide, 0, move)) {
      ++state.counterMoveUses;
    } else if (move.isCapture() &&
               captureHistoryScore(state, board, move) > 0) {
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
    nnue::rewindAccumulator(board);
    board.undoGeneratedMove();
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
      if (failedQuietCount < kMaxHistoryMalusMoves) {
        failedQuietMoves[failedQuietCount++] = move;
      }
    } else if (move.isCapture()) {
      if (failedCaptureCount < kMaxHistoryMalusMoves) {
        failedCaptures[failedCaptureCount++] = move;
      }
    }
    if (score > alpha) alpha = score;
  }

  if (movePicker.overflowed()) {
    result.score = evaluate(board);
    result.hasBestMove = false;
    return copyStats(state, result);
  }
  if (index == 0) {
    result.score = board.isKingInCheck() ? -kMateScore : 0;
    return copyStats(state, result);
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
  prioritizeSearchThread();

  const bool previousHalfmoveTracking = board.tracksGeneratedHalfmoves();
  board.setTrackGeneratedHalfmoves(
      board.halfmoveClock() + kMaxPly >= 100);

  SearchState state;
  state.startTime = Clock::now();
  state.tt =
      limits.useTranspositionTable ? &globalTranspositionTable() : nullptr;
  state.stopSignal = limits.stopSignal;
  state.sharedNodeCounter = limits.sharedNodeCounter;
  state.sharedRootHint = limits.sharedRootHint;
  state.preferSharedRootMove = limits.preferSharedRootMove;
  state.useQuietOrdering = limits.useQuietOrdering;
  state.usePVS = limits.usePVS;
  state.useAspirationWindows = limits.useAspirationWindows;
  state.useNullMovePruning = limits.useNullMovePruning;
  state.useLateMoveReductions = limits.useLateMoveReductions;
  state.useStaticExchangeEvaluation = limits.useStaticExchangeEvaluation;
  state.useFutilityPruning = limits.useFutilityPruning;
  state.useLateMovePruning = limits.useLateMovePruning;
  state.useSingularExtensions = limits.useSingularExtensions;
  state.useProbCut = limits.useProbCut;
  state.useReverseFutilityPruning = limits.useReverseFutilityPruning;
  state.useRazoring = limits.useRazoring;
  state.useInternalIterativeReduction = limits.useInternalIterativeReduction;
  state.useCorrectionHistory = limits.useCorrectionHistory;
  state.aspirationWindow = std::max(1, limits.aspirationWindow);
  state.nullMoveReduction = std::max(1, limits.nullMoveReduction);
  state.lmrMinDepth = std::max(1, limits.lmrMinDepth);
  state.lmrMinMoveNumber = std::max(1, limits.lmrMinMoveNumber);
  state.futilityMargin = std::max(1, limits.futilityMargin);
  state.lateMovePruningMaxDepth = std::max(1, limits.lateMovePruningMaxDepth);
  state.lateMovePruningBaseMoveCount =
      std::max(1, limits.lateMovePruningBaseMoveCount);
  state.singularMinDepth = std::max(2, limits.singularMinDepth);
  state.singularMarginPerDepth = std::max(1, limits.singularMarginPerDepth);
  state.probCutMinDepth = std::max(3, limits.probCutMinDepth);
  state.probCutReduction = std::max(2, limits.probCutReduction);
  state.probCutMargin = std::max(1, limits.probCutMargin);
  state.reverseFutilityMargin = std::max(1, limits.reverseFutilityMargin);
  state.razorMargin = std::max(1, limits.razorMargin);
  state.rootMoveSeed = limits.rootMoveSeed;
  if (limits.timeLimitMs > 0) {
    state.hasDeadline = true;
    state.deadline = Clock::now() + std::chrono::milliseconds(
                                        static_cast<int>(limits.timeLimitMs));
  }

  SearchResult result;
  const int firstDepth = limits.iterativeDeepening
                             ? std::clamp(limits.iterativeStartDepth, 1,
                                          limits.depth)
                             : limits.depth;
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
    if (result.hasBestMove && state.sharedRootHint != nullptr) {
      const std::uint32_t packed =
          (static_cast<std::uint32_t>(depth) << 16U) | result.bestMove.raw();
      std::uint32_t observed =
          state.sharedRootHint->load(std::memory_order_relaxed);
      while (static_cast<int>(observed >> 16U) < depth &&
             !state.sharedRootHint->compare_exchange_weak(
                 observed, packed, std::memory_order_relaxed,
                 std::memory_order_relaxed)) {
      }
    }
    if (limits.onDepthComplete != nullptr) {
      SearchResult info = result;
      info.nodes = aggregateNodeCount(state);
      limits.onDepthComplete(info, limits.infoContext);
    }
    if (!result.hasBestMove) break;
  }

  publishNodeProgress(state);
  board.setTrackGeneratedHalfmoves(previousHalfmoveTracking);
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
  target.singularSearches += source.singularSearches;
  target.singularExtensions += source.singularExtensions;
  target.probCutAttempts += source.probCutAttempts;
  target.probCutPrunes += source.probCutPrunes;
  target.reverseFutilityPrunes += source.reverseFutilityPrunes;
  target.razorAttempts += source.razorAttempts;
  target.razorPrunes += source.razorPrunes;
  target.internalIterativeReductions += source.internalIterativeReductions;
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
  // Most helpers exploit the deepest shared root result; every third helper
  // remains fully diversified to retain Lazy SMP tree coverage.
  limits.preferSharedRootMove = workerId % 3 != 0;
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
  const std::uint64_t singularSearches = stats.singularSearches;
  const std::uint64_t singularExtensions = stats.singularExtensions;
  const std::uint64_t probCutAttempts = stats.probCutAttempts;
  const std::uint64_t probCutPrunes = stats.probCutPrunes;
  const std::uint64_t reverseFutilityPrunes = stats.reverseFutilityPrunes;
  const std::uint64_t razorAttempts = stats.razorAttempts;
  const std::uint64_t razorPrunes = stats.razorPrunes;
  const std::uint64_t internalIterativeReductions =
      stats.internalIterativeReductions;
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
  best.singularSearches = singularSearches;
  best.singularExtensions = singularExtensions;
  best.probCutAttempts = probCutAttempts;
  best.probCutPrunes = probCutPrunes;
  best.reverseFutilityPrunes = reverseFutilityPrunes;
  best.razorAttempts = razorAttempts;
  best.razorPrunes = razorPrunes;
  best.internalIterativeReductions = internalIterativeReductions;
  best.timeMs = stats.timeMs;
  best.stopped = best.stopped || stats.stopped;
  return best;
}

namespace {

class SearchWorkerPool {
 public:
  SearchWorkerPool() = default;
  SearchWorkerPool(const SearchWorkerPool&) = delete;
  SearchWorkerPool& operator=(const SearchWorkerPool&) = delete;

  ~SearchWorkerPool() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      shuttingDown_ = true;
    }
    startCondition_.notify_all();
    for (int index = 0; index < startedWorkers_; ++index) {
      workers_[static_cast<std::size_t>(index)].join();
    }
  }

  void prepare(int workerCount) {
    workerCount = std::clamp(workerCount, 0, 127);
    while (startedWorkers_ < workerCount) {
      const int workerIndex = startedWorkers_++;
      workers_[static_cast<std::size_t>(workerIndex)] =
          std::thread([this, workerIndex]() { workerLoop(workerIndex); });
    }
  }

  void start(Board& board, SearchLimits limits, SearchResult* results,
             int workerCount) {
    prepare(workerCount);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      sourceBoard_ = &board;
      limits_ = limits;
      results_ = results;
      activeWorkers_ = workerCount;
      completedWorkers_ = 0;
      ++generation_;
    }
    startCondition_.notify_all();
  }

  void wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    finishCondition_.wait(
        lock, [this]() { return completedWorkers_ == activeWorkers_; });
    sourceBoard_ = nullptr;
    results_ = nullptr;
  }

 private:
  void workerLoop(int workerIndex) {
    std::uint64_t seenGeneration = 0;
    while (true) {
      std::unique_lock<std::mutex> lock(mutex_);
      startCondition_.wait(lock, [this, seenGeneration]() {
        return shuttingDown_ || generation_ != seenGeneration;
      });
      if (shuttingDown_) return;

      seenGeneration = generation_;
      if (workerIndex >= activeWorkers_) continue;

      Board workerBoard = *sourceBoard_;
      SearchLimits workerLimits = limits_;
      SearchResult* results = results_;
      const int workerId = workerIndex + 1;
      lock.unlock();

      workerLimits.threads = 1;
      workerLimits.iterativeStartDepth = 1 + workerId % 3;
      diversifyWorkerLimits(workerLimits, workerId);
      results[static_cast<std::size_t>(workerId)] =
          searchBestMoveSingle(workerBoard, workerLimits);

      lock.lock();
      ++completedWorkers_;
      if (completedWorkers_ == activeWorkers_) finishCondition_.notify_one();
    }
  }

  std::array<std::thread, 127> workers_;
  std::mutex mutex_;
  std::condition_variable startCondition_;
  std::condition_variable finishCondition_;
  Board* sourceBoard_ = nullptr;
  SearchResult* results_ = nullptr;
  SearchLimits limits_;
  std::uint64_t generation_ = 0;
  int startedWorkers_ = 0;
  int activeWorkers_ = 0;
  int completedWorkers_ = 0;
  bool shuttingDown_ = false;
};

SearchWorkerPool& searchWorkerPool() {
  static SearchWorkerPool pool;
  return pool;
}

}  // namespace

void prepareSearchThreads(int threadCount) {
  searchWorkerPool().prepare(std::clamp(threadCount, 1, 128) - 1);
}

SearchResult searchBestMove(Board& board, SearchLimits limits) {
  if (limits.useTranspositionTable) globalTranspositionTable().newSearch();
  const int threadCount = std::clamp(limits.threads, 1, 128);
  if (threadCount == 1) {
    return searchBestMoveSingle(board, limits);
  }

  std::atomic_bool localStopSignal{false};
  std::atomic<std::uint64_t> sharedNodeCounter{0};
  std::atomic<std::uint32_t> sharedRootHint{0};
  std::atomic_bool* sharedStopSignal =
      limits.stopSignal != nullptr ? limits.stopSignal : &localStopSignal;
  limits.stopSignal = sharedStopSignal;
  limits.sharedNodeCounter = &sharedNodeCounter;
  limits.sharedRootHint = &sharedRootHint;

  // Fixed-capacity result storage and persistent workers keep every repeated
  // search allocation-free after thread-pool initialization.
  std::array<SearchResult, 128> results{};
  SearchWorkerPool& workerPool = searchWorkerPool();
  workerPool.start(board, limits, results.data(), threadCount - 1);

  Board mainBoard = board;
  SearchLimits mainLimits = limits;
  mainLimits.threads = 1;
  results[0] = searchBestMoveSingle(mainBoard, mainLimits);

  workerPool.wait();

  return combineThreadResults(results.data(), threadCount);
}
