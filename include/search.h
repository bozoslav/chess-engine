#pragma once

#include <atomic>
#include <cstdint>

#include "board.h"

constexpr int kSearchMaxPvLength = 96;

struct SearchResult {
  Move bestMove{};
  Move principalVariation[kSearchMaxPvLength] = {};
  int score = 0;
  int depth = 0;
  int pvLength = 0;
  std::uint64_t nodes = 0;
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
  std::uint64_t timeMs = 0;
  bool hasBestMove = false;
  bool stopped = false;
};

using SearchInfoCallback = void (*)(const SearchResult& result, void* context);

struct SearchLimits {
  int depth = 1;
  bool useTranspositionTable = true;
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
  bool iterativeDeepening = true;
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
  int threads = 1;
  int rootMoveSeed = 0;
  std::uint64_t timeLimitMs = 0;
  std::atomic_bool* stopSignal = nullptr;
  // Internal Lazy SMP telemetry. Workers publish in coarse batches so UCI NPS
  // reflects aggregate throughput without an atomic increment per node.
  std::atomic<std::uint64_t>* sharedNodeCounter = nullptr;
  std::atomic<std::uint32_t>* sharedRootHint = nullptr;
  int iterativeStartDepth = 1;
  bool preferSharedRootMove = true;
  SearchInfoCallback onDepthComplete = nullptr;
  void* infoContext = nullptr;
};

void clearSearchState();
void prepareSearchThreads(int threadCount);
SearchResult searchBestMove(Board& board, SearchLimits limits);
