#pragma once

#include <cstdint>

#include "board.h"

constexpr int kSearchMaxPvLength = 96;

struct SearchResult {
  Move bestMove;
  Move principalVariation[kSearchMaxPvLength];
  int score = 0;
  int depth = 0;
  int pvLength = 0;
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
  bool iterativeDeepening = true;
  int aspirationWindow = 50;
  std::uint64_t timeLimitMs = 0;
  SearchInfoCallback onDepthComplete = nullptr;
  void* infoContext = nullptr;
};

void clearSearchState();
SearchResult searchBestMove(Board& board, SearchLimits limits);
