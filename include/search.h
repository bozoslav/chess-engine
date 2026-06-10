#pragma once

#include <cstdint>

#include "board.h"

struct SearchResult {
  Move bestMove;
  int score = 0;
  int depth = 0;
  std::uint64_t nodes = 0;
  std::uint64_t ttHits = 0;
  std::uint64_t ttCutoffs = 0;
  std::uint64_t ttStores = 0;
  std::uint64_t ttMoveUses = 0;
  std::uint64_t killerMoveUses = 0;
  std::uint64_t historyMoveUses = 0;
  std::uint64_t quietCutoffs = 0;
  bool hasBestMove = false;
};

using SearchInfoCallback = void (*)(const SearchResult& result,
                                    void* context);

struct SearchLimits {
  int depth = 1;
  bool useTranspositionTable = true;
  bool useQuietOrdering = true;
  bool iterativeDeepening = true;
  SearchInfoCallback onDepthComplete = nullptr;
  void* infoContext = nullptr;
};

void clearSearchState();
SearchResult searchBestMove(Board& board, SearchLimits limits);
