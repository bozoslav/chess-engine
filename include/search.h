#pragma once

#include <cstdint>

#include "board.h"

struct SearchLimits {
  int depth = 1;
  bool useTranspositionTable = true;
};

struct SearchResult {
  Move bestMove;
  int score = 0;
  int depth = 0;
  std::uint64_t nodes = 0;
  std::uint64_t ttHits = 0;
  std::uint64_t ttCutoffs = 0;
  std::uint64_t ttStores = 0;
  std::uint64_t ttMoveUses = 0;
  bool hasBestMove = false;
};

void clearSearchState();
SearchResult searchBestMove(Board& board, SearchLimits limits);
