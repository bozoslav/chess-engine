#pragma once

#include <cstdint>

#include "board.h"

struct SearchLimits {
  int depth = 1;
};

struct SearchResult {
  Move bestMove;
  int score = 0;
  int depth = 0;
  std::uint64_t nodes = 0;
  bool hasBestMove = false;
};

SearchResult searchBestMove(Board& board, SearchLimits limits);
