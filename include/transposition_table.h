#pragma once

#include <cstddef>
#include <cstdint>

#include "move.h"

enum class TranspositionBound : std::uint8_t {
  Exact = 0,
  Lower = 1,
  Upper = 2,
};

struct TranspositionProbe {
  Move bestMove;
  int score = 0;
  int depth = 0;
  TranspositionBound bound = TranspositionBound::Exact;
  bool hit = false;
  bool hasBestMove = false;
};

class TranspositionTable {
 public:
  static constexpr std::size_t kBucketCount = 1u << 18;
  static constexpr std::size_t kAssociativity = 4;
  static constexpr std::size_t kBytes = kBucketCount * 64;

  void clear();
  bool probe(std::uint64_t key, TranspositionProbe& out) const;
  void store(std::uint64_t key, int depth, int score,
             TranspositionBound bound, Move bestMove);
};

TranspositionTable& globalTranspositionTable();
