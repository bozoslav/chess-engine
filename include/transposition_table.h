#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "move.h"

enum class TranspositionBound : std::uint8_t {
  Exact = 0,
  Lower = 1,
  Upper = 2,
};

struct TranspositionProbe {
  Move bestMove{};
  int score = 0;
  int staticEval = 0;
  int depth = 0;
  TranspositionBound bound = TranspositionBound::Exact;
  bool hit = false;
  bool hasBestMove = false;
  bool hasStaticEval = false;
};

class TranspositionTable {
 public:
  struct TTSlot;
  struct TTBucket;

  static constexpr std::size_t kAssociativity = 4;
  static constexpr std::size_t kDefaultHashMb = 128;
  static constexpr std::size_t kMinHashMb = 1;
  static constexpr std::size_t kMaxHashMb = 4096;

  TranspositionTable();
  ~TranspositionTable();

  void clear();
  void resize(std::size_t megabytes);
  void newSearch();
  bool probe(std::uint64_t key, TranspositionProbe& out) const;
  void store(std::uint64_t key, int depth, int score, TranspositionBound bound,
             Move bestMove, int staticEval = kNoStaticEval);
  std::size_t hashSizeMb() const;
  std::size_t bucketCount() const;
  std::size_t bytes() const;
  int hashfullPermill() const;

  static constexpr int kNoStaticEval = 32767;

 private:
  std::unique_ptr<TTBucket[]> buckets_;
  std::size_t bucketCount_ = 0;
  std::size_t bucketMask_ = 0;
  std::size_t bytes_ = 0;
  std::atomic<std::uint8_t> generation_{0};
};

TranspositionTable& globalTranspositionTable();
