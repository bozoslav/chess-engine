#include "transposition_table.h"

#include <algorithm>

namespace {

constexpr std::uint8_t kOccupiedFlag = 0x80;
constexpr std::uint8_t kBoundMask = 0x03;

struct TTSlot {
  std::uint64_t key = 0;
  std::uint16_t move = 0;
  std::int16_t score = 0;
  std::uint8_t depth = 0;
  std::uint8_t flags = 0;
  std::uint16_t padding = 0;
};

struct alignas(64) TTBucket {
  TTSlot slots[TranspositionTable::kAssociativity];
};

static_assert(sizeof(TTSlot) == 16);
static_assert(sizeof(TTBucket) == 64);
static_assert((TranspositionTable::kBucketCount &
               (TranspositionTable::kBucketCount - 1)) == 0);

TTBucket gBuckets[TranspositionTable::kBucketCount];

bool occupied(const TTSlot& slot) { return (slot.flags & kOccupiedFlag) != 0; }

std::uint8_t encodeFlags(TranspositionBound bound) {
  return static_cast<std::uint8_t>(kOccupiedFlag |
                                   static_cast<std::uint8_t>(bound));
}

TranspositionBound decodeBound(const TTSlot& slot) {
  return static_cast<TranspositionBound>(slot.flags & kBoundMask);
}

std::size_t bucketIndex(std::uint64_t key) {
  return key & (TranspositionTable::kBucketCount - 1);
}

std::int16_t clampScore(int score) {
  constexpr int kMinScore = -32768;
  constexpr int kMaxScore = 32767;
  return static_cast<std::int16_t>(
      std::clamp(score, kMinScore, kMaxScore));
}

std::uint8_t clampDepth(int depth) {
  constexpr int kMaxDepth = 255;
  if (depth <= 0) return 0;
  return static_cast<std::uint8_t>(std::min(depth, kMaxDepth));
}

TTSlot* replacementSlot(TTBucket& bucket, std::uint64_t key) {
  TTSlot* shallowest = &bucket.slots[0];

  for (TTSlot& slot : bucket.slots) {
    if (occupied(slot) && slot.key == key) return &slot;
    if (!occupied(slot)) return &slot;
    if (slot.depth < shallowest->depth) shallowest = &slot;
  }

  return shallowest;
}

}  // namespace

void TranspositionTable::clear() {
  for (TTBucket& bucket : gBuckets) {
    for (TTSlot& slot : bucket.slots) slot = {};
  }
}

bool TranspositionTable::probe(std::uint64_t key,
                               TranspositionProbe& out) const {
  const TTBucket& bucket = gBuckets[bucketIndex(key)];
  for (const TTSlot& slot : bucket.slots) {
    if (!occupied(slot) || slot.key != key) continue;

    out.bestMove = Move::fromRaw(slot.move);
    out.score = slot.score;
    out.depth = slot.depth;
    out.bound = decodeBound(slot);
    out.hit = true;
    out.hasBestMove = slot.move != 0;
    return true;
  }

  out = {};
  return false;
}

void TranspositionTable::store(std::uint64_t key, int depth, int score,
                               TranspositionBound bound, Move bestMove) {
  TTBucket& bucket = gBuckets[bucketIndex(key)];
  TTSlot* slot = replacementSlot(bucket, key);
  const std::uint8_t storedDepth = clampDepth(depth);

  if (occupied(*slot) && slot->key == key && storedDepth + 2 < slot->depth &&
      bound != TranspositionBound::Exact) {
    return;
  }

  slot->key = key;
  slot->move = bestMove.raw();
  slot->score = clampScore(score);
  slot->depth = storedDepth;
  slot->flags = encodeFlags(bound);
  slot->padding = 0;
}

TranspositionTable& globalTranspositionTable() {
  static TranspositionTable table;
  return table;
}
