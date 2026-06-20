#include "transposition_table.h"

#include <algorithm>
#include <atomic>
#include <limits>
#include <memory>

struct TranspositionTable::TTSlot {
  std::atomic<std::uint64_t> key = 0;
  std::atomic<std::uint64_t> data = 0;
};

struct alignas(64) TranspositionTable::TTBucket {
  TTSlot slots[TranspositionTable::kAssociativity];
};

namespace {

constexpr std::uint8_t kOccupiedFlag = 0x80;
constexpr std::uint8_t kBoundMask = 0x03;
constexpr unsigned kGenerationShift = 48U;

using TTSlot = TranspositionTable::TTSlot;
using TTBucket = TranspositionTable::TTBucket;

static_assert(sizeof(TranspositionTable::TTSlot) == 16);
static_assert(sizeof(TranspositionTable::TTBucket) == 64);

bool occupied(std::uint64_t data) {
  return ((data >> 40U) & kOccupiedFlag) != 0;
}

std::uint8_t encodeFlags(TranspositionBound bound) {
  return static_cast<std::uint8_t>(kOccupiedFlag |
                                   static_cast<std::uint8_t>(bound));
}

TranspositionBound decodeBound(std::uint64_t data) {
  return static_cast<TranspositionBound>((data >> 40U) & kBoundMask);
}

std::int16_t clampScore(int score) {
  constexpr int kMinScore = -32768;
  constexpr int kMaxScore = 32767;
  return static_cast<std::int16_t>(std::clamp(score, kMinScore, kMaxScore));
}

std::uint8_t clampDepth(int depth) {
  constexpr int kMaxDepth = 255;
  if (depth <= 0) return 0;
  return static_cast<std::uint8_t>(std::min(depth, kMaxDepth));
}

std::uint64_t packData(int depth, int score, TranspositionBound bound,
                       Move bestMove) {
  return static_cast<std::uint64_t>(bestMove.raw()) |
         (static_cast<std::uint64_t>(
              static_cast<std::uint16_t>(clampScore(score)))
          << 16U) |
         (static_cast<std::uint64_t>(clampDepth(depth)) << 32U) |
         (static_cast<std::uint64_t>(encodeFlags(bound)) << 40U);
}

std::uint16_t unpackMove(std::uint64_t data) {
  return static_cast<std::uint16_t>(data & 0xffffU);
}

std::int16_t unpackScore(std::uint64_t data) {
  return static_cast<std::int16_t>((data >> 16U) & 0xffffU);
}

std::uint8_t unpackDepth(std::uint64_t data) {
  return static_cast<std::uint8_t>((data >> 32U) & 0xffU);
}

std::uint8_t unpackGeneration(std::uint64_t data) {
  return static_cast<std::uint8_t>((data >> kGenerationShift) & 0xffU);
}

std::size_t floorPowerOfTwo(std::size_t value) {
  if (value == 0) return 0;
  std::size_t power = 1;
  while (power <= value / 2) power <<= 1U;
  return power;
}

std::size_t bucketCountForMegabytes(std::size_t megabytes) {
  constexpr std::size_t kBytesPerMegabyte = 1024U * 1024U;
  megabytes = std::clamp(megabytes, TranspositionTable::kMinHashMb,
                         TranspositionTable::kMaxHashMb);
  const std::size_t requestedBuckets =
      (megabytes * kBytesPerMegabyte) / sizeof(TranspositionTable::TTBucket);
  return std::max<std::size_t>(1, floorPowerOfTwo(requestedBuckets));
}

std::size_t replacementSlot(TranspositionTable::TTBucket& bucket,
                            std::uint64_t key, std::uint8_t currentGeneration) {
  std::size_t replacement = 0;
  int lowestQuality = std::numeric_limits<int>::max();

  for (std::size_t index = 0; index < TranspositionTable::kAssociativity;
       ++index) {
    const TTSlot& slot = bucket.slots[index];
    const std::uint64_t signature = slot.key.load(std::memory_order_relaxed);
    const std::uint64_t data = slot.data.load(std::memory_order_relaxed);
    const std::uint64_t slotKey = signature ^ data;
    if (occupied(data) && slotKey == key) return index;
    if (!occupied(data)) return index;
    const int age =
        static_cast<std::uint8_t>(currentGeneration - unpackGeneration(data));
    const int exactBonus =
        decodeBound(data) == TranspositionBound::Exact ? 4 : 0;
    const int quality = static_cast<int>(unpackDepth(data)) + exactBonus -
                        std::min(age, 31) * 8;
    if (quality < lowestQuality) {
      lowestQuality = quality;
      replacement = index;
    }
  }

  return replacement;
}

}  // namespace

TranspositionTable::TranspositionTable() { resize(kDefaultHashMb); }

TranspositionTable::~TranspositionTable() = default;

void TranspositionTable::clear() {
  if (buckets_ == nullptr) return;

  for (std::size_t bucketIndex = 0; bucketIndex < bucketCount_; ++bucketIndex) {
    TTBucket& bucket = buckets_[bucketIndex];
    for (TTSlot& slot : bucket.slots) {
      slot.data.store(0, std::memory_order_relaxed);
      slot.key.store(0, std::memory_order_relaxed);
    }
  }
}

void TranspositionTable::resize(std::size_t megabytes) {
  const std::size_t newBucketCount = bucketCountForMegabytes(megabytes);
  if (buckets_ != nullptr && newBucketCount == bucketCount_) {
    clear();
    return;
  }

  buckets_ = std::make_unique<TTBucket[]>(newBucketCount);
  bucketCount_ = newBucketCount;
  bucketMask_ = newBucketCount - 1;
  bytes_ = newBucketCount * sizeof(TTBucket);
  clear();
}

void TranspositionTable::newSearch() {
  generation_.fetch_add(1, std::memory_order_relaxed);
}

bool TranspositionTable::probe(std::uint64_t key,
                               TranspositionProbe& out) const {
  if (buckets_ == nullptr || bucketCount_ == 0) {
    out = {};
    return false;
  }

  const TTBucket& bucket = buckets_[key & bucketMask_];
  for (const TTSlot& slot : bucket.slots) {
    const std::uint64_t signature = slot.key.load(std::memory_order_relaxed);
    const std::uint64_t data = slot.data.load(std::memory_order_relaxed);
    if (!occupied(data) || (signature ^ data) != key) continue;

    const std::uint16_t move = unpackMove(data);
    out.bestMove = Move::fromRaw(move);
    out.score = unpackScore(data);
    out.depth = unpackDepth(data);
    out.bound = decodeBound(data);
    out.hit = true;
    out.hasBestMove = move != 0;
    return true;
  }

  out = {};
  return false;
}

void TranspositionTable::store(std::uint64_t key, int depth, int score,
                               TranspositionBound bound, Move bestMove) {
  if (buckets_ == nullptr || bucketCount_ == 0) return;

  const std::uint8_t generation = generation_.load(std::memory_order_relaxed);
  TTBucket& bucket = buckets_[key & bucketMask_];
  TTSlot& slot = bucket.slots[replacementSlot(bucket, key, generation)];
  const std::uint64_t previousSignature =
      slot.key.load(std::memory_order_relaxed);
  const std::uint64_t previousData = slot.data.load(std::memory_order_relaxed);
  const std::uint64_t previousKey = previousSignature ^ previousData;
  const std::uint8_t storedDepth = clampDepth(depth);

  if (occupied(previousData) && previousKey == key &&
      storedDepth + 2 < unpackDepth(previousData) &&
      bound != TranspositionBound::Exact) {
    return;
  }

  const std::uint64_t newData =
      packData(storedDepth, score, bound, bestMove) |
      (static_cast<std::uint64_t>(generation) << kGenerationShift);
  slot.data.store(newData, std::memory_order_relaxed);
  slot.key.store(key ^ newData, std::memory_order_relaxed);
}

std::size_t TranspositionTable::hashSizeMb() const {
  constexpr std::size_t kBytesPerMegabyte = 1024U * 1024U;
  return std::max<std::size_t>(1, bytes_ / kBytesPerMegabyte);
}

std::size_t TranspositionTable::bucketCount() const { return bucketCount_; }

std::size_t TranspositionTable::bytes() const { return bytes_; }

int TranspositionTable::hashfullPermill() const {
  if (buckets_ == nullptr || bucketCount_ == 0) return 0;

  constexpr std::size_t kSampleBuckets = 1000;
  const std::size_t bucketsToSample = std::min(bucketCount_, kSampleBuckets);
  std::size_t occupiedSlots = 0;
  std::size_t sampledSlots = 0;
  const std::uint8_t generation = generation_.load(std::memory_order_relaxed);

  for (std::size_t bucketIndex = 0; bucketIndex < bucketsToSample;
       ++bucketIndex) {
    const TTBucket& bucket = buckets_[bucketIndex];
    for (const TTSlot& slot : bucket.slots) {
      const std::uint64_t data = slot.data.load(std::memory_order_relaxed);
      occupiedSlots +=
          occupied(data) && unpackGeneration(data) == generation ? 1U : 0U;
      ++sampledSlots;
    }
  }

  if (sampledSlots == 0) return 0;
  return static_cast<int>((occupiedSlots * 1000U) / sampledSlots);
}

TranspositionTable& globalTranspositionTable() {
  static TranspositionTable table;
  return table;
}
