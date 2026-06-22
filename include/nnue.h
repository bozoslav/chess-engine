#pragma once

#include <cstdint>

#include "board.h"

namespace nnue {

// Stockfish SFNNv13 as emitted by official-stockfish/nnue-pytorch for
// Full_Threats+HalfKAv2_hm^, L1=1024, L2=32, L3=32.
constexpr std::uint32_t kFileVersion = 0x6A448AFAU;
constexpr std::uint32_t kL1Size = 1024;
constexpr std::uint32_t kL2Size = 32;
constexpr std::uint32_t kL3Size = 32;
constexpr std::uint32_t kPsqtBuckets = 8;
constexpr std::uint32_t kLayerStackBuckets = 8;
constexpr std::uint32_t kThreatFeatureCount = 60720;
constexpr std::uint32_t kHalfKaFeatureCount = 22528;
constexpr std::uint32_t kMaxThreatFeatures = 128;
constexpr std::uint32_t kMaxHalfKaFeatures = 32;

bool loadNetwork(const char* path);
void clearNetwork();
bool networkLoaded();
const char* lastError();
void resetAccumulatorCache();
// Rewinds a materialized child accumulator before the board undoes that move.
// No work is performed when the child returned without evaluating.
void rewindAccumulator(const Board& child);

// Returns false only when no compatible network is loaded.
bool evaluate(const Board& board, int& score);

}  // namespace nnue
