#pragma once

#include <cstdint>

using Bitboard = std::uint64_t;
using Square = int;

namespace bitboard {

constexpr int kFileCount = 8;
constexpr int kRankCount = 8;
constexpr int kSquareCount = kFileCount * kRankCount;

constexpr Square squareFromCoords(int x, int y) noexcept {
  return (kRankCount - 1 - x) * kFileCount + y;
}

constexpr int coordX(Square square) noexcept {
  return kRankCount - 1 - square / kFileCount;
}

constexpr int coordY(Square square) noexcept { return square % kFileCount; }

constexpr int rankOf(Square square) noexcept { return square >> 3; }

constexpr int fileOf(Square square) noexcept { return square & 7; }

constexpr Bitboard bit(Square square) noexcept { return Bitboard{1} << square; }

inline int popcount(Bitboard value) noexcept {
#if defined(__clang__) || defined(__GNUC__)
  return __builtin_popcountll(value);
#else
  int count = 0;
  while (value != 0) {
    value &= value - 1;
    ++count;
  }
  return count;
#endif
}

inline Square lsb(Bitboard value) noexcept {
#if defined(__clang__) || defined(__GNUC__)
  return __builtin_ctzll(value);
#else
  Square square = 0;
  while ((value & 1) == 0) {
    value >>= 1;
    ++square;
  }
  return square;
#endif
}

inline Square popLsb(Bitboard& value) noexcept {
  const Square square = lsb(value);
  value &= value - 1;
  return square;
}

}  // namespace bitboard
