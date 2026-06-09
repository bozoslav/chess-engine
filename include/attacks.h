#pragma once

#include <cstdint>

#include "bitboard.h"
#include "types.h"

namespace attack_detail {

constexpr std::uint32_t kRookAttackTableSize = 102400;
constexpr std::uint32_t kBishopAttackTableSize = 5248;
constexpr std::uint32_t kSliderAttackTableSize =
    kRookAttackTableSize + kBishopAttackTableSize;

struct MagicEntry {
  Bitboard mask;
  Bitboard magic;
  std::uint32_t offset;
  std::uint8_t shift;
  std::uint8_t relevantBits;
};

extern Bitboard knightAttackTable[bitboard::kSquareCount];
extern Bitboard kingAttackTable[bitboard::kSquareCount];
extern Bitboard pawnAttackTable[2][bitboard::kSquareCount];
extern MagicEntry rookEntries[bitboard::kSquareCount];
extern MagicEntry bishopEntries[bitboard::kSquareCount];
extern Bitboard sliderAttackTable[kSliderAttackTableSize];

}

class AttackTables final {
 public:
  AttackTables() = delete;

  static void init() noexcept;
  static bool initialized() noexcept;

  static Bitboard knightAttacks(Square square) noexcept;
  static Bitboard kingAttacks(Square square) noexcept;
  static Bitboard pawnAttacks(Color color, Square square) noexcept;
  static Bitboard rookAttacks(Square square, Bitboard occupancy) noexcept;
  static Bitboard bishopAttacks(Square square, Bitboard occupancy) noexcept;
  static Bitboard queenAttacks(Square square, Bitboard occupancy) noexcept;
};

inline Bitboard AttackTables::knightAttacks(Square square) noexcept {
  return attack_detail::knightAttackTable[square];
}

inline Bitboard AttackTables::kingAttacks(Square square) noexcept {
  return attack_detail::kingAttackTable[square];
}

inline Bitboard AttackTables::pawnAttacks(Color color, Square square) noexcept {
  return attack_detail::pawnAttackTable[static_cast<int>(color)][square];
}

inline Bitboard AttackTables::rookAttacks(Square square,
                                          Bitboard occupancy) noexcept {
  const attack_detail::MagicEntry& entry = attack_detail::rookEntries[square];
  const auto index = static_cast<std::uint32_t>(
      ((occupancy & entry.mask) * entry.magic) >> entry.shift);
  return attack_detail::sliderAttackTable[entry.offset + index];
}

inline Bitboard AttackTables::bishopAttacks(Square square,
                                            Bitboard occupancy) noexcept {
  const attack_detail::MagicEntry& entry = attack_detail::bishopEntries[square];
  const auto index = static_cast<std::uint32_t>(
      ((occupancy & entry.mask) * entry.magic) >> entry.shift);
  return attack_detail::sliderAttackTable[entry.offset + index];
}

inline Bitboard AttackTables::queenAttacks(Square square,
                                           Bitboard occupancy) noexcept {
  return rookAttacks(square, occupancy) | bishopAttacks(square, occupancy);
}
