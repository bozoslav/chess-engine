#include <cstdint>
#include <iostream>

#include "attacks.h"

namespace {

bool expectBitboard(const char* name, Bitboard actual, Bitboard expected) {
  if (actual == expected) {
    std::cout << "[PASS] " << name << '\n';
    return true;
  }

  std::cout << "[FAIL] " << name << ": expected 0x" << std::hex << expected
            << ", got 0x" << actual << std::dec << '\n';
  return false;
}

Bitboard rookMask(Square square) {
  const int rank = bitboard::rankOf(square);
  const int file = bitboard::fileOf(square);
  Bitboard mask = 0;

  for (int r = rank + 1; r <= 6; ++r) {
    mask |= bitboard::bit(r * bitboard::kFileCount + file);
  }
  for (int r = rank - 1; r >= 1; --r) {
    mask |= bitboard::bit(r * bitboard::kFileCount + file);
  }
  for (int f = file + 1; f <= 6; ++f) {
    mask |= bitboard::bit(rank * bitboard::kFileCount + f);
  }
  for (int f = file - 1; f >= 1; --f) {
    mask |= bitboard::bit(rank * bitboard::kFileCount + f);
  }

  return mask;
}

Bitboard bishopMask(Square square) {
  const int rank = bitboard::rankOf(square);
  const int file = bitboard::fileOf(square);
  Bitboard mask = 0;

  for (int r = rank + 1, f = file + 1; r <= 6 && f <= 6; ++r, ++f) {
    mask |= bitboard::bit(r * bitboard::kFileCount + f);
  }
  for (int r = rank + 1, f = file - 1; r <= 6 && f >= 1; ++r, --f) {
    mask |= bitboard::bit(r * bitboard::kFileCount + f);
  }
  for (int r = rank - 1, f = file + 1; r >= 1 && f <= 6; --r, ++f) {
    mask |= bitboard::bit(r * bitboard::kFileCount + f);
  }
  for (int r = rank - 1, f = file - 1; r >= 1 && f >= 1; --r, --f) {
    mask |= bitboard::bit(r * bitboard::kFileCount + f);
  }

  return mask;
}

Bitboard occupancyFromIndex(std::uint32_t index, Bitboard mask) {
  Bitboard occupancy = 0;
  int bitIndex = 0;

  while (mask != 0) {
    const Square square = bitboard::popLsb(mask);
    if ((index & (std::uint32_t{1} << bitIndex)) != 0) {
      occupancy |= bitboard::bit(square);
    }
    ++bitIndex;
  }

  return occupancy;
}

Bitboard rookAttacksSlow(Square square, Bitboard occupancy) {
  const int rank = bitboard::rankOf(square);
  const int file = bitboard::fileOf(square);
  Bitboard attacks = 0;

  for (int r = rank + 1; r <= 7; ++r) {
    const Bitboard target = bitboard::bit(r * bitboard::kFileCount + file);
    attacks |= target;
    if ((occupancy & target) != 0) break;
  }
  for (int r = rank - 1; r >= 0; --r) {
    const Bitboard target = bitboard::bit(r * bitboard::kFileCount + file);
    attacks |= target;
    if ((occupancy & target) != 0) break;
  }
  for (int f = file + 1; f <= 7; ++f) {
    const Bitboard target = bitboard::bit(rank * bitboard::kFileCount + f);
    attacks |= target;
    if ((occupancy & target) != 0) break;
  }
  for (int f = file - 1; f >= 0; --f) {
    const Bitboard target = bitboard::bit(rank * bitboard::kFileCount + f);
    attacks |= target;
    if ((occupancy & target) != 0) break;
  }

  return attacks;
}

Bitboard bishopAttacksSlow(Square square, Bitboard occupancy) {
  const int rank = bitboard::rankOf(square);
  const int file = bitboard::fileOf(square);
  Bitboard attacks = 0;

  for (int r = rank + 1, f = file + 1; r <= 7 && f <= 7; ++r, ++f) {
    const Bitboard target = bitboard::bit(r * bitboard::kFileCount + f);
    attacks |= target;
    if ((occupancy & target) != 0) break;
  }
  for (int r = rank + 1, f = file - 1; r <= 7 && f >= 0; ++r, --f) {
    const Bitboard target = bitboard::bit(r * bitboard::kFileCount + f);
    attacks |= target;
    if ((occupancy & target) != 0) break;
  }
  for (int r = rank - 1, f = file + 1; r >= 0 && f <= 7; --r, ++f) {
    const Bitboard target = bitboard::bit(r * bitboard::kFileCount + f);
    attacks |= target;
    if ((occupancy & target) != 0) break;
  }
  for (int r = rank - 1, f = file - 1; r >= 0 && f >= 0; --r, --f) {
    const Bitboard target = bitboard::bit(r * bitboard::kFileCount + f);
    attacks |= target;
    if ((occupancy & target) != 0) break;
  }

  return attacks;
}

bool runLeaperTests() {
  bool ok = true;

  ok &= expectBitboard("coords a1",
                       bitboard::bit(bitboard::squareFromCoords(7, 0)),
                       0x0000000000000001ULL);
  ok &= expectBitboard("coords h8",
                       bitboard::bit(bitboard::squareFromCoords(0, 7)),
                       0x8000000000000000ULL);

  ok &= expectBitboard("knight a1", AttackTables::knightAttacks(0),
                       bitboard::bit(10) | bitboard::bit(17));
  ok &= expectBitboard("king a1", AttackTables::kingAttacks(0),
                       bitboard::bit(1) | bitboard::bit(8) | bitboard::bit(9));
  ok &= expectBitboard("white pawn e2",
                       AttackTables::pawnAttacks(Color::White, 12),
                       bitboard::bit(19) | bitboard::bit(21));
  ok &= expectBitboard("black pawn e7",
                       AttackTables::pawnAttacks(Color::Black, 52),
                       bitboard::bit(43) | bitboard::bit(45));

  return ok;
}

bool runSliderTests(bool rook) {
  bool ok = true;

  for (Square square = 0; square < bitboard::kSquareCount; ++square) {
    const Bitboard mask = rook ? rookMask(square) : bishopMask(square);
    const int relevantBits = bitboard::popcount(mask);
    const std::uint32_t permutations = std::uint32_t{1} << relevantBits;

    for (std::uint32_t index = 0; index < permutations; ++index) {
      const Bitboard occupancy = occupancyFromIndex(index, mask);
      const Bitboard actual =
          rook ? AttackTables::rookAttacks(square, occupancy)
               : AttackTables::bishopAttacks(square, occupancy);
      const Bitboard expected = rook ? rookAttacksSlow(square, occupancy)
                                     : bishopAttacksSlow(square, occupancy);

      if (actual != expected) {
        std::cout << "[FAIL] " << (rook ? "rook" : "bishop") << " square "
                  << square << " index " << index << ": expected 0x" << std::hex
                  << expected << ", got 0x" << actual << std::dec << '\n';
        ok = false;
        return ok;
      }
    }
  }

  std::cout << "[PASS] exhaustive " << (rook ? "rook" : "bishop")
            << " magic attacks\n";
  return ok;
}

}  // namespace

int main() {
  AttackTables::init();

  bool ok = AttackTables::initialized();
  ok &= runLeaperTests();
  ok &= runSliderTests(true);
  ok &= runSliderTests(false);

  if (!ok) return 1;

  std::cout << "All attack table checks passed.\n";
  return 0;
}
