#include <array>
#include <cstdint>
#include <iomanip>
#include <iostream>

namespace {

using Bitboard = std::uint64_t;

constexpr int kSquares = 64;
constexpr int kMaxBlockerPermutations = 4096;

constexpr Bitboard bit(int square) { return Bitboard{1} << square; }

int fileOf(int square) { return square & 7; }

int rankOf(int square) { return square >> 3; }

int popcount(Bitboard value) { return __builtin_popcountll(value); }

int popLsb(Bitboard& value) {
  const int square = __builtin_ctzll(value);
  value &= value - 1;
  return square;
}

Bitboard rookMask(int square) {
  const int rank = rankOf(square);
  const int file = fileOf(square);
  Bitboard mask = 0;

  for (int r = rank + 1; r <= 6; ++r) mask |= bit(r * 8 + file);
  for (int r = rank - 1; r >= 1; --r) mask |= bit(r * 8 + file);
  for (int f = file + 1; f <= 6; ++f) mask |= bit(rank * 8 + f);
  for (int f = file - 1; f >= 1; --f) mask |= bit(rank * 8 + f);

  return mask;
}

Bitboard bishopMask(int square) {
  const int rank = rankOf(square);
  const int file = fileOf(square);
  Bitboard mask = 0;

  for (int r = rank + 1, f = file + 1; r <= 6 && f <= 6; ++r, ++f) {
    mask |= bit(r * 8 + f);
  }
  for (int r = rank + 1, f = file - 1; r <= 6 && f >= 1; ++r, --f) {
    mask |= bit(r * 8 + f);
  }
  for (int r = rank - 1, f = file + 1; r >= 1 && f <= 6; --r, ++f) {
    mask |= bit(r * 8 + f);
  }
  for (int r = rank - 1, f = file - 1; r >= 1 && f >= 1; --r, --f) {
    mask |= bit(r * 8 + f);
  }

  return mask;
}

Bitboard rookAttacksSlow(int square, Bitboard occupancy) {
  const int rank = rankOf(square);
  const int file = fileOf(square);
  Bitboard attacks = 0;

  for (int r = rank + 1; r <= 7; ++r) {
    const Bitboard target = bit(r * 8 + file);
    attacks |= target;
    if ((occupancy & target) != 0) break;
  }
  for (int r = rank - 1; r >= 0; --r) {
    const Bitboard target = bit(r * 8 + file);
    attacks |= target;
    if ((occupancy & target) != 0) break;
  }
  for (int f = file + 1; f <= 7; ++f) {
    const Bitboard target = bit(rank * 8 + f);
    attacks |= target;
    if ((occupancy & target) != 0) break;
  }
  for (int f = file - 1; f >= 0; --f) {
    const Bitboard target = bit(rank * 8 + f);
    attacks |= target;
    if ((occupancy & target) != 0) break;
  }

  return attacks;
}

Bitboard bishopAttacksSlow(int square, Bitboard occupancy) {
  const int rank = rankOf(square);
  const int file = fileOf(square);
  Bitboard attacks = 0;

  for (int r = rank + 1, f = file + 1; r <= 7 && f <= 7; ++r, ++f) {
    const Bitboard target = bit(r * 8 + f);
    attacks |= target;
    if ((occupancy & target) != 0) break;
  }
  for (int r = rank + 1, f = file - 1; r <= 7 && f >= 0; ++r, --f) {
    const Bitboard target = bit(r * 8 + f);
    attacks |= target;
    if ((occupancy & target) != 0) break;
  }
  for (int r = rank - 1, f = file + 1; r >= 0 && f <= 7; --r, ++f) {
    const Bitboard target = bit(r * 8 + f);
    attacks |= target;
    if ((occupancy & target) != 0) break;
  }
  for (int r = rank - 1, f = file - 1; r >= 0 && f >= 0; --r, --f) {
    const Bitboard target = bit(r * 8 + f);
    attacks |= target;
    if ((occupancy & target) != 0) break;
  }

  return attacks;
}

Bitboard occupancyFromIndex(int index, Bitboard mask) {
  Bitboard occupancy = 0;
  int bitIndex = 0;

  while (mask != 0) {
    const int square = popLsb(mask);
    if ((index & (1 << bitIndex)) != 0) occupancy |= bit(square);
    ++bitIndex;
  }

  return occupancy;
}

class SplitMix64 {
 public:
  explicit SplitMix64(std::uint64_t seed) : state_(seed) {}

  std::uint64_t next() {
    std::uint64_t value = (state_ += 0x9e3779b97f4a7c15ULL);
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
  }

  std::uint64_t sparse() { return next() & next() & next(); }

 private:
  std::uint64_t state_;
};

Bitboard findMagic(int square, bool rook) {
  const Bitboard mask = rook ? rookMask(square) : bishopMask(square);
  const int relevantBits = popcount(mask);
  const int permutations = 1 << relevantBits;
  const int shift = 64 - relevantBits;

  std::array<Bitboard, kMaxBlockerPermutations> occupancies{};
  std::array<Bitboard, kMaxBlockerPermutations> attacks{};
  std::array<Bitboard, kMaxBlockerPermutations> used{};
  std::array<bool, kMaxBlockerPermutations> occupied{};

  for (int index = 0; index < permutations; ++index) {
    occupancies[index] = occupancyFromIndex(index, mask);
    attacks[index] = rook ? rookAttacksSlow(square, occupancies[index])
                          : bishopAttacksSlow(square, occupancies[index]);
  }

  SplitMix64 rng(0xd1b54a32d192ed03ULL ^
                 (static_cast<std::uint64_t>(square) * 0x9e3779b97f4a7c15ULL) ^
                 (rook ? 0x94d049bb133111ebULL : 0xbf58476d1ce4e5b9ULL));

  for (std::uint64_t attempt = 1; attempt <= 100000000ULL; ++attempt) {
    const Bitboard magic = rng.sparse();
    if (magic == 0) continue;
    if (popcount((mask * magic) & 0xff00000000000000ULL) < 6) continue;

    for (int index = 0; index < permutations; ++index) occupied[index] = false;

    bool failed = false;
    for (int index = 0; index < permutations; ++index) {
      const auto key = static_cast<int>((occupancies[index] * magic) >> shift);
      if (!occupied[key]) {
        occupied[key] = true;
        used[key] = attacks[index];
      } else if (used[key] != attacks[index]) {
        failed = true;
        break;
      }
    }

    if (!failed) return magic;
  }

  std::cerr << "failed to find " << (rook ? "rook" : "bishop")
            << " magic for square " << square << '\n';
  return 0;
}

void printArray(const char* name,
                const std::array<Bitboard, kSquares>& values) {
  std::cout << "constexpr std::array<Bitboard, 64> " << name << " = {{\n";

  for (int square = 0; square < kSquares; ++square) {
    std::cout << "    0x" << std::hex << std::setw(16) << std::setfill('0')
              << values[square] << "ULL" << std::dec;
    if (square != kSquares - 1) std::cout << ',';
    std::cout << '\n';
  }

  std::cout << "}};\n\n";
}

}  // namespace

int main() {
  std::array<Bitboard, kSquares> rookMagics{};
  std::array<Bitboard, kSquares> bishopMagics{};

  for (int square = 0; square < kSquares; ++square) {
    rookMagics[square] = findMagic(square, true);
    if (rookMagics[square] == 0) return 1;
    std::cerr << "rook " << square << '\n';
  }

  for (int square = 0; square < kSquares; ++square) {
    bishopMagics[square] = findMagic(square, false);
    if (bishopMagics[square] == 0) return 1;
    std::cerr << "bishop " << square << '\n';
  }

  std::cout << "#include <array>\n\n";
  std::cout << "using Bitboard = std::uint64_t;\n\n";
  printArray("kRookMagics", rookMagics);
  printArray("kBishopMagics", bishopMagics);

  return 0;
}
