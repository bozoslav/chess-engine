#include <array>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>

#include "attacks.h"

namespace {

constexpr int kOccupancyCount = 256;
constexpr int kIterations = 100000000;

std::array<Bitboard, kOccupancyCount> makeOccupancies() {
  std::array<Bitboard, kOccupancyCount> occupancies{};
  std::uint64_t state = 0x3c79ac492ba7b653ULL;

  for (Bitboard& occupancy : occupancies) {
    state += 0x9e3779b97f4a7c15ULL;
    std::uint64_t value = state;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    occupancy = value ^ (value >> 31);
  }

  return occupancies;
}

double elapsedSeconds(std::chrono::steady_clock::time_point start,
                      std::chrono::steady_clock::time_point finish) {
  return std::chrono::duration<double>(finish - start).count();
}

}  // namespace

int main() {
  AttackTables::init();

  const std::array<Bitboard, kOccupancyCount> occupancies = makeOccupancies();
  Bitboard checksum = 0;

  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < kIterations; ++i) {
    const Square square = (i * 37) & 63;
    const Bitboard occupancy = occupancies[i & (kOccupancyCount - 1)];
    checksum += AttackTables::rookAttacks(square, occupancy);
    checksum ^= AttackTables::bishopAttacks(square, occupancy) << 1;
    checksum += AttackTables::queenAttacks(square, occupancy) >> 1;
    checksum ^= AttackTables::knightAttacks(square) << 7;
    checksum += AttackTables::kingAttacks(square) >> 3;
  }
  const auto finish = std::chrono::steady_clock::now();

  const double seconds = elapsedSeconds(start, finish);
  const double calls = static_cast<double>(kIterations) * 5.0;

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "iterations,attack_calls,seconds,calls_per_second,checksum\n";
  std::cout << kIterations << ',' << static_cast<std::uint64_t>(calls) << ','
            << seconds << ',' << (seconds == 0.0 ? 0.0 : calls / seconds)
            << ",0x" << std::hex << checksum << std::dec << '\n';

  return 0;
}
