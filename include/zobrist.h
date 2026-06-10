#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "bitboard.h"
#include "types.h"

namespace zobrist {

using Key = std::uint64_t;

struct Tables {
  std::array<std::array<Key, bitboard::kSquareCount>, 17> piece;
  std::array<Key, 16> castling;
  std::array<Key, bitboard::kFileCount> enPassantFile;
  Key blackToMove;
};

constexpr Key splitmix64(Key& state) noexcept {
  state += 0x9e3779b97f4a7c15ULL;
  Key value = state;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

constexpr Tables makeTables() noexcept {
  Tables tables = {};
  Key state = 0x6c8e9cf570932bd5ULL;

  for (auto& pieceTable : tables.piece) {
    for (Key& key : pieceTable) {
      key = splitmix64(state);
    }
  }

  tables.castling[0] = 0;
  for (std::size_t i = 1; i < tables.castling.size(); ++i) {
    tables.castling[i] = splitmix64(state);
  }

  for (Key& key : tables.enPassantFile) {
    key = splitmix64(state);
  }

  tables.blackToMove = splitmix64(state);
  return tables;
}

alignas(64) inline constexpr Tables kTables = makeTables();

constexpr Key piece(Piece piece, Square square) noexcept {
  return kTables.piece[static_cast<std::uint8_t>(piece)][square];
}

constexpr Key castling(int rightsMask) noexcept {
  return kTables.castling[rightsMask & 0x0f];
}

constexpr Key enPassant(Square square) noexcept {
  return square >= 0 ? kTables.enPassantFile[bitboard::fileOf(square)] : 0;
}

constexpr Key sideToMove() noexcept { return kTables.blackToMove; }

}  // namespace zobrist
