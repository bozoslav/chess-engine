#pragma once

#include <cstddef>
#include <cstdint>

#include "types.h"

class Move {
 public:
  constexpr Move() = default;

  constexpr Move(int fromX, int fromY, int toX, int toY,
                 PieceType promo = PieceType::None)
      : data_(pack(fromX, fromY, toX, toY, promo)) {}

  constexpr int fromX() const { return fromSquare() / 8; }
  constexpr int fromY() const { return fromSquare() % 8; }
  constexpr int toX() const { return toSquare() / 8; }
  constexpr int toY() const { return toSquare() % 8; }
  constexpr PieceType promo() const {
    return static_cast<PieceType>((data_ >> kPromoShift) & kPromoMask);
  }
  constexpr std::uint16_t raw() const { return data_; }

 private:
  static constexpr int kToShift = 6;
  static constexpr int kPromoShift = 12;
  static constexpr std::uint16_t kSquareMask = 0x3f;
  static constexpr std::uint16_t kPromoMask = 0x0f;

  static constexpr std::uint16_t square(int x, int y) {
    return static_cast<std::uint16_t>((x << 3) | y);
  }

  static constexpr std::uint16_t pack(int fromX, int fromY, int toX, int toY,
                                      PieceType promo) {
    return static_cast<std::uint16_t>(
        square(fromX, fromY) | (square(toX, toY) << kToShift) |
        (static_cast<std::uint16_t>(promo) << kPromoShift));
  }

  constexpr int fromSquare() const { return data_ & kSquareMask; }
  constexpr int toSquare() const { return (data_ >> kToShift) & kSquareMask; }

  std::uint16_t data_ = 0;
};

static_assert(sizeof(Move) == sizeof(std::uint16_t));

class MoveList {
 public:
  static constexpr std::size_t kCapacity = 256;

  void clear() {
    size_ = 0;
    overflowed_ = false;
  }

  bool push(Move move) {
    if (size_ >= kCapacity) {
      overflowed_ = true;
      return false;
    }

    moves_[size_++] = move;
    return true;
  }

  bool emplace(int fromX, int fromY, int toX, int toY,
               PieceType promo = PieceType::None) {
    return push(Move(fromX, fromY, toX, toY, promo));
  }

  std::size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }
  bool overflowed() const { return overflowed_; }

  const Move& operator[](std::size_t index) const { return moves_[index]; }
  Move& operator[](std::size_t index) { return moves_[index]; }

  const Move* begin() const { return moves_; }
  const Move* end() const { return moves_ + size_; }
  Move* begin() { return moves_; }
  Move* end() { return moves_ + size_; }

 private:
  Move moves_[kCapacity];
  std::size_t size_ = 0;
  bool overflowed_ = false;
};
