#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "bitboard.h"
#include "types.h"

enum class MoveFlag : std::uint8_t {
  Quiet = 0,
  DoublePawnPush = 1,
  KingCastle = 2,
  QueenCastle = 3,
  Capture = 4,
  EnPassant = 5,
  KnightPromotion = 8,
  BishopPromotion = 9,
  RookPromotion = 10,
  QueenPromotion = 11,
  KnightPromotionCapture = 12,
  BishopPromotionCapture = 13,
  RookPromotionCapture = 14,
  QueenPromotionCapture = 15,
};

class Move {
 public:
  constexpr Move() = default;

  constexpr Move(int fromX, int fromY, int toX, int toY,
                 PieceType promo = PieceType::None)
      : data_(pack(bitboard::squareFromCoords(fromX, fromY),
                   bitboard::squareFromCoords(toX, toY),
                   promotionFlag(promo, false))) {}

  constexpr Move(int fromX, int fromY, int toX, int toY, MoveFlag flag)
      : data_(pack(bitboard::squareFromCoords(fromX, fromY),
                   bitboard::squareFromCoords(toX, toY), flag)) {}

  static constexpr Move fromSquares(Square from, Square to,
                                    MoveFlag flag = MoveFlag::Quiet) {
    return Move(pack(from, to, flag));
  }

  static constexpr Move fromRaw(std::uint16_t raw) { return Move(raw); }

  constexpr Square fromSquare() const { return data_ & kSquareMask; }
  constexpr Square toSquare() const { return (data_ >> kToShift) & kSquareMask; }
  constexpr int fromX() const { return bitboard::coordX(fromSquare()); }
  constexpr int fromY() const { return bitboard::coordY(fromSquare()); }
  constexpr int toX() const { return bitboard::coordX(toSquare()); }
  constexpr int toY() const { return bitboard::coordY(toSquare()); }
  constexpr MoveFlag flag() const {
    return static_cast<MoveFlag>((data_ >> kFlagShift) & kFlagMask);
  }
  constexpr PieceType promo() const { return promotionType(flag()); }
  constexpr bool isCapture() const { return isCaptureFlag(flag()); }
  constexpr bool isPromotion() const { return isPromotionFlag(flag()); }
  constexpr bool isCastle() const {
    return flag() == MoveFlag::KingCastle || flag() == MoveFlag::QueenCastle;
  }
  constexpr bool isKingCastle() const { return flag() == MoveFlag::KingCastle; }
  constexpr bool isQueenCastle() const {
    return flag() == MoveFlag::QueenCastle;
  }
  constexpr bool isEnPassant() const { return flag() == MoveFlag::EnPassant; }
  constexpr bool isDoublePawnPush() const {
    return flag() == MoveFlag::DoublePawnPush;
  }
  constexpr std::uint16_t raw() const { return data_; }

  std::size_t writeUci(char* out) const {
    out[0] = static_cast<char>('a' + bitboard::fileOf(fromSquare()));
    out[1] = static_cast<char>('1' + bitboard::rankOf(fromSquare()));
    out[2] = static_cast<char>('a' + bitboard::fileOf(toSquare()));
    out[3] = static_cast<char>('1' + bitboard::rankOf(toSquare()));

    if (!isPromotion()) return 4;

    out[4] = promotionChar(promo());
    return 5;
  }

  std::string toUci() const {
    char text[5] = {};
    const std::size_t size = writeUci(text);
    return std::string(text, size);
  }

  static constexpr MoveFlag promotionFlag(PieceType promo,
                                          bool isCapture) {
    switch (promo) {
      case PieceType::Knight:
        return isCapture ? MoveFlag::KnightPromotionCapture
                         : MoveFlag::KnightPromotion;
      case PieceType::Bishop:
        return isCapture ? MoveFlag::BishopPromotionCapture
                         : MoveFlag::BishopPromotion;
      case PieceType::Rook:
        return isCapture ? MoveFlag::RookPromotionCapture
                         : MoveFlag::RookPromotion;
      case PieceType::Queen:
        return isCapture ? MoveFlag::QueenPromotionCapture
                         : MoveFlag::QueenPromotion;
      case PieceType::None:
      case PieceType::Pawn:
      case PieceType::King:
      default:
        return MoveFlag::Quiet;
    }
  }

 private:
  static constexpr int kToShift = 6;
  static constexpr int kFlagShift = 12;
  static constexpr std::uint16_t kSquareMask = 0x3f;
  static constexpr std::uint16_t kFlagMask = 0x0f;

  constexpr explicit Move(std::uint16_t raw) : data_(raw) {}

  static constexpr std::uint16_t pack(Square from, Square to, MoveFlag flag) {
    return static_cast<std::uint16_t>(
        static_cast<std::uint16_t>(from) |
        (static_cast<std::uint16_t>(to) << kToShift) |
        (static_cast<std::uint16_t>(flag) << kFlagShift));
  }

  static constexpr bool isPromotionFlag(MoveFlag flag) {
    return static_cast<std::uint8_t>(flag) >=
           static_cast<std::uint8_t>(MoveFlag::KnightPromotion);
  }

  static constexpr bool isCaptureFlag(MoveFlag flag) {
    return flag == MoveFlag::Capture || flag == MoveFlag::EnPassant ||
           static_cast<std::uint8_t>(flag) >=
               static_cast<std::uint8_t>(MoveFlag::KnightPromotionCapture);
  }

  static constexpr PieceType promotionType(MoveFlag flag) {
    switch (flag) {
      case MoveFlag::KnightPromotion:
      case MoveFlag::KnightPromotionCapture:
        return PieceType::Knight;
      case MoveFlag::BishopPromotion:
      case MoveFlag::BishopPromotionCapture:
        return PieceType::Bishop;
      case MoveFlag::RookPromotion:
      case MoveFlag::RookPromotionCapture:
        return PieceType::Rook;
      case MoveFlag::QueenPromotion:
      case MoveFlag::QueenPromotionCapture:
        return PieceType::Queen;
      case MoveFlag::Quiet:
      case MoveFlag::DoublePawnPush:
      case MoveFlag::KingCastle:
      case MoveFlag::QueenCastle:
      case MoveFlag::Capture:
      case MoveFlag::EnPassant:
      default:
        return PieceType::None;
    }
  }

  static constexpr char promotionChar(PieceType promo) {
    switch (promo) {
      case PieceType::Knight:
        return 'n';
      case PieceType::Bishop:
        return 'b';
      case PieceType::Rook:
        return 'r';
      case PieceType::Queen:
        return 'q';
      case PieceType::None:
      case PieceType::Pawn:
      case PieceType::King:
      default:
        return '\0';
    }
  }

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

  bool emplace(int fromX, int fromY, int toX, int toY, MoveFlag flag) {
    return push(Move(fromX, fromY, toX, toY, flag));
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
