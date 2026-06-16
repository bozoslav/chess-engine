#include "attacks.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace {

constexpr std::array<Bitboard, 64> kRookMagics = {{
    0x0880008040001024ULL, 0x8440004810002000ULL, 0x8080200080100008ULL,
    0x0100040900201000ULL, 0x8200100420080201ULL, 0x8200010200100408ULL,
    0x0480008001000200ULL, 0x8a80046280004100ULL, 0x0810800040002090ULL,
    0x0040802000400088ULL, 0x6040801000200080ULL, 0x0040800800100080ULL,
    0x8449002800510044ULL, 0x9212004402004810ULL, 0x0441000401008200ULL,
    0x0012800100004480ULL, 0x020180800a24c008ULL, 0x1004820045002204ULL,
    0x4020410011082000ULL, 0x0100808010000800ULL, 0x4281010010040801ULL,
    0x0001010004000208ULL, 0x0530010100040200ULL, 0x0080020000a04104ULL,
    0x0620218180004008ULL, 0x0610208100400108ULL, 0x0010200080100089ULL,
    0x81b0002500090010ULL, 0x000c040080080081ULL, 0x0002000200040810ULL,
    0x0001040101000200ULL, 0x0585108e000c004dULL, 0x48400096a1800341ULL,
    0x0000804002802000ULL, 0x0012401103002008ULL, 0x1000100009002100ULL,
    0x000200702a000420ULL, 0x001a000280800400ULL, 0x4000510204000890ULL,
    0x0108040042002095ULL, 0x0088984000208000ULL, 0x01c0080030002000ULL,
    0x0052200500410010ULL, 0x0678001000808008ULL, 0x0c40080011010004ULL,
    0x20820008100a0064ULL, 0x4009000200010004ULL, 0x0000006297020004ULL,
    0x2440008002402280ULL, 0x000100f0400a8100ULL, 0x0025120228804200ULL,
    0x0818068008100080ULL, 0x0008000402004040ULL, 0x0142001104080200ULL,
    0x0400100862010400ULL, 0x1480008064010200ULL, 0x2201a10544108001ULL,
    0x8002854421001202ULL, 0x1220200044110009ULL, 0xc601001528a01001ULL,
    0x0802002084908802ULL, 0x10a1003400081229ULL, 0x800a0040a8040102ULL,
    0x4001069021024402ULL,
}};

constexpr std::array<Bitboard, 64> kBishopMagics = {{
    0x04b0100088044440ULL, 0x0620018200810000ULL, 0x0050044081249004ULL,
    0x0054440080442120ULL, 0x8002021000240000ULL, 0x0000882008000000ULL,
    0x8402080404040104ULL, 0x0080208228014008ULL, 0x8b00a04410008102ULL,
    0x0080084204040420ULL, 0x0008083881160601ULL, 0x0c42880841020800ULL,
    0x0018011040470000ULL, 0x0010042444402400ULL, 0x40000e4144202003ULL,
    0x0218413284042001ULL, 0x824050a004258a18ULL, 0x0005582801040400ULL,
    0x1808090400461600ULL, 0x108c000801409000ULL, 0x0044009084a00110ULL,
    0x0001001202423202ULL, 0x6420601411081800ULL, 0x0011080084008214ULL,
    0x003840100a100100ULL, 0x8008201284040080ULL, 0x2202020081180200ULL,
    0x2001040008020960ULL, 0x1225010000104000ULL, 0x8018020000410080ULL,
    0x0014808884044430ULL, 0x0804420100820123ULL, 0x1010108802143880ULL,
    0x0c0c04020020a684ULL, 0x9a00202400080810ULL, 0x0000200800050104ULL,
    0x0020440400404100ULL, 0x00042c4080441000ULL, 0x0010920202008090ULL,
    0x1808008020488a04ULL, 0x0004100808000540ULL, 0x02005c2208002008ULL,
    0x0022020201000210ULL, 0x0303804200801810ULL, 0x03004000b2023302ULL,
    0x1008100088210200ULL, 0x1c240d840108cc08ULL, 0xc0b0840440400282ULL,
    0x0002025004044040ULL, 0x408a410c01608442ULL, 0x0a18c04208041280ULL,
    0x09001600c2160000ULL, 0x0020001020220000ULL, 0x8c000910100084a0ULL,
    0x8140042820812090ULL, 0x0620410121030000ULL, 0x2002028414120220ULL,
    0x1004002c04020830ULL, 0x0094005020841000ULL, 0x0810600418208800ULL,
    0x1020001090202200ULL, 0x0800084882488a00ULL, 0x0000080208081102ULL,
    0x8804910441020200ULL,
}};

static_assert(static_cast<int>(Color::White) == 0);
static_assert(static_cast<int>(Color::Black) == 1);

constexpr bool inside(int rank, int file) noexcept {
  return rank >= 0 && rank < bitboard::kRankCount && file >= 0 &&
         file < bitboard::kFileCount;
}

constexpr int constexprPopcount(Bitboard value) noexcept {
  int count = 0;
  while (value != 0) {
    value &= value - 1;
    ++count;
  }
  return count;
}

constexpr Bitboard rookMask(Square square) noexcept {
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

constexpr Bitboard bishopMask(Square square) noexcept {
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

constexpr std::uint32_t sliderTableSize(bool rook) noexcept {
  std::uint32_t size = 0;
  for (Square square = 0; square < bitboard::kSquareCount; ++square) {
    const Bitboard mask = rook ? rookMask(square) : bishopMask(square);
    size += std::uint32_t{1} << constexprPopcount(mask);
  }
  return size;
}

static_assert(sliderTableSize(true) == attack_detail::kRookAttackTableSize);
static_assert(sliderTableSize(false) == attack_detail::kBishopAttackTableSize);

Bitboard knightAttacksFrom(Square square) noexcept {
  static constexpr int kOffsets[8][2] = {
      {1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2},
  };

  const int rank = bitboard::rankOf(square);
  const int file = bitboard::fileOf(square);
  Bitboard attacks = 0;

  for (const auto& offset : kOffsets) {
    const int targetRank = rank + offset[0];
    const int targetFile = file + offset[1];
    if (inside(targetRank, targetFile)) {
      attacks |= bitboard::bit(targetRank * bitboard::kFileCount + targetFile);
    }
  }

  return attacks;
}

Bitboard kingAttacksFrom(Square square) noexcept {
  const int rank = bitboard::rankOf(square);
  const int file = bitboard::fileOf(square);
  Bitboard attacks = 0;

  for (int dr = -1; dr <= 1; ++dr) {
    for (int df = -1; df <= 1; ++df) {
      if (dr == 0 && df == 0) continue;

      const int targetRank = rank + dr;
      const int targetFile = file + df;
      if (inside(targetRank, targetFile)) {
        attacks |=
            bitboard::bit(targetRank * bitboard::kFileCount + targetFile);
      }
    }
  }

  return attacks;
}

Bitboard pawnAttacksFrom(Color color, Square square) noexcept {
  const int rank = bitboard::rankOf(square);
  const int file = bitboard::fileOf(square);
  const int direction = color == Color::White ? 1 : -1;
  const int targetRank = rank + direction;
  Bitboard attacks = 0;

  if (inside(targetRank, file - 1)) {
    attacks |= bitboard::bit(targetRank * bitboard::kFileCount + file - 1);
  }
  if (inside(targetRank, file + 1)) {
    attacks |= bitboard::bit(targetRank * bitboard::kFileCount + file + 1);
  }

  return attacks;
}

Bitboard rookAttacksSlow(Square square, Bitboard occupancy) noexcept {
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

Bitboard bishopAttacksSlow(Square square, Bitboard occupancy) noexcept {
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

Bitboard occupancyFromIndex(std::uint32_t index, Bitboard mask) noexcept {
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

void initLeapers() noexcept {
  for (Square square = 0; square < bitboard::kSquareCount; ++square) {
    attack_detail::knightAttackTable[square] = knightAttacksFrom(square);
    attack_detail::kingAttackTable[square] = kingAttacksFrom(square);
    attack_detail::pawnAttackTable[static_cast<int>(Color::White)][square] =
        pawnAttacksFrom(Color::White, square);
    attack_detail::pawnAttackTable[static_cast<int>(Color::Black)][square] =
        pawnAttacksFrom(Color::Black, square);
  }
}

void initSliderTable(bool rook, std::uint32_t& offset) noexcept {
  for (Square square = 0; square < bitboard::kSquareCount; ++square) {
    const Bitboard mask = rook ? rookMask(square) : bishopMask(square);
    const int relevantBits = bitboard::popcount(mask);
    attack_detail::MagicEntry& entry =
        rook ? attack_detail::rookEntries[square]
             : attack_detail::bishopEntries[square];

    entry.mask = mask;
    entry.magic = rook ? kRookMagics[square] : kBishopMagics[square];
    entry.offset = offset;
    entry.shift = static_cast<std::uint8_t>(64 - relevantBits);
    entry.relevantBits = static_cast<std::uint8_t>(relevantBits);

    const std::uint32_t permutations = std::uint32_t{1} << relevantBits;
    for (std::uint32_t index = 0; index < permutations; ++index) {
      const Bitboard occupancy = occupancyFromIndex(index, mask);
      const auto attackIndex = static_cast<std::uint32_t>(
          ((occupancy & entry.mask) * entry.magic) >> entry.shift);
      attack_detail::sliderAttackTable[entry.offset + attackIndex] =
          rook ? rookAttacksSlow(square, occupancy)
               : bishopAttacksSlow(square, occupancy);
    }

    offset += permutations;
  }
}

bool g_initialized = false;

}  // namespace

namespace attack_detail {

alignas(64) Bitboard knightAttackTable[bitboard::kSquareCount] = {};
alignas(64) Bitboard kingAttackTable[bitboard::kSquareCount] = {};
alignas(64) Bitboard pawnAttackTable[2][bitboard::kSquareCount] = {};
alignas(64) MagicEntry rookEntries[bitboard::kSquareCount] = {};
alignas(64) MagicEntry bishopEntries[bitboard::kSquareCount] = {};
alignas(64) Bitboard sliderAttackTable[kSliderAttackTableSize] = {};

}  // namespace attack_detail

void AttackTables::init() noexcept {
  if (g_initialized) return;

  initLeapers();

  std::uint32_t offset = 0;
  initSliderTable(true, offset);
  initSliderTable(false, offset);

  g_initialized = true;
}

bool AttackTables::initialized() noexcept { return g_initialized; }
