#include "nnue.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "attacks.h"
#include "bitboard.h"
#include "types.h"

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace nnue {
namespace {

constexpr std::uint32_t kFullThreatsHash = 0x8F234CB8U;
constexpr std::uint32_t kHalfKaV2HmHash = 0x7F234CB8U;
constexpr std::uint32_t kFeatureHash = 0x6165D5C9U;
constexpr std::uint32_t kFeatureTransformerHash = 0x6165DDC9U;
constexpr std::uint32_t kFullyConnectedHash = 0xA3337111U;
constexpr std::uint32_t kNetworkHash = 0xC256ACD8U;
constexpr std::size_t kDescriptionLimit = 1U << 20;
constexpr std::size_t kLeb128MarkerSize = 17;
constexpr char kLeb128Marker[] = "COMPRESSED_LEB128";
constexpr int kFeatureTransformerMax = 255;
constexpr int kHiddenMax = 127;
constexpr std::int64_t kOutputQuantizationScale = 9600;
constexpr std::int64_t kDenseOutputDenominator = 16384;
constexpr int kOutputScale = 16;

static_assert(sizeof(kLeb128Marker) - 1 == kLeb128MarkerSize);
static_assert(kFeatureHash ==
              ((((kFullThreatsHash << 1U) | (kFullThreatsHash >> 31U)) &
                0xFFFFFFFFU) ^
               kHalfKaV2HmHash));

constexpr int kKingBuckets[64] = {
    -1, -1, -1, -1, 31, 30, 29, 28, -1, -1, -1, -1, 27, 26, 25, 24,
    -1, -1, -1, -1, 23, 22, 21, 20, -1, -1, -1, -1, 19, 18, 17, 16,
    -1, -1, -1, -1, 15, 14, 13, 12, -1, -1, -1, -1, 11, 10, 9,  8,
    -1, -1, -1, -1, 7,  6,  5,  4,  -1, -1, -1, -1, 3,  2,  1,  0,
};

constexpr int kValidThreatTargets[12] = {6, 6, 10, 10, 8, 8,
                                         8, 8, 10, 10, 0, 0};

constexpr int kThreatTargetMap[6][6] = {
    {0, 1, -1, 2, -1, -1}, {0, 1, 2, 3, 4, -1}, {0, 1, 2, 3, -1, -1},
    {0, 1, 2, 3, -1, -1},  {0, 1, 2, 3, 4, -1}, {-1, -1, -1, -1, -1, -1},
};

struct alignas(64) DenseBucket {
  std::array<std::int32_t, kL2Size + 1> l1Bias{};
  std::array<std::int8_t, (kL2Size + 1) * kL1Size> l1Weights{};
  std::array<std::int32_t, kL3Size> l2Bias{};
  std::array<std::int8_t, kL3Size*(kL2Size * 2)> l2Weights{};
  std::int32_t outputBias = 0;
  std::array<std::int8_t, kL3Size> outputWeights{};
};

struct Network {
  std::uint32_t generation = 0;
  std::array<std::int16_t, kL1Size> featureBias{};
  std::vector<std::int8_t> threatWeights;
  std::vector<std::int32_t> threatPsqt;
  std::vector<std::int16_t> halfKaWeights;
  std::vector<std::int32_t> halfKaPsqt;
  std::array<DenseBucket, kLayerStackBuckets> buckets{};
};

struct ActiveFeatures {
  std::array<std::uint16_t, kMaxThreatFeatures> threats;
  std::array<std::uint16_t, kMaxHalfKaFeatures> halfKa;
  std::uint32_t threatCount = 0;
  std::uint32_t halfKaCount = 0;
  bool overflow = false;
};

struct EvaluationScratch {
  alignas(64) std::uint8_t pooled[kL1Size]{};
  alignas(64) std::uint8_t l2Input[kL2Size * 2]{};
  alignas(64) std::uint8_t l3Input[kL3Size]{};
};

struct AccumulatorCacheEntry {
  alignas(64) std::int16_t accumulation[2][kL1Size]{};
  alignas(64) std::int64_t psqt[2][kPsqtBuckets]{};
  std::array<std::array<std::uint16_t, kMaxThreatFeatures>, 2> threats{};
  std::uint32_t threatCount[2]{};
  std::uint64_t key = 0;
  std::uint32_t generation = 0;
  bool valid = false;
};

struct ThreatLayout {
  int pieceBase[12]{};
  int squareOffset[12][64]{};
  int squareSpan[12]{};
  std::int8_t targetOrdinal[12][64][64]{};
};

std::unique_ptr<Network> g_network;
std::string g_lastError;
std::uint32_t g_nextGeneration = 1;
thread_local EvaluationScratch t_scratch;
thread_local std::array<AccumulatorCacheEntry, Board::kMaxHistory + 1>
    t_accumulatorStack;

class NetworkReader {
 public:
  explicit NetworkReader(const char* path) : input_(path, std::ios::binary) {
    if (!input_) throw std::runtime_error("cannot open network file");
  }

  std::uint32_t readU32() {
    unsigned char bytes[4]{};
    readExact(bytes, sizeof(bytes));
    return static_cast<std::uint32_t>(bytes[0]) |
           (static_cast<std::uint32_t>(bytes[1]) << 8U) |
           (static_cast<std::uint32_t>(bytes[2]) << 16U) |
           (static_cast<std::uint32_t>(bytes[3]) << 24U);
  }

  std::string readString(std::size_t size) {
    if (size > kDescriptionLimit)
      throw std::runtime_error("network description is unreasonably large");
    std::string value(size, '\0');
    if (size != 0) readExact(value.data(), size);
    return value;
  }

  template <typename T>
  void readTensor(T* output, std::size_t count) {
    static_assert(std::is_integral<T>::value, "integer tensor required");
    if (hasLeb128Marker()) {
      readLeb128Tensor(output, count);
      return;
    }
    readRawTensor(output, count);
  }

  void requireEnd() {
    if (input_.peek() != std::char_traits<char>::eof())
      throw std::runtime_error("unexpected trailing data in network file");
  }

 private:
  void readExact(void* output, std::size_t size) {
    input_.read(static_cast<char*>(output), static_cast<std::streamsize>(size));
    if (!input_) throw std::runtime_error("unexpected end of network file");
  }

  bool hasLeb128Marker() {
    const std::streampos position = input_.tellg();
    char marker[kLeb128MarkerSize]{};
    input_.read(marker, static_cast<std::streamsize>(sizeof(marker)));
    const bool matches =
        input_.gcount() == static_cast<std::streamsize>(sizeof(marker)) &&
        std::memcmp(marker, kLeb128Marker, sizeof(marker)) == 0;
    input_.clear();
    input_.seekg(position);
    if (matches) {
      readExact(marker, sizeof(marker));
    }
    return matches;
  }

  template <typename T>
  void readRawTensor(T* output, std::size_t count) {
    if constexpr (sizeof(T) == 1) {
      readExact(output, count);
    } else {
      std::array<unsigned char, sizeof(T)> bytes{};
      for (std::size_t i = 0; i < count; ++i) {
        readExact(bytes.data(), bytes.size());
        using Unsigned = typename std::make_unsigned<T>::type;
        Unsigned value = 0;
        for (std::size_t byte = 0; byte < bytes.size(); ++byte)
          value |= static_cast<Unsigned>(bytes[byte]) << (byte * 8U);
        output[i] = static_cast<T>(value);
      }
    }
  }

  template <typename T>
  void readLeb128Tensor(T* output, std::size_t count) {
    const std::uint32_t byteCount = readU32();
    std::vector<unsigned char> encoded(byteCount);
    if (byteCount != 0) readExact(encoded.data(), encoded.size());

    std::size_t cursor = 0;
    for (std::size_t i = 0; i < count; ++i) {
      std::uint64_t value = 0;
      unsigned shift = 0;
      unsigned char byte = 0;
      do {
        if (cursor >= encoded.size() || shift >= 64)
          throw std::runtime_error("invalid signed LEB128 tensor");
        byte = encoded[cursor++];
        value |= static_cast<std::uint64_t>(byte & 0x7FU) << shift;
        shift += 7;
      } while ((byte & 0x80U) != 0);

      if ((byte & 0x40U) != 0 && shift < 64)
        value |= (~std::uint64_t{0}) << shift;
      const std::int64_t signedValue = static_cast<std::int64_t>(value);
      if (signedValue <
              static_cast<std::int64_t>(std::numeric_limits<T>::min()) ||
          signedValue >
              static_cast<std::int64_t>(std::numeric_limits<T>::max()))
        throw std::runtime_error("signed LEB128 value is outside tensor range");
      output[i] = static_cast<T>(signedValue);
    }
    if (cursor != encoded.size())
      throw std::runtime_error("signed LEB128 tensor length mismatch");
  }

  std::ifstream input_;
};

int colorIndex(Color color) { return color == Color::White ? 0 : 1; }

Color pieceColor(Piece piece) {
  return isWhitePiece(piece) ? Color::White : Color::Black;
}

int stockfishPieceType(Piece piece) {
  return static_cast<int>(pieceType(piece)) - static_cast<int>(PieceType::Pawn);
}

int stockfishPieceId(Piece piece) {
  return stockfishPieceType(piece) * 2 + colorIndex(pieceColor(piece));
}

Piece pieceAt(const Board& board, Square square) {
  return board.at(square);
}

Bitboard pseudoAttacks(PieceType type, Color color, Square square) {
  switch (type) {
    case PieceType::Pawn:
      return AttackTables::pawnAttacks(color, square);
    case PieceType::Knight:
      return AttackTables::knightAttacks(square);
    case PieceType::Bishop:
      return AttackTables::bishopAttacks(square, 0);
    case PieceType::Rook:
      return AttackTables::rookAttacks(square, 0);
    case PieceType::Queen:
      return AttackTables::queenAttacks(square, 0);
    case PieceType::King:
      return AttackTables::kingAttacks(square);
    case PieceType::None:
    default:
      return 0;
  }
}

Bitboard actualAttacks(PieceType type, Color color, Square square,
                       Bitboard occupied) {
  switch (type) {
    case PieceType::Knight:
      return AttackTables::knightAttacks(square);
    case PieceType::Bishop:
      return AttackTables::bishopAttacks(square, occupied);
    case PieceType::Rook:
      return AttackTables::rookAttacks(square, occupied);
    case PieceType::Queen:
      return AttackTables::queenAttacks(square, occupied);
    case PieceType::King:
      return AttackTables::kingAttacks(square);
    case PieceType::Pawn:
      return AttackTables::pawnAttacks(color, square);
    case PieceType::None:
    default:
      return 0;
  }
}

const ThreatLayout& threatLayout() {
  static const ThreatLayout layout = [] {
    ThreatLayout result{};
    std::memset(result.targetOrdinal, -1, sizeof(result.targetOrdinal));
    int pieceBase = 0;
    for (int colorValue = 0; colorValue < 2; ++colorValue) {
      for (int typeValue = 0; typeValue < 6; ++typeValue) {
        const int id = typeValue * 2 + colorValue;
        result.pieceBase[id] = pieceBase;
        const PieceType type = static_cast<PieceType>(typeValue + 1);
        const Color color = colorValue == 0 ? Color::White : Color::Black;
        int squareOffset = 0;
        for (Square from = 0; from < 64; ++from) {
          result.squareOffset[id][from] = squareOffset;
          Bitboard attacks = pseudoAttacks(type, color, from);
          if (type == PieceType::Pawn) {
            if (bitboard::rankOf(from) == 0 || bitboard::rankOf(from) == 7)
              continue;
            const int push = color == Color::White ? 8 : -8;
            attacks |= bitboard::bit(from + push);
          }
          int ordinal = 0;
          while (attacks != 0) {
            const Square to = bitboard::popLsb(attacks);
            result.targetOrdinal[id][from][to] =
                static_cast<std::int8_t>(ordinal++);
          }
          squareOffset += ordinal;
        }
        result.squareSpan[id] = squareOffset;
        pieceBase += kValidThreatTargets[id] * squareOffset;
      }
    }
    if (pieceBase != static_cast<int>(kThreatFeatureCount))
      throw std::runtime_error("internal Full Threats layout mismatch");
    return result;
  }();
  return layout;
}

Square orientHalfKa(Color perspective, Square square, Square kingSquare) {
  const bool flipHorizontal = bitboard::fileOf(kingSquare) < 4;
  return square ^ (perspective == Color::Black ? 56 : 0) ^
         (flipHorizontal ? 7 : 0);
}

std::uint32_t halfKaIndex(Color perspective, Square kingSquare, Square square,
                          Piece piece) {
  const Square orientedKing = orientHalfKa(perspective, kingSquare, kingSquare);
  const Square orientedPiece = orientHalfKa(perspective, square, kingSquare);
  const int bucket = kKingBuckets[orientedKing];
  const int sourcePlane = stockfishPieceType(piece) * 2 +
                          (pieceColor(piece) != perspective ? 1 : 0);
  const int exportPlane = std::min(sourcePlane, 10);
  return static_cast<std::uint32_t>(bucket * 704 + exportPlane * 64 +
                                    orientedPiece);
}

int threatIndex(Color perspective, Piece attacker, Square from, Square to,
                Piece attacked, Square kingSquare) {
  const bool enemy = pieceColor(attacker) != pieceColor(attacked);
  const int orientation = (perspective == Color::White ? 0 : 56) ^
                          (bitboard::fileOf(kingSquare) < 4 ? 0 : 7);
  from ^= orientation;
  to ^= orientation;

  int attackerId = stockfishPieceId(attacker);
  int attackedId = stockfishPieceId(attacked);
  if (perspective == Color::Black) {
    attackerId ^= 1;
    attackedId ^= 1;
  }

  const int attackerType = attackerId / 2;
  const int attackedType = attackedId / 2;
  const int targetClass = kThreatTargetMap[attackerType][attackedType];
  if (targetClass < 0 || (attackerType == attackedType &&
                          (enemy || attackerType != 0) && from < to))
    return -1;

  const ThreatLayout& layout = threatLayout();
  const int targetOrdinal = layout.targetOrdinal[attackerId][from][to];
  if (targetOrdinal < 0) return -1;
  return layout.pieceBase[attackerId] +
         ((attackedId & 1) * (kValidThreatTargets[attackerId] / 2) +
          targetClass) *
             layout.squareSpan[attackerId] +
         layout.squareOffset[attackerId][from] + targetOrdinal;
}

void addThreatFeature(ActiveFeatures& active, int index) {
  if (index < 0) return;
  if (index >= static_cast<int>(kThreatFeatureCount) ||
      active.threatCount >= active.threats.size()) {
    active.overflow = true;
    return;
  }
  active.threats[active.threatCount++] = static_cast<std::uint16_t>(index);
}

template <std::size_t Size>
void sortFeatures(std::array<std::uint16_t, Size>& features,
                  std::uint32_t count) {
  for (std::uint32_t index = 1; index < count; ++index) {
    const std::uint16_t value = features[index];
    std::uint32_t insertion = index;
    while (insertion > 0 && features[insertion - 1] > value) {
      features[insertion] = features[insertion - 1];
      --insertion;
    }
    features[insertion] = value;
  }
}

void sortActiveFeatures(ActiveFeatures (&active)[2]) {
  for (ActiveFeatures& features : active) {
    sortFeatures(features.threats, features.threatCount);
  }
}

void collectFeatures(const Board& board, ActiveFeatures (&active)[2],
                     bool collectHalfKa) {
  const Square kings[2] = {board.kingSquare(Color::White),
                           board.kingSquare(Color::Black)};
  if (kings[0] < 0 || kings[1] < 0) {
    active[0].overflow = true;
    active[1].overflow = true;
    return;
  }

  if (collectHalfKa) {
    ActiveFeatures& white = active[0];
    ActiveFeatures& black = active[1];
    Bitboard pieces = board.allPieces();
    while (pieces != 0) {
      const Square square = bitboard::popLsb(pieces);
      const Piece piece = pieceAt(board, square);
      if (white.halfKaCount >= white.halfKa.size() ||
          black.halfKaCount >= black.halfKa.size()) {
        white.overflow = true;
        black.overflow = true;
        return;
      }
      white.halfKa[white.halfKaCount++] =
          halfKaIndex(Color::White, kings[0], square, piece);
      black.halfKa[black.halfKaCount++] =
          halfKaIndex(Color::Black, kings[1], square, piece);
    }
  }

  const Bitboard occupied = board.allPieces();
  const Bitboard pawns = board.pieces(Color::White, PieceType::Pawn) |
                         board.pieces(Color::Black, PieceType::Pawn);
  Bitboard attackers = occupied;
  while (attackers != 0) {
    const Square from = bitboard::popLsb(attackers);
    const Piece attacker = pieceAt(board, from);
    const PieceType type = pieceType(attacker);
    const Color color = pieceColor(attacker);
    Bitboard targets = actualAttacks(type, color, from, occupied) & occupied;
    if (type == PieceType::Pawn) {
      const Square forward = from + (color == Color::White ? 8 : -8);
      if (forward >= 0 && forward < bitboard::kSquareCount &&
          (pawns & bitboard::bit(forward)) != 0) {
        targets |= bitboard::bit(forward);
      }
    }
    while (targets != 0) {
      const Square to = bitboard::popLsb(targets);
      const Piece attacked = pieceAt(board, to);
      addThreatFeature(active[0], threatIndex(Color::White, attacker, from, to,
                                              attacked, kings[0]));
      addThreatFeature(active[1], threatIndex(Color::Black, attacker, from, to,
                                              attacked, kings[1]));
    }
    if (active[0].overflow || active[1].overflow) {
      return;
    }
  }
}

void loadFeatureTransformer(NetworkReader& reader, Network& network) {
  reader.readTensor(network.featureBias.data(), network.featureBias.size());

  network.threatWeights.resize(static_cast<std::size_t>(kThreatFeatureCount) *
                               kL1Size);
  network.threatPsqt.resize(static_cast<std::size_t>(kThreatFeatureCount) *
                            kPsqtBuckets);
  network.halfKaWeights.resize(static_cast<std::size_t>(kHalfKaFeatureCount) *
                               kL1Size);
  network.halfKaPsqt.resize(static_cast<std::size_t>(kHalfKaFeatureCount) *
                            kPsqtBuckets);

  reader.readTensor(network.threatWeights.data(), network.threatWeights.size());
  reader.readTensor(network.threatPsqt.data(), network.threatPsqt.size());
  reader.readTensor(network.halfKaWeights.data(), network.halfKaWeights.size());
  reader.readTensor(network.halfKaPsqt.data(), network.halfKaPsqt.size());
}

void loadDenseBucket(NetworkReader& reader, DenseBucket& bucket) {
  if (reader.readU32() != kFullyConnectedHash)
    throw std::runtime_error("fully-connected layer hash mismatch");
  reader.readTensor(bucket.l1Bias.data(), bucket.l1Bias.size());
  reader.readTensor(bucket.l1Weights.data(), bucket.l1Weights.size());
  reader.readTensor(bucket.l2Bias.data(), bucket.l2Bias.size());
  reader.readTensor(bucket.l2Weights.data(), bucket.l2Weights.size());
  reader.readTensor(&bucket.outputBias, 1);
  reader.readTensor(bucket.outputWeights.data(), bucket.outputWeights.size());
}

void applyThreatFeature(const Network& network, AccumulatorCacheEntry& entry,
                        int perspective, std::uint32_t feature, int sign) {
  const std::int8_t* weights =
      &network.threatWeights[static_cast<std::size_t>(feature) * kL1Size];
#if defined(__ARM_NEON)
  std::size_t i = 0;
  std::int16_t* target = entry.accumulation[perspective];
  if (sign > 0) {
    for (; i + 32 <= kL1Size; i += 32) {
      const int8x16_t p0 = vld1q_s8(weights + i);
      const int8x16_t p1 = vld1q_s8(weights + i + 16);

      const int16x8_t w0 = vmovl_s8(vget_low_s8(p0));
      const int16x8_t w1 = vmovl_s8(vget_high_s8(p0));
      const int16x8_t w2 = vmovl_s8(vget_low_s8(p1));
      const int16x8_t w3 = vmovl_s8(vget_high_s8(p1));

      const int16x8_t c0 = vld1q_s16(target + i);
      const int16x8_t c1 = vld1q_s16(target + i + 8);
      const int16x8_t c2 = vld1q_s16(target + i + 16);
      const int16x8_t c3 = vld1q_s16(target + i + 24);

      vst1q_s16(target + i, vaddq_s16(c0, w0));
      vst1q_s16(target + i + 8, vaddq_s16(c1, w1));
      vst1q_s16(target + i + 16, vaddq_s16(c2, w2));
      vst1q_s16(target + i + 24, vaddq_s16(c3, w3));
    }
    for (; i + 16 <= kL1Size; i += 16) {
      const int8x16_t packed = vld1q_s8(weights + i);
      const int16x8_t low16 = vmovl_s8(vget_low_s8(packed));
      const int16x8_t high16 = vmovl_s8(vget_high_s8(packed));
      vst1q_s16(target + i, vaddq_s16(vld1q_s16(target + i), low16));
      vst1q_s16(target + i + 8, vaddq_s16(vld1q_s16(target + i + 8), high16));
    }
  } else {
    for (; i + 32 <= kL1Size; i += 32) {
      const int8x16_t p0 = vld1q_s8(weights + i);
      const int8x16_t p1 = vld1q_s8(weights + i + 16);

      const int16x8_t w0 = vmovl_s8(vget_low_s8(p0));
      const int16x8_t w1 = vmovl_s8(vget_high_s8(p0));
      const int16x8_t w2 = vmovl_s8(vget_low_s8(p1));
      const int16x8_t w3 = vmovl_s8(vget_high_s8(p1));

      const int16x8_t c0 = vld1q_s16(target + i);
      const int16x8_t c1 = vld1q_s16(target + i + 8);
      const int16x8_t c2 = vld1q_s16(target + i + 16);
      const int16x8_t c3 = vld1q_s16(target + i + 24);

      vst1q_s16(target + i, vsubq_s16(c0, w0));
      vst1q_s16(target + i + 8, vsubq_s16(c1, w1));
      vst1q_s16(target + i + 16, vsubq_s16(c2, w2));
      vst1q_s16(target + i + 24, vsubq_s16(c3, w3));
    }
    for (; i + 16 <= kL1Size; i += 16) {
      const int8x16_t packed = vld1q_s8(weights + i);
      const int16x8_t low16 = vmovl_s8(vget_low_s8(packed));
      const int16x8_t high16 = vmovl_s8(vget_high_s8(packed));
      vst1q_s16(target + i, vsubq_s16(vld1q_s16(target + i), low16));
      vst1q_s16(target + i + 8, vsubq_s16(vld1q_s16(target + i + 8), high16));
    }
  }
  for (; i < kL1Size; ++i)
#else
  for (std::size_t i = 0; i < kL1Size; ++i)
#endif
    entry.accumulation[perspective][i] += sign * weights[i];
  const std::int32_t* psqt =
      &network.threatPsqt[static_cast<std::size_t>(feature) * kPsqtBuckets];
  for (std::size_t bucket = 0; bucket < kPsqtBuckets; ++bucket)
    entry.psqt[perspective][bucket] +=
        static_cast<std::int64_t>(sign) * psqt[bucket];
}

void applyHalfKaFeature(const Network& network, AccumulatorCacheEntry& entry,
                        int perspective, std::uint32_t feature, int sign) {
  const std::int16_t* weights =
      &network.halfKaWeights[static_cast<std::size_t>(feature) * kL1Size];
#if defined(__ARM_NEON)
  std::size_t i = 0;
  std::int16_t* target = entry.accumulation[perspective];
  if (sign > 0) {
    for (; i + 32 <= kL1Size; i += 32) {
      const int16x8_t w0 = vld1q_s16(weights + i);
      const int16x8_t w1 = vld1q_s16(weights + i + 8);
      const int16x8_t w2 = vld1q_s16(weights + i + 16);
      const int16x8_t w3 = vld1q_s16(weights + i + 24);
      const int16x8_t c0 = vld1q_s16(target + i);
      const int16x8_t c1 = vld1q_s16(target + i + 8);
      const int16x8_t c2 = vld1q_s16(target + i + 16);
      const int16x8_t c3 = vld1q_s16(target + i + 24);
      vst1q_s16(target + i, vaddq_s16(c0, w0));
      vst1q_s16(target + i + 8, vaddq_s16(c1, w1));
      vst1q_s16(target + i + 16, vaddq_s16(c2, w2));
      vst1q_s16(target + i + 24, vaddq_s16(c3, w3));
    }
    for (; i + 8 <= kL1Size; i += 8) {
      vst1q_s16(target + i, vaddq_s16(vld1q_s16(target + i), vld1q_s16(weights + i)));
    }
  } else {
    for (; i + 32 <= kL1Size; i += 32) {
      const int16x8_t w0 = vld1q_s16(weights + i);
      const int16x8_t w1 = vld1q_s16(weights + i + 8);
      const int16x8_t w2 = vld1q_s16(weights + i + 16);
      const int16x8_t w3 = vld1q_s16(weights + i + 24);
      const int16x8_t c0 = vld1q_s16(target + i);
      const int16x8_t c1 = vld1q_s16(target + i + 8);
      const int16x8_t c2 = vld1q_s16(target + i + 16);
      const int16x8_t c3 = vld1q_s16(target + i + 24);
      vst1q_s16(target + i, vsubq_s16(c0, w0));
      vst1q_s16(target + i + 8, vsubq_s16(c1, w1));
      vst1q_s16(target + i + 16, vsubq_s16(c2, w2));
      vst1q_s16(target + i + 24, vsubq_s16(c3, w3));
    }
    for (; i + 8 <= kL1Size; i += 8) {
      vst1q_s16(target + i, vsubq_s16(vld1q_s16(target + i), vld1q_s16(weights + i)));
    }
  }
  for (; i < kL1Size; ++i)
#else
  for (std::size_t i = 0; i < kL1Size; ++i)
#endif
    entry.accumulation[perspective][i] += sign * weights[i];
  const std::int32_t* psqt =
      &network.halfKaPsqt[static_cast<std::size_t>(feature) * kPsqtBuckets];
  for (std::size_t bucket = 0; bucket < kPsqtBuckets; ++bucket)
    entry.psqt[perspective][bucket] +=
        static_cast<std::int64_t>(sign) * psqt[bucket];
}

void storeThreatFeatures(AccumulatorCacheEntry& entry,
                         const ActiveFeatures& active, int perspective) {
  entry.threatCount[perspective] = active.threatCount;
  std::copy_n(active.threats.begin(), active.threatCount,
              entry.threats[perspective].begin());
}

void buildAccumulatorFresh(const Network& network,
                           const ActiveFeatures (&active)[2],
                           AccumulatorCacheEntry& entry) {
  for (int perspective = 0; perspective < 2; ++perspective) {
    for (std::size_t i = 0; i < kL1Size; ++i)
      entry.accumulation[perspective][i] = network.featureBias[i];
    for (std::size_t bucket = 0; bucket < kPsqtBuckets; ++bucket)
      entry.psqt[perspective][bucket] = 0;

    for (std::uint32_t n = 0; n < active[perspective].threatCount; ++n)
      applyThreatFeature(network, entry, perspective,
                         active[perspective].threats[n], 1);
    for (std::uint32_t n = 0; n < active[perspective].halfKaCount; ++n)
      applyHalfKaFeature(network, entry, perspective,
                         active[perspective].halfKa[n], 1);
    storeThreatFeatures(entry, active[perspective], perspective);
  }
}

void updateHalfKaMove(const Network& network, const Board& board,
                      AccumulatorCacheEntry& entry) {
  std::array<Board::PieceDelta, 5> deltas{};
  const int count =
      board.lastMovePieceDeltas(deltas.data(), static_cast<int>(deltas.size()));
  for (int perspective = 0; perspective < 2; ++perspective) {
    const Color color = perspective == 0 ? Color::White : Color::Black;
    const Square king = board.kingSquare(color);
    for (int i = 0; i < count; ++i) {
      const Board::PieceDelta& delta = deltas[i];
      if (delta.from >= 0) {
        applyHalfKaFeature(network, entry, perspective,
                           halfKaIndex(color, king, delta.from, delta.piece),
                           -1);
      }
      if (delta.to >= 0) {
        applyHalfKaFeature(network, entry, perspective,
                           halfKaIndex(color, king, delta.to, delta.piece), 1);
      }
    }
  }
}

template <std::size_t Size, typename ApplyFeature>
void transformFeatureList(std::array<std::uint16_t, Size>& current,
                          std::uint32_t& currentCount,
                          const std::array<std::uint16_t, Size>& target,
                          std::uint32_t targetCount,
                          ApplyFeature applyFeature) {
  auto oldIt = current.begin();
  const auto oldEnd = oldIt + currentCount;
  auto newIt = target.begin();
  const auto newEnd = newIt + targetCount;
  while (oldIt != oldEnd || newIt != newEnd) {
    if (newIt == newEnd || (oldIt != oldEnd && *oldIt < *newIt)) {
      applyFeature(*oldIt++, -1);
    } else if (oldIt == oldEnd || *newIt < *oldIt) {
      applyFeature(*newIt++, 1);
    } else {
      ++oldIt;
      ++newIt;
    }
  }
  currentCount = targetCount;
  std::copy_n(target.begin(), targetCount, current.begin());
}

void transformThreats(const Network& network, AccumulatorCacheEntry& entry,
                      const ActiveFeatures (&active)[2]) {
  for (int perspective = 0; perspective < 2; ++perspective) {
    transformFeatureList(
        entry.threats[perspective], entry.threatCount[perspective],
        active[perspective].threats, active[perspective].threatCount,
        [&](std::uint16_t feature, int sign) {
          applyThreatFeature(network, entry, perspective, feature, sign);
        });
  }
}

AccumulatorCacheEntry* prepareAccumulator(const Network& network,
                                          const Board& board) {
  const int ply = board.ply();
  if (ply < 0 || ply > Board::kMaxHistory) return nullptr;

  AccumulatorCacheEntry& entry = t_accumulatorStack[ply];
  if (entry.valid && entry.key == board.key() &&
      entry.generation == network.generation)
    return &entry;

  const AccumulatorCacheEntry* parent = nullptr;
  if (ply > 0) {
    const AccumulatorCacheEntry& candidate = t_accumulatorStack[ply - 1];
    if (candidate.valid && candidate.key == board.previousKey() &&
        candidate.generation == network.generation) {
      parent = &candidate;
    }
  }

  if (parent != nullptr && board.lastMoveWasNull()) {
    entry = *parent;
  } else {
    Color kingColor = Color::White;
    const bool rebuild =
        parent == nullptr || board.lastMoveChangedKingSquare(kingColor);
    if (rebuild) {
      ActiveFeatures active[2];
      collectFeatures(board, active, true);
      if (active[0].overflow || active[1].overflow) return nullptr;
      sortActiveFeatures(active);
      buildAccumulatorFresh(network, active, entry);
    } else {
      entry = *parent;
      updateHalfKaMove(network, board, entry);
      ActiveFeatures active[2];
      collectFeatures(board, active, false);
      if (active[0].overflow || active[1].overflow) return nullptr;
      sortActiveFeatures(active);
      transformThreats(network, entry, active);
    }
  }

  entry.key = board.key();
  entry.generation = network.generation;
  entry.valid = true;
  return &entry;
}

std::int32_t dot(const std::int8_t* weights, const std::uint8_t* inputs,
                 std::size_t count, std::int32_t bias) {
#if defined(__AVX2__)
  const __m256i ones = _mm256_set1_epi16(1);
  __m256i vectorSum = _mm256_setzero_si256();
  std::size_t i = 0;
  for (; i + 32 <= count; i += 32) {
    const __m256i input =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(inputs + i));
    const __m256i weight =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(weights + i));
    const __m256i pairs = _mm256_maddubs_epi16(input, weight);
    vectorSum = _mm256_add_epi32(vectorSum, _mm256_madd_epi16(pairs, ones));
  }
  const __m128i low = _mm256_castsi256_si128(vectorSum);
  const __m128i high = _mm256_extracti128_si256(vectorSum, 1);
  __m128i sum128 = _mm_add_epi32(low, high);
  sum128 = _mm_hadd_epi32(sum128, sum128);
  sum128 = _mm_hadd_epi32(sum128, sum128);
  std::int32_t sum = bias + _mm_cvtsi128_si32(sum128);
  for (; i < count; ++i)
    sum += static_cast<std::int32_t>(weights[i]) * inputs[i];
  return sum;
#elif defined(__ARM_NEON)
  int32x4_t vectorSum = vdupq_n_s32(0);
  std::size_t i = 0;
  for (; i + 16 <= count; i += 16) {
    const int8x16_t weight = vld1q_s8(weights + i);
    const int8x16_t input = vreinterpretq_s8_u8(vld1q_u8(inputs + i));
#if defined(__ARM_FEATURE_DOTPROD)
    vectorSum = vdotq_s32(vectorSum, weight, input);
#else
    const int16x8_t low = vmull_s8(vget_low_s8(weight), vget_low_s8(input));
    const int16x8_t high = vmull_s8(vget_high_s8(weight), vget_high_s8(input));
    vectorSum = vaddq_s32(vectorSum, vpaddlq_s16(low));
    vectorSum = vaddq_s32(vectorSum, vpaddlq_s16(high));
#endif
  }
  std::int32_t sum = bias + vaddvq_s32(vectorSum);
  for (; i < count; ++i)
    sum += static_cast<std::int32_t>(weights[i]) * inputs[i];
  return sum;
#else
  std::int32_t sum = bias;
  for (std::size_t i = 0; i < count; ++i)
    sum += static_cast<std::int32_t>(weights[i]) * inputs[i];
  return sum;
#endif
}

void dot8(const std::int8_t* weights, std::size_t weightStride,
          const std::uint8_t* inputs, std::size_t count,
          const std::int32_t* biases, std::int32_t* sums) {
#if defined(__ARM_NEON)
  int32x4_t vectorSums[8] = {vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0),
                             vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0),
                             vdupq_n_s32(0), vdupq_n_s32(0)};
  std::size_t i = 0;
  for (; i + 16 <= count; i += 16) {
    const int8x16_t input = vreinterpretq_s8_u8(vld1q_u8(inputs + i));
#if defined(__ARM_FEATURE_DOTPROD)
    vectorSums[0] = vdotq_s32(vectorSums[0], vld1q_s8(weights + i), input);
    vectorSums[1] =
        vdotq_s32(vectorSums[1], vld1q_s8(weights + weightStride + i), input);
    vectorSums[2] = vdotq_s32(vectorSums[2],
                              vld1q_s8(weights + weightStride * 2 + i), input);
    vectorSums[3] = vdotq_s32(vectorSums[3],
                              vld1q_s8(weights + weightStride * 3 + i), input);
    vectorSums[4] = vdotq_s32(vectorSums[4],
                              vld1q_s8(weights + weightStride * 4 + i), input);
    vectorSums[5] = vdotq_s32(vectorSums[5],
                              vld1q_s8(weights + weightStride * 5 + i), input);
    vectorSums[6] = vdotq_s32(vectorSums[6],
                              vld1q_s8(weights + weightStride * 6 + i), input);
    vectorSums[7] = vdotq_s32(vectorSums[7],
                              vld1q_s8(weights + weightStride * 7 + i), input);
#else
    for (std::size_t output = 0; output < 8; ++output) {
      const int8x16_t weight = vld1q_s8(weights + output * weightStride + i);
      const int16x8_t low = vmull_s8(vget_low_s8(weight), vget_low_s8(input));
      const int16x8_t high =
          vmull_s8(vget_high_s8(weight), vget_high_s8(input));
      vectorSums[output] = vaddq_s32(vectorSums[output], vpaddlq_s16(low));
      vectorSums[output] = vaddq_s32(vectorSums[output], vpaddlq_s16(high));
    }
#endif
  }
  for (std::size_t output = 0; output < 8; ++output) {
    std::int32_t sum = biases[output] + vaddvq_s32(vectorSums[output]);
    const std::int8_t* row = weights + output * weightStride;
    for (std::size_t tail = i; tail < count; ++tail)
      sum += static_cast<std::int32_t>(row[tail]) * inputs[tail];
    sums[output] = sum;
  }
#else
  for (std::size_t output = 0; output < 8; ++output) {
    sums[output] =
        dot(weights + output * weightStride, inputs, count, biases[output]);
  }
#endif
}

void productPool(const std::int16_t* input, std::uint8_t* output) {
#if defined(__ARM_NEON)
  const int16x8_t zero = vdupq_n_s16(0);
  const int16x8_t maximum = vdupq_n_s16(kFeatureTransformerMax);
  for (std::size_t i = 0; i < kL1Size / 2; i += 16) {
    int16x8_t firstLow = vld1q_s16(input + i);
    int16x8_t firstHigh = vld1q_s16(input + i + 8);
    int16x8_t secondLow = vld1q_s16(input + kL1Size / 2 + i);
    int16x8_t secondHigh =
        vld1q_s16(input + kL1Size / 2 + i + 8);
    firstLow = vminq_s16(vmaxq_s16(firstLow, zero), maximum);
    firstHigh = vminq_s16(vmaxq_s16(firstHigh, zero), maximum);
    secondLow = vminq_s16(vmaxq_s16(secondLow, zero), maximum);
    secondHigh = vminq_s16(vmaxq_s16(secondHigh, zero), maximum);

    const uint16x8_t pooledLow = vcombine_u16(
        vqmovun_s32(vshrq_n_s32(
            vmull_s16(vget_low_s16(firstLow), vget_low_s16(secondLow)), 9)),
        vqmovun_s32(vshrq_n_s32(
            vmull_s16(vget_high_s16(firstLow), vget_high_s16(secondLow)), 9)));
    const uint16x8_t pooledHigh = vcombine_u16(
        vqmovun_s32(vshrq_n_s32(
            vmull_s16(vget_low_s16(firstHigh), vget_low_s16(secondHigh)), 9)),
        vqmovun_s32(vshrq_n_s32(
            vmull_s16(vget_high_s16(firstHigh), vget_high_s16(secondHigh)),
            9)));
    vst1q_u8(output + i,
             vcombine_u8(vqmovn_u16(pooledLow), vqmovn_u16(pooledHigh)));
  }
#else
  for (std::size_t i = 0; i < kL1Size / 2; ++i) {
    const int first = std::clamp(input[i], 0, kFeatureTransformerMax);
    const int second =
        std::clamp(input[i + kL1Size / 2], 0, kFeatureTransformerMax);
    output[i] = static_cast<std::uint8_t>((first * second) >> 9);
  }
#endif
}

int denseEvaluation(const Network& network, const Board& board,
                    const AccumulatorCacheEntry& accumulator,
                    EvaluationScratch& scratch) {
  const int side = colorIndex(board.sideToMove());
  const int other = side ^ 1;
  productPool(accumulator.accumulation[side], scratch.pooled);
  productPool(accumulator.accumulation[other], scratch.pooled + kL1Size / 2);

  const int pieceCount = bitboard::popcount(board.allPieces());
  const int bucketIndex = std::clamp((pieceCount - 1) / 4, 0, 7);
  const DenseBucket& bucket = network.buckets[bucketIndex];

  for (std::size_t output = 0; output < kL2Size; output += 8) {
    std::int32_t sums[8];
    dot8(&bucket.l1Weights[output * kL1Size], kL1Size, scratch.pooled, kL1Size,
         &bucket.l1Bias[output], sums);
    for (std::size_t lane = 0; lane < 8; ++lane) {
      const std::int32_t sum = sums[lane];
      const std::int64_t square = static_cast<std::int64_t>(sum) * sum;
      scratch.l2Input[output + lane] = static_cast<std::uint8_t>(
          std::min<std::int64_t>(kHiddenMax, square >> 21));
      scratch.l2Input[kL2Size + output + lane] =
          static_cast<std::uint8_t>(std::clamp(sum >> 7, 0, kHiddenMax));
    }
  }
  const std::int32_t skip =
      dot(&bucket.l1Weights[kL2Size * kL1Size], scratch.pooled, kL1Size,
          bucket.l1Bias[kL2Size]);

  for (std::size_t output = 0; output < kL3Size; output += 8) {
    std::int32_t sums[8];
    dot8(&bucket.l2Weights[output * (kL2Size * 2)], kL2Size * 2,
         scratch.l2Input, kL2Size * 2, &bucket.l2Bias[output], sums);
    for (std::size_t lane = 0; lane < 8; ++lane) {
      scratch.l3Input[output + lane] =
          static_cast<std::uint8_t>(std::clamp(sums[lane] >> 6, 0, kHiddenMax));
    }
  }

  const std::int32_t main = dot(bucket.outputWeights.data(), scratch.l3Input,
                                kL3Size, bucket.outputBias);
  const std::int64_t positionalRaw =
      ((static_cast<std::int64_t>(main) + skip) * kOutputQuantizationScale) /
      kDenseOutputDenominator;

  const std::int64_t psqtDifference =
      accumulator.psqt[0][bucketIndex] - accumulator.psqt[1][bucketIndex];
  const std::int64_t signedPsqt =
      board.sideToMove() == Color::White ? psqtDifference : -psqtDifference;
  return static_cast<int>((positionalRaw + signedPsqt / 2) / kOutputScale);
}

}  // namespace

bool loadNetwork(const char* path) {
  g_lastError.clear();
  if (path == nullptr || path[0] == '\0') {
    g_lastError = "empty network path";
    return false;
  }

  try {
    NetworkReader reader(path);
    if (reader.readU32() != kFileVersion)
      throw std::runtime_error("unsupported Stockfish NNUE version");
    if (reader.readU32() != kNetworkHash)
      throw std::runtime_error(
          "network architecture hash does not match SFNNv13 h1024-32-32");

    auto candidate = std::make_unique<Network>();
    (void)reader.readString(reader.readU32());
    if (reader.readU32() != kFeatureTransformerHash)
      throw std::runtime_error("feature-transformer hash mismatch");
    loadFeatureTransformer(reader, *candidate);
    for (DenseBucket& bucket : candidate->buckets)
      loadDenseBucket(reader, bucket);
    reader.requireEnd();

    candidate->generation = g_nextGeneration++;
    if (g_nextGeneration == 0) g_nextGeneration = 1;
    g_network = std::move(candidate);
    return true;
  } catch (const std::exception& error) {
    g_lastError = error.what();
    return false;
  }
}

void clearNetwork() { g_network.reset(); }

bool networkLoaded() { return g_network != nullptr; }

const char* lastError() { return g_lastError.c_str(); }

void resetAccumulatorCache() {
  for (AccumulatorCacheEntry& entry : t_accumulatorStack) entry.valid = false;
}

void rewindAccumulator(const Board&) {}

bool evaluate(const Board& board, int& score) {
  if (g_network == nullptr) return false;
  if (!AttackTables::initialized()) AttackTables::init();

  const AccumulatorCacheEntry* accumulator =
      prepareAccumulator(*g_network, board);
  if (accumulator == nullptr) return false;
  score = denseEvaluation(*g_network, board, *accumulator, t_scratch);
  return true;
}

}  // namespace nnue
