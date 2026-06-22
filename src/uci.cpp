#include "uci.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <istream>
#include <mutex>
#include <ostream>
#include <sstream>
#include <string>
#include <thread>
#include <utility>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

#include "evaluate.h"
#include "movegen.h"
#include "nnue.h"
#include "search.h"
#include "transposition_table.h"

namespace {

constexpr Square kNoSquare = -1;
constexpr int kTimedSearchDepth = 64;
constexpr int kMaxThreads = 128;
constexpr int kMaxMoveOverheadMs = 5000;
constexpr const char* kStartFen =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
#ifndef CHESS_ENGINE_DEFAULT_EVALFILE
#define CHESS_ENGINE_DEFAULT_EVALFILE ""
#endif

struct BenchCase {
  const char* name;
  const char* fen;
  int depth;
  std::uint64_t expectedNodes;
};

struct GoOptions {
  int depth = 1;
  int movetimeMs = 0;
  int whiteTimeMs = 0;
  int blackTimeMs = 0;
  int whiteIncrementMs = 0;
  int blackIncrementMs = 0;
  int movesToGo = 0;
  bool hasDepth = false;
  bool perftMode = false;
  bool infinite = false;
};

int detectDefaultThreads() {
#if defined(__APPLE__)
  int performanceCores = 0;
  std::size_t size = sizeof(performanceCores);
  if (sysctlbyname("hw.perflevel0.logicalcpu", &performanceCores, &size,
                   nullptr, 0) == 0 &&
      performanceCores > 0) {
    return std::clamp(performanceCores, 1, kMaxThreads);
  }
#endif

  const unsigned hardwareThreads = std::thread::hardware_concurrency();
  if (hardwareThreads == 0) return 1;
  return std::clamp(static_cast<int>(hardwareThreads), 1, kMaxThreads);
}

int gSearchThreads = detectDefaultThreads();
bool gUseSingularExtensions = true;
int gMoveOverheadMs = 30;

constexpr BenchCase kBenchCases[] = {
    {
        "startpos",
        kStartFen,
        5,
        4865609,
    },
    {
        "kiwipete",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/"
        "PPPBBPPP/R3K2R w KQkq - 0 1",
        3,
        97862,
    },
    {
        "endgame",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        4,
        43238,
    },
};

std::string_view trimLeft(std::string_view text) {
  while (!text.empty() && text.front() == ' ') text.remove_prefix(1);
  return text;
}

std::string_view trim(std::string_view text) {
  text = trimLeft(text);
  while (!text.empty() && text.back() == ' ') text.remove_suffix(1);
  return text;
}

bool startsWith(std::string_view text, std::string_view prefix) {
  return text.size() >= prefix.size() &&
         text.substr(0, prefix.size()) == prefix;
}

Square parseSquare(char file, char rank) {
  if (file < 'a' || file > 'h') return kNoSquare;
  if (rank < '1' || rank > '8') return kNoSquare;
  return (rank - '1') * bitboard::kFileCount + (file - 'a');
}

PieceType parsePromotion(char promotion) {
  switch (promotion) {
    case 'n':
      return PieceType::Knight;
    case 'b':
      return PieceType::Bishop;
    case 'r':
      return PieceType::Rook;
    case 'q':
      return PieceType::Queen;
    default:
      return PieceType::None;
  }
}

bool applyMoveList(Board& board, std::string_view text) {
  text = trim(text);
  if (text.empty()) return true;
  if (!startsWith(text, "moves")) return false;

  text.remove_prefix(5);
  text = trim(text);
  while (!text.empty()) {
    const std::size_t end = text.find(' ');
    const std::string_view moveText =
        end == std::string_view::npos ? text : text.substr(0, end);

    Move move;
    if (!moveFromUci(board, moveText, move)) return false;
    // UCI move lists represent played game moves, so update FEN counters too.
    if (!board.makeMove(move)) return false;

    if (end == std::string_view::npos) break;
    text.remove_prefix(end + 1);
    text = trimLeft(text);
  }

  return true;
}

int parsePositiveInt(std::string_view text, int fallback) {
  std::istringstream stream{std::string(text)};
  int value = fallback;
  stream >> value;
  return value > 0 ? value : fallback;
}

int parseNonNegativeInt(std::string_view text, int fallback, int maximum) {
  std::istringstream stream{std::string(text)};
  int value = fallback;
  if (!(stream >> value)) return fallback;
  return std::clamp(value, 0, maximum);
}

void writeMoveUci(std::ostream& out, const Move& move) {
  char text[5] = {};
  const std::size_t size = move.writeUci(text);
  out.write(text, static_cast<std::streamsize>(size));
}

void writeInfoLine(const Board& board, const SearchResult& result,
                   std::ostream& out) {
  if (!result.hasBestMove) return;
  out << "info depth " << result.depth << " nodes " << result.nodes << " time "
      << result.timeMs;
  if (result.timeMs > 0) {
    out << " nps " << (result.nodes * 1000ULL) / result.timeMs;
  }
  out << " hashfull " << globalTranspositionTable().hashfullPermill()
      << " score cp " << result.score;
  if (result.pvLength > 0) {
    out << " pv";
    for (int index = 0; index < result.pvLength; ++index) {
      out << ' ';
      writeMoveUci(out, result.principalVariation[index]);
    }
  }
  out << " string tt_hits " << result.ttHits << " tt_cutoffs "
      << result.ttCutoffs << " tt_move_uses " << result.ttMoveUses
      << " killer_uses " << result.killerMoveUses << " counter_uses "
      << result.counterMoveUses << " history_uses " << result.historyMoveUses
      << " continuation_uses " << result.continuationHistoryUses
      << " capture_history_uses " << result.captureHistoryUses
      << " quiet_cutoffs " << result.quietCutoffs << " pvs_researches "
      << result.pvsResearches << " aspiration_researches "
      << result.aspirationResearches << " null_attempts "
      << result.nullMoveAttempts << " null_prunes " << result.nullMovePrunes
      << " lmr_attempts " << result.lmrAttempts << " lmr_researches "
      << result.lmrResearches << " see_prunes " << result.seePrunes
      << " futility_prunes " << result.futilityPrunes << " lmp_prunes "
      << result.lateMovePrunes << " singular_searches "
      << result.singularSearches << " singular_extensions "
      << result.singularExtensions << " probcut_attempts "
      << result.probCutAttempts << " probcut_prunes " << result.probCutPrunes
      << " rfp_prunes " << result.reverseFutilityPrunes
      << " razor_attempts " << result.razorAttempts << " razor_prunes "
      << result.razorPrunes << " iir_reductions "
      << result.internalIterativeReductions;
  if (board.hasRepeatedPosition()) {
    out << " repetition " << board.repetitionCount();
  }
  out << '\n';
}

void writeBestMoveLine(const SearchResult& result, std::ostream& out) {
  if (!result.hasBestMove) {
    out << "bestmove 0000\n";
    return;
  }

  out << "bestmove ";
  writeMoveUci(out, result.bestMove);
  out << '\n';
}

void chooseFallbackBestMove(const Board& board, SearchResult& result) {
  if (result.hasBestMove) return;

  Board copy = board;
  MoveList moves;
  genLegalMoves(copy, moves);
  if (moves.empty() || moves.overflowed()) return;

  result.bestMove = moves[0];
  result.principalVariation[0] = moves[0];
  result.pvLength = 1;
  result.hasBestMove = true;
}

struct SearchInfoOutput {
  const Board* board;
  std::ostream* out;
  std::mutex* outputMutex;
};

void writeDepthInfo(const SearchResult& result, void* context) {
  SearchInfoOutput* output = static_cast<SearchInfoOutput*>(context);
  std::lock_guard<std::mutex> lock(*output->outputMutex);
  writeInfoLine(*output->board, result, *output->out);
  output->out->flush();
}

int allocateTimeMs(const Board& board, const GoOptions& options) {
  if (options.movetimeMs > 0) return options.movetimeMs;

  const bool white = board.sideToMove() == Color::White;
  const int remaining = white ? options.whiteTimeMs : options.blackTimeMs;
  const int increment =
      white ? options.whiteIncrementMs : options.blackIncrementMs;
  if (remaining <= 0) return 0;

  // A fixed remaining/30 budget is unsafe in long zero-increment games: the
  // clock also pays process, bridge, and network latency on every move. Reserve
  // that cost for the expected moves still to play before dividing the clock.
  const int divisor = options.movesToGo > 0
                          ? std::clamp(options.movesToGo, 1, 80)
                          : (increment > 0 ? 30
                                           : (board.fullmoveNumber() >= 40 ? 50
                                                                          : 40));
  const std::int64_t overheadReserve =
      static_cast<std::int64_t>(gMoveOverheadMs) * divisor;
  const int spendable = static_cast<int>(std::max<std::int64_t>(
      1, static_cast<std::int64_t>(remaining) - overheadReserve));
  const int base = std::max(1, spendable / divisor);
  const int incrementUse =
      std::max(0, increment - gMoveOverheadMs) * 3 / 4;
  const int conservativeCap =
      std::max(1, (remaining - gMoveOverheadMs) / 4);
  const int rawBudget = std::max(1, base + incrementUse);
  const int cappedBudget = std::min(rawBudget, conservativeCap);
  return std::min(cappedBudget,
                  std::max(1, remaining - gMoveOverheadMs));
}

SearchLimits makeSearchLimits(const Board& board, const GoOptions& options,
                              SearchInfoOutput& output,
                              std::atomic_bool* stopSignal) {
  SearchLimits limits;
  const int allocatedTimeMs = allocateTimeMs(board, options);
  limits.timeLimitMs = static_cast<std::uint64_t>(allocatedTimeMs);
  if (options.hasDepth) {
    limits.depth = options.depth;
  } else if (allocatedTimeMs > 0 || options.infinite) {
    limits.depth = kTimedSearchDepth;
  } else {
    limits.depth = 1;
  }
  limits.threads = gSearchThreads;
  limits.useSingularExtensions = gUseSingularExtensions;
  limits.onDepthComplete = writeDepthInfo;
  limits.infoContext = &output;
  limits.stopSignal = stopSignal;
  return limits;
}

bool isSynchronousSearch(const GoOptions& options) {
  return options.hasDepth && !options.infinite && options.movetimeMs == 0 &&
         options.whiteTimeMs == 0 && options.blackTimeMs == 0;
}

struct SearchController {
  std::thread worker;
  std::atomic_bool stopSignal{false};

  ~SearchController() { stopAndJoin(); }

  void stopAndJoin() {
    if (!worker.joinable()) return;
    stopSignal.store(true, std::memory_order_relaxed);
    worker.join();
    stopSignal.store(false, std::memory_order_relaxed);
  }

  void wait() {
    if (!worker.joinable()) return;
    worker.join();
    stopSignal.store(false, std::memory_order_relaxed);
  }

  void start(Board board, GoOptions options, std::ostream& out,
             std::mutex& outputMutex) {
    stopAndJoin();
    stopSignal.store(false, std::memory_order_relaxed);
    worker = std::thread([this, board = std::move(board), options, &out,
                          &outputMutex]() mutable {
      SearchInfoOutput output{&board, &out, &outputMutex};
      SearchLimits limits =
          makeSearchLimits(board, options, output, &stopSignal);
      SearchResult result = searchBestMove(board, limits);
      chooseFallbackBestMove(board, result);

      std::lock_guard<std::mutex> lock(outputMutex);
      writeBestMoveLine(result, out);
      out.flush();
    });
  }
};

void runPerftLine(Board& board, int depth, std::ostream& out) {
  const std::uint64_t nodes = perft(board, depth);
  out << "perft depth " << depth << " nodes " << nodes << '\n';
}

void writeFenLine(const Board& board, std::ostream& out) {
  out << "fen " << board.toFen() << '\n';
}

void writeNnueEval(const Board& board, std::ostream& out) {
  int score = 0;
  const bool loaded = nnue::evaluate(board, score);
  out << "nnueeval loaded " << (loaded ? 1 : 0) << " score "
      << (loaded ? score : 0) << '\n';
}

void writeEval(const Board& board, std::ostream& out) {
  out << "NNUE evaluation " << evaluate(board)
      << " (side to move, internal units)\n";
}

std::uint64_t mixChecksum(std::uint64_t checksum, std::uint64_t value) {
  checksum ^= value + 0x9e3779b97f4a7c15ULL + (checksum << 6) + (checksum >> 2);
  return checksum;
}

GoOptions parseGoOptions(std::string_view command) {
  std::istringstream stream{std::string(command)};
  std::string token;
  GoOptions options;

  stream >> token;
  while (stream >> token) {
    if (token == "depth") {
      stream >> options.depth;
      options.hasDepth = true;
    } else if (token == "perft") {
      stream >> options.depth;
      options.hasDepth = true;
      options.perftMode = true;
    } else if (token == "movetime") {
      stream >> options.movetimeMs;
    } else if (token == "wtime") {
      stream >> options.whiteTimeMs;
    } else if (token == "btime") {
      stream >> options.blackTimeMs;
    } else if (token == "winc") {
      stream >> options.whiteIncrementMs;
    } else if (token == "binc") {
      stream >> options.blackIncrementMs;
    } else if (token == "movestogo") {
      stream >> options.movesToGo;
    } else if (token == "infinite") {
      options.infinite = true;
    }
  }

  if (options.depth <= 0) options.depth = 1;
  if (options.movetimeMs < 0) options.movetimeMs = 0;
  if (options.whiteTimeMs < 0) options.whiteTimeMs = 0;
  if (options.blackTimeMs < 0) options.blackTimeMs = 0;
  if (options.whiteIncrementMs < 0) options.whiteIncrementMs = 0;
  if (options.blackIncrementMs < 0) options.blackIncrementMs = 0;
  if (options.movesToGo < 0) options.movesToGo = 0;

  return options;
}

void handleSetOption(std::string_view command, std::ostream& out) {
  command = trim(command);
  if (startsWith(command, "setoption")) {
    command.remove_prefix(9);
    command = trim(command);
  }
  if (!startsWith(command, "name")) {
    out << "info string invalid setoption command\n";
    return;
  }

  command.remove_prefix(4);
  command = trim(command);
  const std::size_t valuePos = command.find(" value ");
  const std::string_view name =
      trim(valuePos == std::string_view::npos ? command
                                              : command.substr(0, valuePos));
  const std::string_view value = valuePos == std::string_view::npos
                                     ? std::string_view{}
                                     : trim(command.substr(valuePos + 7));

  if (name == "EvalFile") {
    if (value.empty() || value == "<empty>") {
      clearNnueFile();
      out << "info string NNUE eval file cleared\n";
      return;
    }

    const std::string path{value};
    if (loadNnueFile(path.c_str())) {
      out << "info string Stockfish NNUE loaded " << nnueEvalFile() << '\n';
    } else {
      out << "info string failed to load Stockfish NNUE " << path << ": "
          << nnueLoadError() << '\n';
    }
    return;
  }

  if (name == "Threads") {
    gSearchThreads =
        std::clamp(parsePositiveInt(value, gSearchThreads), 1, kMaxThreads);
    prepareSearchThreads(gSearchThreads);
    out << "info string Threads set to " << gSearchThreads << '\n';
    return;
  }

  if (name == "Hash") {
    const int fallback =
        static_cast<int>(globalTranspositionTable().hashSizeMb());
    const int hashMb = parsePositiveInt(value, fallback);
    globalTranspositionTable().resize(static_cast<std::size_t>(hashMb));
    out << "info string Hash set to " << globalTranspositionTable().hashSizeMb()
        << " MB\n";
    return;
  }

  if (name == "SingularExtensions") {
    gUseSingularExtensions = value == "true" || value == "1" || value == "on";
    out << "info string SingularExtensions set to "
        << (gUseSingularExtensions ? "true" : "false") << '\n';
    return;
  }

  if (name == "MoveOverhead") {
    gMoveOverheadMs =
        parseNonNegativeInt(value, gMoveOverheadMs, kMaxMoveOverheadMs);
    out << "info string MoveOverhead set to " << gMoveOverheadMs << " ms\n";
    return;
  }

  out << "info string unknown option " << name << '\n';
}

void configureDefaultsFromEnvironment() {
  const char* evalFileOverride = std::getenv("CHESS_ENGINE_EVALFILE");
  if (evalFileOverride != nullptr) {
    if (evalFileOverride[0] == '\0') {
      clearNnueFile();
    } else {
      loadNnueFile(evalFileOverride);
    }
  } else if constexpr (std::string_view(CHESS_ENGINE_DEFAULT_EVALFILE).size() >
                       0) {
    loadNnueFile(CHESS_ENGINE_DEFAULT_EVALFILE);
  }

  if (const char* threads = std::getenv("CHESS_ENGINE_THREADS")) {
    gSearchThreads =
        std::clamp(parsePositiveInt(threads, gSearchThreads), 1, kMaxThreads);
  }

  if (const char* hashMb = std::getenv("CHESS_ENGINE_HASH_MB")) {
    const int fallback =
        static_cast<int>(globalTranspositionTable().hashSizeMb());
    globalTranspositionTable().resize(
        static_cast<std::size_t>(parsePositiveInt(hashMb, fallback)));
  }
}

}  // namespace

bool moveFromUci(const Board& board, std::string_view text, Move& move) {
  if (text.size() != 4 && text.size() != 5) return false;

  const Square from = parseSquare(text[0], text[1]);
  const Square to = parseSquare(text[2], text[3]);
  if (from == kNoSquare || to == kNoSquare) return false;

  const PieceType promotion =
      text.size() == 5 ? parsePromotion(text[4]) : PieceType::None;
  if (text.size() == 5 && promotion == PieceType::None) return false;

  MoveList moves;
  genLegalMoves(board, moves);
  if (moves.overflowed()) return false;

  for (const Move candidate : moves) {
    if (candidate.fromSquare() == from && candidate.toSquare() == to &&
        candidate.promo() == promotion) {
      move = candidate;
      return true;
    }
  }

  return false;
}

bool setPositionFromUci(Board& board, std::string_view command) {
  command = trim(command);
  if (startsWith(command, "position")) {
    command.remove_prefix(8);
    command = trim(command);
  }

  Board next;
  std::string_view moveText;
  if (startsWith(command, "startpos")) {
    command.remove_prefix(8);
    next.setFromFen(kStartFen);
    moveText = trim(command);
  } else if (startsWith(command, "fen")) {
    command.remove_prefix(3);
    command = trim(command);
    const std::size_t movesPos = command.find(" moves ");
    const std::string_view fen = movesPos == std::string_view::npos
                                     ? command
                                     : command.substr(0, movesPos);
    if (!next.setFromFen(trim(fen))) return false;
    moveText = movesPos == std::string_view::npos
                   ? std::string_view{}
                   : command.substr(movesPos + 1);
  } else {
    return false;
  }

  if (!applyMoveList(next, moveText)) return false;
  board = next;
  return true;
}

bool runBench(std::ostream& out) {
  using Clock = std::chrono::steady_clock;

  bool ok = true;
  std::uint64_t totalNodes = 0;
  std::uint64_t checksum = 0xcbf29ce484222325ULL;
  const auto start = Clock::now();

  for (const BenchCase& testCase : kBenchCases) {
    Board board;
    if (!board.setFromFen(testCase.fen)) {
      out << "bench " << testCase.name << " invalid_fen\n";
      ok = false;
      continue;
    }

    const std::uint64_t nodes = perft(board, testCase.depth);
    const bool matches = nodes == testCase.expectedNodes;
    ok &= matches;
    totalNodes += nodes;
    checksum = mixChecksum(checksum, board.key());
    checksum = mixChecksum(checksum, nodes);
    checksum =
        mixChecksum(checksum, static_cast<std::uint64_t>(testCase.depth));

    out << "bench " << testCase.name << " depth " << testCase.depth << " nodes "
        << nodes << ' ' << (matches ? "ok" : "node_mismatch") << '\n';
  }

  const std::chrono::duration<double> elapsed = Clock::now() - start;
  const double seconds = elapsed.count();
  const double nps =
      seconds > 0.0 ? static_cast<double>(totalNodes) / seconds : 0.0;

  out << "bench total nodes " << totalNodes << " seconds " << std::fixed
      << std::setprecision(6) << seconds << " nps "
      << static_cast<std::uint64_t>(nps) << " checksum 0x" << std::hex
      << checksum << std::dec << ' ' << (ok ? "ok" : "node_mismatch") << '\n';

  return ok;
}

void runUci(std::istream& in, std::ostream& out) {
  configureDefaultsFromEnvironment();
  prepareSearchThreads(gSearchThreads);

  Board board;
  std::string line;
  std::mutex outputMutex;
  SearchController search;

  auto writeLocked = [&](auto writer) {
    std::lock_guard<std::mutex> lock(outputMutex);
    writer();
    out.flush();
  };

  while (std::getline(in, line)) {
    const std::string_view command = trim(line);
    if (command.empty()) continue;

    if (command == "uci") {
      writeLocked([&]() {
        out << "id name chess_engine\n";
        out << "id author Leon Mamic\n";
        out << "option name EvalFile type string default "
            << (nnueReady() ? nnueEvalFile() : "<empty>") << '\n';
        out << "option name Threads type spin default " << gSearchThreads
            << " min 1 max " << kMaxThreads << '\n';
        out << "option name Hash type spin default "
            << globalTranspositionTable().hashSizeMb() << " min "
            << TranspositionTable::kMinHashMb << " max "
            << TranspositionTable::kMaxHashMb << '\n';
        out << "option name SingularExtensions type check default "
            << (gUseSingularExtensions ? "true" : "false") << '\n';
        out << "option name MoveOverhead type spin default " << gMoveOverheadMs
            << " min 0 max " << kMaxMoveOverheadMs << '\n';
        out << "uciok\n";
      });
    } else if (command == "isready") {
      writeLocked([&]() { out << "readyok\n"; });
    } else if (command == "ucinewgame") {
      search.stopAndJoin();
      board = Board{};
      clearSearchState();
    } else if (startsWith(command, "position")) {
      search.stopAndJoin();
      if (!setPositionFromUci(board, command)) {
        writeLocked([&]() { out << "info string invalid position command\n"; });
      }
    } else if (startsWith(command, "go")) {
      const GoOptions options = parseGoOptions(command);
      if (options.perftMode) {
        search.stopAndJoin();
        Board copy = board;
        writeLocked([&]() { runPerftLine(copy, options.depth, out); });
      } else {
        search.start(board, options, out, outputMutex);
        if (isSynchronousSearch(options)) {
          search.wait();
        }
      }
    } else if (startsWith(command, "setoption")) {
      search.stopAndJoin();
      writeLocked([&]() { handleSetOption(command, out); });
    } else if (startsWith(command, "perft")) {
      search.stopAndJoin();
      const int depth = parsePositiveInt(trim(command.substr(5)), 1);
      writeLocked([&]() { runPerftLine(board, depth, out); });
    } else if (command == "fen") {
      writeLocked([&]() { writeFenLine(board, out); });
    } else if (command == "nnueeval") {
      writeLocked([&]() { writeNnueEval(board, out); });
    } else if (command == "eval") {
      writeLocked([&]() { writeEval(board, out); });
    } else if (command == "bench") {
      search.stopAndJoin();
      writeLocked([&]() { runBench(out); });
    } else if (command == "stop") {
      search.stopAndJoin();
    } else if (command == "quit") {
      search.stopAndJoin();
      break;
    } else {
      writeLocked([&]() { out << "info string unknown command\n"; });
    }
  }

  search.stopAndJoin();
}
