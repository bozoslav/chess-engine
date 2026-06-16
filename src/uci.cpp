#include "uci.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <istream>
#include <ostream>
#include <sstream>
#include <string>

#include "movegen.h"
#include "search.h"

namespace {

constexpr Square kNoSquare = -1;
constexpr int kTimedSearchDepth = 64;
constexpr const char* kStartFen =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

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
  bool hasDepth = false;
  bool perftMode = false;
  bool infinite = false;
};

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

void writeMoveUci(std::ostream& out, const Move& move) {
  char text[5] = {};
  const std::size_t size = move.writeUci(text);
  out.write(text, static_cast<std::streamsize>(size));
}

void writeInfoLine(const Board& board, const SearchResult& result,
                   std::ostream& out) {
  if (!result.hasBestMove) return;
  out << "info depth " << result.depth << " nodes " << result.nodes
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
      << " killer_uses " << result.killerMoveUses << " history_uses "
      << result.historyMoveUses << " quiet_cutoffs " << result.quietCutoffs
      << " pvs_researches " << result.pvsResearches << " aspiration_researches "
      << result.aspirationResearches;
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

struct SearchInfoOutput {
  const Board* board;
  std::ostream* out;
};

void writeDepthInfo(const SearchResult& result, void* context) {
  SearchInfoOutput* output = static_cast<SearchInfoOutput*>(context);
  writeInfoLine(*output->board, result, *output->out);
}

int allocateTimeMs(const Board& board, const GoOptions& options) {
  if (options.movetimeMs > 0) return options.movetimeMs;

  const bool white = board.sideToMove() == Color::White;
  const int remaining = white ? options.whiteTimeMs : options.blackTimeMs;
  const int increment =
      white ? options.whiteIncrementMs : options.blackIncrementMs;
  if (remaining <= 0) return 0;

  const int base = remaining / 30;
  const int incrementUse = increment / 2;
  const int conservativeCap = std::max(1, remaining / 4);
  const int reserve = remaining > 200 ? 50 : std::max(1, remaining / 10);
  const int rawBudget = std::max(1, base + incrementUse);
  const int cappedBudget = std::min(rawBudget, conservativeCap);
  return std::min(cappedBudget, std::max(1, remaining - reserve));
}

void writeBestMove(Board& board, const GoOptions& options, std::ostream& out) {
  SearchInfoOutput output{&board, &out};
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
  limits.onDepthComplete = writeDepthInfo;
  limits.infoContext = &output;

  const SearchResult result = searchBestMove(board, limits);
  writeBestMoveLine(result, out);
}

void runPerftLine(Board& board, int depth, std::ostream& out) {
  const std::uint64_t nodes = perft(board, depth);
  out << "perft depth " << depth << " nodes " << nodes << '\n';
}

std::uint64_t mixChecksum(std::uint64_t checksum, std::uint64_t value) {
  checksum ^= value + 0x9e3779b97f4a7c15ULL + (checksum << 6) + (checksum >> 2);
  return checksum;
}

void handleGo(Board& board, std::string_view command, std::ostream& out) {
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

  if (options.perftMode) {
    Board copy = board;
    runPerftLine(copy, options.depth, out);
    return;
  }

  writeBestMove(board, options, out);
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
  Board board;
  std::string line;

  while (std::getline(in, line)) {
    const std::string_view command = trim(line);
    if (command.empty()) continue;

    if (command == "uci") {
      out << "id name chess_engine\n";
      out << "id author Leon Mamic\n";
      out << "uciok\n";
    } else if (command == "isready") {
      out << "readyok\n";
    } else if (command == "ucinewgame") {
      board = Board{};
      clearSearchState();
    } else if (startsWith(command, "position")) {
      if (!setPositionFromUci(board, command)) {
        out << "info string invalid position command\n";
      }
    } else if (startsWith(command, "go")) {
      handleGo(board, command, out);
    } else if (startsWith(command, "perft")) {
      const int depth = parsePositiveInt(trim(command.substr(5)), 1);
      runPerftLine(board, depth, out);
    } else if (command == "bench") {
      runBench(out);
    } else if (command == "stop") {
      out << "bestmove 0000\n";
    } else if (command == "quit") {
      break;
    } else {
      out << "info string unknown command\n";
    }

    out.flush();
  }
}
