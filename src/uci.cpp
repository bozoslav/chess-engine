#include "uci.h"

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
constexpr const char* kStartFen =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

struct BenchCase {
  const char* name;
  const char* fen;
  int depth;
  std::uint64_t expectedNodes;
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

void writeSearchResult(const Board& board, SearchResult result,
                       std::ostream& out) {
  if (!result.hasBestMove) {
    out << "bestmove 0000\n";
    return;
  }

  out << "info depth " << result.depth << " nodes " << result.nodes
      << " score cp " << result.score;
  if (board.hasRepeatedPosition()) {
    out << " string repetition " << board.repetitionCount();
  }
  out << '\n';
  out << "bestmove " << result.bestMove.toUci() << '\n';
}

void writeBestMove(Board& board, int depth, std::ostream& out) {
  const SearchResult result = searchBestMove(board, {depth});
  writeSearchResult(board, result, out);
}

void runPerftLine(Board& board, int depth, std::ostream& out) {
  const std::uint64_t nodes = perft(board, depth);
  out << "perft depth " << depth << " nodes " << nodes << '\n';
}

std::uint64_t mixChecksum(std::uint64_t checksum, std::uint64_t value) {
  checksum ^=
      value + 0x9e3779b97f4a7c15ULL + (checksum << 6) + (checksum >> 2);
  return checksum;
}

void handleGo(Board& board, std::string_view command, std::ostream& out) {
  std::istringstream stream{std::string(command)};
  std::string token;
  int depth = 1;
  bool perftMode = false;

  stream >> token;
  while (stream >> token) {
    if (token == "depth") {
      stream >> depth;
    } else if (token == "perft") {
      stream >> depth;
      perftMode = true;
    }
  }

  if (depth <= 0) depth = 1;

  if (perftMode) {
    Board copy = board;
    runPerftLine(copy, depth, out);
    return;
  }

  writeBestMove(board, depth, out);
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
    const std::string_view fen =
        movesPos == std::string_view::npos ? command
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
    checksum = mixChecksum(checksum, static_cast<std::uint64_t>(testCase.depth));

    out << "bench " << testCase.name << " depth " << testCase.depth
        << " nodes " << nodes << ' ' << (matches ? "ok" : "node_mismatch")
        << '\n';
  }

  const std::chrono::duration<double> elapsed = Clock::now() - start;
  const double seconds = elapsed.count();
  const double nps =
      seconds > 0.0 ? static_cast<double>(totalNodes) / seconds : 0.0;

  out << "bench total nodes " << totalNodes << " seconds " << std::fixed
      << std::setprecision(6) << seconds << " nps "
      << static_cast<std::uint64_t>(nps) << " checksum 0x" << std::hex
      << checksum << std::dec << ' ' << (ok ? "ok" : "node_mismatch")
      << '\n';

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
