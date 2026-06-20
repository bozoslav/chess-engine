#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <string_view>

#include "board.h"
#include "nnue.h"
#include "uci.h"

namespace {

constexpr const char* kStartFen =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

constexpr const char* kFenCases[] = {
    kStartFen,
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/"
    "PPPBBPPP/R3K2R w KQkq - 0 1",
    "4rrk1/pp3ppp/2n1b3/3qP3/3P4/2PB1N2/PP3PPP/R2QR1K1 w - - 0 18",
    "2rq1rk1/pb3ppp/1p2pn2/2bp4/3N4/2PBPN2/PPQ2PPP/R1B2RK1 b - - 3 12",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
};

constexpr std::string_view kMoveLine[] = {
    "e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6",
    "e1g1", "f8e7", "f1e1", "b7b5", "a4b3", "d7d6", "c2c3", "e8g8",
    "h2h3", "c6b8", "d2d4", "b8d7", "b1d2", "c7c5", "d4d5", "c5c4",
};

double secondsSince(std::chrono::steady_clock::time_point start) {
  const auto finish = std::chrono::steady_clock::now();
  return std::chrono::duration<double>(finish - start).count();
}

void printResult(const char* name, std::uint64_t evals, double seconds,
                 std::int64_t checksum) {
  const double eps = seconds > 0.0 ? static_cast<double>(evals) / seconds : 0.0;
  std::cout << name << ',' << evals << ',' << seconds << ',' << eps << ','
            << checksum << '\n';
}

bool applyMove(Board& board, std::string_view text) {
  Move move;
  return moveFromUci(board, text, move) && board.makeGeneratedMove(move);
}

bool runFenBenchmark(std::uint64_t repeats) {
  Board boards[sizeof(kFenCases) / sizeof(kFenCases[0])];
  for (std::size_t index = 0; index < std::size(kFenCases); ++index) {
    if (!boards[index].setFromFen(kFenCases[index])) return false;
  }

  std::int64_t checksum = 0;
  std::uint64_t evals = 0;
  const auto start = std::chrono::steady_clock::now();
  for (std::uint64_t repeat = 0; repeat < repeats; ++repeat) {
    for (const Board& board : boards) {
      int score = 0;
      if (!nnue::evaluate(board, score)) return false;
      checksum += score;
      ++evals;
    }
  }

  printResult("fen_eval", evals, secondsSince(start), checksum);
  return true;
}

bool runIncrementalBenchmark(std::uint64_t repeats) {
  std::int64_t checksum = 0;
  std::uint64_t evals = 0;
  const auto start = std::chrono::steady_clock::now();
  for (std::uint64_t repeat = 0; repeat < repeats; ++repeat) {
    Board board;
    for (std::string_view text : kMoveLine) {
      if (!applyMove(board, text)) return false;
      int score = 0;
      if (!nnue::evaluate(board, score)) return false;
      checksum += score;
      ++evals;
    }
  }

  printResult("incremental_eval", evals, secondsSince(start), checksum);
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: chess_engine_nnue_benchmark <network.nnue> "
                 "[fen_repeats] [incremental_repeats]\n";
    return 2;
  }

  const std::uint64_t fenRepeats = argc >= 3 ? std::stoull(argv[2]) : 100000ULL;
  const std::uint64_t incrementalRepeats =
      argc >= 4 ? std::stoull(argv[3]) : 50000ULL;

  if (!nnue::loadNetwork(argv[1])) {
    std::cerr << "failed to load NNUE network: " << argv[1] << '\n';
    return 1;
  }

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "case,evals,seconds,evals_per_second,checksum\n";
  if (!runFenBenchmark(fenRepeats)) return 1;
  if (!runIncrementalBenchmark(incrementalRepeats)) return 1;
  return 0;
}
