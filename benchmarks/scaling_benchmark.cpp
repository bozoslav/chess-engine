#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>

#include "board.h"
#include "nnue.h"
#include "search.h"

namespace {

int parsePositive(const char* text, int fallback) {
  const int value = std::atoi(text);
  return value > 0 ? value : fallback;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || argc > 4) {
    std::cerr << "usage: chess_engine_scaling_benchmark <network.nnue> "
                 "[threads] [milliseconds]\n";
    return 2;
  }
  if (!nnue::loadNetwork(argv[1])) {
    std::cerr << "failed to load Stockfish NNUE: " << nnue::lastError()
              << '\n';
    return 1;
  }

  const int threads = argc >= 3 ? parsePositive(argv[2], 1) : 1;
  const int milliseconds = argc >= 4 ? parsePositive(argv[3], 5000) : 5000;

  Board board;
  clearSearchState();
  SearchLimits limits;
  limits.depth = kSearchMaxPvLength;
  limits.threads = std::clamp(threads, 1, 128);
  limits.timeLimitMs = static_cast<std::uint64_t>(milliseconds);

  const auto start = std::chrono::steady_clock::now();
  const SearchResult result = searchBestMove(board, limits);
  const auto finish = std::chrono::steady_clock::now();
  const std::chrono::duration<double> elapsed = finish - start;
  const double nps = elapsed.count() > 0.0
                         ? static_cast<double>(result.nodes) / elapsed.count()
                         : 0.0;

  std::cout << std::fixed << std::setprecision(3)
            << "threads,requested_ms,wall_seconds,depth,nodes,nps,bestmove,"
               "score,stopped\n"
            << limits.threads << ',' << milliseconds << ',' << elapsed.count()
            << ',' << result.depth << ',' << result.nodes << ',' << nps << ','
            << (result.hasBestMove ? result.bestMove.toUci() : "0000") << ','
            << result.score << ',' << (result.stopped ? "yes" : "no") << '\n';

  return result.hasBestMove ? 0 : 1;
}
