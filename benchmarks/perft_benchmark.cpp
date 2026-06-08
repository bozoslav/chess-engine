#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>

#include "board.h"
#include "movegen.h"

namespace {

struct BenchmarkCase {
  const char* name;
  const char* fen;
  int depth;
  std::uint64_t expectedNodes;
};

struct BenchmarkResult {
  const BenchmarkCase* testCase;
  std::uint64_t nodes;
  double bestSeconds;
  double meanSeconds;
  int runs;
};

std::uint64_t runPerftOnce(const BenchmarkCase& testCase, double& seconds) {
  Board board;
  if (!board.setFromFen(testCase.fen)) return 0;
  const auto start = std::chrono::steady_clock::now();
  const std::uint64_t nodes = perft(board, testCase.depth);
  const auto finish = std::chrono::steady_clock::now();
  const std::chrono::duration<double> elapsed = finish - start;
  seconds = elapsed.count();
  return nodes;
}

BenchmarkResult runBenchmark(const BenchmarkCase& testCase, int runs) {
  double firstSeconds = 0.0;
  const std::uint64_t firstNodes = runPerftOnce(testCase, firstSeconds);
  if (firstNodes == 0) {
    std::cout << "invalid FEN for benchmark case: " << testCase.name << '\n';
    return {&testCase, 0, 0.0, 0.0, 0};
  }

  std::uint64_t nodes = firstNodes;
  double bestSeconds = firstSeconds;
  double totalSeconds = firstSeconds;

  for (int run = 1; run < runs; ++run) {
    double seconds = 0.0;
    const std::uint64_t runNodes = runPerftOnce(testCase, seconds);
    if (runNodes != nodes) nodes = runNodes;
    if (seconds < bestSeconds) bestSeconds = seconds;
    totalSeconds += seconds;
  }

  return {&testCase, nodes, bestSeconds, totalSeconds / runs, runs};
}

double nodesPerSecond(std::uint64_t nodes, double seconds) {
  if (seconds == 0.0) return 0.0;
  return static_cast<double>(nodes) / seconds;
}

}  // namespace

int main() {
  constexpr int kRunsPerCase = 3;
  constexpr const char* kStartFen =
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

  constexpr BenchmarkCase cases[] = {
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

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "case,depth,nodes,best_seconds,best_nodes_per_second,"
               "mean_seconds,mean_nodes_per_second,runs,status\n";

  bool ok = true;
  std::uint64_t totalNodes = 0;
  double totalBestSeconds = 0.0;
  double totalMeanSeconds = 0.0;

  for (const BenchmarkCase& testCase : cases) {
    const BenchmarkResult result = runBenchmark(testCase, kRunsPerCase);
    const bool matches = result.nodes == testCase.expectedNodes;
    ok &= matches;
    totalNodes += result.nodes;
    totalBestSeconds += result.bestSeconds;
    totalMeanSeconds += result.meanSeconds;

    std::cout << testCase.name << ',' << testCase.depth << ',' << result.nodes
              << ',' << result.bestSeconds << ','
              << nodesPerSecond(result.nodes, result.bestSeconds) << ','
              << result.meanSeconds << ','
              << nodesPerSecond(result.nodes, result.meanSeconds) << ','
              << result.runs << ',' << (matches ? "ok" : "node_mismatch")
              << '\n';
  }

  std::cout << "total,-," << totalNodes << ',' << totalBestSeconds << ','
            << nodesPerSecond(totalNodes, totalBestSeconds) << ','
            << totalMeanSeconds << ','
            << nodesPerSecond(totalNodes, totalMeanSeconds) << ','
            << kRunsPerCase << ',' << (ok ? "ok" : "node_mismatch") << '\n';

  return ok ? 0 : 1;
}
