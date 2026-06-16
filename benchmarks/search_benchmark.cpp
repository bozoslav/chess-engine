#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>

#include "board.h"
#include "search.h"

namespace {

struct BenchmarkCase {
  const char* name;
  const char* fen;
  int depth;
  bool useTT;
  bool useQuietOrdering;
  bool usePVS;
  bool useAspirationWindows;
  bool useNullMovePruning;
};

struct BenchmarkResult {
  const BenchmarkCase* testCase;
  SearchResult search;
  double bestSeconds;
  double meanSeconds;
  double meanNodes;
  double meanTTHits;
  double meanTTCutoffs;
  double meanTTMoveUses;
  double meanKillerMoveUses;
  double meanHistoryMoveUses;
  double meanQuietCutoffs;
  double meanPVSResearches;
  double meanAspirationResearches;
  double meanNullMoveAttempts;
  double meanNullMovePrunes;
  int runs;
  bool ok;
};

SearchResult runSearchOnce(const BenchmarkCase& testCase, double& seconds) {
  Board board;
  if (!board.setFromFen(testCase.fen)) {
    seconds = 0.0;
    return {};
  }

  clearSearchState();
  const auto start = std::chrono::steady_clock::now();
  SearchLimits limits;
  limits.depth = testCase.depth;
  limits.useTranspositionTable = testCase.useTT;
  limits.useQuietOrdering = testCase.useQuietOrdering;
  limits.usePVS = testCase.usePVS;
  limits.useAspirationWindows = testCase.useAspirationWindows;
  limits.useNullMovePruning = testCase.useNullMovePruning;
  const SearchResult result = searchBestMove(board, limits);
  const auto finish = std::chrono::steady_clock::now();
  const std::chrono::duration<double> elapsed = finish - start;
  seconds = elapsed.count();
  return result;
}

BenchmarkResult runBenchmark(const BenchmarkCase& testCase, int runs) {
  double firstSeconds = 0.0;
  const SearchResult first = runSearchOnce(testCase, firstSeconds);
  if (!first.hasBestMove) {
    return {&testCase, first, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0,
            0.0,       0.0,   0.0, 0.0, 0.0, 0.0, 0,   false};
  }

  SearchResult best = first;
  double bestSeconds = firstSeconds;
  double totalSeconds = firstSeconds;
  double totalNodes = static_cast<double>(first.nodes);
  double totalTTHits = static_cast<double>(first.ttHits);
  double totalTTCutoffs = static_cast<double>(first.ttCutoffs);
  double totalTTMoveUses = static_cast<double>(first.ttMoveUses);
  double totalKillerMoveUses = static_cast<double>(first.killerMoveUses);
  double totalHistoryMoveUses = static_cast<double>(first.historyMoveUses);
  double totalQuietCutoffs = static_cast<double>(first.quietCutoffs);
  double totalPVSResearches = static_cast<double>(first.pvsResearches);
  double totalAspirationResearches =
      static_cast<double>(first.aspirationResearches);
  double totalNullMoveAttempts = static_cast<double>(first.nullMoveAttempts);
  double totalNullMovePrunes = static_cast<double>(first.nullMovePrunes);

  for (int run = 1; run < runs; ++run) {
    double seconds = 0.0;
    const SearchResult result = runSearchOnce(testCase, seconds);
    if (!result.hasBestMove) {
      return {&testCase,
              result,
              bestSeconds,
              totalSeconds / run,
              totalNodes / run,
              totalTTHits / run,
              totalTTCutoffs / run,
              totalTTMoveUses / run,
              totalKillerMoveUses / run,
              totalHistoryMoveUses / run,
              totalQuietCutoffs / run,
              totalPVSResearches / run,
              totalAspirationResearches / run,
              totalNullMoveAttempts / run,
              totalNullMovePrunes / run,
              run,
              false};
    }

    if (seconds < bestSeconds) {
      bestSeconds = seconds;
      best = result;
    }

    totalSeconds += seconds;
    totalNodes += static_cast<double>(result.nodes);
    totalTTHits += static_cast<double>(result.ttHits);
    totalTTCutoffs += static_cast<double>(result.ttCutoffs);
    totalTTMoveUses += static_cast<double>(result.ttMoveUses);
    totalKillerMoveUses += static_cast<double>(result.killerMoveUses);
    totalHistoryMoveUses += static_cast<double>(result.historyMoveUses);
    totalQuietCutoffs += static_cast<double>(result.quietCutoffs);
    totalPVSResearches += static_cast<double>(result.pvsResearches);
    totalAspirationResearches +=
        static_cast<double>(result.aspirationResearches);
    totalNullMoveAttempts += static_cast<double>(result.nullMoveAttempts);
    totalNullMovePrunes += static_cast<double>(result.nullMovePrunes);
  }

  return {&testCase,
          best,
          bestSeconds,
          totalSeconds / runs,
          totalNodes / runs,
          totalTTHits / runs,
          totalTTCutoffs / runs,
          totalTTMoveUses / runs,
          totalKillerMoveUses / runs,
          totalHistoryMoveUses / runs,
          totalQuietCutoffs / runs,
          totalPVSResearches / runs,
          totalAspirationResearches / runs,
          totalNullMoveAttempts / runs,
          totalNullMovePrunes / runs,
          runs,
          true};
}

double nodesPerSecond(double nodes, double seconds) {
  if (seconds == 0.0) return 0.0;
  return nodes / seconds;
}

}  // namespace

int main() {
  constexpr int kRunsPerCase = 3;
  constexpr const char* kStartFen =
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

  constexpr BenchmarkCase cases[] = {
      {
          "startpos_tt_plain",
          kStartFen,
          5,
          true,
          false,
          false,
          false,
          false,
      },
      {
          "startpos_tt_ordered_ab",
          kStartFen,
          5,
          true,
          true,
          false,
          false,
          false,
      },
      {
          "startpos_tt_ordered_pvs_asp_no_null",
          kStartFen,
          5,
          true,
          true,
          true,
          true,
          false,
      },
      {
          "startpos_tt_ordered_pvs_asp_null",
          kStartFen,
          5,
          true,
          true,
          true,
          true,
          true,
      },
      {
          "kiwipete_tt_plain",
          "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/"
          "PPPBBPPP/R3K2R w KQkq - 0 1",
          4,
          true,
          false,
          false,
          false,
          false,
      },
      {
          "kiwipete_tt_ordered_ab",
          "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/"
          "PPPBBPPP/R3K2R w KQkq - 0 1",
          4,
          true,
          true,
          false,
          false,
          false,
      },
      {
          "kiwipete_tt_ordered_pvs_asp_no_null",
          "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/"
          "PPPBBPPP/R3K2R w KQkq - 0 1",
          4,
          true,
          true,
          true,
          true,
          false,
      },
      {
          "kiwipete_tt_ordered_pvs_asp_null",
          "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/"
          "PPPBBPPP/R3K2R w KQkq - 0 1",
          4,
          true,
          true,
          true,
          true,
          true,
      },
      {
          "open_king_tt_plain",
          "4k3/8/8/8/8/5q2/8/4KQ2 w - - 0 1",
          6,
          true,
          false,
          false,
          false,
          false,
      },
      {
          "open_king_tt_ordered_ab",
          "4k3/8/8/8/8/5q2/8/4KQ2 w - - 0 1",
          6,
          true,
          true,
          false,
          false,
          false,
      },
      {
          "open_king_tt_ordered_pvs_asp_no_null",
          "4k3/8/8/8/8/5q2/8/4KQ2 w - - 0 1",
          6,
          true,
          true,
          true,
          true,
          false,
      },
      {
          "open_king_tt_ordered_pvs_asp_null",
          "4k3/8/8/8/8/5q2/8/4KQ2 w - - 0 1",
          6,
          true,
          true,
          true,
          true,
          true,
      },
  };

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "case,depth,bestmove,score,best_nodes,best_seconds,"
               "best_nodes_per_second,mean_nodes,mean_seconds,"
               "mean_nodes_per_second,mean_tt_hits,mean_tt_cutoffs,"
               "mean_tt_move_uses,mean_killer_uses,mean_history_uses,"
               "mean_quiet_cutoffs,mean_pvs_researches,"
               "mean_aspiration_researches,mean_null_attempts,"
               "mean_null_prunes,runs,status\n";

  bool ok = true;
  double totalMeanNodes = 0.0;
  double totalMeanSeconds = 0.0;

  for (const BenchmarkCase& testCase : cases) {
    const BenchmarkResult result = runBenchmark(testCase, kRunsPerCase);
    ok &= result.ok;
    totalMeanNodes += result.meanNodes;
    totalMeanSeconds += result.meanSeconds;

    std::cout << testCase.name << ',' << testCase.depth << ',';
    if (result.search.hasBestMove) {
      std::cout << result.search.bestMove.toUci();
    } else {
      std::cout << "0000";
    }

    std::cout << ',' << result.search.score << ',' << result.search.nodes << ','
              << result.bestSeconds << ','
              << nodesPerSecond(static_cast<double>(result.search.nodes),
                                result.bestSeconds)
              << ',' << result.meanNodes << ',' << result.meanSeconds << ','
              << nodesPerSecond(result.meanNodes, result.meanSeconds) << ','
              << result.meanTTHits << ',' << result.meanTTCutoffs << ','
              << result.meanTTMoveUses << ',' << result.meanKillerMoveUses
              << ',' << result.meanHistoryMoveUses << ','
              << result.meanQuietCutoffs << ',' << result.meanPVSResearches
              << ',' << result.meanAspirationResearches << ','
              << result.meanNullMoveAttempts << ',' << result.meanNullMovePrunes
              << ',' << result.runs << ','
              << (result.ok ? "ok" : "search_failed") << '\n';
  }

  std::cout << "total,-,-,0,0,0.000000,0.000000," << totalMeanNodes << ','
            << totalMeanSeconds << ','
            << nodesPerSecond(totalMeanNodes, totalMeanSeconds)
            << ",0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,"
               "0.000000,0.000000,0.000000,0.000000,"
            << kRunsPerCase << ',' << (ok ? "ok" : "search_failed") << '\n';

  return ok ? 0 : 1;
}
