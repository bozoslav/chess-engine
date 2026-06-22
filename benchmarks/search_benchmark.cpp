#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>

#include "board.h"
#include "nnue.h"
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
  bool useLateMoveReductions;
  bool useStaticExchangeEvaluation;
  bool useFutilityPruning;
  bool useLateMovePruning;
  bool useSingularExtensions = false;
  bool useProbCut = false;
  bool useReverseFutilityPruning = false;
  bool useRazoring = false;
  bool useInternalIterativeReduction = false;
  bool useCorrectionHistory = false;
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
  double meanCounterMoveUses;
  double meanHistoryMoveUses;
  double meanContinuationHistoryUses;
  double meanCaptureHistoryUses;
  double meanQuietCutoffs;
  double meanPVSResearches;
  double meanAspirationResearches;
  double meanNullMoveAttempts;
  double meanNullMovePrunes;
  double meanLMRAttempts;
  double meanLMRResearches;
  double meanSeePrunes;
  double meanFutilityPrunes;
  double meanLateMovePrunes;
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
  limits.useLateMoveReductions = testCase.useLateMoveReductions;
  limits.useStaticExchangeEvaluation = testCase.useStaticExchangeEvaluation;
  limits.useFutilityPruning = testCase.useFutilityPruning;
  limits.useLateMovePruning = testCase.useLateMovePruning;
  limits.useSingularExtensions = testCase.useSingularExtensions;
  limits.useProbCut = testCase.useProbCut;
  limits.useReverseFutilityPruning = testCase.useReverseFutilityPruning;
  limits.useRazoring = testCase.useRazoring;
  limits.useInternalIterativeReduction =
      testCase.useInternalIterativeReduction;
  limits.useCorrectionHistory = testCase.useCorrectionHistory;
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
    return {&testCase, first, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0,       0.0,   0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
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
  double totalCounterMoveUses = static_cast<double>(first.counterMoveUses);
  double totalHistoryMoveUses = static_cast<double>(first.historyMoveUses);
  double totalContinuationHistoryUses =
      static_cast<double>(first.continuationHistoryUses);
  double totalCaptureHistoryUses =
      static_cast<double>(first.captureHistoryUses);
  double totalQuietCutoffs = static_cast<double>(first.quietCutoffs);
  double totalPVSResearches = static_cast<double>(first.pvsResearches);
  double totalAspirationResearches =
      static_cast<double>(first.aspirationResearches);
  double totalNullMoveAttempts = static_cast<double>(first.nullMoveAttempts);
  double totalNullMovePrunes = static_cast<double>(first.nullMovePrunes);
  double totalLMRAttempts = static_cast<double>(first.lmrAttempts);
  double totalLMRResearches = static_cast<double>(first.lmrResearches);
  double totalSeePrunes = static_cast<double>(first.seePrunes);
  double totalFutilityPrunes = static_cast<double>(first.futilityPrunes);
  double totalLateMovePrunes = static_cast<double>(first.lateMovePrunes);

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
              totalCounterMoveUses / run,
              totalHistoryMoveUses / run,
              totalContinuationHistoryUses / run,
              totalCaptureHistoryUses / run,
              totalQuietCutoffs / run,
              totalPVSResearches / run,
              totalAspirationResearches / run,
              totalNullMoveAttempts / run,
              totalNullMovePrunes / run,
              totalLMRAttempts / run,
              totalLMRResearches / run,
              totalSeePrunes / run,
              totalFutilityPrunes / run,
              totalLateMovePrunes / run,
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
    totalCounterMoveUses += static_cast<double>(result.counterMoveUses);
    totalHistoryMoveUses += static_cast<double>(result.historyMoveUses);
    totalContinuationHistoryUses +=
        static_cast<double>(result.continuationHistoryUses);
    totalCaptureHistoryUses += static_cast<double>(result.captureHistoryUses);
    totalQuietCutoffs += static_cast<double>(result.quietCutoffs);
    totalPVSResearches += static_cast<double>(result.pvsResearches);
    totalAspirationResearches +=
        static_cast<double>(result.aspirationResearches);
    totalNullMoveAttempts += static_cast<double>(result.nullMoveAttempts);
    totalNullMovePrunes += static_cast<double>(result.nullMovePrunes);
    totalLMRAttempts += static_cast<double>(result.lmrAttempts);
    totalLMRResearches += static_cast<double>(result.lmrResearches);
    totalSeePrunes += static_cast<double>(result.seePrunes);
    totalFutilityPrunes += static_cast<double>(result.futilityPrunes);
    totalLateMovePrunes += static_cast<double>(result.lateMovePrunes);
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
          totalCounterMoveUses / runs,
          totalHistoryMoveUses / runs,
          totalContinuationHistoryUses / runs,
          totalCaptureHistoryUses / runs,
          totalQuietCutoffs / runs,
          totalPVSResearches / runs,
          totalAspirationResearches / runs,
          totalNullMoveAttempts / runs,
          totalNullMovePrunes / runs,
          totalLMRAttempts / runs,
          totalLMRResearches / runs,
          totalSeePrunes / runs,
          totalFutilityPrunes / runs,
          totalLateMovePrunes / runs,
          runs,
          true};
}

double nodesPerSecond(double nodes, double seconds) {
  if (seconds == 0.0) return 0.0;
  return nodes / seconds;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: chess_engine_search_benchmark <network.nnue>\n";
    return 2;
  }
  if (!nnue::loadNetwork(argv[1])) {
    std::cerr << "failed to load Stockfish NNUE: " << nnue::lastError() << '\n';
    return 1;
  }

  constexpr int kRunsPerCase = 3;
  constexpr const char* kStartFen =
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
  constexpr const char* kKiwipeteFen =
      "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/"
      "PPPBBPPP/R3K2R w KQkq - 0 1";
  constexpr const char* kOpenKingFen = "4k3/8/8/8/8/5q2/8/4KQ2 w - - 0 1";

  constexpr BenchmarkCase cases[] = {
      {"startpos_tt_plain", kStartFen, 5, true, false, false, false, false,
       false, false, false, false},
      {"startpos_tt_ordered_ab", kStartFen, 5, true, true, false, false, false,
       false, false, false, false},
      {"startpos_tt_ordered_pvs_asp_no_null", kStartFen, 5, true, true, true,
       true, false, false, false, false, false},
      {"startpos_tt_ordered_pvs_asp_null", kStartFen, 5, true, true, true, true,
       true, false, false, false, false},
      {"startpos_tt_ordered_pvs_asp_null_lmr", kStartFen, 5, true, true, true,
       true, true, true, false, false, false},
      {"startpos_tt_ordered_pvs_asp_null_lmr_see_selective", kStartFen, 5, true,
       true, true, true, true, true, true, true, true},
      {"kiwipete_tt_plain", kKiwipeteFen, 4, true, false, false, false, false,
       false, false, false, false},
      {"kiwipete_tt_ordered_ab", kKiwipeteFen, 4, true, true, false, false,
       false, false, false, false, false},
      {"kiwipete_tt_ordered_pvs_asp_no_null", kKiwipeteFen, 4, true, true, true,
       true, false, false, false, false, false},
      {"kiwipete_tt_ordered_pvs_asp_null", kKiwipeteFen, 4, true, true, true,
       true, true, false, false, false, false},
      {"kiwipete_tt_ordered_pvs_asp_null_lmr", kKiwipeteFen, 4, true, true,
       true, true, true, true, false, false, false},
      {"kiwipete_tt_ordered_pvs_asp_null_lmr_see_selective", kKiwipeteFen, 4,
       true, true, true, true, true, true, true, true, true},
      {"open_king_tt_plain", kOpenKingFen, 6, true, false, false, false, false,
       false, false, false, false},
      {"open_king_tt_ordered_ab", kOpenKingFen, 6, true, true, false, false,
       false, false, false, false, false},
      {"open_king_tt_ordered_pvs_asp_no_null", kOpenKingFen, 6, true, true,
       true, true, false, false, false, false},
      {"open_king_tt_ordered_pvs_asp_null", kOpenKingFen, 6, true, true, true,
       true, true, false, false, false, false},
      {"open_king_tt_ordered_pvs_asp_null_lmr", kOpenKingFen, 6, true, true,
       true, true, true, true, false, false, false},
      {"open_king_tt_ordered_pvs_asp_null_lmr_see_selective", kOpenKingFen, 6,
       true, true, true, true, true, true, true, true, true},
      {"startpos_elite_off", kStartFen, 8, true, true, true, true, true, true,
       true, true, true, false, false},
      {"startpos_probcut_only", kStartFen, 8, true, true, true, true, true,
       true, true, true, true, false, true},
      {"startpos_singular_only", kStartFen, 8, true, true, true, true, true,
       true, true, true, true, true, false},
      {"startpos_elite_on", kStartFen, 8, true, true, true, true, true, true,
       true, true, true, true, true},
      {"startpos_modern_base", kStartFen, 8, true, true, true, true, true,
       true, true, true, true, false, true, false, false, false, false},
      {"startpos_modern_rfp", kStartFen, 8, true, true, true, true, true,
       true, true, true, true, false, true, true, false, false, false},
      {"startpos_modern_rfp_razor", kStartFen, 8, true, true, true, true,
       true, true, true, true, true, false, true, true, true, false, false},
      {"startpos_modern_rfp_razor_iir", kStartFen, 8, true, true, true, true,
       true, true, true, true, true, false, true, true, true, true, false},
      {"startpos_modern_final", kStartFen, 8, true, true, true, true, true,
       true, true, true, true, true, true, true, true, true, false},
      {"startpos_modern_all", kStartFen, 8, true, true, true, true, true,
       true, true, true, true, false, true, true, true, true, true},
  };

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "case,depth,bestmove,score,best_nodes,best_seconds,"
               "best_nodes_per_second,mean_nodes,mean_seconds,"
               "mean_nodes_per_second,mean_tt_hits,mean_tt_cutoffs,"
               "mean_tt_move_uses,mean_killer_uses,mean_counter_uses,"
               "mean_history_uses,mean_continuation_uses,"
               "mean_capture_history_uses,"
               "mean_quiet_cutoffs,mean_pvs_researches,"
               "mean_aspiration_researches,mean_null_attempts,"
               "mean_null_prunes,mean_lmr_attempts,mean_lmr_researches,"
               "mean_see_prunes,mean_futility_prunes,mean_lmp_prunes,"
               "runs,status\n";

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
              << ',' << result.meanCounterMoveUses << ','
              << result.meanHistoryMoveUses << ','
              << result.meanContinuationHistoryUses << ','
              << result.meanCaptureHistoryUses << ',' << result.meanQuietCutoffs
              << ',' << result.meanPVSResearches << ','
              << result.meanAspirationResearches << ','
              << result.meanNullMoveAttempts << ',' << result.meanNullMovePrunes
              << ',' << result.meanLMRAttempts << ','
              << result.meanLMRResearches << ',' << result.meanSeePrunes << ','
              << result.meanFutilityPrunes << ',' << result.meanLateMovePrunes
              << ',' << result.runs << ','
              << (result.ok ? "ok" : "search_failed") << '\n';
  }

  std::cout << "total,-,-,0,0,0.000000,0.000000," << totalMeanNodes << ','
            << totalMeanSeconds << ','
            << nodesPerSecond(totalMeanNodes, totalMeanSeconds)
            << ",0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,"
               "0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,"
               "0.000000,0.000000,0.000000,0.000000,0.000000,"
            << kRunsPerCase << ',' << (ok ? "ok" : "search_failed") << '\n';

  return ok ? 0 : 1;
}
