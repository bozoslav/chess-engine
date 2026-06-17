#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>

#include "board.h"
#include "evaluate.h"
#include "movegen.h"
#include "search.h"
#include "uci.h"

namespace {

constexpr const char* kStartFen =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

struct PerftExpectation {
  int depth;
  std::uint64_t nodes;
};

struct PerftPosition {
  const char* name;
  const char* fen;
  const PerftExpectation* expectations;
  int expectationCount;
};

bool expectBool(const char* name, bool actual, bool expected) {
  if (actual == expected) {
    std::cout << "[PASS] " << name << '\n';
    return true;
  }

  std::cout << "[FAIL] " << name << ": expected " << expected << ", got "
            << actual << '\n';
  return false;
}

bool containsMove(const MoveList& moves, int fromX, int fromY, int toX, int toY,
                  PieceType promo = PieceType::None) {
  for (const Move move : moves) {
    if (move.fromX() == fromX && move.fromY() == fromY && move.toX() == toX &&
        move.toY() == toY && move.promo() == promo) {
      return true;
    }
  }

  return false;
}

bool expectGeneratedMove(const char* name, const char* fen, int fromX,
                         int fromY, int toX, int toY, bool expected) {
  Board board;
  if (!board.setFromFen(fen)) {
    std::cout << "[FAIL] " << name << ": invalid FEN\n";
    return false;
  }

  MoveList moves;
  genLegalMoves(board, moves);
  const bool actual = containsMove(moves, fromX, fromY, toX, toY);
  return expectBool(name, actual, expected);
}

bool expectUciMove(const char* name, const char* fen, std::string_view text,
                   MoveFlag expectedFlag,
                   PieceType expectedPromotion = PieceType::None) {
  Board board;
  if (!board.setFromFen(fen)) {
    std::cout << "[FAIL] " << name << ": invalid FEN\n";
    return false;
  }

  Move move;
  bool ok = expectBool(name, moveFromUci(board, text, move), true);
  if (!ok) return false;

  ok &= expectBool("uci round trip", move.toUci() == text, true);
  ok &= expectBool("uci flag", move.flag() == expectedFlag, true);
  ok &= expectBool("uci promotion", move.promo() == expectedPromotion, true);
  return ok;
}

bool expectRejectedUciMove(const char* name, const char* fen,
                           std::string_view text) {
  Board board;
  if (!board.setFromFen(fen)) {
    std::cout << "[FAIL] " << name << ": invalid FEN\n";
    return false;
  }

  Move move;
  return expectBool(name, moveFromUci(board, text, move), false);
}

bool expectMakeUndoRestoresKey(const char* name, const char* fen,
                               std::string_view text) {
  Board board;
  if (!board.setFromFen(fen)) {
    std::cout << "[FAIL] " << name << ": invalid FEN\n";
    return false;
  }

  Move move;
  bool ok = expectBool(name, moveFromUci(board, text, move), true);
  if (!ok) return false;

  const std::uint64_t initialKey = board.key();
  ok &= expectBool("hash make move", board.makeMove(move), true);
  ok &= expectBool("hash changes after move", board.key() != initialKey, true);
  ok &= expectBool("hash undo move", board.undoMove(), true);
  ok &= expectBool("hash restored after undo", board.key() == initialKey, true);
  return ok;
}

bool applyUciMove(Board& board, std::string_view text) {
  Move move;
  return moveFromUci(board, text, move) && board.makeMove(move);
}

bool expectTextContains(const char* name, const std::string& text,
                        std::string_view needle) {
  return expectBool(name, text.find(needle) != std::string::npos, true);
}

bool runMoveEncodingAndHashTests() {
  bool ok = true;

  Move doublePush(6, 4, 4, 4, MoveFlag::DoublePawnPush);
  ok &= expectBool("move uci write", doublePush.toUci() == "e2e4", true);
  ok &= expectBool("move from square",
                   doublePush.fromSquare() == bitboard::squareFromCoords(6, 4),
                   true);
  ok &= expectBool("move double flag", doublePush.isDoublePawnPush(), true);
  ok &= expectBool("move size stays 16-bit",
                   sizeof(Move) == sizeof(std::uint16_t), true);

  Board start;
  Board fenStart;
  ok &= expectBool("load startpos for hash", fenStart.setFromFen(kStartFen),
                   true);
  ok &= expectBool("constructor and FEN hash match",
                   start.key() == fenStart.key(), true);

  Board blackToMove;
  ok &= expectBool("load black-to-move FEN",
                   blackToMove.setFromFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/"
                                          "RNBQKBNR b KQkq - 0 1"),
                   true);
  ok &= expectBool("side-to-move changes hash",
                   blackToMove.key() != fenStart.key(), true);

  ok &= expectUciMove("uci quiet knight", kStartFen, "g1f3", MoveFlag::Quiet);
  ok &= expectUciMove("uci double pawn push", kStartFen, "e2e4",
                      MoveFlag::DoublePawnPush);
  ok &= expectUciMove("uci capture", "8/8/8/3p4/4P3/8/8/k6K w - - 0 1", "e4d5",
                      MoveFlag::Capture);
  ok &= expectUciMove("uci castle", "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
                      "e1g1", MoveFlag::KingCastle);
  ok &= expectUciMove("uci en passant", "k7/8/8/3pP3/8/8/8/7K w - d6 0 1",
                      "e5d6", MoveFlag::EnPassant);
  ok &= expectUciMove("uci promotion", "8/P7/8/8/8/8/8/k6K w - - 0 1", "a7a8q",
                      MoveFlag::QueenPromotion, PieceType::Queen);
  ok &=
      expectUciMove("uci promotion capture", "1r6/P7/8/8/8/8/8/k6K w - - 0 1",
                    "a7b8q", MoveFlag::QueenPromotionCapture, PieceType::Queen);
  ok &= expectRejectedUciMove("reject illegal uci", kStartFen, "e2e5");

  ok &= expectMakeUndoRestoresKey("hash quiet make undo", kStartFen, "g1f3");
  ok &= expectMakeUndoRestoresKey("hash double push make undo", kStartFen,
                                  "e2e4");
  ok &= expectMakeUndoRestoresKey(
      "hash castle make undo", "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1");
  ok &= expectMakeUndoRestoresKey("hash en passant make undo",
                                  "k7/8/8/3pP3/8/8/8/7K w - d6 0 1", "e5d6");
  ok &= expectMakeUndoRestoresKey("hash promotion make undo",
                                  "8/P7/8/8/8/8/8/k6K w - - 0 1", "a7a8q");

  Board nullMoveStart;
  const std::uint64_t nullStartKey = nullMoveStart.key();
  ok &= expectBool("null move from start", nullMoveStart.makeNullMove(), true);
  ok &= expectBool("null move flips side",
                   nullMoveStart.sideToMove() == Color::Black, true);
  ok &= expectBool("null move changes key", nullMoveStart.key() != nullStartKey,
                   true);
  ok &= expectBool("undo null move", nullMoveStart.undoNullMove(), true);
  ok &= expectBool("undo null restores side",
                   nullMoveStart.sideToMove() == Color::White, true);
  ok &= expectBool("undo null restores key",
                   nullMoveStart.key() == nullStartKey, true);

  Board nullMoveEp;
  ok &= expectBool("null ep setup", applyUciMove(nullMoveEp, "e2e4"), true);
  const std::uint64_t epKey = nullMoveEp.key();
  ok &= expectBool("double push creates ep", nullMoveEp.hasEnPassant(), true);
  ok &= expectBool("null clears ep", nullMoveEp.makeNullMove(), true);
  ok &= expectBool("ep cleared after null", nullMoveEp.hasEnPassant(), false);
  ok &= expectBool("undo null restores ep", nullMoveEp.undoNullMove(), true);
  ok &= expectBool("ep restored after undo null", nullMoveEp.hasEnPassant(),
                   true);
  ok &= expectBool("ep key restored after undo null", nullMoveEp.key() == epKey,
                   true);

  Board nullInCheck;
  ok &= expectBool("load null in check",
                   nullInCheck.setFromFen("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1"),
                   true);
  ok &= expectBool("reject null while in check", nullInCheck.makeNullMove(),
                   false);

  return ok;
}

bool runRepetitionTests() {
  bool ok = true;

  Board board;
  ok &= expectBool("initial repetition count", board.repetitionCount(), 1);
  ok &= expectBool("initial not repeated", board.hasRepeatedPosition(), false);

  ok &= expectBool("rep move 1", applyUciMove(board, "g1f3"), true);
  ok &= expectBool("rep move 2", applyUciMove(board, "g8f6"), true);
  ok &= expectBool("rep move 3", applyUciMove(board, "f3g1"), true);
  ok &= expectBool("rep move 4", applyUciMove(board, "f6g8"), true);
  ok &= expectBool("one repetition cycle count", board.repetitionCount(), 2);
  ok &= expectBool("position repeated once", board.hasRepeatedPosition(), true);
  ok &= expectBool("not threefold yet", board.isThreefoldRepetition(), false);

  ok &= expectBool("rep move 5", applyUciMove(board, "g1f3"), true);
  ok &= expectBool("rep move 6", applyUciMove(board, "g8f6"), true);
  ok &= expectBool("rep move 7", applyUciMove(board, "f3g1"), true);
  ok &= expectBool("rep move 8", applyUciMove(board, "f6g8"), true);
  ok &= expectBool("threefold count", board.repetitionCount(), 3);
  ok &= expectBool("threefold repetition", board.isThreefoldRepetition(), true);

  const std::uint64_t repeatedKey = board.key();
  ok &= expectBool("undo repeated move", board.undoMove(), true);
  ok &= expectBool("undo changes key", board.key() != repeatedKey, true);
  ok &= expectBool("reapply repeated move", applyUciMove(board, "f6g8"), true);
  ok &= expectBool("reapply restores threefold", board.isThreefoldRepetition(),
                   true);

  Board fromCommand;
  ok &= expectBool(
      "position command with moves",
      setPositionFromUci(fromCommand,
                         "position startpos moves g1f3 g8f6 f3g1 f6g8"),
      true);
  ok &= expectBool("position command repetition",
                   fromCommand.hasRepeatedPosition(), true);

  return ok;
}

bool runUciProtocolTests() {
  bool ok = true;

  Board board;
  ok &= expectBool("set startpos command",
                   setPositionFromUci(board, "position startpos moves e2e4"),
                   true);
  ok &=
      expectBool("set startpos side", board.sideToMove() == Color::Black, true);

  ok &= expectBool(
      "set fen command",
      setPositionFromUci(board, "position fen 8/8/8/8/8/8/8/k6K w - - 0 1"),
      true);
  ok &= expectBool("reject bad position",
                   setPositionFromUci(board, "position startpos moves e2e5"),
                   false);

  std::istringstream input(
      "uci\n"
      "isready\n"
      "position startpos moves e2e4 e7e5\n"
      "go depth 2\n"
      "go movetime 1\n"
      "go perft 2\n"
      "quit\n");
  std::ostringstream output;
  runUci(input, output);
  const std::string text = output.str();
  ok &= expectTextContains("uci id", text, "id name chess_engine");
  ok &= expectTextContains("uci ok", text, "uciok");
  ok &= expectTextContains("uci ready", text, "readyok");
  ok &= expectTextContains("uci depth 1 info", text, "info depth 1");
  ok &= expectTextContains("uci depth 2 info", text, "info depth 2");
  ok &= expectTextContains("uci score", text, "score cp ");
  ok &= expectTextContains("uci pv", text, " pv ");
  ok &= expectTextContains("uci quiet stats", text, "quiet_cutoffs ");
  ok &= expectTextContains("uci pvs stats", text, "pvs_researches ");
  ok &= expectTextContains("uci aspiration stats", text,
                           "aspiration_researches ");
  ok &= expectTextContains("uci null stats", text, "null_attempts ");
  ok &= expectTextContains("uci lmr stats", text, "lmr_attempts ");
  ok &= expectTextContains("uci bestmove", text, "bestmove ");
  ok &= expectTextContains("uci perft", text, "perft depth 2 nodes ");

  std::ostringstream benchOutput;
  ok &= expectBool("bench command", runBench(benchOutput), true);
  ok &= expectTextContains("bench total", benchOutput.str(), "bench total");

  return ok;
}

bool runEvaluationAndSearchTests() {
  bool ok = true;

  Board start;
  ok &= expectBool("start eval is equal", evaluate(start), 0);

  Board whiteMaterial;
  ok &= expectBool("load white material edge",
                   whiteMaterial.setFromFen("4k3/8/8/8/8/8/8/4KQ2 w - - 0 1"),
                   true);
  ok &= expectBool("white material eval positive",
                   evaluate(whiteMaterial) > 800, true);

  Board blackToMoveMaterial;
  ok &= expectBool(
      "load black material edge",
      blackToMoveMaterial.setFromFen("4k3/8/8/8/8/8/8/4KQ2 b - - 0 1"), true);
  ok &= expectBool("black side eval negative",
                   evaluate(blackToMoveMaterial) < -800, true);

  Board passedPawn;
  Board blockedPassedPawn;
  ok &= expectBool("load passed pawn eval",
                   passedPawn.setFromFen("k7/8/p7/4P3/8/8/8/4K3 w - - 0 1"),
                   true);
  ok &= expectBool(
      "load blocked passed pawn eval",
      blockedPassedPawn.setFromFen("k7/8/3p4/4P3/8/8/8/4K3 w - - 0 1"), true);
  ok &= expectBool("passed pawn beats blocked pawn",
                   evaluate(passedPawn) > evaluate(blockedPassedPawn), true);

  Board connectedPawns;
  Board doubledPawns;
  ok &= expectBool(
      "load connected pawns",
      connectedPawns.setFromFen("4k3/8/8/8/8/8/2PP4/4K3 w - - 0 1"), true);
  ok &= expectBool("load doubled pawns",
                   doubledPawns.setFromFen("4k3/8/8/8/8/2P5/2P5/4K3 w - - 0 1"),
                   true);
  ok &= expectBool("connected pawns beat doubled isolated pawns",
                   evaluate(connectedPawns) > evaluate(doubledPawns), true);

  Board rookOpenFile;
  Board rookBlockedFile;
  ok &= expectBool("load rook open file",
                   rookOpenFile.setFromFen("4k3/8/8/8/8/8/1P6/R3K3 w - - 0 1"),
                   true);
  ok &= expectBool(
      "load rook blocked file",
      rookBlockedFile.setFromFen("4k3/8/8/8/8/8/P7/R3K3 w - - 0 1"), true);
  ok &= expectBool("rook open file scores higher",
                   evaluate(rookOpenFile) > evaluate(rookBlockedFile), true);

  Board captureQueen;
  ok &= expectBool("load queen capture search",
                   captureQueen.setFromFen("4k3/8/8/8/8/5q2/8/4KQ2 w - - 0 1"),
                   true);
  const SearchResult result = searchBestMove(captureQueen, {1});
  ok &= expectBool("search has best move", result.hasBestMove, true);
  ok &= expectBool("search captures queen", result.bestMove.toUci() == "f1f3",
                   true);
  ok &= expectBool("search records pv", result.pvLength > 0, true);
  ok &= expectBool("search pv starts with best move",
                   result.principalVariation[0].raw() == result.bestMove.raw(),
                   true);
  ok &= expectBool("search visits nodes", result.nodes > 0, true);
  ok &= expectBool("search stores tt entries", result.ttStores > 0, true);

  clearSearchState();
  Board firstTTSearch;
  Board secondTTSearch;
  ok &= expectBool("load tt first search", firstTTSearch.setFromFen(kStartFen),
                   true);
  ok &= expectBool("load tt second search",
                   secondTTSearch.setFromFen(kStartFen), true);
  const SearchResult first = searchBestMove(firstTTSearch, {3});
  const SearchResult second = searchBestMove(secondTTSearch, {3});
  ok &= expectBool("tt first search has best move", first.hasBestMove, true);
  ok &= expectBool("tt second search has best move", second.hasBestMove, true);
  ok &= expectBool("tt preserves best move",
                   second.bestMove.raw() == first.bestMove.raw(), true);
  ok &= expectBool("tt second search hits table", second.ttHits > 0, true);

  clearSearchState();
  Board orderedStart;
  SearchLimits orderingLimits;
  orderingLimits.depth = 3;
  const SearchResult ordered = searchBestMove(orderedStart, orderingLimits);
  ok &= expectBool("quiet ordering produces cutoffs", ordered.quietCutoffs > 0,
                   true);
  ok &= expectBool("quiet ordering records uses",
                   ordered.killerMoveUses > 0 || ordered.historyMoveUses > 0,
                   true);

  clearSearchState();
  Board alphaBetaStart;
  Board pvsStart;
  SearchLimits alphaBetaLimits;
  alphaBetaLimits.depth = 4;
  alphaBetaLimits.useTranspositionTable = false;
  alphaBetaLimits.usePVS = false;
  alphaBetaLimits.useAspirationWindows = false;
  alphaBetaLimits.useLateMoveReductions = false;
  SearchLimits pvsLimits = alphaBetaLimits;
  pvsLimits.usePVS = true;
  ok &= expectBool("load ab start", alphaBetaStart.setFromFen(kStartFen), true);
  ok &= expectBool("load pvs start", pvsStart.setFromFen(kStartFen), true);
  const SearchResult alphaBeta =
      searchBestMove(alphaBetaStart, alphaBetaLimits);
  const SearchResult pvs = searchBestMove(pvsStart, pvsLimits);
  ok &= expectBool("ab has best move", alphaBeta.hasBestMove, true);
  ok &= expectBool("pvs has best move", pvs.hasBestMove, true);
  ok &= expectBool("pvs preserves score", pvs.score == alphaBeta.score, true);
  ok &= expectBool("pvs records pv", pvs.pvLength > 0, true);
  ok &= expectBool("pvs pv starts with best move",
                   pvs.principalVariation[0].raw() == pvs.bestMove.raw(), true);

  clearSearchState();
  Board aspirationStart;
  SearchLimits aspirationLimits;
  aspirationLimits.depth = 3;
  aspirationLimits.useTranspositionTable = false;
  aspirationLimits.usePVS = false;
  aspirationLimits.useAspirationWindows = true;
  aspirationLimits.useLateMoveReductions = false;
  aspirationLimits.aspirationWindow = 1;
  const SearchResult aspiration =
      searchBestMove(aspirationStart, aspirationLimits);
  ok &= expectBool("aspiration has best move", aspiration.hasBestMove, true);
  ok &= expectBool("aspiration researches tight window",
                   aspiration.aspirationResearches > 0, true);

  clearSearchState();
  Board timedStart;
  SearchLimits timedLimits;
  timedLimits.depth = 64;
  timedLimits.timeLimitMs = 1;
  const SearchResult timed = searchBestMove(timedStart, timedLimits);
  ok &= expectBool("timed search has best move", timed.hasBestMove, true);
  ok &= expectBool("timed search visits nodes", timed.nodes > 0, true);

  clearSearchState();
  Board noNullStart;
  Board nullSearchStart;
  SearchLimits noNullLimits;
  noNullLimits.depth = 5;
  noNullLimits.useTranspositionTable = false;
  noNullLimits.useAspirationWindows = false;
  noNullLimits.useNullMovePruning = false;
  noNullLimits.useLateMoveReductions = false;
  SearchLimits nullLimits = noNullLimits;
  nullLimits.useNullMovePruning = true;
  const SearchResult noNull = searchBestMove(noNullStart, noNullLimits);
  const SearchResult nullSearch = searchBestMove(nullSearchStart, nullLimits);
  ok &= expectBool("null search has best move", nullSearch.hasBestMove, true);
  ok &= expectBool("null search preserves score",
                   nullSearch.score == noNull.score, true);
  ok &= expectBool("null search attempts pruning",
                   nullSearch.nullMoveAttempts > 0, true);

  clearSearchState();
  Board noLmrStart;
  Board lmrStart;
  SearchLimits noLmrLimits;
  noLmrLimits.depth = 5;
  noLmrLimits.useTranspositionTable = false;
  noLmrLimits.useAspirationWindows = false;
  noLmrLimits.useNullMovePruning = false;
  noLmrLimits.useLateMoveReductions = false;
  SearchLimits lmrLimits = noLmrLimits;
  lmrLimits.useLateMoveReductions = true;
  const SearchResult noLmr = searchBestMove(noLmrStart, noLmrLimits);
  const SearchResult lmr = searchBestMove(lmrStart, lmrLimits);
  ok &= expectBool("lmr baseline has best move", noLmr.hasBestMove, true);
  ok &= expectBool("lmr search has best move", lmr.hasBestMove, true);
  ok &= expectBool("lmr attempts reductions", lmr.lmrAttempts > 0, true);
  ok &= expectBool("lmr researches no more than attempts",
                   lmr.lmrResearches <= lmr.lmrAttempts, true);

  return ok;
}

bool runRuleSmokeTests() {
  bool ok = true;

  Board legalKnight;
  ok &=
      expectBool("legal knight move", legalKnight.makeMove({7, 1, 5, 2}), true);

  Board illegalKnight;
  ok &= expectBool("illegal knight move", illegalKnight.makeMove({7, 1, 5, 1}),
                   false);

  Board wrongSide;
  ok &= expectBool("wrong side move", wrongSide.makeMove({1, 3, 2, 3}), false);

  Board ownPiece;
  ok &=
      expectBool("move onto own piece", ownPiece.makeMove({7, 0, 6, 0}), false);

  Board blockedRook;
  ok &= expectBool("blocked rook move", blockedRook.makeMove({7, 0, 4, 0}),
                   false);

  Board simplePawn;
  ok &= expectBool("simple pawn move", simplePawn.makeMove({6, 3, 5, 3}), true);

  Board checkTest;
  ok &= expectBool("prep white pawn", checkTest.makeMove({6, 4, 5, 4}), true);
  ok &= expectBool("prep black pawn", checkTest.makeMove({1, 3, 3, 3}), true);
  ok &=
      expectBool("bishop gives check", checkTest.makeMove({7, 5, 3, 1}), true);
  ok &= expectBool("black king in check", checkTest.isKingInCheck(), true);
  ok &= expectBool("reject move in check", checkTest.makeMove({1, 0, 2, 0}),
                   false);
  ok &= expectBool("block the check", checkTest.makeMove({1, 2, 2, 2}), true);
  ok &= expectBool("black king safe again", checkTest.isKingInCheck(), false);

  Board undoTest;
  ok &= expectBool("undo without moves", undoTest.undoMove(), false);
  ok &= expectBool("make move before undo", undoTest.makeMove({7, 1, 5, 2}),
                   true);
  ok &= expectBool("undo last move", undoTest.undoMove(), true);
  ok &= expectBool("move again after undo", undoTest.makeMove({7, 1, 5, 2}),
                   true);

  Board fenTest;
  ok &= expectBool("load startpos FEN", fenTest.setFromFen(kStartFen), true);

  ok &= expectGeneratedMove("double pawn push blocks bishop check",
                            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/"
                            "R2Q1RK1 w kq - 0 1",
                            6, 3, 4, 3, true);

  ok &= expectGeneratedMove("reject en passant discovered check",
                            "k7/8/8/K2pP2r/8/8/8/8 w - d6 0 1", 3, 4, 2, 3,
                            false);

  return ok;
}

bool runPerftPosition(const PerftPosition& position) {
  bool ok = true;

  for (int i = 0; i < position.expectationCount; ++i) {
    Board board;
    if (!board.setFromFen(position.fen)) {
      std::cout << "[FAIL] " << position.name << ": invalid FEN\n";
      return false;
    }

    const PerftExpectation expectation = position.expectations[i];
    const std::uint64_t actual = perft(board, expectation.depth);
    if (actual != expectation.nodes) {
      std::cout << "[FAIL] " << position.name << " depth " << expectation.depth
                << ": expected " << expectation.nodes << ", got " << actual
                << '\n';
      ok = false;
    } else {
      std::cout << "[PASS] " << position.name << " depth " << expectation.depth
                << ": " << actual << '\n';
    }
  }

  return ok;
}

}  // namespace

int main() {
  static constexpr PerftExpectation startpos[] = {
      {1, 20},
      {2, 400},
      {3, 8902},
      {4, 197281},
  };

  static constexpr PerftExpectation kiwipete[] = {
      {1, 48},
      {2, 2039},
      {3, 97862},
  };

  static constexpr PerftExpectation endgame[] = {
      {1, 14},
      {2, 191},
      {3, 2812},
      {4, 43238},
  };

  static constexpr PerftExpectation promotionAndCastle[] = {
      {1, 6},
      {2, 264},
      {3, 9467},
  };

  static constexpr PerftPosition positions[] = {
      {
          "startpos",
          kStartFen,
          startpos,
          static_cast<int>(sizeof(startpos) / sizeof(startpos[0])),
      },
      {
          "kiwipete",
          "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/"
          "PPPBBPPP/R3K2R w KQkq - 0 1",
          kiwipete,
          static_cast<int>(sizeof(kiwipete) / sizeof(kiwipete[0])),
      },
      {
          "endgame",
          "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
          endgame,
          static_cast<int>(sizeof(endgame) / sizeof(endgame[0])),
      },
      {
          "promotion_and_castle",
          "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/"
          "R2Q1RK1 w kq - 0 1",
          promotionAndCastle,
          static_cast<int>(sizeof(promotionAndCastle) /
                           sizeof(promotionAndCastle[0])),
      },
  };

  bool ok = runMoveEncodingAndHashTests();
  ok &= runRepetitionTests();
  ok &= runUciProtocolTests();
  ok &= runEvaluationAndSearchTests();
  ok &= runRuleSmokeTests();
  for (const PerftPosition& position : positions) {
    ok &= runPerftPosition(position);
  }

  if (!ok) return 1;

  std::cout << "All perft correctness checks passed.\n";
  return 0;
}
