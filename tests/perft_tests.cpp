#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>

#include "board.h"
#include "evaluate.h"
#include "movegen.h"
#include "nnue.h"
#include "search.h"
#include "see.h"
#include "transposition_table.h"
#include "uci.h"

namespace {

constexpr const char* kStartFen =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

#ifndef CHESS_ENGINE_TEST_EVALFILE
#define CHESS_ENGINE_TEST_EVALFILE ""
#endif

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

bool containsRawMove(const MoveList& moves, Move move) {
  for (const Move candidate : moves) {
    if (candidate.raw() == move.raw()) return true;
  }
  return false;
}

bool runStagedMoveGenerationTests() {
  constexpr const char* kPositions[] = {
      kStartFen,
      "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/"
      "PPPBBPPP/R3K2R w KQkq - 0 1",
      "4k3/8/8/8/8/8/4r3/4K3 w - - 0 1",
      "7k/P7/8/8/8/8/8/7K w - - 0 1",
  };

  bool ok = true;
  for (const char* fen : kPositions) {
    Board board;
    ok &=
        expectBool("load staged movegen position", board.setFromFen(fen), true);

    MoveList all;
    MoveList captures;
    MoveList quiets;
    genLegalMoves(board, all);
    genLegalCaptures(board, captures);
    genLegalQuiets(board, quiets);

    ok &= expectBool(
        "staged movegen has no overflow",
        !all.overflowed() && !captures.overflowed() && !quiets.overflowed(),
        true);
    ok &= expectBool("captures and quiets partition all legal moves",
                     captures.size() + quiets.size() == all.size(), true);
    for (const Move move : captures) {
      ok &=
          expectBool("capture stage contains captures", move.isCapture(), true);
      ok &= expectBool("capture stage move is legal",
                       containsRawMove(all, move), true);
    }
    for (const Move move : quiets) {
      ok &=
          expectBool("quiet stage excludes captures", move.isCapture(), false);
      ok &= expectBool("quiet stage move is legal", containsRawMove(all, move),
                       true);
    }
    for (const Move move : all) {
      ok &= expectBool(
          "all legal moves appear in one staged list",
          containsRawMove(captures, move) || containsRawMove(quiets, move),
          true);
    }
  }

  Board indexed;
  ok &= expectBool("flat mailbox a1 matches coordinate access",
                   indexed.at(0) == indexed.at(7, 0), true);
  ok &= expectBool("flat mailbox h8 matches coordinate access",
                   indexed.at(63) == indexed.at(0, 7), true);
  return ok;
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

bool expectSeeScore(const char* name, const char* fen, std::string_view text,
                    int expected) {
  Board board;
  if (!board.setFromFen(fen)) {
    std::cout << "[FAIL] " << name << ": invalid FEN\n";
    return false;
  }

  Move move;
  bool ok = expectBool(name, moveFromUci(board, text, move), true);
  if (!ok) return false;

  const int actual = staticExchangeEvaluation(board, move);
  const bool thresholdExact = staticExchangeNonLosing(board, move, expected);
  const bool thresholdAbove =
      staticExchangeNonLosing(board, move, expected + 1);
  if (actual == expected && thresholdExact && !thresholdAbove) {
    std::cout << "[PASS] " << name << " SEE: " << actual << '\n';
    return true;
  }

  std::cout << "[FAIL] " << name << " SEE: expected " << expected << ", got "
            << actual << ", threshold_exact " << thresholdExact
            << ", threshold_above " << thresholdAbove << '\n';
  return false;
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

  ok &=
      expectBool("value-initialized move is sentinel", Move{}.raw() == 0, true);
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
  ok &= expectBool("start FEN serializes", start.toFen() == kStartFen, true);

  Board epFen;
  ok &= expectBool("load ep FEN",
                   epFen.setFromFen("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/"
                                    "RNBQKBNR w KQkq e6 0 2"),
                   true);
  ok &= expectBool("ep FEN preserves counters",
                   epFen.toFen() ==
                       "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/"
                       "RNBQKBNR w KQkq e6 0 2",
                   true);

  Board clocks;
  ok &= expectBool("load FEN clocks",
                   clocks.setFromFen("8/8/8/8/8/8/6k1/K7 w - - 99 42"),
                   true);
  ok &= expectBool("halfmove clock parsed", clocks.halfmoveClock(), 99);
  ok &= expectBool("fullmove number parsed", clocks.fullmoveNumber(), 42);
  ok &= expectBool("quiet move increments clock", applyUciMove(clocks, "a1b1"),
                   true);
  ok &= expectBool("quiet move reaches 50-move boundary",
                   clocks.isFiftyMoveDraw(), true);
  ok &= expectBool("white move preserves fullmove number",
                   clocks.fullmoveNumber(), 42);
  ok &= expectBool("undo clock move", clocks.undoMove(), true);
  ok &= expectBool("undo restores halfmove clock", clocks.halfmoveClock(), 99);
  ok &= expectBool("undo restores fullmove number", clocks.fullmoveNumber(),
                   42);

  Board blackClock;
  ok &= expectBool("load black clock",
                   blackClock.setFromFen("7k/8/8/8/8/8/8/K7 b - - 7 42"),
                   true);
  ok &= expectBool("black quiet move", applyUciMove(blackClock, "h8g8"), true);
  ok &= expectBool("black quiet increments halfmove",
                   blackClock.halfmoveClock(), 8);
  ok &= expectBool("black move increments fullmove", blackClock.fullmoveNumber(),
                   43);

  Board pawnClock;
  ok &= expectBool("load pawn clock",
                   pawnClock.setFromFen("7k/8/8/8/8/8/P7/7K w - - 99 42"),
                   true);
  ok &= expectBool("pawn clock move", applyUciMove(pawnClock, "a2a3"), true);
  ok &= expectBool("pawn move resets halfmove", pawnClock.halfmoveClock(), 0);

  Board invalidClock;
  ok &= expectBool("reject negative halfmove clock",
                   invalidClock.setFromFen("8/8/8/8/8/8/6k1/K7 w - - -1 1"),
                   false);
  ok &= expectBool("reject zero fullmove number",
                   invalidClock.setFromFen("8/8/8/8/8/8/6k1/K7 w - - 0 0"),
                   false);

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
  ok &= expectBool("null move preserves halfmove clock",
                   nullMoveStart.halfmoveClock(), 0);
  ok &= expectBool("null move preserves fullmove number",
                   nullMoveStart.fullmoveNumber(), 1);
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

bool runEndgameRuleTests() {
  bool ok = true;

  Board bareKings;
  ok &= expectBool("load bare kings",
                   bareKings.setFromFen("7k/8/8/8/8/8/8/K7 w - - 0 1"),
                   true);
  ok &= expectBool("bare kings insufficient", bareKings.isInsufficientMaterial(),
                   true);

  Board bishopOnly;
  ok &= expectBool("load bishop versus king",
                   bishopOnly.setFromFen("7k/8/8/8/8/8/8/K1B5 w - - 0 1"),
                   true);
  ok &= expectBool("bishop versus king insufficient",
                   bishopOnly.isInsufficientMaterial(), true);

  Board bishopKnight;
  ok &= expectBool(
      "load bishop and knight",
      bishopKnight.setFromFen("7k/8/8/8/8/8/8/K1BN4 w - - 0 1"), true);
  ok &= expectBool("bishop and knight has mating material",
                   bishopKnight.isInsufficientMaterial(), false);

  Board sameColorBishops;
  ok &= expectBool(
      "load same-color bishops",
      sameColorBishops.setFromFen("5b1k/8/8/8/8/8/8/K1B5 w - - 0 1"), true);
  ok &= expectBool("same-color bishops insufficient",
                   sameColorBishops.isInsufficientMaterial(), true);

  Board oppositeColorBishops;
  ok &= expectBool(
      "load opposite-color bishops",
      oppositeColorBishops.setFromFen("2b4k/8/8/8/8/8/8/K1B5 w - - 0 1"),
      true);
  ok &= expectBool("opposite-color bishops retain mating material",
                   oppositeColorBishops.isInsufficientMaterial(), false);

  Board mateAtBoundary;
  ok &= expectBool(
      "load mate before 50-move boundary",
      mateAtBoundary.setFromFen("7k/8/5KQ1/8/8/8/8/8 w - - 99 1"), true);
  clearSearchState();
  const SearchResult mateResult = searchBestMove(mateAtBoundary, {2});
  ok &= expectBool("mate overrides 50-move boundary", mateResult.score > 29000,
                   true);

  std::ifstream network(CHESS_ENGINE_TEST_EVALFILE, std::ios::binary);
  if (network.good()) {
    network.close();
    ok &= expectBool("load NNUE for rule50 conversion test",
                     loadNnueFile(CHESS_ENGINE_TEST_EVALFILE), true);
    Board conversion;
    ok &= expectBool(
        "load conversion position",
        conversion.setFromFen(
            "8/8/5pp1/7p/5BkP/6P1/2r3K1/8 b - - 99 152"),
        true);
    clearSearchState();
    const SearchResult conversionResult = searchBestMove(conversion, {2});
    ok &= expectBool("conversion search has a move", conversionResult.hasBestMove,
                     true);
    ok &= expectBool(
        "conversion search chooses a zeroing pawn move",
        conversionResult.hasBestMove &&
            pieceType(conversion.at(conversionResult.bestMove.fromSquare())) ==
                PieceType::Pawn,
        true);
    ok &= expectBool("conversion remains winning", conversionResult.score > 0,
                     true);
    clearNnueFile();
  }

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
      "setoption name EvalFile value <empty>\n"
      "setoption name SingularExtensions value false\n"
      "setoption name MoveOverhead value 250\n"
      "setoption name EvalFile value /tmp/chess_engine_missing.nnue\n"
      "position startpos moves e2e4 e7e5\n"
      "fen\n"
      "nnueeval\n"
      "eval\n"
      "go depth 2\n"
      "go movetime 1\n"
      "go perft 2\n"
      "quit\n");
  std::ostringstream output;
  runUci(input, output);
  const std::string text = output.str();
  ok &= expectTextContains("uci id", text, "id name chess_engine");
  ok &=
      expectTextContains("uci eval file option", text, "option name EvalFile");
  ok &= expectTextContains("uci singular option", text,
                           "option name SingularExtensions type check");
  ok &= expectTextContains("uci move overhead option", text,
                           "option name MoveOverhead type spin");
  ok &= expectTextContains("uci move overhead set", text,
                           "MoveOverhead set to 250 ms");
  ok &= expectTextContains("uci ok", text, "uciok");
  ok &= expectTextContains("uci ready", text, "readyok");
  ok &= expectTextContains("uci bad nnue file", text,
                           "failed to load Stockfish NNUE");
  ok &= expectTextContains("uci fen", text,
                           "fen rnbqkbnr/pppp1ppp/8/4p3/4P3/8/"
                           "PPPP1PPP/RNBQKBNR w KQkq e6 0 2");
  ok &= expectTextContains("uci nnue eval", text, "nnueeval loaded 0 score 0");
  ok &= expectTextContains("uci eval command", text,
                           "NNUE evaluation 0 (side to move, internal units)");
  ok &= expectTextContains("uci depth 1 info", text, "info depth 1");
  ok &= expectTextContains("uci depth 2 info", text, "info depth 2");
  ok &= expectTextContains("uci score", text, "score cp ");
  ok &= expectTextContains("uci pv", text, " pv ");
  ok &= expectTextContains("uci quiet stats", text, "quiet_cutoffs ");
  ok &= expectTextContains("uci counter stats", text, "counter_uses ");
  ok &=
      expectTextContains("uci continuation stats", text, "continuation_uses ");
  ok &= expectTextContains("uci capture history stats", text,
                           "capture_history_uses ");
  ok &= expectTextContains("uci pvs stats", text, "pvs_researches ");
  ok &= expectTextContains("uci aspiration stats", text,
                           "aspiration_researches ");
  ok &= expectTextContains("uci null stats", text, "null_attempts ");
  ok &= expectTextContains("uci lmr stats", text, "lmr_attempts ");
  ok &= expectTextContains("uci see stats", text, "see_prunes ");
  ok &= expectTextContains("uci futility stats", text, "futility_prunes ");
  ok &= expectTextContains("uci lmp stats", text, "lmp_prunes ");
  ok &= expectTextContains("uci singular stats", text, "singular_searches ");
  ok &= expectTextContains("uci probcut stats", text, "probcut_attempts ");
  ok &= expectTextContains("uci rfp stats", text, "rfp_prunes ");
  ok &= expectTextContains("uci razor stats", text, "razor_attempts ");
  ok &= expectTextContains("uci iir stats", text, "iir_reductions ");
  ok &= expectTextContains("uci bestmove", text, "bestmove ");
  ok &= expectTextContains("uci perft", text, "perft depth 2 nodes ");

  std::istringstream stopInput(
      "position startpos\n"
      "go infinite\n"
      "isready\n"
      "stop\n"
      "quit\n");
  std::ostringstream stopOutput;
  runUci(stopInput, stopOutput);
  const std::string stopText = stopOutput.str();
  ok &= expectTextContains("uci async ready while searching", stopText,
                           "readyok");
  ok &= expectTextContains("uci async stop bestmove", stopText, "bestmove ");
  ok &= expectBool("uci async stop has legal fallback",
                   stopText.find("bestmove 0000") == std::string::npos, true);

  std::ostringstream benchOutput;
  ok &= expectBool("bench command", runBench(benchOutput), true);
  ok &= expectTextContains("bench total", benchOutput.str(), "bench total");

  return ok;
}

bool runNnueTests() {
  bool ok = true;

  clearNnueFile();
  Board start;
  int score = 123;
  ok &= expectBool("nnue unavailable before load", nnue::evaluate(start, score),
                   false);
  ok &= expectBool("neutral evaluation without network", evaluate(start) == 0,
                   true);

  constexpr const char* kInvalidPath = "/tmp/chess_engine_invalid.nnue";
  {
    std::ofstream output(kInvalidPath, std::ios::binary);
    output.write("bad", 3);
  }
  ok &= expectBool("reject malformed Stockfish network",
                   loadNnueFile(kInvalidPath), false);
  ok &=
      expectBool("malformed network leaves NNUE disabled", nnueReady(), false);
  ok &= expectBool("malformed network reports an error",
                   std::string_view(nnueLoadError()).empty(), false);
  std::remove(kInvalidPath);

  std::ifstream network(CHESS_ENGINE_TEST_EVALFILE, std::ios::binary);
  if (network.good()) {
    network.close();
    ok &= expectBool("load Stockfish-format integration network",
                     loadNnueFile(CHESS_ENGINE_TEST_EVALFILE), true);
    Board incremental;
    constexpr std::string_view moves[] = {
        "e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6",
        "b5a4", "g8f6", "e1g1", "f8e7", "f1e1", "b7b5",
    };
    for (std::string_view move : moves) {
      ok &= expectBool("NNUE integration move", applyUciMove(incremental, move),
                       true);
      int incrementalScore = 0;
      int rebuiltScore = 0;
      ok &= expectBool("incremental Stockfish NNUE evaluation",
                       nnue::evaluate(incremental, incrementalScore), true);
      nnue::resetAccumulatorCache();
      ok &= expectBool("rebuilt Stockfish NNUE evaluation",
                       nnue::evaluate(incremental, rebuiltScore), true);
      ok &= expectBool("incremental NNUE equals rebuild",
                       incrementalScore == rebuiltScore, true);
    }

    Board siblingTransitions;
    int rootScore = 0;
    ok &= expectBool("seed sibling NNUE accumulator",
                     nnue::evaluate(siblingTransitions, rootScore), true);
    ok &= expectBool("first sibling move",
                     applyUciMove(siblingTransitions, "e2e4"), true);
    int firstSiblingScore = 0;
    ok &= expectBool("evaluate first sibling",
                     nnue::evaluate(siblingTransitions, firstSiblingScore),
                     true);
    nnue::rewindAccumulator(siblingTransitions);
    ok &= expectBool("undo first sibling", siblingTransitions.undoMove(), true);
    ok &= expectBool("second sibling move",
                     applyUciMove(siblingTransitions, "d2d4"), true);
    int siblingIncrementalScore = 0;
    ok &= expectBool(
        "incrementally transform between siblings",
        nnue::evaluate(siblingTransitions, siblingIncrementalScore), true);
    nnue::resetAccumulatorCache();
    int siblingRebuiltScore = 0;
    ok &= expectBool("rebuild second sibling",
                     nnue::evaluate(siblingTransitions, siblingRebuiltScore),
                     true);
    ok &= expectBool("sibling NNUE transition equals rebuild",
                     siblingIncrementalScore == siblingRebuiltScore, true);

    Board randomized;
    nnue::resetAccumulatorCache();
    int seededScore = 0;
    ok &= expectBool("seed randomized NNUE accumulator",
                     nnue::evaluate(randomized, seededScore), true);
    std::uint32_t randomState = 0x9e3779b9U;
    for (int step = 0; step < 512; ++step) {
      MoveList legalMoves;
      genLegalMoves(randomized, legalMoves);
      if (legalMoves.empty()) {
        randomized = Board();
        nnue::resetAccumulatorCache();
        ok &= expectBool("reseed randomized NNUE accumulator",
                         nnue::evaluate(randomized, seededScore), true);
        continue;
      }

      randomState ^= randomState << 13U;
      randomState ^= randomState >> 17U;
      randomState ^= randomState << 5U;
      const Move move = legalMoves[randomState % legalMoves.size()];
      if (!randomized.makeGeneratedMove(move)) {
        std::cerr << "FAILED: randomized NNUE move at step " << step << '\n';
        ok = false;
        break;
      }

      int incrementalScore = 0;
      int rebuiltScore = 0;
      if (!nnue::evaluate(randomized, incrementalScore)) {
        std::cerr << "FAILED: randomized incremental NNUE evaluation at step "
                  << step << '\n';
        ok = false;
        break;
      }
      nnue::resetAccumulatorCache();
      if (!nnue::evaluate(randomized, rebuiltScore)) {
        std::cerr << "FAILED: randomized rebuilt NNUE evaluation at step "
                  << step << '\n';
        ok = false;
        break;
      }
      if (incrementalScore != rebuiltScore) {
        std::cerr << "FAILED: randomized incremental NNUE parity at step "
                  << step << ": incremental " << incrementalScore
                  << ", rebuilt " << rebuiltScore << '\n';
        ok = false;
        break;
      }
    }
    clearNnueFile();
  }
  return ok;
}

bool runEvaluationAndSearchTests() {
  bool ok = true;

  TranspositionTable evalTable;
  evalTable.resize(1);
  evalTable.store(0x123456789abcdef0ULL, 7, 42,
                  TranspositionBound::Exact, {}, 321);
  TranspositionProbe evalProbe;
  ok &= expectBool("tt static eval probe",
                   evalTable.probe(0x123456789abcdef0ULL, evalProbe), true);
  ok &= expectBool("tt static eval present", evalProbe.hasStaticEval, true);
  ok &= expectBool("tt static eval quantization",
                   evalProbe.staticEval >= 304 && evalProbe.staticEval <= 336,
                   true);

  clearNnueFile();
  Board start;
  ok &= expectBool("start eval is equal", evaluate(start), 0);

  ok &= expectSeeScore("see free queen capture",
                       "4k3/8/8/8/8/5q2/8/4KQ2 w - - 0 1", "f1f3", 900);
  ok &= expectSeeScore("see bad queen capture",
                       "7k/8/2p5/3p4/4Q3/8/8/7K w - - 0 1", "e4d5", -800);
  ok &= expectSeeScore("see en passant capture",
                       "k7/8/8/3pP3/8/8/8/7K w - d6 0 1", "e5d6", 100);
  ok &= expectSeeScore("see promotion gain", "7k/P7/8/8/8/8/8/7K w - - 0 1",
                       "a7a8q", 800);

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

  clearSearchState();
  Board selectiveStart;
  SearchLimits selectiveLimits;
  selectiveLimits.depth = 6;
  selectiveLimits.useSingularExtensions = true;
  selectiveLimits.singularMinDepth = 4;
  selectiveLimits.useProbCut = false;
  const SearchResult selective =
      searchBestMove(selectiveStart, selectiveLimits);
  ok &= expectBool("elite selective search has best move",
                   selective.hasBestMove, true);
  ok &= expectBool("singular verification searches execute",
                   selective.singularSearches > 0, true);
  ok &= expectBool("singular extensions bounded by searches",
                   selective.singularExtensions <= selective.singularSearches,
                   true);

  Board probCutStart;
  SearchLimits probCutLimits;
  probCutLimits.depth = 5;
  probCutLimits.useTranspositionTable = false;
  probCutLimits.useNullMovePruning = false;
  probCutLimits.useSingularExtensions = false;
  probCutLimits.probCutMinDepth = 3;
  probCutLimits.probCutReduction = 2;
  probCutLimits.probCutMargin = 30000;
  const SearchResult probCut = searchBestMove(probCutStart, probCutLimits);
  ok &= expectBool("probcut search has best move", probCut.hasBestMove, true);
  ok &=
      expectBool("probcut searches execute", probCut.probCutAttempts > 0, true);
  ok &= expectBool("probcut prunes bounded by attempts",
                   probCut.probCutPrunes <= probCut.probCutAttempts, true);

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
  ok &= runStagedMoveGenerationTests();
  ok &= runRepetitionTests();
  ok &= runEndgameRuleTests();
  ok &= runUciProtocolTests();
  ok &= runNnueTests();
  ok &= runEvaluationAndSearchTests();
  ok &= runRuleSmokeTests();
  for (const PerftPosition& position : positions) {
    ok &= runPerftPosition(position);
  }

  if (!ok) return 1;

  std::cout << "All perft correctness checks passed.\n";
  return 0;
}
