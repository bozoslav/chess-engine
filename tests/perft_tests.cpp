#include <cstdint>
#include <iostream>

#include "board.h"
#include "movegen.h"

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

  bool ok = runRuleSmokeTests();
  for (const PerftPosition& position : positions) {
    ok &= runPerftPosition(position);
  }

  if (!ok) return 1;

  std::cout << "All perft correctness checks passed.\n";
  return 0;
}
