#include <iostream>

#include "board.h"
#include "movegen.h"

void printResult(const char* testName, bool actual, bool expected) {
  std::cout << testName << ": " << ((actual == expected) ? "PASS" : "FAIL")
            << '\n';
}

int main() {
  Board legalKnight;
  printResult("Legal knight move", legalKnight.makeMove({7, 1, 5, 2}), true);

  Board illegalKnight;
  printResult("Illegal knight move", illegalKnight.makeMove({7, 1, 5, 1}),
              false);

  Board wrongSide;
  printResult("Wrong side move", wrongSide.makeMove({1, 3, 2, 3}), false);

  Board ownPiece;
  printResult("Move onto own piece", ownPiece.makeMove({7, 0, 6, 0}), false);

  Board blockedRook;
  printResult("Blocked rook move", blockedRook.makeMove({7, 0, 4, 0}), false);

  Board simplePawn;
  printResult("Simple pawn move", simplePawn.makeMove({6, 3, 5, 3}), true);

  Board checkTest;
  printResult("Prep white pawn", checkTest.makeMove({6, 4, 5, 4}), true);
  printResult("Prep black pawn", checkTest.makeMove({1, 3, 3, 3}), true);
  printResult("Bishop gives check", checkTest.makeMove({7, 5, 3, 1}), true);
  printResult("Black king in check", checkTest.isKingInCheck(), true);
  printResult("Reject move in check", checkTest.makeMove({1, 0, 2, 0}), false);
  printResult("Block the check", checkTest.makeMove({1, 2, 2, 2}), true);
  printResult("Black king safe again", checkTest.isKingInCheck(), false);

  Board undoTest;
  printResult("Undo without moves", undoTest.undoMove(), false);
  printResult("Make move before undo", undoTest.makeMove({7, 1, 5, 2}), true);
  printResult("Undo last move", undoTest.undoMove(), true);
  printResult("Move again after undo", undoTest.makeMove({7, 1, 5, 2}), true);

  Board basic;
  std::cout << perft(basic, 1) << "\n";
  std::cout << perft(basic, 2) << "\n";
  std::cout << perft(basic, 3) << "\n";
  std::cout << perft(basic, 4) << "\n";
  std::cout << perft(basic, 5) << "\n";
}
