#include <iostream>

#include "board.h"
#include "movegen.h"

int main() {
  Board board;
  std::cout << "chess_engine baseline ready\n";
  std::cout << "startpos perft(4): " << perft(board, 4) << '\n';
}
