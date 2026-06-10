#include <cstdlib>
#include <iostream>
#include <string_view>

#include "board.h"
#include "movegen.h"
#include "uci.h"

namespace {

int parseDepth(const char* text, int fallback) {
  if (text == nullptr) return fallback;

  char* end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (end == text || value <= 0) return fallback;
  return static_cast<int>(value);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 1 && std::string_view(argv[1]) == "bench") {
    return runBench(std::cout) ? 0 : 1;
  }

  if (argc > 1 && std::string_view(argv[1]) == "perft") {
    const int depth = argc > 2 ? parseDepth(argv[2], 4) : 4;
    Board board;
    std::cout << "perft depth " << depth << " nodes " << perft(board, depth)
              << '\n';
    return 0;
  }

  runUci(std::cin, std::cout);
  return 0;
}
