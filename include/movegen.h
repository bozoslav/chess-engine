#pragma once

#include <cstdint>
#include <vector>

#include "board.h"

std::vector<Move> genLegalMoves(Board& board);
std::uint64_t perft(Board& board, int depth);
