#pragma once

#include <cstdint>

#include "board.h"

void genLegalMoves(Board& board, MoveList& moves);
std::uint64_t perft(Board& board, int depth);
