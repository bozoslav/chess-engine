#pragma once

#include <cstdint>

#include "board.h"

void genLegalMoves(const Board& board, MoveList& moves);
std::uint64_t perft(Board& board, int depth);
