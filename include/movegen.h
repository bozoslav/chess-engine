#pragma once

#include <vector>

#include "board.h"

std::vector<Move> genLegalMoves(Board& board);
int perft(Board& board, int depth);