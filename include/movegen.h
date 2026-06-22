#pragma once

#include <cstdint>

#include "board.h"

void genLegalMoves(const Board& board, MoveList& moves);
void genLegalCaptures(const Board& board, MoveList& moves);
void genLegalQuiets(const Board& board, MoveList& moves);
void appendLegalQuiets(const Board& board, MoveList& moves);
// Generates captures, en passant, and promotions. The position must not be in
// check; check evasions require genLegalMoves().
void genLegalNoisyMoves(const Board& board, MoveList& moves);
std::uint64_t perft(Board& board, int depth);
