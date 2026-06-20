#pragma once

#include "board.h"

int staticExchangeEvaluation(const Board& board, Move move);
bool staticExchangeNonLosing(const Board& board, Move move, int threshold = 0);
