#pragma once

#include <string_view>

#include "board.h"

bool moveFromUci(const Board& board, std::string_view text, Move& move);
