#pragma once

#include <iosfwd>
#include <string_view>

#include "board.h"

bool moveFromUci(const Board& board, std::string_view text, Move& move);
bool setPositionFromUci(Board& board, std::string_view command);
bool runBench(std::ostream& out);
void runUci(std::istream& in, std::ostream& out);
