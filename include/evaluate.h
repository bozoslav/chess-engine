#pragma once

#include "board.h"

// Material values are used only for move ordering and static exchange
// evaluation. Position evaluation itself is NNUE-only.
int pieceValue(PieceType type);
int evaluate(const Board& board);

bool loadNnueFile(const char* path);
void clearNnueFile();
bool nnueReady();
const char* nnueEvalFile();
const char* nnueLoadError();
