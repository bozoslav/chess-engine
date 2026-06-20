#include "evaluate.h"

#include <string>

#include "nnue.h"

namespace {

std::string g_evalFile;

}  // namespace

int pieceValue(PieceType type) {
  switch (type) {
    case PieceType::Pawn:
      return 100;
    case PieceType::Knight:
      return 320;
    case PieceType::Bishop:
      return 330;
    case PieceType::Rook:
      return 500;
    case PieceType::Queen:
      return 900;
    case PieceType::King:
    case PieceType::None:
    default:
      return 0;
  }
}

int evaluate(const Board& board) {
  int score = 0;
  return nnue::evaluate(board, score) ? score : 0;
}

bool loadNnueFile(const char* path) {
  if (path == nullptr || path[0] == '\0') {
    clearNnueFile();
    return false;
  }
  if (!nnue::loadNetwork(path)) return false;
  g_evalFile = path;
  return true;
}

void clearNnueFile() {
  nnue::clearNetwork();
  g_evalFile.clear();
}

bool nnueReady() { return nnue::networkLoaded(); }

const char* nnueEvalFile() { return g_evalFile.c_str(); }

const char* nnueLoadError() { return nnue::lastError(); }
