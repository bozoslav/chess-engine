#pragma once

#include "types.h"

struct Move {
  int fromX;
  int fromY;
  int toX;
  int toY;
  PieceType promo = PieceType::None;
};
