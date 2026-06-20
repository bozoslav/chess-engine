#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GEN_BUILD="$ROOT/cmake-build-pgo-generate"
RELEASE_BUILD="$ROOT/cmake-build-release"
PROFILE_RUN="$ROOT/cmake-build-pgo-profiles/$(date +%Y%m%d-%H%M%S)"
RAW_PATTERN="$PROFILE_RUN/engine-%p.profraw"
PROFILE="$PROFILE_RUN/chess-engine.profdata"
NETWORK="$ROOT/data/stockfish/export/latest.nnue"

if [[ ! -f "$NETWORK" ]]; then
  echo "NNUE file does not exist: $NETWORK" >&2
  exit 1
fi

mkdir -p "$PROFILE_RUN"

cmake -S "$ROOT" -B "$GEN_BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF \
  -DCHESS_ENGINE_NATIVE_M4=ON \
  -DCHESS_ENGINE_IPO=ON \
  -DCHESS_ENGINE_PGO=GENERATE \
  -DCHESS_ENGINE_PGO_PROFILE="$RAW_PATTERN"

cmake --build "$GEN_BUILD" -j 10 --target \
  chess_engine_search_benchmark \
  chess_engine_nnue_benchmark \
  chess_engine_perft_benchmark \
  chess_engine_scaling_benchmark

"$GEN_BUILD/chess_engine_search_benchmark" "$NETWORK"
"$GEN_BUILD/chess_engine_nnue_benchmark" "$NETWORK"
"$GEN_BUILD/chess_engine_perft_benchmark" 3
"$GEN_BUILD/chess_engine_scaling_benchmark" "$NETWORK" 1 10000

RAW_PROFILES=("$PROFILE_RUN"/engine-*.profraw)
if [[ ! -e ${RAW_PROFILES[0]} ]]; then
  echo "PGO training produced no raw profiles under $PROFILE_RUN" >&2
  exit 1
fi
xcrun llvm-profdata merge -sparse "${RAW_PROFILES[@]}" -o "$PROFILE"

cmake -S "$ROOT" -B "$RELEASE_BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DCHESS_ENGINE_NATIVE_M4=ON \
  -DCHESS_ENGINE_IPO=ON \
  -DCHESS_ENGINE_PGO=USE \
  -DCHESS_ENGINE_PGO_PROFILE="$PROFILE"
cmake --build "$RELEASE_BUILD" -j 10
ctest --test-dir "$RELEASE_BUILD" --output-on-failure

echo "PGO-optimized engine: $RELEASE_BUILD/chess_engine"
echo "Merged profile: $PROFILE"
