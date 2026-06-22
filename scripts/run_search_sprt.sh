#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export ENGINE_NEW="${ENGINE_NEW:-${repo_root}/cmake-build-release/chess_engine}"
export ENGINE_OLD="${ENGINE_OLD:-${repo_root}/cmake-build-search-baseline/chess_engine}"
export GAMES="${GAMES:-10000}"
export CONCURRENCY="${CONCURRENCY:-4}"
export TC="${TC:-1+0.01}"
export THREADS="${THREADS:-1}"
export HASH_MB="${HASH_MB:-64}"
export OPENINGS="${OPENINGS:-${repo_root}/benchmarks/openings_smoke.epd}"
export SPRT_ELO0="${SPRT_ELO0:-0}"
export SPRT_ELO1="${SPRT_ELO1:-5}"
export SPRT_ALPHA="${SPRT_ALPHA:-0.05}"
export SPRT_BETA="${SPRT_BETA:-0.05}"

exec bash "${repo_root}/scripts/run_match.sh"
