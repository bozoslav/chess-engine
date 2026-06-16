#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cutechess="${CUTECHESS_CLI:-${repo_root}/external/bin/cutechess-cli}"
engine_new="${ENGINE_NEW:-${repo_root}/cmake-build-release/chess_engine}"
engine_old="${ENGINE_OLD:-}"
baseline_commit="${BASELINE_COMMIT:-HEAD}"
games="${GAMES:-100}"
concurrency="${CONCURRENCY:-1}"
tc="${TC:-10+0.1}"
timestamp="$(date +%Y%m%d-%H%M%S)"
out_dir="${repo_root}/docs/rating/matches"
pgn="${out_dir}/match-${timestamp}.pgn"
log="${out_dir}/match-${timestamp}.log"

if [[ ! -x "${cutechess}" ]]; then
  echo "cutechess-cli not found at ${cutechess}" >&2
  echo "Run: bash scripts/setup_cutechess.sh" >&2
  exit 1
fi

if [[ ! -x "${engine_new}" ]]; then
  echo "new engine not found at ${engine_new}" >&2
  echo "Run: cmake --build cmake-build-release --target chess_engine" >&2
  exit 1
fi

if [[ -z "${engine_old}" ]]; then
  engine_old="$(bash "${repo_root}/scripts/build_baseline.sh" "${baseline_commit}")"
fi

if [[ ! -x "${engine_old}" ]]; then
  echo "baseline engine not found at ${engine_old}" >&2
  exit 1
fi

mkdir -p "${out_dir}"

"${cutechess}" \
  -engine name=current cmd="${engine_new}" proto=uci \
  -engine name=baseline cmd="${engine_old}" proto=uci \
  -each tc="${tc}" \
  -games "${games}" \
  -rounds 1 \
  -repeat \
  -concurrency "${concurrency}" \
  -recover \
  -pgnout "${pgn}" \
  2>&1 | tee "${log}"

echo "PGN: ${pgn}"
echo "Log: ${log}"
