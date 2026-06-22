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
threads="${THREADS:-1}"
hash_mb="${HASH_MB:-64}"
openings="${OPENINGS:-${repo_root}/benchmarks/openings_smoke.epd}"
sprt_elo0="${SPRT_ELO0:-}"
sprt_elo1="${SPRT_ELO1:-}"
sprt_alpha="${SPRT_ALPHA:-0.05}"
sprt_beta="${SPRT_BETA:-0.05}"
current_singular="${CURRENT_SINGULAR_EXTENSIONS:-}"
baseline_singular="${BASELINE_SINGULAR_EXTENSIONS:-}"
current_move_overhead="${CURRENT_MOVE_OVERHEAD:-}"
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

extra_args=()
if [[ -n "${openings}" ]]; then
  [[ -f "${openings}" ]] || {
    echo "openings file not found: ${openings}" >&2
    exit 1
  }
  extra_args+=(
    -openings "file=${openings}" format=epd order=random policy=round
  )
fi

if [[ -n "${sprt_elo0}" || -n "${sprt_elo1}" ]]; then
  [[ -n "${sprt_elo0}" && -n "${sprt_elo1}" ]] || {
    echo "SPRT_ELO0 and SPRT_ELO1 must be set together" >&2
    exit 1
  }
  extra_args+=(
    -sprt "elo0=${sprt_elo0}" "elo1=${sprt_elo1}"
    "alpha=${sprt_alpha}" "beta=${sprt_beta}"
  )
fi

current_engine=(-engine name=current "cmd=${engine_new}" proto=uci)
baseline_engine=(-engine name=baseline "cmd=${engine_old}" proto=uci)
if [[ -n "${current_singular}" ]]; then
  current_engine+=("option.SingularExtensions=${current_singular}")
fi
if [[ -n "${current_move_overhead}" ]]; then
  current_engine+=("option.MoveOverhead=${current_move_overhead}")
fi
if [[ -n "${baseline_singular}" ]]; then
  baseline_engine+=("option.SingularExtensions=${baseline_singular}")
fi

"${cutechess}" \
  "${current_engine[@]}" \
  "${baseline_engine[@]}" \
  -each tc="${tc}" "option.Threads=${threads}" "option.Hash=${hash_mb}" \
  -games "${games}" \
  -rounds 1 \
  -repeat \
  -concurrency "${concurrency}" \
  "${extra_args[@]}" \
  -recover \
  -pgnout "${pgn}" \
  2>&1 | tee "${log}"

echo "PGN: ${pgn}"
echo "Log: ${log}"
