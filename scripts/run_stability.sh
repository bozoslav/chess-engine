#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
log_file="${STABILITY_LOG:-${repo_root}/docs/rating/stability_log.md}"
baseline_commit="${BASELINE_COMMIT:-HEAD}"
concurrency="${CONCURRENCY:-4}"
fast_games="${GAMES_FAST:-20}"
blitz_games="${GAMES_BLITZ:-20}"
run_blitz="${RUN_BLITZ:-0}"

mkdir -p "$(dirname "${log_file}")"

if [[ ! -f "${log_file}" ]]; then
  {
    echo "# Stability Log"
    echo
    echo "Private Cutechess stability history. This file is intentionally under ignored docs/."
    echo
    echo "Default protocol:"
    echo
    echo "- Fast smoke: 20 games at 1+0.01."
    echo "- Optional blitz check: set RUN_BLITZ=1 for 20 games at 5+0.05."
    echo "- Override GAMES_FAST, GAMES_BLITZ, CONCURRENCY, BASELINE_COMMIT, and STABILITY_LOG as needed."
  } >"${log_file}"
fi

run_case() {
  local name="$1"
  local games="$2"
  local tc="$3"
  local tmp
  local status="ok"

  tmp="$(mktemp)"
  echo "Running ${name}: ${games} games at ${tc}"
  if ! GAMES="${games}" TC="${tc}" CONCURRENCY="${concurrency}" \
    BASELINE_COMMIT="${baseline_commit}" \
    bash "${repo_root}/scripts/run_match.sh" 2>&1 | tee "${tmp}"; then
    status="failed"
  fi

  local score
  local elo
  local pgn
  local match_log
  local warnings
  score="$(grep -E "Score of current vs baseline" "${tmp}" | tail -1 || true)"
  elo="$(grep -E "Elo difference" "${tmp}" | tail -1 || true)"
  pgn="$(grep -E "^PGN:" "${tmp}" | tail -1 | sed 's/^PGN: //' || true)"
  match_log="$(grep -E "^Log:" "${tmp}" | tail -1 | sed 's/^Log: //' || true)"
  warnings="$(grep -Ei "illegal|crash|disconnect|timeout|stall|forfeit" "${tmp}" | tail -5 || true)"

  {
    echo
    echo "## $(date +%Y-%m-%d) ${name}"
    echo
    echo "- Command: GAMES=${games} TC=${tc} CONCURRENCY=${concurrency} BASELINE_COMMIT=${baseline_commit} bash scripts/run_match.sh"
    echo "- Status: ${status}"
    echo "- Score: ${score:-not reported}"
    echo "- Elo: ${elo:-not reported}"
    echo "- PGN: ${pgn:-not reported}"
    echo "- Log: ${match_log:-not reported}"
    if [[ -n "${warnings}" ]]; then
      echo "- Warning lines:"
      echo "${warnings}" | sed 's/^/  - /'
    else
      echo "- Warning lines: none detected"
    fi
  } >>"${log_file}"

  rm -f "${tmp}"

  if [[ "${status}" != "ok" ]]; then
    echo "${name} failed; see ${log_file}" >&2
    exit 1
  fi
}

run_case "fast-smoke" "${fast_games}" "1+0.01"

if [[ "${run_blitz}" == "1" ]]; then
  run_case "blitz-check" "${blitz_games}" "5+0.05"
fi

echo "Stability log: ${log_file}"
