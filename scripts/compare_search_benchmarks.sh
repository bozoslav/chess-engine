#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
baseline_dir="${BASELINE_DIR:-${repo_root}/cmake-build-search-baseline}"
candidate_dir="${CANDIDATE_DIR:-${repo_root}/cmake-build-release}"
network="${NETWORK:-${repo_root}/data/stockfish/export/latest.nnue}"
scaling_ms="${SCALING_MS:-3000}"
scaling_runs="${SCALING_RUNS:-3}"

for dir in "${baseline_dir}" "${candidate_dir}"; do
  [[ -x "${dir}/chess_engine_search_benchmark" ]] || {
    echo "missing search benchmark: ${dir}/chess_engine_search_benchmark" >&2
    exit 1
  }
  [[ -x "${dir}/chess_engine_scaling_benchmark" ]] || {
    echo "missing scaling benchmark: ${dir}/chess_engine_scaling_benchmark" >&2
    exit 1
  }
done

tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT

run_search() {
  local dir="$1"
  local output="$2"
  "${dir}/chess_engine_search_benchmark" "${network}" >"${output}"
}

run_scaling() {
  local dir="$1"
  local threads="$2"
  local output="$3"
  : >"${output}"
  for ((run = 0; run < scaling_runs; ++run)); do
    "${dir}/chess_engine_scaling_benchmark" "${network}" "${threads}" \
      "${scaling_ms}" | tail -n 1 >>"${output}"
  done
}

mean_column() {
  local file="$1"
  local column="$2"
  awk -F, -v column="${column}" \
    '{ total += $column; count += 1 } END { printf "%.6f", total / count }' \
    "${file}"
}

percent() {
  local baseline="$1"
  local candidate="$2"
  awk -v baseline="${baseline}" -v candidate="${candidate}" \
    'BEGIN { printf "%+.2f", 100.0 * (candidate / baseline - 1.0) }'
}

reduction_percent() {
  local baseline="$1"
  local candidate="$2"
  awk -v baseline="${baseline}" -v candidate="${candidate}" \
    'BEGIN { printf "%+.2f", 100.0 * (1.0 - candidate / baseline) }'
}

run_search "${baseline_dir}" "${tmp_dir}/baseline-search.csv"
run_search "${candidate_dir}" "${tmp_dir}/candidate-search.csv"
run_scaling "${baseline_dir}" 1 "${tmp_dir}/baseline-1t.csv"
run_scaling "${candidate_dir}" 1 "${tmp_dir}/candidate-1t.csv"
run_scaling "${baseline_dir}" 10 "${tmp_dir}/baseline-10t.csv"
run_scaling "${candidate_dir}" 10 "${tmp_dir}/candidate-10t.csv"

baseline_case="${BASELINE_SEARCH_CASE:-startpos_elite_off}"
candidate_case="${CANDIDATE_SEARCH_CASE:-startpos_modern_final}"
baseline_search="$(awk -F, -v name="${baseline_case}" '$1 == name { print $10 }' "${tmp_dir}/baseline-search.csv")"
candidate_search="$(awk -F, -v name="${candidate_case}" '$1 == name { print $10 }' "${tmp_dir}/candidate-search.csv")"
baseline_nodes="$(awk -F, -v name="${baseline_case}" '$1 == name { print $8 }' "${tmp_dir}/baseline-search.csv")"
candidate_nodes="$(awk -F, -v name="${candidate_case}" '$1 == name { print $8 }' "${tmp_dir}/candidate-search.csv")"
baseline_time="$(awk -F, -v name="${baseline_case}" '$1 == name { print $9 }' "${tmp_dir}/baseline-search.csv")"
candidate_time="$(awk -F, -v name="${candidate_case}" '$1 == name { print $9 }' "${tmp_dir}/candidate-search.csv")"
baseline_1t="$(mean_column "${tmp_dir}/baseline-1t.csv" 6)"
candidate_1t="$(mean_column "${tmp_dir}/candidate-1t.csv" 6)"
baseline_10t="$(mean_column "${tmp_dir}/baseline-10t.csv" 6)"
candidate_10t="$(mean_column "${tmp_dir}/candidate-10t.csv" 6)"

printf '%-24s %14s %14s %10s\n' metric baseline candidate change
printf '%-24s %14.0f %14.0f %9s%%\n' search_nps "${baseline_search}" \
  "${candidate_search}" "$(percent "${baseline_search}" "${candidate_search}")"
printf '%-24s %14.0f %14.0f %9s%%\n' search_node_reduction \
  "${baseline_nodes}" "${candidate_nodes}" \
  "$(reduction_percent "${baseline_nodes}" "${candidate_nodes}")"
printf '%-24s %14.6f %14.6f %9s%%\n' search_time_reduction \
  "${baseline_time}" "${candidate_time}" \
  "$(reduction_percent "${baseline_time}" "${candidate_time}")"
printf '%-24s %14.0f %14.0f %9s%%\n' scaling_1t_nps "${baseline_1t}" \
  "${candidate_1t}" "$(percent "${baseline_1t}" "${candidate_1t}")"
printf '%-24s %14.0f %14.0f %9s%%\n' scaling_10t_nps "${baseline_10t}" \
  "${candidate_10t}" "$(percent "${baseline_10t}" "${candidate_10t}")"
