#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python="${PYTHON:-${repo_root}/.venv/bin/python}"
if [[ ! -x "${python}" ]]; then
  python="${PYTHON:-python3}"
fi

raw_input="${RAW_INPUT:-${repo_root}/data/lichess_db_eval.jsonl.zst}"
name="${NAME:-lichess-stockfish-trainer}"
seed="${SEED:-$(date -u +%Y%m%d%H%M%S)}"
limit="${LIMIT:-${IMPORT_LIMIT:-0}}"
sample_rate="${SAMPLE_RATE:-1.0}"
min_depth="${MIN_DEPTH:-0}"
max_abs_cp="${MAX_ABS_CP:-0}"
input_score_pov="${INPUT_SCORE_POV:-white}"
default_ply="${DEFAULT_PLY:-30}"
force_default_ply="${FORCE_DEFAULT_PLY:-1}"
report_every="${REPORT_EVERY:-100000}"
validate="${VALIDATE:-1}"
overwrite="${OVERWRITE:-1}"
keep_plain="${KEEP_PLAIN:-0}"
skip_mates="${SKIP_MATES:-0}"
skip_invalid="${SKIP_INVALID:-1}"
shard_size="${SHARD_SIZE:-10000000}"

plain="${PLAIN:-${repo_root}/data/stockfish/intermediate/${name}.plain}"
binpack="${OUTPUT:-${repo_root}/data/stockfish/lichess/${name}.binpack}"

overwrite_args=()
if [[ "${overwrite}" != "0" ]]; then
  overwrite_args+=(--overwrite)
fi

if [[ ! -f "${raw_input}" ]]; then
  echo "Raw Lichess eval input not found at ${raw_input}" >&2
  echo "Set RAW_INPUT=/path/to/lichess_db_eval.jsonl.zst" >&2
  exit 1
fi

plain_cmd=(
  "${python}" "${repo_root}/tools/stockfish_data/lichess_eval_to_stockfish_plain.py"
  --input "${raw_input}"
  --output "${plain}"
  --limit "${limit}"
  --sample-rate "${sample_rate}"
  --seed "${seed}"
  --min-depth "${min_depth}"
  --max-abs-cp "${max_abs_cp}"
  --input-score-pov "${input_score_pov}"
  --default-ply "${default_ply}"
  --report-every "${report_every}"
)

if [[ "${force_default_ply}" != "0" ]]; then
  plain_cmd+=(--force-default-ply)
fi
if [[ "${skip_mates}" == "0" ]]; then
  plain_cmd+=(--keep-mates)
fi
plain_cmd+=("${overwrite_args[@]}")

"${plain_cmd[@]}"

tool="$("${repo_root}/scripts/build_stockfish_binpack_tool.sh")"
mkdir -p "$(dirname "${binpack}")"

convert_cmd=("${tool}" plain-to-binpack "${plain}" "${binpack}")
if [[ "${validate}" == "0" ]]; then
  convert_cmd+=(--no-validate)
fi
if [[ "${skip_invalid}" != "0" ]]; then
  convert_cmd+=(--skip-invalid)
fi
if [[ "${shard_size}" != "0" ]]; then
  convert_cmd+=(--shard-size "${shard_size}")
fi
"${convert_cmd[@]}"

echo
if [[ "${shard_size}" == "0" ]]; then
  echo "Stockfish trainer binpack ready:"
  echo "  ${binpack}"
else
  echo "Stockfish trainer binpack shards ready:"
  echo "  ${binpack%.binpack}-*.binpack"
fi
if [[ "${keep_plain}" == "0" ]]; then
  rm -f "${plain}"
else
  echo "Plain intermediate:"
  echo "  ${plain}"
fi
echo
echo "For static eval-only data, train with --start-lambda=1.0 --end-lambda=1.0."
