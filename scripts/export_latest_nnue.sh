#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS="$ROOT/data/stockfish/runs"
TRAINER="$ROOT/external/nnue-pytorch"
EXPORT="$ROOT/data/stockfish/export/latest.nnue"
CHECKPOINT="${1:-}"

if [[ -z $CHECKPOINT ]]; then
  while IFS= read -r candidate; do
    if [[ -z $CHECKPOINT || $candidate -nt $CHECKPOINT ]]; then
      CHECKPOINT="$candidate"
    fi
  done < <(find "$RUNS" -type f -path '*/checkpoints/last.ckpt' -print)
fi

if [[ -z $CHECKPOINT || ! -f $CHECKPOINT ]]; then
  echo "No last.ckpt was found under $RUNS" >&2
  exit 1
fi

eval "$(micromamba shell hook --shell bash)"
micromamba activate sf-nnue

mkdir -p "$(dirname "$EXPORT")"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT/data/stockfish/.matplotlib-cache}"
mkdir -p "$MPLCONFIGDIR"
TEMP_EXPORT="$EXPORT.$$.nnue"
trap 'rm -f "$TEMP_EXPORT"' EXIT

cd "$TRAINER"
python -u serialize.py "$CHECKPOINT" "$TEMP_EXPORT" \
  --features='Full_Threats+HalfKAv2_hm^' \
  --l1=1024 --l2=32 --l3=32 \
  --ft_compression=leb128 --device=mps --loader_num_workers=1

mv -f "$TEMP_EXPORT" "$EXPORT"
trap - EXIT
echo "Exported Stockfish NNUE from $CHECKPOINT"
echo "Latest network: $EXPORT"
