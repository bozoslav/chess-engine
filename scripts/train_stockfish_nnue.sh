#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAINER="$ROOT/external/nnue-pytorch"
DATA="$ROOT/data/stockfish/lichess"
RUNS="$ROOT/data/stockfish/runs"
RUN="$RUNS/sfnnv13-h1024-$(date +%Y%m%d-%H%M%S)"

FRESH=false
if [[ ${1:-} == "--fresh" ]]; then
  FRESH=true
  shift
fi
if [[ $# -ne 0 ]]; then
  echo "Usage: $0 [--fresh]" >&2
  exit 2
fi

eval "$(micromamba shell hook --shell bash)"
micromamba activate sf-nnue

TRAIN=(
  "$DATA"/lichess-stockfish-trainer-0[0-2][0-9].binpack
  "$DATA"/lichess-stockfish-trainer-03[0-7].binpack
)
VALIDATION="$DATA/lichess-stockfish-trainer-038.binpack"

if [[ ${#TRAIN[@]} -ne 38 || ! -f "$VALIDATION" ]]; then
  echo "Expected 38 training shards and one validation shard under $DATA" >&2
  exit 1
fi

cd "$TRAINER"
echo "Training output: $RUN"

CHECKPOINT="$RUN/lightning_logs/version_0/checkpoints/last.ckpt"
export_checkpoint() {
  if [[ ! -f "$CHECKPOINT" ]]; then
    echo "No completed checkpoint to export from this run." >&2
    return 0
  fi
  "$ROOT/scripts/export_latest_nnue.sh" "$CHECKPOINT"
}

RESUME_ARGS=()
if [[ $FRESH == false && -d "$RUNS" ]]; then
  RESUME=""
  while IFS= read -r checkpoint; do
    if [[ -z $RESUME || $checkpoint -nt $RESUME ]]; then
      RESUME="$checkpoint"
    fi
  done < <(find "$RUNS" -type f -path '*/checkpoints/last.ckpt' -print)

  if [[ -n $RESUME ]]; then
    echo "Resuming checkpoint: $RESUME"
    RESUME_ARGS+=("--resume-from-checkpoint=$RESUME")
  else
    echo "No checkpoint found; starting a fresh network."
  fi
else
  echo "Starting a fresh network."
fi

trap 'export_checkpoint || echo "Checkpoint export failed; run scripts/export_latest_nnue.sh manually." >&2' EXIT

caffeinate -i python -u train.py "${TRAIN[@]}" \
  --validation-datasets="$VALIDATION" \
  --features='Full_Threats+HalfKAv2_hm^' \
  --l1=1024 --l2=32 --l3=32 \
  --accelerator=mps --num_workers=3 --threads=8 \
  --batch-size=8192 --epoch-size=10000000 \
  --validation-size=1000000 --check-val-every-n-epoch=1 \
  --max_epochs=800 --max-time=00:06:00:00 \
  --network-save-period=1 --save-top-k=-1 --save-last-network=True \
  --start-lambda=1.0 --end-lambda=1.0 \
  "${RESUME_ARGS[@]}" \
  --default_root_dir="$RUN"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Training finished without a last checkpoint at $CHECKPOINT" >&2
  exit 1
fi

export_checkpoint
trap - EXIT
