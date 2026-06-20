# Chess Engine

Original C++ chess engine focused on correctness, low-latency search
infrastructure, and measurable strength improvements.

## Build

```bash
cmake -S . -B cmake-build-release -DCMAKE_BUILD_TYPE=Release
cmake --build cmake-build-release
```

## Test

```bash
ctest --test-dir cmake-build-release --output-on-failure
```

## Run

```bash
./cmake-build-release/chess_engine
```

Basic UCI session:

```text
uci
isready
position startpos
go depth 8
```

NNUE development uses the official Stockfish `nnue-pytorch` trainer under the
ignored `external/nnue-pytorch/` checkout. Start a timestamped MPS training run
with:

```bash
./scripts/train_stockfish_nnue.sh
```

The script trains on shards 000-037, validates on shard 038, retains every
checkpoint, and exports the final Stockfish-format network to:

```text
data/stockfish/export/latest.nnue
```

Clean builds load that file automatically. A different Stockfish-format network
with the same `Full_Threats+HalfKAv2_hm^`, 1024-32-32 architecture can be loaded
at runtime through UCI:

```text
setoption name EvalFile value /absolute/path/to/network.nnue
```

The `CHESS_ENGINE_EVALFILE` environment variable overrides the compiled default.

## Benchmarks

```bash
./cmake-build-release/chess_engine bench
./cmake-build-release/chess_engine_perft_benchmark
./cmake-build-release/chess_engine_search_benchmark data/stockfish/export/latest.nnue
./cmake-build-release/chess_engine_nnue_benchmark data/stockfish/export/latest.nnue
```

## Cutechess

Build the local Cutechess CLI into ignored `external/` directories:

```bash
bash scripts/setup_cutechess.sh
```

Run a small baseline match:

```bash
GAMES=20 TC=1+0.01 CONCURRENCY=4 bash scripts/run_match.sh
```

## Lichess bot

The repository includes an official `lichess-bot`-based deployment, safe token
handling, challenge filters, local integration checks, and service templates.
Start with [lichess/README.md](lichess/README.md).
