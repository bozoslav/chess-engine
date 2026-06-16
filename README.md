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

## Benchmarks

```bash
./cmake-build-release/chess_engine bench
./cmake-build-release/chess_engine_perft_benchmark
./cmake-build-release/chess_engine_search_benchmark
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
