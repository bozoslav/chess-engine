#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLIENT_DIR="${ROOT_DIR}/external/lichess-bot"
PYTHON_BIN="${ROOT_DIR}/.venv-lichess/bin/python"
ENGINE="${ROOT_DIR}/cmake-build-release/chess_engine"
NETWORK="${ROOT_DIR}/data/stockfish/export/latest.nnue"
CONFIG="${ROOT_DIR}/lichess/config.yml"

[[ -x "${PYTHON_BIN}" ]] || {
  echo "error: run scripts/setup_lichess_bot.sh first" >&2
  exit 1
}
[[ -f "${CLIENT_DIR}/lichess-bot.py" ]] || {
  echo "error: official lichess-bot checkout is missing" >&2
  exit 1
}
[[ -x "${ENGINE}" ]] || { echo "error: Release engine is missing" >&2; exit 1; }
[[ -f "${NETWORK}" ]] || { echo "error: NNUE network is missing" >&2; exit 1; }

cd "${ROOT_DIR}"
PYTHONPATH="${CLIENT_DIR}" \
LICHESS_BOT_TOKEN="local-validation-token" \
ROOT_DIR="${ROOT_DIR}" \
"${PYTHON_BIN}" - <<'PY'
import os
from collections import Counter, defaultdict
from pathlib import Path

import chess
import chess.engine

from lib.blocklist import OnlineBlocklist
from lib.config import load_config
from lib.model import Challenge

root = Path(os.environ["ROOT_DIR"])
config = load_config(str(root / "lichess/config.yml"))

assert config.engine.protocol == "uci"
assert config.engine.ponder is True
assert config.challenge.concurrency == 1
assert not config.challenge.allow_list
assert config.matchmaking.allow_matchmaking is True
assert config.matchmaking.allow_during_games is False
assert config.matchmaking.challenge_timeout == 5
assert config.challenge.accept_bot is True
assert config.challenge.variants == ["standard"]
assert config.challenge.time_controls == ["bullet", "blitz", "rapid", "classical"]
assert set(config.challenge.modes) == {"rated"}

profile = {"username": "LocalValidationBot", "perfs": {}}

def supported(speed, base, increment, *, rated=True, variant="standard", bot=False):
    info = {
        "id": f"{speed}-{base}-{increment}-{variant}",
        "rated": rated,
        "variant": {"key": variant, "name": variant},
        "perf": {"name": speed},
        "speed": speed,
        "timeControl": {"type": "clock", "limit": base, "increment": increment},
        "challenger": {"name": "Opponent", "rating": 1800,
                       "title": "BOT" if bot else None},
        "destUser": {"name": profile["username"]},
        "color": "random",
        "finalColor": "white",
        "initialFen": "startpos",
    }
    challenge = Challenge(info, profile)
    accepted, _ = challenge.is_supported(
        config.challenge,
        defaultdict(list),
        Counter(),
        OnlineBlocklist([]),
        profile,
    )
    return accepted

assert supported("bullet", 60, 0, bot=True)
assert supported("blitz", 180, 2)
assert supported("rapid", 600, 5)
assert supported("classical", 10800, 180)
assert not supported("ultraBullet", 15, 0)
assert not supported("bullet", 30, 0)
assert not supported("classical", 10801, 0)
assert not supported("blitz", 180, 181)
assert not supported("blitz", 180, 2, variant="chess960")

engine = chess.engine.SimpleEngine.popen_uci(
    str(root / "cmake-build-release/chess_engine"), cwd=str(root), timeout=60.0
)
try:
    engine.configure({
        "EvalFile": str(root / "data/stockfish/export/latest.nnue"),
        "Threads": 1,
        "Hash": 16,
        "SingularExtensions": True,
    })
    board = chess.Board()
    result = engine.play(board, chess.engine.Limit(time=0.1))
    assert result.move in board.legal_moves
finally:
    engine.quit()

print("lichess-bot config, challenge filters, and UCI move smoke passed")
PY
