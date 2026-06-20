#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLIENT_DIR="${ROOT_DIR}/external/lichess-bot"
VENV_DIR="${ROOT_DIR}/.venv-lichess"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Known-good upstream revision, version 2026.6.19.1. Override with
# LICHESS_BOT_REF=master when intentionally testing a newer upstream version.
LICHESS_BOT_REF="${LICHESS_BOT_REF:-f1fdc347a5fe24cfa972ebaadda4685f24a0cf85}"
LICHESS_BOT_REPOSITORY="https://github.com/lichess-bot-devs/lichess-bot.git"

command -v git >/dev/null || { echo "error: git is required" >&2; exit 1; }
command -v cmake >/dev/null || { echo "error: cmake is required" >&2; exit 1; }
command -v "${PYTHON_BIN}" >/dev/null || {
  echo "error: ${PYTHON_BIN} is required (Python 3.10 or newer)" >&2
  exit 1
}

"${PYTHON_BIN}" -c 'import sys; raise SystemExit(sys.version_info < (3, 10))' || {
  echo "error: lichess-bot requires Python 3.10 or newer" >&2
  exit 1
}

if [[ -e "${CLIENT_DIR}" && ! -d "${CLIENT_DIR}/.git" ]]; then
  echo "error: ${CLIENT_DIR} exists but is not a git checkout" >&2
  exit 1
fi

if [[ ! -d "${CLIENT_DIR}/.git" ]]; then
  git clone --filter=blob:none "${LICHESS_BOT_REPOSITORY}" "${CLIENT_DIR}"
fi

git -C "${CLIENT_DIR}" fetch origin "${LICHESS_BOT_REF}"
git -C "${CLIENT_DIR}" checkout --detach FETCH_HEAD

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install -r "${CLIENT_DIR}/requirements.txt"

cmake -S "${ROOT_DIR}" -B "${ROOT_DIR}/cmake-build-release" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "${ROOT_DIR}/cmake-build-release" --target chess_engine

if [[ ! -f "${ROOT_DIR}/data/stockfish/export/latest.nnue" ]]; then
  echo "error: missing data/stockfish/export/latest.nnue" >&2
  echo "Export or copy the supported Stockfish NNUE network before running the bot." >&2
  exit 1
fi

mkdir -p "${ROOT_DIR}/lichess/game_records" "${ROOT_DIR}/lichess/logs"

if [[ ! -f "${ROOT_DIR}/lichess/.env" ]]; then
  cp "${ROOT_DIR}/lichess/.env.example" "${ROOT_DIR}/lichess/.env"
  chmod 600 "${ROOT_DIR}/lichess/.env"
  echo "Created lichess/.env. Replace its placeholder with the bot account token."
fi

"${ROOT_DIR}/scripts/check_lichess_bot.sh"

echo
echo "Lichess bot setup is ready."
echo "Next: edit lichess/.env, then run scripts/run_lichess_bot.sh -u exactly once."
