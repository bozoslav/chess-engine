#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLIENT_DIR="${ROOT_DIR}/external/lichess-bot"
PYTHON_BIN="${ROOT_DIR}/.venv-lichess/bin/python"
CONFIG_FILE="${LICHESS_BOT_CONFIG:-${ROOT_DIR}/lichess/config.yml}"
ENV_FILE="${LICHESS_BOT_ENV:-${ROOT_DIR}/lichess/.env}"

if [[ -z "${LICHESS_BOT_TOKEN:-}" && -f "${ENV_FILE}" ]]; then
  while IFS='=' read -r key value; do
    if [[ "${key}" == "LICHESS_BOT_TOKEN" ]]; then
      export LICHESS_BOT_TOKEN="${value}"
      break
    fi
  done < "${ENV_FILE}"
fi

if [[ -z "${LICHESS_BOT_TOKEN:-}" || \
      "${LICHESS_BOT_TOKEN}" == "replace_with_the_bot_accounts_token" ]]; then
  echo "error: set LICHESS_BOT_TOKEN in lichess/.env or the environment" >&2
  exit 1
fi

[[ -x "${PYTHON_BIN}" ]] || {
  echo "error: lichess-bot environment is missing; run scripts/setup_lichess_bot.sh" >&2
  exit 1
}
[[ -f "${CLIENT_DIR}/lichess-bot.py" ]] || {
  echo "error: official lichess-bot checkout is missing; run scripts/setup_lichess_bot.sh" >&2
  exit 1
}
[[ -x "${ROOT_DIR}/cmake-build-release/chess_engine" ]] || {
  echo "error: Release engine is missing; run scripts/setup_lichess_bot.sh" >&2
  exit 1
}
[[ -f "${ROOT_DIR}/data/stockfish/export/latest.nnue" ]] || {
  echo "error: data/stockfish/export/latest.nnue is missing" >&2
  exit 1
}
[[ -f "${CONFIG_FILE}" ]] || {
  echo "error: config file does not exist: ${CONFIG_FILE}" >&2
  exit 1
}

mkdir -p "${ROOT_DIR}/lichess/game_records" "${ROOT_DIR}/lichess/logs"
cd "${ROOT_DIR}"
exec "${PYTHON_BIN}" "${CLIENT_DIR}/lichess-bot.py" \
  --config "${CONFIG_FILE}" "$@"
