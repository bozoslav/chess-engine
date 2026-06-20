#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
nnue_pytorch_dir="${NNUE_PYTORCH_DIR:-${repo_root}/external/nnue-pytorch}"
header_dir="${nnue_pytorch_dir}/data_loader/cpp/lib"
source_file="${repo_root}/tools/stockfish_binpack_tool.cpp"
output="${OUTPUT:-${repo_root}/external/bin/stockfish_binpack_tool}"
cxx="${CXX:-c++}"

if [[ ! -d "${header_dir}" ]]; then
  echo "nnue-pytorch headers not found at ${header_dir}" >&2
  echo "Clone first: git clone https://github.com/official-stockfish/nnue-pytorch.git external/nnue-pytorch" >&2
  exit 1
fi

mkdir -p "$(dirname "${output}")"

"${cxx}" \
  -std=c++20 \
  -O3 \
  -DNDEBUG \
  -Wall \
  -Wextra \
  -I "${header_dir}" \
  "${source_file}" \
  -o "${output}"

echo "${output}"
