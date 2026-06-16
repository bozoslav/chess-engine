#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
external_dir="${repo_root}/external"
src_dir="${external_dir}/cutechess-src"
build_dir="${external_dir}/cutechess-build"
bin_dir="${external_dir}/bin"
binary="${bin_dir}/cutechess-cli"

mkdir -p "${external_dir}" "${bin_dir}"

if ! command -v cmake >/dev/null 2>&1; then
  echo "cmake is required. Install with: brew install cmake" >&2
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  echo "git is required." >&2
  exit 1
fi

if [[ ! -d "${src_dir}/.git" ]]; then
  git clone https://github.com/cutechess/cutechess.git "${src_dir}"
else
  git -C "${src_dir}" fetch --depth 1 origin master
  git -C "${src_dir}" reset --hard origin/master
fi

cmake -S "${src_dir}" -B "${build_dir}"
cmake --build "${build_dir}" --target cli

found_binary="$(find "${build_dir}" -type f -name cutechess-cli | head -n 1)"
if [[ -z "${found_binary}" ]]; then
  echo "cutechess-cli was not found under ${build_dir}" >&2
  exit 1
fi

ln -sf "${found_binary}" "${binary}"
"${binary}" --version
