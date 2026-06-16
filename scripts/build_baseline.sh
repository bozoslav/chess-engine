#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
commit="${1:-HEAD}"
baseline_root="${repo_root}/external/baselines"
worktree_dir="${baseline_root}/${commit//[^A-Za-z0-9._-]/_}"
build_dir="${worktree_dir}/cmake-build-release"

mkdir -p "${baseline_root}"

if [[ ! -e "${worktree_dir}/.git" ]]; then
  git -C "${repo_root}" worktree add --detach "${worktree_dir}" "${commit}" >&2
fi

cmake -S "${worktree_dir}" -B "${build_dir}" -DCMAKE_BUILD_TYPE=Release >&2
cmake --build "${build_dir}" --target chess_engine >&2

echo "${build_dir}/chess_engine"
