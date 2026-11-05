#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
pin="$(sed -n 's/^llama_commit = "\([0-9a-f]*\)"$/\1/p' "$repo_dir/configs/native/models.toml")"
commit="${LLAMA_COMMIT:-$pin}"
deps_dir="${A2E_HOME:-$repo_dir/deps}"
source_dir="$deps_dir/llama.cpp"
build_dir="$source_dir/build-a2e"
jobs="${A2E_BUILD_JOBS:-1}"

if [[ ! "$commit" =~ ^[a-f0-9]{40}$ || ! "$jobs" =~ ^[1-9][0-9]*$ ]]; then
    echo "Use a full llama.cpp commit SHA and a positive A2E_BUILD_JOBS value." >&2
    exit 2
fi

mkdir -p "$deps_dir"
if [[ ! -d "$source_dir" ]]; then
    git clone --no-checkout --depth 1 https://github.com/ggml-org/llama.cpp.git "$source_dir"
fi
if ! git -C "$source_dir" cat-file -e "$commit^{commit}" 2>/dev/null; then
    git -C "$source_dir" fetch --depth 1 origin "$commit"
fi
if [[ "$(git -C "$source_dir" rev-parse HEAD)" != "$commit" || ! -f "$source_dir/CMakeLists.txt" ]]; then
    git -C "$source_dir" checkout --detach "$commit"
fi

metal=OFF
if [[ "$(uname -s)" == Darwin ]]; then
    metal=ON
fi
cmake -S "$source_dir" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
    -DGGML_CCACHE=OFF -DGGML_NATIVE=ON -DGGML_METAL="$metal" -DLLAMA_BUILD_TESTS=OFF
cmake --build "$build_dir" --parallel "$jobs" --target llama-server llama-bench
"$build_dir/bin/llama-server" --version
