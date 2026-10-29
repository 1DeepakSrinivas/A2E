#!/usr/bin/env bash
# Build llama.cpp with Metal on Apple Silicon and install the harness.
set -euo pipefail
LLAMA_COMMIT="${LLAMA_COMMIT:?pin a llama.cpp commit, e.g. LLAMA_COMMIT=<sha>}"
ROOT="${A2E_HOME:-$HOME/a2e-deps}"
mkdir -p "$ROOT" && cd "$ROOT"
[ -d llama.cpp ] || git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp && git fetch -q && git checkout "$LLAMA_COMMIT"
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target llama-server llama-bench
echo "llama-server: $ROOT/llama.cpp/build/bin/llama-server (commit $LLAMA_COMMIT)"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
echo "Energy sampling needs: sudo powermetrics (run the harness with sudo or a sudoers rule)."
