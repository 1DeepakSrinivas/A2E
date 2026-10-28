#!/usr/bin/env bash
# Build llama.cpp (CPU/NEON) on Raspberry Pi 5 (aarch64) and install the harness.
set -euo pipefail
LLAMA_COMMIT="${LLAMA_COMMIT:?pin a llama.cpp commit, e.g. LLAMA_COMMIT=<sha>}"
ROOT="${A2E_HOME:-$HOME/a2e-deps}"
sudo apt-get update -y && sudo apt-get install -y git cmake build-essential libcurl4-openssl-dev linux-perf
mkdir -p "$ROOT" && cd "$ROOT"
[ -d llama.cpp ] || git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp && git fetch -q && git checkout "$LLAMA_COMMIT"
cmake -B build -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4 --target llama-server llama-bench
echo "llama-server: $ROOT/llama.cpp/build/bin/llama-server (commit $LLAMA_COMMIT)"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
echo "Check power/thermal tooling: vcgencmd pmic_read_adc; vcgencmd get_throttled"
