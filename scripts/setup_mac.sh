#!/usr/bin/env bash
# Build llama.cpp with Metal on Apple Silicon and install the harness.
set -euo pipefail
bash "$(dirname -- "${BASH_SOURCE[0]}")/build_llama.sh"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
echo "Energy sampling needs: sudo powermetrics (run the harness with sudo or a sudoers rule)."
