#!/usr/bin/env bash
# Build llama.cpp (CPU/NEON) on Raspberry Pi 5 (aarch64) and install the harness.
set -euo pipefail
sudo apt-get update -y && sudo apt-get install -y git cmake build-essential libcurl4-openssl-dev linux-perf
bash "$(dirname -- "${BASH_SOURCE[0]}")/build_llama.sh"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
echo "Check power/thermal tooling: vcgencmd pmic_read_adc; vcgencmd get_throttled"
