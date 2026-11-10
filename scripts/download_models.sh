#!/usr/bin/env bash
# Download the pinned GGUF files from configs/models.toml and verify their SHA-256.
#   scripts/download_models.sh [models_dir]      default: models_dir from the config
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

if [[ -n "${1:-}" ]]; then
    export A2E_NATIVE_MODELS_DIR="$1"
fi
binary="${A2E_NATIVE_BINARY:-target/release/a2e}"
if [[ ! -x "$binary" ]]; then
    echo "Build the runner first: cargo build --release --locked" >&2
    exit 2
fi
exec "$binary" models fetch
