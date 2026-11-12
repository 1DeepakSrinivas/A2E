#!/usr/bin/env bash
# Build target/release/a2e. Called by setup.sh.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

export PATH="$HOME/.cargo/bin:$PATH"
rust="${A2E_RUST:-1.96.0}"
jobs="${A2E_BUILD_JOBS:-1}"

# Link-time optimization needs more memory than the 1 GB Pi has.
if [[ "${A2E_PLATFORM:-}" == pi ]]; then
    export CARGO_PROFILE_RELEASE_LTO=false
fi
cargo "+$rust" build --release --locked --jobs "$jobs"
