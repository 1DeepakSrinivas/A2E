#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ "$(uname -s)" != Linux || "$(uname -m)" != aarch64 ]] ||
   ! command -v apt-get >/dev/null || [[ "$(dpkg --print-architecture)" != arm64 ]]; then
    echo "Install 64-bit Raspberry Pi OS before running setup-pi.sh." >&2
    exit 2
fi
if [[ "$(id -u)" == 0 ]]; then
    echo "Run setup-pi.sh as your normal user. The package installation uses sudo." >&2
    exit 2
fi

sudo apt-get update
sudo apt-get install -y build-essential cmake git curl ca-certificates

export PATH="$HOME/.cargo/bin:$PATH"
if ! command -v rustup >/dev/null; then
    curl --proto '=https' --tlsv1.2 -fsSL https://sh.rustup.rs |
        sh -s -- -y --profile minimal --default-toolchain none
fi
rustup toolchain install 1.96.0 --profile minimal

# Keep installation within the Pi's memory budget.
export A2E_BUILD_JOBS="${A2E_BUILD_JOBS:-1}"
A2E_HOME="$PWD/deps" LLAMA_COMMIT="" bash scripts/build_llama.sh
CARGO_PROFILE_RELEASE_LTO=false cargo +1.96.0 build --release --locked --jobs "$A2E_BUILD_JOBS"

bash scripts/init_sandbox.sh
target/release/a2e models fetch
target/release/a2e doctor
printf '\nTo run the smoke suite after disabling swap:\n  target/release/a2e run\n'
