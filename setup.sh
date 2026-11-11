#!/usr/bin/env bash
# Set up this machine: packages, llama.cpp, the a2e binary, models, sandbox and checks.
#   ./setup.sh [--skip-llama] [--skip-models]
# Runs on 64-bit Raspberry Pi OS and Apple Silicon macOS. Each step is a script in scripts/setup
# that can be run alone, and every step can be repeated.
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

skip_llama=0
skip_models=0
for arg in "$@"; do
    case "$arg" in
        --skip-llama) skip_llama=1 ;;
        --skip-models) skip_models=1 ;;
        *) echo "usage: ./setup.sh [--skip-llama] [--skip-models]" >&2; exit 2 ;;
    esac
done

if [[ "$(id -u)" == 0 ]]; then
    echo "Run setup.sh as your normal user. Package installation uses sudo." >&2
    exit 2
fi
if [[ "$(uname -s)" == Linux && "$(uname -m)" == aarch64 ]] && command -v apt-get >/dev/null &&
   [[ "$(dpkg --print-architecture)" == arm64 ]]; then
    A2E_PLATFORM=pi
    A2E_BUILD_JOBS="${A2E_BUILD_JOBS:-1}"
elif [[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]]; then
    A2E_PLATFORM=mac
    A2E_BUILD_JOBS="${A2E_BUILD_JOBS:-$(sysctl -n hw.ncpu)}"
else
    echo "Use 64-bit Raspberry Pi OS or an Apple Silicon Mac." >&2
    exit 2
fi
export A2E_PLATFORM A2E_BUILD_JOBS
export A2E_RUST="${A2E_RUST:-1.96.0}"
export PATH="$HOME/.cargo/bin:$PATH"

step() { printf '\n==> %s\n' "$1"; }

step "Packages and Rust $A2E_RUST"
bash scripts/setup/deps.sh

if [[ "$skip_llama" == 0 ]]; then
    step "llama.cpp at the pinned commit"
    A2E_HOME="$PWD/deps" LLAMA_COMMIT="" bash scripts/build_llama.sh
fi

step "a2e binary"
bash scripts/setup/build.sh

if [[ "$skip_models" == 0 ]]; then
    step "Models"
    target/release/a2e models fetch
fi

step "File sandbox"
bash scripts/init_sandbox.sh

step "Checks"
bash scripts/setup/checks.sh

printf '\nSetup done. Start a suite in tmux:\n  scripts/run.sh start smoke\n'
