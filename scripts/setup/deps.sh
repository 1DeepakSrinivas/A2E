#!/usr/bin/env bash
# Install build packages, tmux and the Rust toolchain. Called by setup.sh.
set -euo pipefail

rust="${A2E_RUST:-1.96.0}"
export PATH="$HOME/.cargo/bin:$PATH"

case "${A2E_PLATFORM:-}" in
    pi)
        sudo apt-get update
        sudo apt-get install -y build-essential cmake git curl ca-certificates tmux
        ;;
    mac)
        if ! xcode-select -p >/dev/null 2>&1; then
            echo "Install the Xcode command line tools: xcode-select --install" >&2
            exit 2
        fi
        if ! command -v brew >/dev/null; then
            echo "Install Homebrew (https://brew.sh), then run setup.sh again." >&2
            exit 2
        fi
        for pkg in cmake tmux; do
            brew list --formula "$pkg" >/dev/null 2>&1 || brew install "$pkg"
        done
        ;;
    *)
        echo "Run this through setup.sh." >&2
        exit 2
        ;;
esac

if ! command -v rustup >/dev/null; then
    curl --proto '=https' --tlsv1.2 -fsSL https://sh.rustup.rs |
        sh -s -- -y --profile minimal --default-toolchain none
fi
rustup toolchain install "$rust" --profile minimal
