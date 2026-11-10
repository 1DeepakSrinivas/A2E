#!/usr/bin/env bash
# Generate and verify the workspace tree that the file cases read and write.
#   scripts/init_sandbox.sh [--seed N] [--force]      output: $A2E_SANDBOX or ./sandbox
# Uses target/release/a2e-native when present, otherwise builds it.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="${A2E_SANDBOX:-sandbox}"
BIN=target/release/a2e-native
if [ ! -x "$BIN" ]; then
  cargo build --release --locked
fi

"$BIN" sandbox init --out "$OUT" "$@"
"$BIN" sandbox verify --dir "$OUT"
