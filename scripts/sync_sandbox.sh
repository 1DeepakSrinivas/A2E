#!/usr/bin/env bash
# Copy a sandbox tree built on this machine to a Pi and verify it there.
#   scripts/sync_sandbox.sh <user@host> [remote project dir]      default dir: ~/A2E
# Build the tree first with scripts/init_sandbox.sh.
set -euo pipefail
cd "$(dirname "$0")/.."

HOST="${1:?usage: scripts/sync_sandbox.sh <user@host> [remote project dir]}"
REMOTE="${2:-A2E}"
OUT="${A2E_SANDBOX:-sandbox}"

[ -d "$OUT/base" ] || { echo "no $OUT/base; run scripts/init_sandbox.sh first" >&2; exit 1; }
rsync -a --delete "$OUT/" "$HOST:$REMOTE/sandbox/"
ssh "$HOST" "cd $REMOTE && target/release/a2e-native sandbox verify"
