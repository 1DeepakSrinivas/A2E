#!/usr/bin/env bash
# Check the machine, then run (or resume) a suite. Usage: scripts/run_suite.sh <suite> [a2e run options...]
#   scripts/run_suite.sh smoke
#   scripts/run_suite.sh pilot --limit 10
# On a Mac, energy needs root:  sudo -E scripts/run_suite.sh pilot
# Stop at any time with Ctrl+C; run the same command again to resume where it stopped.
set -euo pipefail
cd "$(dirname "$0")/.."
suite="${1:?usage: scripts/run_suite.sh <suite> [options]}"; shift
uv run a2e doctor || { echo "doctor found a problem (see FAIL lines above); fix it or run 'uv run a2e run $suite' directly"; exit 1; }
uv run a2e plan "$suite"
exec uv run a2e run "$suite" "$@"
