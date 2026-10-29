#!/usr/bin/env bash
# Run a named test group from configs/tests.toml. Usage: scripts/run_tests.sh [group] [pytest args...]
# Groups: `scripts/run_tests.sh --list`. Default group: quick.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ "${1:-}" = "--list" ]; then exec uv run a2e test --list; fi
group="${1:-quick}"; shift || true
exec uv run a2e test "$group" "$@"
