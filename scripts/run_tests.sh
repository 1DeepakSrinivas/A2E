#!/usr/bin/env bash
# Grouped test runs. Usage: scripts/run_tests.sh [unit|integration|sweep|hardware|quick|all] [pytest args...]
#   unit         pure logic, <1s
#   integration  fake llama-server over HTTP, sampler, store (~30s)
#   sweep        full grid structure + baseline protocol mini sweeps (~30s)
#   hardware     real device / real llama-server; self-skips without opt-in env vars (see tests/hardware/conftest.py)
#   quick        unit + integration
#   all          everything (hardware tests skip themselves when not applicable)
set -euo pipefail
cd "$(dirname "$0")/.."
group="${1:-quick}"; shift || true
case "$group" in
  unit|integration|sweep|hardware) uv run pytest -m "$group" -ra "$@" ;;
  quick) uv run pytest -m "unit or integration" -ra "$@" ;;
  all)   uv run pytest -ra "$@" ;;
  *) echo "unknown group: $group" >&2; exit 2 ;;
esac
