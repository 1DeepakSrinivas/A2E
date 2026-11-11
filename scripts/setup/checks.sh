#!/usr/bin/env bash
# Validate cases and tools, then run doctor. Called by setup.sh.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

bin=target/release/a2e
"$bin" tools validate
"$bin" cases validate

doctor=(doctor)
if [[ "${A2E_PLATFORM:-}" == mac ]]; then
    doctor+=(--probe)
fi
if ! "$bin" "${doctor[@]}"; then
    if [[ "${A2E_PLATFORM:-}" == pi ]]; then
        echo "doctor failed. The Pi baseline needs swap off: sudo dphys-swapfile swapoff && sudo systemctl disable dphys-swapfile" >&2
    fi
    exit 1
fi

if [[ "${A2E_PLATFORM:-}" == pi ]]; then
    if command -v vcgencmd >/dev/null; then
        echo "throttled: $(vcgencmd get_throttled)  (0x0 is clean)"
    fi
    if [[ ! -f .env ]]; then
        echo "No .env. Results stay in results/outbox until you copy .env.example to .env and set WORKER_URL and WRITE_TOKEN."
    fi
fi
