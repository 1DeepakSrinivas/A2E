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
"$bin" "${doctor[@]}"

if [[ "${A2E_PLATFORM:-}" == pi ]]; then
    if [[ -n "$(swapon --show --noheadings 2>/dev/null)" ]]; then
        echo "Swap is on. The Pi baseline refuses to run with swap on: sudo dphys-swapfile swapoff && sudo systemctl disable dphys-swapfile"
    fi
    if command -v vcgencmd >/dev/null; then
        echo "throttled: $(vcgencmd get_throttled)  (0x0 is clean)"
    fi
    # The 1.2B model needs nearly all of the 1 GB, so the desktop and the GPU split matter
    if [[ "$(systemctl get-default 2>/dev/null)" == graphical.target ]]; then
        echo "The desktop is enabled and holds memory the models need: sudo systemctl set-default multi-user.target && sudo reboot"
    fi
    if command -v vcgencmd >/dev/null; then
        gpu="$(vcgencmd get_mem gpu 2>/dev/null | tr -dc '0-9')"
        if [[ -n "$gpu" && "$gpu" -gt 16 ]]; then
            echo "GPU memory is ${gpu}M. Set gpu_mem=16 in /boot/firmware/config.txt and reboot to give it to the models."
        fi
    fi
    awk '/MemAvailable/ {printf "available memory: %d MiB (doctor shows the headroom per model)\n", $2/1024}' /proc/meminfo
    if [[ ! -f .env ]]; then
        echo "No .env. Results stay in results/outbox until you copy .env.example to .env and set WORKER_URL and WRITE_TOKEN."
    fi
fi
