#!/usr/bin/env bash
# Allows the runner's powermetrics command through sudo without a password.
# Remove with: sudo rm /etc/sudoers.d/a2e-powermetrics
set -euo pipefail

if [ "$(uname -s)" != "Darwin" ]; then
    echo "This script is for macOS only." >&2
    exit 1
fi
if [ "$(id -u)" -eq 0 ]; then
    echo "Run this script as your normal user, not with sudo." >&2
    exit 1
fi

rule_file=/etc/sudoers.d/a2e-powermetrics
tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT
# Keep these arguments in sync with src/measure.rs.
sampler=(/usr/bin/powermetrics --samplers "cpu_power,gpu_power,thermal" -i 500)
sampler_rule="${sampler[*]}"
printf '%s ALL=(root) NOPASSWD: %s\n' "$(id -un)" "${sampler_rule//,/\\,}" > "$tmp"

sudo visudo -cf "$tmp"
sudo install -m 0440 -o root -g wheel "$tmp" "$rule_file"
sudo -n -l -- "${sampler[@]}" > /dev/null
echo "Installed $rule_file. Check with: target/release/a2e doctor --probe"
