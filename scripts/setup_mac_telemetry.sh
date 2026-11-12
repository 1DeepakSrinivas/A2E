#!/usr/bin/env bash
# Allows the current user to run /usr/bin/powermetrics through sudo without a password.
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
printf '%s ALL=(root) NOPASSWD: /usr/bin/powermetrics\n' "$(id -un)" > "$tmp"

sudo visudo -cf "$tmp"
sudo install -m 0440 -o root -g wheel "$tmp" "$rule_file"
sudo -n -l /usr/bin/powermetrics > /dev/null
echo "Installed $rule_file. Check with: target/release/a2e doctor --probe"
