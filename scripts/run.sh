#!/usr/bin/env bash
# Run a suite in a detached tmux session so it survives a closed terminal or lost SSH link.
#   scripts/run.sh start <suite> [--local] [a2e run options]   e.g. start smoke --retry-failed
#   scripts/run.sh attach|status|logs|stop <suite>
# The session is named a2e-<suite>. Output is also written to results/logs. --local sets
# WORKER_URL=http://localhost:1, so results stay in results/outbox instead of reaching the Worker.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

usage() { sed -n '2,6p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }

[[ $# -ge 2 ]] || usage
action="$1"
suite="$2"
shift 2
[[ "$suite" =~ ^[a-z0-9_]+$ ]] || { echo "Suite names use lowercase letters, digits and underscores." >&2; exit 2; }
session="a2e-$suite"
bin=target/release/a2e
suite_file="configs/suites/$suite.toml"

need_tmux() {
    command -v tmux >/dev/null || { echo "tmux is not installed; run ./setup.sh." >&2; exit 1; }
}
running() { tmux has-session -t "$session" 2>/dev/null; }

case "$action" in
    start)
        need_tmux
        [[ -x "$bin" ]] || { echo "$bin is missing; run ./setup.sh." >&2; exit 1; }
        [[ -f "$suite_file" ]] || { echo "No suite file $suite_file." >&2; exit 1; }
        if running; then
            echo "Session $session is already running. Attach with: scripts/run.sh attach $suite" >&2
            exit 1
        fi
        if tmux list-sessions -F '#S' 2>/dev/null | grep -q '^a2e-'; then
            echo "Another a2e session is running; only one run can hold the runtime lock:" >&2
            tmux list-sessions -F '#S' | grep '^a2e-' >&2
            exit 1
        fi

        local_only=0
        args=()
        for arg in "$@"; do
            if [[ "$arg" == --local ]]; then local_only=1; else args+=("$arg"); fi
        done
        env_prefix=()
        if [[ "$local_only" == 1 ]]; then
            env_prefix=(env WORKER_URL=http://localhost:1)
            echo "Local only: results stay in results/outbox."
        else
            echo "Results are uploaded to the Worker named by WORKER_URL or .env. Use --local to keep them off the Worker."
        fi

        "$bin" sandbox verify >/dev/null 2>&1 || {
            echo "The sandbox is missing or does not match its manifest; run scripts/init_sandbox.sh." >&2
            exit 1
        }
        "$bin" plan "$suite_file" >/dev/null

        mkdir -p results/logs
        log="results/logs/$suite-$(date +%Y%m%d-%H%M%S).log"
        cmd="${env_prefix[*]:-} $bin run $suite_file ${args[*]:-} 2>&1 | tee $log"
        tmux new-session -d -s "$session" "bash -c '$cmd'"
        echo "Started $session. Log: $log"
        echo "Attach: scripts/run.sh attach $suite   (detach with Ctrl-b d)"
        ;;
    attach)
        need_tmux
        running || { echo "No session $session." >&2; exit 1; }
        tmux attach -t "$session"
        ;;
    status)
        if command -v tmux >/dev/null && running; then
            echo "$session is running."
        else
            echo "$session is not running."
        fi
        "$bin" upload --status
        ;;
    logs)
        latest="$(ls -t results/logs/"$suite"-*.log 2>/dev/null | head -n 1 || true)"
        [[ -n "$latest" ]] || { echo "No logs for $suite." >&2; exit 1; }
        tail -n 40 -f "$latest"
        ;;
    stop)
        need_tmux
        running || { echo "No session $session." >&2; exit 1; }
        tmux send-keys -t "$session" C-c
        echo "Sent Ctrl-C to $session. The session closes when the runner exits."
        ;;
    *) usage ;;
esac
