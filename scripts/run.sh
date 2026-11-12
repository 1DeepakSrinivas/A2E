#!/usr/bin/env bash
# Run suites in a detached tmux session so they survive a closed terminal or lost SSH link.
#   scripts/run.sh                                       interactive menu
#   scripts/run.sh start <suite>[,<suite>...] [--local] [a2e run options]
#   scripts/run.sh attach|status|logs|stop [name]
# One suite runs in the session a2e-<suite>. Several suites run one after another in a2e-queue,
# because only one run can hold the runtime lock. Output is also written to results/logs.
# --local sets WORKER_URL=http://localhost:1, so results stay in results/outbox.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

bin=target/release/a2e

usage() { sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }
die() { echo "$1" >&2; exit 1; }

need_tmux() { command -v tmux >/dev/null || die "tmux is not installed; run ./setup.sh."; }
running() { tmux has-session -t "a2e-$1" 2>/dev/null; }
sessions() { tmux list-sessions -F '#S' 2>/dev/null | sed -n 's/^a2e-//p' || true; }
suite_list() { local f; for f in configs/suites/*.toml; do basename "$f" .toml; done; }

# Ask for one name from the lines on stdin; a single line is chosen without asking.
choose() {
    local prompt="$1" items=() line i=1 reply
    while IFS= read -r line; do items+=("$line"); done
    [[ ${#items[@]} -gt 0 ]] || return 1
    if [[ ${#items[@]} -eq 1 ]]; then echo "${items[0]}"; return; fi
    for line in "${items[@]}"; do printf '  %d) %s\n' "$i" "$line" >&2; i=$((i + 1)); done
    read -r -p "$prompt " reply </dev/tty
    [[ "$reply" =~ ^[0-9]+$ && "$reply" -ge 1 && "$reply" -le ${#items[@]} ]] || die "Not a choice."
    echo "${items[$((reply - 1))]}"
}

yes_no() {
    local reply
    read -r -p "$1 [y/N] " reply </dev/tty
    [[ "$reply" =~ ^[yY] ]]
}

# launch <local 0|1> <suite list, space separated> [a2e run options]
launch() {
    local local_only="$1" suites="$2" session name s cmd="" log
    shift 2
    need_tmux
    [[ -x "$bin" ]] || die "$bin is missing; run ./setup.sh."
    for s in $suites; do
        [[ "$s" =~ ^[a-z0-9_]+$ ]] || die "Suite names use lowercase letters, digits and underscores."
        [[ -f "configs/suites/$s.toml" ]] || die "No suite file configs/suites/$s.toml."
    done
    set -- $suites
    if [[ $# -eq 1 ]]; then name="$1"; else name=queue; fi
    session="a2e-$name"
    [[ -z "$(sessions)" ]] || die "Another a2e session is running ($(sessions | tr '\n' ' ')); only one run can hold the runtime lock."

    "$bin" sandbox verify >/dev/null 2>&1 ||
        die "The sandbox is missing or does not match its manifest; run scripts/init_sandbox.sh."
    for s in $suites; do "$bin" plan "configs/suites/$s.toml" >/dev/null; done

    local prefix=""
    if [[ "$local_only" == 1 ]]; then
        prefix="env WORKER_URL=http://localhost:1"
        echo "Local only: results stay in results/outbox."
    else
        echo "Results are uploaded to the Worker named by WORKER_URL or .env. Use --local to keep them off the Worker."
    fi
    for s in $suites; do cmd+="$prefix $bin run configs/suites/$s.toml ${RUN_ARGS:-}; "; done

    mkdir -p results/logs
    log="results/logs/$name-$(date +%Y%m%d-%H%M%S).log"
    tmux new-session -d -s "$session" "bash -c '{ $cmd} 2>&1 | tee $log'"
    echo "Started $session ($suites). Log: $log"
    echo "Attach: scripts/run.sh attach $name   (detach with Ctrl-b d)"
}

start_cmd() {
    local suites="${1//,/ }" local_only=0 arg
    shift
    RUN_ARGS=""
    for arg in "$@"; do
        if [[ "$arg" == --local ]]; then local_only=1; else RUN_ARGS+="$arg "; fi
    done
    launch "$local_only" "$suites"
}

attach_cmd() { need_tmux; running "$1" || die "No session a2e-$1."; tmux attach -t "a2e-$1"; }
status_cmd() {
    if command -v tmux >/dev/null && running "$1"; then echo "a2e-$1 is running."; else echo "a2e-$1 is not running."; fi
    "$bin" upload --status
}
logs_cmd() {
    local latest
    latest="$(ls -t results/logs/"$1"-*.log 2>/dev/null | head -n 1 || true)"
    [[ -n "$latest" ]] || die "No logs for $1."
    tail -n 40 -f "$latest"
}
stop_cmd() {
    need_tmux
    running "$1" || die "No session a2e-$1."
    tmux send-keys -t "a2e-$1" C-c
    echo "Sent Ctrl-C to a2e-$1. The session closes when the runner exits."
}

interactive() {
    [[ -t 0 && -t 1 ]] || usage
    local action name picked suites i=1 s reply local_only=0
    echo "A2E runs"
    action="$(printf '%s\n' "Start suites" "Attach to a run" "Status" "Show log" "Stop a run" "Quit" | choose "Choose:")"
    case "$action" in
        "Start suites")
            echo "Suites:"
            for s in $(suite_list); do printf '  %d) %s\n' "$i" "$s"; i=$((i + 1)); done
            read -r -p "Numbers separated by spaces, or 'a' for all: " reply </dev/tty
            suites=""
            if [[ "$reply" == a ]]; then
                suites="$(suite_list | tr '\n' ' ')"
            else
                for picked in $reply; do
                    [[ "$picked" =~ ^[0-9]+$ ]] || die "Not a number: $picked"
                    s="$(suite_list | sed -n "${picked}p")"
                    [[ -n "$s" ]] || die "No suite number $picked."
                    suites+="$s "
                done
            fi
            [[ -n "$suites" ]] || die "No suites chosen."
            RUN_ARGS=""
            if yes_no "Send results to the Worker? (No keeps them in results/outbox)"; then local_only=0; else local_only=1; fi
            if yes_no "Retry attempts that failed before?"; then RUN_ARGS="--retry-failed"; fi
            echo "Run: $suites| $([[ $local_only == 1 ]] && echo local only || echo upload to Worker) | ${RUN_ARGS:-no retry}"
            yes_no "Start?" || die "Cancelled."
            launch "$local_only" "$suites"
            ;;
        "Attach to a run") need_tmux; name="$(sessions | choose "Run:")" || die "No run is active."; attach_cmd "$name" ;;
        "Status")
            name="$(sessions | choose "Run:" || true)"
            if [[ -n "$name" ]]; then status_cmd "$name"; else echo "No run is active."; "$bin" upload --status; fi
            ;;
        "Show log")
            name="$(for f in results/logs/*.log; do [[ -e "$f" ]] && basename "$f"; done | sed 's/-[0-9]*-[0-9]*\.log$//' | sort -u | choose "Log:")" || die "No logs yet."
            logs_cmd "$name"
            ;;
        "Stop a run") need_tmux; name="$(sessions | choose "Run:")" || die "No run is active."; stop_cmd "$name" ;;
        *) ;;
    esac
}

if [[ $# -eq 0 ]]; then interactive; exit; fi
action="$1"
shift
case "$action" in
    start) [[ $# -ge 1 ]] || usage; start_cmd "$@" ;;
    attach|status|logs|stop)
        [[ $# -ge 1 ]] || usage
        [[ "$1" =~ ^[a-z0-9_]+$ ]] || die "Names use lowercase letters, digits and underscores."
        "${action}_cmd" "$1"
        ;;
    *) usage ;;
esac
