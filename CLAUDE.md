# Working rules for this repository

## Code quality

- Run `cargo fmt --all`, then verify with `cargo fmt --all -- --check` on Rust changes.
- Run `cargo clippy --workspace --all-targets -- -D warnings` on Rust changes.
- Fix findings before handing off, and report checks that could not run.
- Keep comments short and focused on decisions or constraints. Do not add ASCII art or decorative comment separators.

## Branches

Build each functional area on its own branch. Name it `<area>/<topic>`, and add `-<feature>` when the work is a separate feature inside that area. Use lowercase words joined by hyphens.

| Area | Use it for | Examples |
|---|---|---|
| `core` | harness, runner, store, server and client, suites | `core/harness-agentic-loop`, `core/runner-resume` |
| `metrics` | power, thermal, memory sampling, energy | `metrics/power-usb-meter` |
| `benchmarks` | cases, suites, bench and perf | `benchmarks/web-cases` |
| `results` | results queries, reports, plots, exports | `results/plots` |
| `ui` | CLI and TUI | `ui/tui-run-tab` |
| `testing` | tests and test tooling | `testing/unit-tests`, `testing/hardware-tests` |
| `docs` | documents and the methods report | `docs/user-guide` |
| `infra` | setup scripts, configs, dependencies | `infra/setup-scripts` |
| `fix` | a bug fix that belongs to no feature branch | `fix/energy-short-windows` |

Until the existing work is merged into `main`, start new branches from `claude/upbeat-bell-buo31j`. After that, start them from `main`.

## Commits

- Author and committer: `1DeepakSrinivas <90671142+1DeepakSrinivas@users.noreply.github.com>`.
- Do not add `Co-Authored-By`, `Claude-Session` or any other attribution lines to commit messages or pull requests.
