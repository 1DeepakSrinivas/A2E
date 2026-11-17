# Repository notes

Follow the branch and commit rules in `CLAUDE.md`.

Run `cargo fmt --all`, then `cargo fmt --all -- --check` for Rust changes.
Run `cargo clippy --workspace --all-targets -- -D warnings` for Rust changes.
Fix findings before handoff and report any checks that could not run.

Keep comments short. Explain decisions, protocol constraints, and measurement
limitations when the code cannot express them. Avoid ASCII art, decorative
separators, and comments that repeat the implementation.

Cases are TOML files under `configs/cases`. Every case needs `[[golden]]`
calls, and the tests replay them through the agent loop. Run
`a2e cases validate` after any case, rubric, or tool definition change.

Missing measurement data stays unavailable. Do not turn missing power or
inference timings into zeros, and do not report software power assumptions as
measured energy.

Each extension point has one registry. Add a tool in `src/tools` plus
`configs/tools/standard.toml`, a check kind in `src/eval/checks`, a model format in
`src/adapter`, a model in `configs/models.toml`, a suite in `configs/suites`, and a
shared variant in `configs/variants.toml`. `docs/GUIDE.md` lists the steps for each.

Write documents and help text in plain words, with short headings in sentence case. A
test in `tests/docs.rs` fails when a document names a command, flag, environment
variable or path that does not exist, so update the docs in the same change as the code.
