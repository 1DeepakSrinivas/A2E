# Native runtime

The Rust CLI runs FunctionGemma 270M Q4_K_M, Qwen3-0.6B Q4_K_M, and
LFM2.5-1.2B-Instruct QAD Q4_0 on a Pi 3B+ or Apple Silicon Mac. The model
registry records the exact files, revisions, and published hashes.

## Setup

On a Pi 3B+ with 64-bit Raspberry Pi OS, run `./setup-pi.sh` as your normal
user. The script uses sudo for APT packages and builds with one job by default.
Builds may need temporary swap on a 1 GB device. Disable swap before measuring.

On an Apple Silicon Mac with Rust 1.96 or newer, CMake, and the Xcode command
line tools, run the following from the project directory.

```bash
bash scripts/build_llama.sh
cargo build --release --locked
target/release/a2e-native models fetch
target/release/a2e-native doctor --probe
```

The checkout and build live under `deps/llama.cpp`. Git ignores the local
dependencies, models, and results. Run setup on each machine to build for its
architecture. Copy `configs/native/local.example.toml` to
`configs/native/local.toml` to override machine settings. A different memory
limit or GPU setting produces a separate study.

## Run and read results

```bash
target/release/a2e-native plan
target/release/a2e-native run
target/release/a2e-native results
target/release/a2e-native results --json --csv results/native/runs.csv
```

Pass a suite file to `plan` or `run`. The native suites cover the development
smoke run, held-out baseline, harness comparisons, context boundaries, and live
Brave search. Set `BRAVE_SEARCH_API_KEY` for the live
suite. The other suites use recorded fixture data.

Rerun the same command to resume completed work. Use `--retry-failed` to
retry unsuccessful attempts. The database and streamed traces live under
`results/native`. Trace files contain prompts, raw model output, tool results,
and the evidence used for judging.

Latency includes failed attempts. The runtime measures model loading separately
and records inference and harness RSS. Tools run inside the harness process,
so harness RSS includes them. Peak RSS comes from samples. Pi power is unavailable. Mac power comes from privileged
`powermetrics` samples and covers the reported CPU, GPU, and ANE scope.

## Cases

Each benchmark case is one TOML file under `configs/native/cases/<category>/`.
A case sets its scope (prompt and call limits), the tools it may use, fixtures,
typed outcome checks, and optional judge settings. A suite selects cases by
path, split, and tags. Run `a2e-native cases validate` to check every file
against the tool definitions and `a2e-native cases list --ids` to see the counts.
See `configs/native/cases/chain/difference-dev-0.toml` for a worked example.

## Add a tool

All tools are Rust functions that run inside the harness process, so there is
no tool executable, protocol, or second process to measure. A tool has two
parts:

1. A handler in `native/src/tools/<domain>.rs`. It is a function
   `fn(&mut World, &Value) -> Result<Value>` that you register by name in the
   module's `register` function. `World` holds the task state (files, stores,
   clock, database, pages).
2. A definition in `configs/native/tools/standard.toml` with the name, the
   description, and the JSON Schema for the arguments. This is what the model
   sees, so wording changes can be tried without rebuilding.

Then list the tool in a case's `tools.allowed`. Run
`a2e-native tools validate` to check that every definition has a handler and a
valid schema. Keep handlers bounded in time and output, because the harness
stops a task when a call passes `budget.tool_seconds` or returns more than
`budget.tool_output_bytes`.

## Score retrieval summaries

Set `OPENROUTER_API_KEY`, `A2E_JUDGE_MODEL`, and `A2E_JUDGE_PROVIDER`.
Choose a model and provider that support strict JSON Schema output. Run
`target/release/a2e-native judge --study <study-id>` after the measured run.
The command sends the task, captured page excerpts, and answer to OpenRouter.
Judge calls and costs are recorded separately from the edge measurements.

The [rubric](../native/rubric.md) assigns up to 16 points. A provisional
judgment appears in the raw export. Summary comparisons require passing
`--calibration <file>` with a human-scored set. The JSON file contains
`reviewed_by` and at least four `cases`. Each case has a `packet`,
`expected_verdict`, and `scores` keyed by the four rubric dimensions.
Calibration requires at least 80% verdict agreement and at most 0.75 mean
absolute error per dimension.
