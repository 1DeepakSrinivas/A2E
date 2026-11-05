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
smoke run, held-out baseline, harness comparisons, context and tool-count
boundaries, and live Brave search. Set `BRAVE_SEARCH_API_KEY` for the live
suite. The other suites use recorded fixture data.

Rerun the same command to resume completed work. Use `--retry-failed` to
retry unsuccessful attempts. The database and streamed traces live under
`results/native`. Trace files contain prompts, raw model output, tool results,
and the evidence used for judging.

Latency includes failed attempts. The runtime measures model loading separately
and records inference, harness, and tool process RSS. Peak RSS comes from
samples. Pi power is unavailable. Mac power comes from privileged
`powermetrics` samples and covers the reported CPU, GPU, and ANE scope.

## Add a tool

Copy the [example manifest](../configs/native/tools/uppercase.toml). Set the
executable, argument schema, timeout, and maximum response size. Build the
[Rust example](../native/examples/rust_tool.rs) with
`cargo build --example rust_tool`, or compile the
[Go example](../native/examples/go-tool/main.go) with `go build` and point
the manifest at it.

The harness starts the executable once per task. It sends `init`, `call`,
and `snapshot` requests as JSON lines. The response echoes `protocol` and
`id` and contains `ok`, `output`, and `error`. Write diagnostics to stderr.
Use `A2E_TOOL_ROOT` for task files. Register the tool name in a task's
`tools` list, then run `a2e-native tools validate <manifest>` and pass
`--manifest <manifest>` to `plan` and `run`. Run tool executables you trust.

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
