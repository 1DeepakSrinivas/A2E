# Architecture

This document describes how the A2E code is organized, how data moves through it, and where to make common changes. The user guide explains how to use the system. The methods report explains why it was built this way.

## 1. Layers

A2E has five layers. Each layer only calls the layers below it.

1. **Interfaces.** `cli.py` (the `a2e` command) and `tui.py` (the terminal UI). Both call the same functions.
2. **Orchestration.** `suite.py` turns a suite file into jobs. `runner.py` runs the jobs. `results.py` reads results back. `doctor.py`, `env.py`, `settings.py` and `testgroups.py` support them.
3. **Workloads.** `plain.py` (plain generation), `agent/` with `tools/` and `tasks/` (custom tool chains), `bfcl/` (the BFCL benchmark), and `bench.py` (llama-bench and perf).
4. **Measurement.** `server.py` starts llama-server, `client.py` talks to it, `metrics/` samples power, temperature, clock and memory, and `hardware.py` picks the right sources per device.
5. **Storage.** `store.py`, a SQLite database with a versioned schema.

`third_party/bfcl_eval/` holds the official BFCL checker and data, copied unchanged from upstream.

## 2. Modules

### 2.1 Interfaces

| Module | What it does |
|---|---|
| `cli.py` | The `a2e` command. Subcommands are `doctor`, `suites`, `plan`, `run`, `suite-lock`, `results ...`, `test`, `tui` and `bfcl ...`. `format_event` turns runner events into the log lines that the command line and the TUI both print. |
| `tui.py` | A Textual app with five tabs (Suites, Run, Results, Tests, Doctor). Runs and test groups run in worker threads and send updates back with `call_from_thread`. A run can be stopped through a `threading.Event`, which the runner checks between tasks. |

### 2.2 Orchestration

| Module | What it does |
|---|---|
| `settings.py` | Loads `configs/models.toml`, merges `configs/local.toml` and environment variables over it, and resolves paths. `detect_device` and `resolve_device` decide between `pi5` and `mac`. |
| `suite.py` | Pydantic models for suite files, with strict validation. `expand` turns a suite into an ordered list of job dictionaries with keys and skip reasons. `plan_summary` counts them and `lock_entry` makes the lock file entry. |
| `runner.py` | `Runner` opens or resumes an experiment, records jobs, recovers from crashes, starts and reuses llama-server, measures idle power, waits for cooldown, runs each repeat with sampling, computes per-stage energy and writes everything. `EXECUTORS` maps a task kind to its function. `RunnerHooks` lets tests replace the server, the client or the instruments. |
| `results.py` | Builds data frames from the analysis views: `experiments`, `jobs`, `summary`, `stages`, `stability`, `timeline`, `bench_summary`, `kernels` and the markdown `report_markdown`. |
| `env.py` | Captures the machine snapshot stored with each experiment, and fingerprints model files. |
| `doctor.py` | The checks behind `a2e doctor`. |
| `testgroups.py` | Loads `configs/tests.toml` and builds pytest command lines. |

### 2.3 Workloads

| Module | What it does |
|---|---|
| `plain.py` | Picks plain prompts with the seed and runs one generation of exactly `decode_len` tokens. |
| `agent/loop.py` | `model_step` does one model call: apply the chat template, call `/completion`, parse. `TurnRecord` holds a turn's results and `finalize` turns six timestamps into stage times and windows. `AgentLoop` runs a custom chain: call the model, validate, run tools, feed results back, until `final_answer`, a parse error, a server error or `max_turns`. |
| `agent/prompt.py` | The fixed tool prompts (`SYSTEM` for custom chains, `SYSTEM_BFCL` for BFCL) and the tool-response message format. |
| `agent/parse.py` | Reads tool calls out of model text. Handles thinking blocks, code fences, `<tool_call>` tags, single objects, arrays, `parameters` instead of `arguments`, and arguments given as a JSON string. Errors are `empty`, `no_json`, `malformed_json` and `bad_shape`. |
| `agent/grammar.py` | `call_schema` builds the JSON schema of valid calls that grammar mode sends to llama-server. `SUPPORTED_KEYWORDS` lists the JSON Schema keywords the tool schemas may use. |
| `agent/schema_gen.py` | `pad_tools` adds seeded distractor tools until the rendered tool list reaches the target token count. |
| `tools/base.py`, `tools/builtin.py` | The tool registry, schema validation and the real local tools (calculator, notes on disk, document search and read, reminders, `final_answer`), plus the synthetic document collection. |
| `tasks/custom.py`, `tasks/scoring.py` | The chain generators for the three families and the step scoring, including steps whose expected arguments depend on earlier tool output. |
| `bfcl/` | `load.py` (categories, cases, stratified subsampling), `convert.py` (BFCL types to JSON Schema, safe call strings), `score.py` (official checkers), `single_turn.py`, `multi_turn.py`, `runner.py` (case rows, raw outputs, rescoring) and `replay.py`. `paths.py` puts the vendored package on the import path. |
| `bench.py` | Builds llama-bench command lines, runs them optionally under perf, and parses both outputs. |

### 2.4 Measurement and storage

| Module | What it does |
|---|---|
| `server.py` | `build_args` turns a configuration into llama-server arguments. `LlamaServer` starts the process, waits for `/health` and stops it. |
| `client.py` | `Client.complete` calls `/completion` and turns the `timings` block into a `Completion`. Also `apply_template`, `tokenize`, `slots` and `reset_cache`. |
| `metrics/sampler.py` | A background thread that samples memory and CPU of the server process and its children, temperature, clock, power, throttling and thermal pressure at a fixed rate. |
| `metrics/power_*.py`, `metrics/thermal.py` | The Pi PMIC reader, the Mac `powermetrics` reader, the `vcgencmd` parsers and `integrate_energy`. |
| `hardware.py` | `make_instruments` picks the power and thermal source for the device. Also the Pi governor and the cooldown wait. |
| `store.py` | The SQLite store: schema migrations, WAL mode, transactions, inserts, queries and export. |

## 3. Data flow of one run

```text
suite.toml ──expand──▶ jobs ──Runner──▶ for each job:
                                         start llama-server (if its settings changed) ──▶ idle power
                                         for each repeat:
                                           Sampler.start ─────────────────────────────┐
                                           executor (plain | custom | bfcl):          │ (t, rss, cpu,
                                             for each task:                           │  temp, freq,
                                               model_step: /apply-template            │  power, throttle,
                                                           /completion ─▶ timings     │  pressure)
                                                           parse                      │
                                               validate ─▶ run tools                  │
                                               TurnRecord.finalize(stamps)            │
                                           Sampler.stop ◀─────────────────────────────┘
                                           turn_row: integrate power over each stage window
                                           one transaction: turns, cases, samples, run complete
                                         job done
```

Bench jobs follow the same pattern without llama-server. The runner stops the server, runs llama-bench for each test size, and samples the llama-bench process.

## 4. Job and run lifecycle

```text
job:  pending ──▶ running ──▶ done
                     │  └──▶ failed   (error stored; --retry-failed runs it again)
                     │  └──▶ skipped  (schema too large at run time)
                     └──▶ pending     (stopped, or found running after a crash)
       skipped at plan time: memory rule, governor on the Mac

run:  running ──▶ complete
         └─────▶ aborted   (job stopped, failed or crashed; all runs of an unfinished job)
```

The rule that keeps the data clean is that a job's runs only count as a set. When a job starts, any runs left from an earlier attempt are marked aborted. When a job stops before it is done, all its runs are marked aborted. The analysis views only read complete runs.

## 5. Timing model

A turn is timed with six `time.perf_counter` stamps:

| Stamp | Taken |
|---|---|
| t0 | before the `/apply-template` call |
| t1 | after the template call |
| t2 | after the `/completion` call returns |
| t3 | after parsing |
| t4 | after validation |
| t5 | after the tools ran |

`template_ms` is t1 minus t0, `parse_ms` is t3 minus t2, `validate_ms` is t4 minus t3, `tool_wait_ms` is t5 minus t4 and `wall_ms` is t5 minus t0. `prefill_ms` and `decode_ms` come from llama-server's own timings. `harness_ms` is the part of the turn not covered by any of these. The power samples use the same clock, so energy can be computed over any of these windows.

## 6. Design decisions in the code

- **llama-server over HTTP.** The server runs as its own process. A2E can measure its memory from outside, restart it per model, and use the same interface on both devices. The `timings` block in each reply gives prefill and decode times measured inside the server.
- **One fixed prompt format.** All models get the same tool instructions. Only the role markup comes from each model's own chat template, through `/apply-template`. This keeps the prompt identical across models apart from the markup.
- **Grammar mode through `json_schema`.** A2E builds a JSON schema of valid calls and llama-server converts it to a grammar. A2E does not maintain its own grammar converter.
- **Vendored BFCL checker.** The official files are copied byte for byte and checked against a hash manifest. One stub replaces a module that would import every LLM SDK.
- **Safe execution for BFCL multi-turn.** The upstream executor runs `eval()` on call strings. A2E builds those strings itself from whitelisted function names and `repr()` of validated JSON values, so model text never reaches `eval()` directly.
- **Deterministic jobs.** Job keys are hashes of canonical JSON, and the expansion order is fixed. Every random choice uses the suite seed.
- **One transaction per run.** A run is written at once after it finishes, so energy can be computed from the complete power trace and a crash leaves no partial run.

## 7. How to extend

**Add a model.** Add a `[models.<name>]` entry with `params_b` to `configs/models.toml`. Put the files where the template expects them, or set `files` in `local.toml`. Add the model to the suites that should use it and run `uv run a2e suite-lock`.

**Add a task kind.** Add the kind to `Task.kind`, `AXES_FOR_KIND` and `KIND_ORDER` in `suite.py`, add its grid to `expand`, and write an executor `fn(client, job, ctx) -> list[Unit]` in `runner.py` registered in `EXECUTORS`. A `Unit` holds the task's turns and its score.

**Add a custom tool family.** Add the tools to `tools/builtin.py`, a generator to `tasks/custom.py`, and the family's tool list to `FAMILY_TOOLS`.

**Add a BFCL category.** Add its data files to `CATS` in `scripts/sync_bfcl.sh`, run the script, and add the category to `bfcl/load.py`. The oracle sweep in `tests/sweep/test_bfcl_oracle.py` will show whether the official checker accepts the ground truth.

**Add a power source.** Subclass `PowerSource` with `read_watts` and a `scope` string, and return it from `make_instruments` in `hardware.py`. A USB power meter fits here without changing any stored data.

**Change the database schema.** Append a new SQL entry to `MIGRATIONS` in `store.py`. Do not edit an old entry. Existing databases upgrade the next time they are opened.

## 8. Tests

Tests live in four folders, and each folder is a pytest marker:

- `tests/unit` holds pure logic tests.
- `tests/integration` starts real processes. `tests/fakes/fake_llama_server.py` answers `/health`, `/completion`, `/apply-template`, `/tokenize` and `/slots` like llama-server, with prefix cache behaviour and injectable failures. `fake_llama_bench.py` and `fake_perf.py` do the same for llama-bench and perf. `tests/rig.py` builds a complete machine setup in a temporary folder with these fakes.
- `tests/sweep` runs large checks, e.g., every BFCL ground truth answer through the official checker.
- `tests/hardware` runs on a real device and skips itself elsewhere.

`configs/tests.toml` defines named groups on top of these markers.
