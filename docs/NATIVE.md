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

Each device has an optional file `configs/native/hosts/<user>@<hostname>.toml`,
for example `deepak@RPi.toml`. The CLI loads the file that matches the current
user and hostname, or the one named by `A2E_HOST_ID`. The file can hold any
setting from the models file plus a `[roles]` table with the booleans
`controller` and `benchmark_target`, which both default to false. Settings
apply in this order, and the later one wins: the models file, the host file,
`local.toml`, then environment variables. `doctor` prints the host, the loaded
host file and the roles.

## Run and read results

```bash
target/release/a2e-native plan
target/release/a2e-native run
target/release/a2e-native upload --status
```

Pass a suite file to `plan` or `run`. The shipped suites are listed in the
suites section below. Set `BRAVE_SEARCH_API_KEY` for the live suite, either in
the shell or in a `.env` file in the working directory. The CLI reads `.env` at
start-up as `KEY=VALUE` lines, and a variable already set in the shell wins. The
other suites use recorded fixture data.

Run the same command again to resume completed work. Use `--retry-failed` to
retry unsuccessful attempts. The database and the trace files live under
`results/native`. Trace files contain prompts, raw model output, tool results,
and the evidence used for judging.

## Sending results to the cloud

Judging, scoring and reports run in the A2E-Results project, which is checked
out as the submodule `third_party/a2e-results`. Its `docs/API.md` describes the
Worker API. After each attempt the runner posts the study, the model load and the
finished attempt to the Worker. Set `WORKER_URL` and `WRITE_TOKEN` in the shell
or in `.env` (see `.env.example`).

Every request is written to `results/native/outbox` first and removed once the
Worker stores it. If the network is down, or the variables are not set, the
requests stay in the outbox and the run continues. Send them later with
`target/release/a2e-native upload`, and check what is waiting with
`upload --status`. A request the Worker refuses for good moves to
`outbox/rejected` with the error beside it. Resume state is kept in
`results/native/resume.jsonl`. On a development machine run
`git submodule update --init` to fetch the submodule; the Pi does not need it.

Latency includes failed attempts. The runtime measures model loading
separately and records inference and harness RSS. Tools run inside the harness
process, so harness RSS includes them. Peak RSS comes from the samples. Pi
power is unavailable. Mac power comes from privileged `powermetrics` samples
and covers the reported CPU, GPU, and ANE scope.

Greedy decoding is used for every attempt. Repeats measure how latency, memory,
and temperature vary. They do not measure variation in success.

## Cases

Each benchmark case is one TOML file under `configs/native/cases/<category>/`.
The library has 108 cases. The categories follow the BFCL families that suit
models under 2 billion parameters.

| Category | Cases | What it checks |
|---|---|---|
| simple | 16 | One call with the right arguments |
| multiple | 16 | Choosing the right tool among 2, 4, or 8 tools |
| irrelevance | 10 | Making no call when no tool fits |
| missing_param | 10 | Asking for a required value instead of guessing it |
| chain | 24 | Two to six dependent calls, where the final state decides |
| recovery | 12 | Recovering from failed calls, slow tools, and large results |
| robustness | 8 | Ignoring instructions inside files, pages, and database rows |
| web | 8 | Search and fetch over recorded pages |

The four live search cases are in `configs/native/cases/web_live` and run only
from the `web_live` suite.

Run `a2e-native cases validate` to check every file against the tool
definitions. When the submodule is checked out, it also checks judge rubric names against `third_party/a2e-results/cli/rubrics`. Run `a2e-native cases list --ids` to see the
counts by category and split. See `configs/native/cases/chain/chain-copy.toml`
for a worked example.

### Case file

```toml
id = "chain-copy"              # unique, matches the file name
category = "chain"             # one of the categories above
tags = ["tier2", "smoke", "files"]
split = "dev"                  # dev or heldout, never both in one suite
golden_answer = "Copied."      # final answer used by the golden replay

[scope]
prompt = "Read reading.txt and copy its contents to copy.txt."
expected_calls = 2             # optional
max_model_calls = 6            # optional, never above the suite budget
final_answer = "required"      # required or optional

[tools]
allowed = ["read_file", "write_file"]
distractors = 0                # extra unrelated tools added in autonomous mode

[fixtures.files]
"reading.txt" = "Reading: 17"

[[stage]]                      # guided mode only
instruction = "Read reading.txt."
tools = ["read_file"]

[[check]]
kind = "file"
path = "copy.txt"
eq = "Reading: 17"

[output]
files = ["copy.txt"]           # shown in the report

[[golden]]                     # reference calls used only by the tests
tool = "read_file"
args = { path = "reading.txt" }
```

Unknown keys are errors, so a typo is caught by `cases validate`.

### Fixtures

The `[fixtures]` table sets the starting state of one attempt. It can hold
`files`, `devices`, `pages`, `kv`, `notes`, `todos`, a `clock` timestamp, and a
`db` seed. The database seed is SQL text, either inline as `sql` or as a
`sql_file` path relative to the case file. The seeds in
`configs/native/fixtures` are loaded into an in-memory SQLite database for each
attempt.

Use `[[fixtures.faults]]` to break a tool on purpose. Each entry names a `tool`
and a `mode`.

- `fail_n` fails the first `n` calls.
- `rate_limit` returns a rate limit error for the first `n` calls.
- `slow_ms` delays each call by `ms` milliseconds.
- `oversize_bytes` pads the result with `bytes` bytes.
- `malformed` replaces the result with garbled text.

### Checks

Checks are typed and are evaluated after the attempt by a function that needs
no tools or model, so each kind has its own unit tests. An attempt succeeds
when every check passes. A case with no judge ends as `success` or a failure.

- `call` needs a call to `tool`. The `args` match is `exact` or `subset`, and
  numbers compare with a tolerance `tol`. Use `min` and `max` to bound how
  many times it is called.
- `no_call` fails if the tool is called at all.
- `call_order` needs the listed tools to appear in that order.
- `calls_at_most` bounds the total number of tool calls.
- `file` checks a file with `eq`, `contains`, `regex`, `json_eq`, or
  `absent = true`.
- `state` compares a JSON pointer into the final state, e.g.
  `/devices/fan_1/setting`, with `eq`.
- `answer` checks the final answer with `contains`, `not_contains`, `regex`,
  `exact`, or `number = { value = 12, tol = 0.01 }`.
- `fetched_url`, `fetched_source`, and `fetched_count` check what the web
  tools returned.

A failed attempt gets a failure class from the first check or limit that
failed. The classes are `missing_call`, `wrong_tool`, `wrong_args`,
`wrong_order`, `extra_call`, `no_final_answer`, `bad_answer`, `state_mismatch`,
`parse_failure`, `argument_error`, `repeated_action`, `resource_limit`, and
`infrastructure`.

### Guided stages

Guided mode shows one stage at a time, with only that stage's tools. Cases in
the simple, multiple, irrelevance, and missing_param categories get one stage
automatically. Chain, recovery, robustness, and web cases must declare their
stages. A guided variant skips a case that has no stages and the plan reports
it as not applicable.

### Golden replays

Every case lists `[[golden]]` calls that solve it. The test suite replays them
through the real agent loop and requires each offline case to succeed. This
keeps a case from being impossible to solve. Write the golden calls first when
you add a case.

## Suites

A suite picks models, context sizes, variants, repeats, and cases. It selects
cases by directory, by `tags`, and by `split`. Paths are relative to the suite
file. The study id is a hash of the suite, the cases, the tool definitions, the
machine settings, and the binary, so changing any of them starts a new study.

- `smoke` runs the cases tagged `smoke` from the dev split, once.
- `core` runs all held-out cases in autonomous and guided mode, with one warmup
  and three repeats.
- `harness` compares the autonomous variant with guided, filter, recent
  context, constrained decoding, and repair, on the cases tagged `tier2`.
- `web_live` runs the live search cases.

## Add a tool

All tools are Rust functions that run inside the harness process, so there is
no tool executable, protocol, or second process to measure. A tool has two
parts.

1. A handler in `native/src/tools/<domain>.rs`. It is a function
   `fn(&mut World, &Value) -> Result<Value>` that you register by name in the
   module's `register` function. `World` holds the task state, e.g. files,
   stores, the clock, the database, and pages.
2. A definition in `configs/native/tools/standard.toml` with the name, the
   description, and the JSON Schema for the arguments. This is what the model
   sees, so wording changes can be tried without rebuilding.

Then list the tool in a case's `tools.allowed`. Run `a2e-native tools validate`
to check that every definition has a handler and a valid schema. Keep handlers
bounded in time and output, because the harness stops an attempt when a call
passes `budget.tool_seconds` or returns more than `budget.tool_output_bytes`.

## Results, judging and reports

This program only produces results. The A2E-Results project stores them in a
Cloudflare D1 database and does the judging, scoring, summaries, CSV export and
reports. See `third_party/a2e-results/README.md` for its commands and
`third_party/a2e-results/docs/API.md` for the Worker API this program follows.

Each attempt is posted as one request with its turns, tool calls, checks and
resource samples. The ids are `<first 12 characters of the study id>-<first 16
of the job>-<attempt>` for a run and `<first 12 of the study>-load-<n>` for a
model load, so a repeated post stores nothing twice. Samples are written after
an attempt ends and its wall time is fixed, and the upload runs between
attempts, so neither changes what is measured. A test checks that time spent
outside the model, the tools, and parsing stays a small share of wall time.

A case asks for a judge by setting `[judge] enabled = true`. The case then ends
as `awaiting_judge` once its other checks pass, and the cloud judge scores it
later.

```toml
[judge]
enabled = true
rubric = "retrieval_summary"
handoff = ["prompt", "required_facts", "evidence", "answer"]
required_facts = ["The fee is 12 dollars."]
```

The `handoff` list chooses what the judge sees. The fields are `prompt`,
`answer`, `calls`, `evidence`, `files`, `state_diff`, and `required_facts`. The
rubric named in the case is a TOML file in `third_party/a2e-results/cli/rubrics`.
A run with a judged case sends its `answer`, `evidence`, `final_state` and tool
calls, which is all the judge needs.

Data from an older local `native.sqlite3` is loaded with `bun src/cli.ts import
--db PATH` in the A2E-Results project.
