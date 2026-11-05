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
target/release/a2e-native report
```

Pass a suite file to `plan` or `run`. The shipped suites are listed in the
suites section below. Set `BRAVE_SEARCH_API_KEY` for the live suite. The other
suites use recorded fixture data.

Run the same command again to resume completed work. Use `--retry-failed` to
retry unsuccessful attempts. The database and the trace files live under
`results/native`. Trace files contain prompts, raw model output, tool results,
and the evidence used for judging.

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
definitions and the rubric files. Run `a2e-native cases list --ids` to see the
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

## Results database

Results are stored in SQLite at `results/native/native.sqlite3`. Schema version
2 uses normal tables, so you can query them directly. A database from the
earlier layout is refused with a message, and you start a new results
directory instead.

- `studies` and `cases` hold the frozen study definition and each case.
- `model_loads` holds load time and peak memory for each model start.
- `runs` has one row per attempt with every headline number as a column.
- `turns` has one row per model call with token counts and the template,
  tokenize, request, prefill, decode, and parse times.
- `tool_calls` has one row per call with its arguments, validity, result, and
  duration.
- `checks` has one row per check with its label and result.
- `samples` has every resource sample, with memory, CPU time, temperature, and
  power when the device reports them.
- `judgments`, `run_judgments`, and `judge_calls` hold judge packets, verdicts,
  and each OpenRouter call with its tokens, time, and cost.

The views `v_run_summary`, `v_category_success`, `v_failures`, and
`v_resources` count only the latest attempt of each job. A judged attempt
counts as a success only when the judge verdict is `pass` and the judge has
passed calibration.

The harness writes turns, tool calls, checks, and samples in one batch after
an attempt ends and its wall time is fixed. Storage work therefore does not
change what is measured. A test checks that time spent outside the model, the
tools, and parsing stays a small share of wall time.

## Make a report

```bash
target/release/a2e-native report
target/release/a2e-native report --study <id> --out reports/mac --run <run-id>
```

The command reads the newest study unless you pass `--study`, and writes
`report.md` and a `plots` directory. By default they go to
`results/native/reports/<study>`. The `--run` option picks which attempt the
resource timeline shows. Without it, the slowest measured attempt that has
samples is used.

The report has the study header, a headline table with bootstrap intervals,
success by category, failure classes, latency, calls and tokens, memory
against the limit, temperature and energy, judge scores, and a list of failed
attempts with their failed checks and tool calls. A value that was not
measured prints as `n/a`. It is never shown as zero.

The seven SVG plots are written by the program with no extra libraries.
They show success by model and variant, success by category, failure classes,
wall time, peak memory against the limit, resources over one run, and tokens
against wall time. Each file is opaque and readable in any viewer.

## Score with a judge

A case can ask for a judge by setting `[judge] enabled = true`. The case then
ends as `awaiting_judge` once its other checks pass, and it counts as pending
until the judge has scored it.

```toml
[judge]
enabled = true
rubric = "retrieval_summary"
handoff = ["prompt", "required_facts", "evidence", "answer"]
required_facts = ["The fee is 12 dollars."]
```

The `handoff` list chooses what the judge sees. The fields are `prompt`,
`answer`, `calls`, `evidence`, `files`, `state_diff`, and `required_facts`.
The judge gets only the fields the case lists, and the text in them is marked
as data, not instructions.

A rubric is a TOML file in `configs/native/rubrics`. It has instructions,
guidance, a `[verdict]` table with `fail_below`, `pass_total`, and `pass_each`,
and a list of `[[dimensions]]`. Each dimension has five descriptors for the
scores 0 to 4. The JSON schema for the judge response is built from the
dimensions, so adding a dimension needs no code change. The rubric hash is
frozen into the study.

Set `OPENROUTER_API_KEY`, `A2E_JUDGE_MODEL`, and `A2E_JUDGE_PROVIDER`. Choose
a model and provider that support strict JSON Schema output. Run
`target/release/a2e-native judge --study <study-id>` after the measured run.
The command sends the packet for each pending attempt to OpenRouter. Judge
calls and costs are recorded separately from the edge measurements.

The shipped `retrieval_summary` rubric gives up to 16 points over four
dimensions. A judgment is provisional until you pass `--calibration <file>`
with a human-scored set. The JSON file contains `reviewed_by` and at least four
`cases`. Each case has a `packet`, an `expected_verdict`, and `scores` keyed by
the rubric dimensions. Calibration needs at least 80% verdict agreement and at
most 0.75 mean absolute error per dimension. Provisional judgments appear in
the raw export but not in the success summary.
