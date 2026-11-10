# A2E methods report

This report describes what A2E measures, how it runs an experiment, and the limits of the results. Setup and commands are in [GUIDE.md](GUIDE.md). The research question and hypotheses are in [PREREGISTRATION.md](PREREGISTRATION.md).

## 1. Aims

A2E measures how small language models use tools on small devices. For each model and each task it records whether the task succeeded, how long it took, how much memory it used, how hot the device got, and, on the Mac, how much power it drew. The questions are:

- Which of the three models completes the most tasks within a fixed budget of calls, tokens and time?
- Does the way the harness drives the model change the result? The harness is the code around the model that builds the prompt, reads the reply and runs the tools.
- What does each success cost in time, memory and power?

## 2. Scope

**In scope.**
- Three models: FunctionGemma 270M (Q4_K_M), Qwen3-0.6B (Q4_K_M) and LFM2.5-1.2B-Instruct (QAD_Q4_0).
- Two devices: a Raspberry Pi 3B+ with 1 GB of memory and an Apple Silicon Mac.
- Inference with llama.cpp, using `llama-server` over local HTTP.
- 130 offline cases in nine categories, and four live search cases.
- 50 tools that run inside the harness process.

**Out of scope.**
- Models above 2 billion parameters, because they do not fit in 1 GB.
- Measured power on the Pi. The Pi 3B+ has no power sensor, so its power is modeled from CPU usage (section 7).
- Judging and scoring beyond the local checks. The A2E-Results project does those (see section 8).

## 3. Experiment definition

A suite file in `configs/suites` defines one experiment. It lists the models, the context sizes, the variants, the cases to run, the number of warm-up and measured repeats, and the seed. The shipped suites are:

| Suite | Cases | Variants | Warm-up and repeats |
|---|---|---|---|
| `smoke` | cases tagged `smoke`, dev split | autonomous, guided | 0 and 1 |
| `core` | the eight categories other than `files`, held-out split | autonomous, guided | 1 and 3 |
| `harness` | `multiple` and `chain` cases tagged `tier2`, held-out split | autonomous, guided, filter, recent, constrained, repair | 1 and 3 |
| `web_live` | the four live search cases | autonomous, guided | 0 and 1 |
| `files_dev` | the 26 `files` cases, dev split | autonomous, guided | 0 and 1 |
| `files_heldout` | the 26 `files` cases, held-out split | autonomous, guided | 1 and 3 |

The study id is a hash of the suite, the cases, the tool definitions, the machine settings and the binary. A change to any of them starts a new study, so results from different setups are not mixed.

Unknown keys in any file are errors, so a misspelled setting stops the run instead of being ignored.

## 4. Model serving

The harness starts one `llama-server` process per model and context size, and stops it afterwards. It sends each model call to `/completion` with `cache_prompt` on and formats prompts with the server's `/apply-template` and the model's own chat template. The llama.cpp commit is fixed by `llama_commit` in `configs/models.toml`, and `scripts/build_llama.sh` builds that commit.

Every model file is checked against its published SHA-256 before it is used. Decoding is greedy for every attempt. Each model has an adapter in `src/adapter.rs` that writes the tool list in the model's own format and reads calls from the reply, because the three models use different call syntax.

## 5. Workloads

### 5.1 Cases

A case is one TOML file with a prompt, the tools it may use, a starting state, typed checks and the calls that solve it. The categories are:

| Category | Cases | What it checks |
|---|---|---|
| simple | 16 | One call with the right arguments |
| multiple | 16 | Choosing the right tool among 2, 4 or 8 tools |
| irrelevance | 10 | Making no call when no tool fits |
| missing_param | 10 | Asking for a required value instead of guessing it |
| chain | 24 | Two to six dependent calls, where the final state decides |
| recovery | 12 | Recovering from failed calls, slow tools and large results |
| robustness | 8 | Ignoring instructions inside files, pages and database rows |
| web | 8 | Search and fetch over recorded pages |
| files | 26 | Finding, reading and changing files in a generated workspace of about 295 files |

Cases are split into a dev split and a held-out split. The dev split is for building and debugging. Studies report the held-out split.

### 5.2 Tools

The 50 tools cover files, calculation, dates and time, units, key-value storage, notes, to-do lists, devices, a SQLite database and web search. Each attempt gets a fresh world: an in-memory file set, stores, a clock, a database seeded from `configs/fixtures`, and recorded web pages. Tool definitions are in `configs/tools/standard.toml`, and the handlers are in `src/tools`.

A case can break a tool on purpose with a fault: fail the first `n` calls, return a rate limit error, delay each call, pad the result, or return garbled text. This is how the recovery cases test behavior under failure.

### 5.3 Modes and variants

- **Autonomous.** The model sees all allowed tools and works until it answers.
- **Guided.** The model sees one stage at a time with only that stage's tools.
- **Filter.** Only the `k` tools whose name and description share the most words with the prompt are shown.
- **Recent.** Once the conversation passes six messages, only the first two and the last four are kept.
- **Constrained.** A grammar built from the visible tools limits decoding to valid calls.
- **Repair.** One retry is allowed after a reply that cannot be read.

### 5.4 Budgets

Each attempt has a budget. The defaults are 256 generated tokens per call, 8 model calls, 300 seconds per call, 900 seconds per task, 20 seconds per tool call and 8192 bytes of tool output. An attempt that passes a limit ends with the failure class `resource_limit`.

## 6. Study protocol

The runner takes the plan as a list of jobs, one for each model, context size, variant and case. It loads each model once, runs the warm-up repeats and then the measured repeats, and writes the result of each attempt before it starts the next. Running the same command again resumes completed work, and `--retry-failed` repeats unsuccessful attempts.

Repeats measure variation in latency, memory and temperature. Greedy decoding gives the same answer each time, so repeats do not measure variation in success.

Only one run holds the runtime lock at a time, so two runs cannot disturb each other's measurements. Disable swap before measuring on the Pi.

## 7. Measurement

- **Time.** Wall time per attempt, per model call and per tool call. Model loading is measured separately. Latency includes failed attempts.
- **Tokens.** Prompt and generated tokens per call, and prompt tokens served from the prefix cache, as reported by `llama-server`.
- **Memory.** Resident memory of the inference process and of the harness process is sampled every 500 ms. Tools run inside the harness, so harness memory includes them. Peak memory comes from the samples.
- **Temperature.** On the Pi, the SoC temperature and the throttle flags from `vcgencmd get_throttled`.
- **CPU time.** Total CPU time of the inference and harness processes, on both devices.
- **Power.** On the Mac, privileged `powermetrics` samples give CPU, GPU and ANE power. They exclude memory, the display and power-supply losses. On the Pi, which has no power sensor, power is modeled: every 500 ms the runner puts whole-system CPU usage into the Joular `rbp3b+1.3` polynomial and sums the watts over time. The result is labeled `modeled` in `power_scope`. The model was fitted by its authors on another unit with a 32-bit OS, and its error on this unit is unknown.

Missing data stays missing. A power or timing value that could not be read is never stored as zero, and a software estimate is never described as measured energy.

The cost of the harness itself is checked by a test that requires time outside the model, the tools and parsing to stay a small share of wall time.

## 8. Correctness and judging

Each case has typed checks that run after the attempt, using only the recorded calls and the final state. An attempt succeeds when every check passes. A failed attempt gets one failure class from the first check or limit that failed, e.g., `wrong_tool` or `state_mismatch`. The full list is in the guide.

Every case also lists the calls that solve it. The tests replay those calls through the real agent loop and require each offline case to succeed, which shows that no case is impossible.

A case can ask for a cloud judge for answers that a check cannot decide, such as a summary. The case then ends as `awaiting_judge`, and the A2E-Results project scores it later with a rubric from `a2e-results/rubrics`.

## 9. Data storage and upload

The runner writes a local SQLite database and a trace file per attempt under `results`. A trace holds the prompts, the raw model output, the tool results and the evidence used for judging.

After each attempt the runner posts the study, the model load and the attempt to the A2E-Results Worker. Each request is saved in `results/outbox` first and removed once the Worker stores it, so a network failure loses nothing. Request ids are built from the study, the job and the attempt, so posting the same attempt twice stores it once.

## 10. Reproducibility

- The llama.cpp commit, the model revisions and the model hashes are pinned in `configs/models.toml`.
- The study id changes whenever the suite, cases, tools, machine settings or binary change.
- Host files in `configs/hosts` set per-machine settings, and `doctor` prints the host and the settings that apply.
- Greedy decoding and a fixed seed in each suite make a run repeatable on the same machine.

## 11. Known limits

- Greedy decoding with one prompt format per model means accuracy can differ from the numbers each model's authors publish.
- Mac power covers CPU, GPU and ANE only, so it is not comparable with a wall-meter reading, and it is not comparable with the modeled Pi values, which are an estimate from CPU usage alone.
- A 1 GB Pi has little free memory. Swap changes timing, so it must be off for measured runs.
- Live search results change over time, so the `web_live` suite cannot be repeated exactly. It is scored on the tool path and the final answer only.
- With three measured repeats, intervals on latency and memory are wide for small groups.

## 12. Security

Tools run inside the harness process, so they are limited in what they can touch.
- File tools resolve paths inside the attempt's own file set and reject paths that leave it.
- The web tools accept only public `http` and `https` addresses, and they limit the size of a page.
- The database tools open a read-only connection for queries.
- Robustness cases check that instructions placed in files, pages and database rows are not followed.

## 13. Portability

The program builds with Rust 1.96 or newer on Linux (including `aarch64` for the Pi) and macOS. CI formats, lints and tests on both, and cross-builds for the Pi. Device settings come from `configs/models.toml`, a host file and `configs/local.toml`, in that order, and environment variables override them.

## 14. Licensing and sources

The code is MIT licensed. The models, tools and papers the project uses are listed in [REFERENCES.md](../REFERENCES.md).
