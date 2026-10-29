# A2E user guide

This guide explains how to set up A2E, run measurements, and read the results. It assumes you have the repository checked out on a Raspberry Pi 5 or an Apple Silicon Mac. Other documents cover the details:

- [CONFIG_REFERENCE.md](CONFIG_REFERENCE.md) lists every field in every configuration file.
- [DATA_DICTIONARY.md](DATA_DICTIONARY.md) lists every table and column in the results database and explains how each number is computed.
- [ARCHITECTURE.md](ARCHITECTURE.md) describes the code and how data moves through it.
- [REPORT.md](REPORT.md) is the methods report. It explains what the study measures, how, and why each choice was made.

## 1. What A2E does

A2E measures the cost of running small language models as tool-calling agents on edge hardware. A tool-calling agent is a model that answers a request by producing structured calls to functions, e.g., `calculator(expression="12*7")`, instead of only writing text. The program runs the calls, sends the results back to the model, and the model continues until the task is done.

A2E runs the models with llama.cpp on two devices:

- a Raspberry Pi 5 with 8 GB of memory, using only the CPU
- an Apple Silicon Mac, using the GPU through Metal

For every model call, A2E records how long each stage took, how many tokens were processed, how much energy the device used, how much memory the model server used, and how hot the chip was. It also checks whether the model called the right tools with the right arguments.

The stages it separates are:

- prefill, which is the time the model spends reading the prompt, including the tool descriptions
- decode, which is the time the model spends writing its answer
- parse, which is the time spent reading the tool calls out of the answer
- validate, which is the time spent checking the calls against the tool descriptions
- tool wait, which is the time spent running the tools
- template and harness time, which is time spent in A2E itself

The study compares these numbers across models, quantization levels, prompt lengths, decoding settings and the two devices. The goal is to find patterns and real bottlenecks, not to publish single headline numbers.

## 2. Words used in this guide

These words have a fixed meaning in A2E.

- **Suite.** A TOML file in `configs/suites/` that describes a set of measurements. You run a suite with `a2e run <name>`.
- **Job.** One measured configuration. A job fixes the device, the model file, the server settings, one task and one value for every setting the task uses. A suite expands into a list of jobs.
- **Job key.** A short hash of a job's settings. The same settings always give the same key.
- **Experiment.** One suite on one device, stored in the database. Running the same suite again on the same device continues the same experiment.
- **Run.** One repeat of a job. Each job has warm-up runs, which are recorded but not analysed, followed by measured runs.
- **Turn.** One call to the model inside a run, with all of its stage timings, token counts and energy.
- **Case.** One scored task inside a run, e.g., one BFCL test case or one custom tool chain.
- **Sample.** One reading of power, temperature, clock speed, CPU use and memory, taken several times per second during a run.

## 3. Install

### 3.1 On a Raspberry Pi 5

1. Install Raspberry Pi OS (64 bit). Make sure `vcgencmd` works by running `vcgencmd measure_temp`.
2. Pick a llama.cpp commit to use for the whole study and write it down. Every measurement must use the same commit.
3. Build llama.cpp and install uv with the setup script. Replace `<sha>` with the commit.

```bash
LLAMA_COMMIT=<sha> scripts/setup_pi.sh
```

4. Install the Python environment with the optional parts for BFCL and the terminal UI.

```bash
uv sync --extra bfcl --extra tui
```

### 3.2 On an Apple Silicon Mac

1. Install the Xcode command line tools and CMake.
2. Build llama.cpp with Metal at the same commit you use on the Pi.

```bash
LLAMA_COMMIT=<sha> scripts/setup_mac.sh
```

3. Install the Python environment.

```bash
uv sync --extra bfcl --extra tui
```

Energy readings on the Mac come from `powermetrics`, which only works as root. To record energy you run A2E with `sudo -E`. Without root, A2E still runs and records everything except power and thermal pressure, and it prints a line that starts with `WARNING: no energy will be recorded`. To make such a run stop instead, set `power = "required"` in `configs/local.toml`.

### 3.3 Model files

A2E reads GGUF model files. The models in the study are listed in `configs/models.toml`. For each model you need one file per quantization you plan to run: `q4_0`, `q8_0` and `q3_k_m`. By default A2E looks for a file named like this:

```text
<models_dir>/<model>/<model>-<quant>.gguf
e.g. ~/models/qwen3-4b/qwen3-4b-q4_0.gguf
```

If your files have other names, give their exact paths in `configs/local.toml` (see section 4). When you download a model, write down where it came from in the `source` field of `configs/models.toml` and add it to `REFERENCES.md`.

The Pi has 8 GB of memory, so A2E only runs Q8_0 files on the Pi for models of 4 billion parameters or fewer. The 8B models run at Q4_0 on the Pi. Jobs that break this rule are listed as skipped with the reason "not run: memory".

## 4. Configure the machine

Copy the example settings file and edit it.

```bash
cp configs/local.example.toml configs/local.toml
```

The important fields are:

- `llama_server`, which is the path to the `llama-server` binary that the setup script built
- `models_dir`, which is the folder that holds the model files
- `db`, which is the results database file (the default is `results/a2e.db`)
- `device`, which you only set if automatic detection fails (`pi5` or `mac`)
- `cooler`, which records whether the Pi has its active cooler fitted
- `power`, which is `auto` to record energy when possible, `none` to skip it, or `required` to stop with an error when energy cannot be recorded

The `web` task family also needs a Brave Search API key in the environment. A2E never reads the key from a file.

```bash
export BRAVE_SEARCH_API_KEY=<your key>
```

`configs/local.toml` is not committed to git because it describes one machine. Relative paths in it are read from the repository folder. See [CONFIG_REFERENCE.md](CONFIG_REFERENCE.md) for every field.

## 5. Check the machine

Run the doctor before a long run.

```bash
uv run a2e doctor
```

The doctor checks each of these and prints `ok`, `warn` or `FAIL`:

- the Python version
- that the device is a Pi 5 or a Mac
- that the `llama-server` binary exists and can run
- that the `llama-bench` binary exists (only bench jobs need it)
- that each model file exists
- that `vcgencmd` (Pi) or `powermetrics` and root (Mac) are available
- that the CPU governor can be read (Pi)
- that `perf` is installed for kernel profiles (Pi)
- that the cooler setting is recorded (Pi)
- that the BFCL and TUI extras are installed
- that the database can be written and that there is free disk space

A `FAIL` line stops `scripts/run_suite.sh`. A `warn` line does not, e.g., a missing model file only matters if a suite uses it.

## 6. Suites

### 6.1 The shipped suites

| Suite | What it runs | Runnable jobs | Skipped jobs | Runs | Model loads |
|---|---|---|---|---|---|
| `smoke` | One small job of each kind on Qwen3-4B Q4_0, one measured run. Use it to check a new machine. | 8 | 0 | 8 | 2 |
| `pilot` | Qwen3-4B Q4_0 with KV cache f16 and q8_0. Plain generation, every offline custom task family, a slice of every BFCL category, and one llama-bench test with kernel profiling. | 140 | 0 | 560 | 4 |
| `baseline` | All five models at Q4_0 and Q8_0. Plain generation next to tool calling. | 108 | 12 | 432 | 18 |
| `chains` | Chains of 2 to 5 tool calls for every family, and notes with 1, 3 or 5 arguments per call. | 720 | 80 | 2,880 | 18 |
| `factorial` | The main sweep on custom chains over schema length, decode length, grammar and KV cache. | 1,296 | 144 | 5,184 | 36 |
| `bfcl` | Accuracy and cost on every BFCL category in scope for every model, quantization and KV cache setting. | 720 | 80 | 1,440 | 36 |
| `cache_ablation` | Prefix cache on and off, KV cache f16 and q8_0, over long chains and BFCL multi-turn. | 120 | 0 | 480 | 12 |
| `local_tools` | File search, file organizing, math and file plus math chains of 2 to 5 calls, on small BFCL file systems, for every model at Q4_0 and Q8_0. | 1,080 | 120 | 4,320 | 18 |
| `full_eval` | One broad report: 50 tasks per model, quant and grammar setting, from single BFCL calls and multi-turn BFCL to custom chains of 2 to 5 calls. One measured run, no warm-up. Edit the two marked lines for your models and quants. | 336 | 0 | 336 | 8 |
| `harness_ablation` | Eleven harness settings (prompt format, context management, error recovery, stopping rules, and two combinations) against the baseline on the same tasks, for Qwen3-4B and xLAM-2 3B at Q4_0. | 572 | 0 | 2,288 | 4 |
| `web_live` | Live web search through the Brave Search API on Qwen3-4B and Phi-4-mini. Needs a key and network access. | 24 | 0 | 48 | 4 |
| `kernels` | llama-bench prefill and decode tests for every model, quantization and KV cache setting, with the top kernels from Linux perf on the Pi. | 36 | 4 | 36 | 0 |
| `stability` | 40 long chains back to back with no cooldown, with different CPU governors on the Pi. | 8 | 4 | 24 | 4 |

The counts cover both devices together, so each device runs about half of them. They come from `uv run a2e plan <suite>`, which also prints the counts per device and per task kind. A run is one repeat of one job, so runs equal runnable jobs times warm-up plus measured repeats. Skipped jobs are the 8B models at Q8_0 on the Pi and, in the stability suite, governor jobs on the Mac. Some factorial jobs are also skipped at run time when the tools do not fit the schema length.

Run the suites in this order:

1. `smoke` on each device, to check the setup.
2. `pilot` on each device. Its timings tell you how long the larger suites will take. They also decide the BFCL subsample sizes, the Pi context size, the cooldown target, whether to fit the Pi cooler, and whether the extra benchmarks (API-Bank, ToolBench, tau-bench) are affordable.
3. `baseline`, then `bfcl`, `factorial`, `chains`, `local_tools`, `cache_ablation` and `kernels`, in any order.
4. `web_live` whenever the device has network access and `BRAVE_SEARCH_API_KEY` is set.
5. `stability` last, because it heats the device on purpose.

### 6.2 What a suite file contains

A suite file has these parts:

- `[suite]` gives the name, a description and the random seed.
- `[repeats]` gives the number of warm-up runs and measured runs per job. The study default is 1 warm-up run and 3 measured runs.
- `[server]` gives the llama-server context size, the flash attention setting and the thread count.
- `[run]` gives the idle baseline length, the sampling rate, the cooldown target and the turn limits.
- `[matrix]` lists the devices, models, quantizations and KV cache types. Every combination is one model load.
- `[axes]` gives the default values for grammar, prefix cache, decode length, schema length and CPU governor. Tasks can vary these settings.
- Each `[[tasks]]` entry adds one kind of task. A task can override any axis.
- Each optional `[[harness]]` entry is a named harness setting. Every custom and BFCL job runs once under each one, so you can compare how the code around the model changes cost and accuracy. See section 1.7a of the configuration reference.

Here is the smoke suite in full:

```toml
[suite]
name = "smoke"
description = "One small job of each kind on Qwen3-4B Q4_0, one measured run, no warm-up."
seed = 0

[repeats]
warmup = 0
measured = 1

[server]
ctx = 8192

[run]
idle_seconds = 5
cooldown_c = 0

[matrix]
device = ["pi5", "mac"]
model = ["qwen3-4b"]
quant = ["q4_0"]
kv_quant = ["f16"]

[axes]
grammar = [true]
prefix_cache = [true]
decode_len = [128]
schema_len = [512]

[[tasks]]
kind = "plain"
n = 2

[[tasks]]
kind = "custom"
family = "calc"
depth = 2
n = 2

[[tasks]]
kind = "bfcl"
category = ["simple_python", "irrelevance"]
n = 3
```

### 6.2a The full evaluation

`full_eval` is the suite to run when you want one clean report across difficulty levels. It has 50 tasks for each combination of device, model, quant and grammar setting (grammar off and on):

| Group | Tasks | What it shows |
|---|---|---|
| BFCL single turn | 15 | `simple_python` 4, `multiple` 3, `parallel` 3, `parallel_multiple` 2, `irrelevance` 3: one call, choosing a tool, several calls, declining |
| BFCL multi turn | 9 | `multi_turn_base` 4, `multi_turn_miss_func` 2, `multi_turn_miss_param` 3: state across user turns, tools revealed later, missing arguments |
| Custom chains | 26 | `calc` (2 and 4 calls), `notes` (3), `retrieval` and `fs_search` (2 and 4), `fs_organize`, `math` and `fs_math` (3 and 5): longer horizons with real tools |

The two `# EDIT` lines in `configs/suites/full_eval.toml` set the two models and the two quants. Use names from `configs/models.toml`. The suite runs once per job and has no warm-up, so there is no repeat-to-repeat spread. Read the report by category and family totals, because a single cell has only 2 to 4 tasks. The live `web` family is left out because live results change.

```bash
scripts/run_suite.sh full_eval
uv run a2e results list                    # find the experiment id
scripts/report_full_eval.sh 1              # writes results/full_eval_1/
```

`report.md` has every table. The CSV files in the same folder cut the numbers by model, BFCL category, and custom family and depth, and hold the success-per-joule, Pareto, grammar-effect, energy-by-stage and energy-budget views. Set `OUT_DIR=somewhere` to write the folder elsewhere. The run can be stopped and resumed.

### 6.3 The four task kinds

**Plain.** The model writes text for a fixed prompt with no tools. A2E asks llama-server to ignore the end of text token, so every run decodes exactly `decode_len` tokens. This is the baseline that the tool-calling numbers are compared against. The prompts are in `src/a2e/data/plain_prompts.json`.

**Custom.** The model completes a chain of tool calls. Each task has an exact expected call for every step, so A2E can score each step. There are eight families. Three use A2E's own tools:

- `calc` asks for one calculator call per expression. The depth is the number of expressions, from 1 to 5.
- `notes` asks the model to save notes and then list them. `n_args` sets how many fields each note has, from 1 to 5.
- `retrieval` asks the model to search a local document collection, read the best document, compute with a number from it, save a note and list the notes. The depth sets how many of these steps the task has, from 2 to 5.

Four use the simulated file system and math tools from BFCL (`GorillaFileSystem` and `MathAPI`), unchanged:

- `fs_search` (depth 2 to 5) asks the model to find a file, go into its folder one level at a time, show it, show the lines with a given word, count its words and sort it.
- `fs_organize` (depth 2 to 5) asks the model to create a folder, move a file into it, go into it, list it and rename the file. A2E also compares the final file system with the expected one, so a correct sequence of calls that leaves extra damage still fails.
- `math` (depth 1 to 5) gives five numbers and asks for the mean, the standard deviation, rounding, the largest number and a percentage. Later steps use earlier results. A number the model copies from an earlier result counts as correct within 0.1%.
- `fs_math` (depth 3 to 5) asks the model to count the words in two files, add the counts, work out a percentage and round it.

These four start from small file systems taken from BFCL's own multi-turn data. BFCL is written for every model size, so A2E keeps only the 25 of its 52 file systems that have 2 to 6 files, at most 3 folder levels and at most 800 characters of file content. `scripts/curate_fs_scenarios.py` makes this selection and writes `src/a2e/data/fs_scenarios.json`. Each task picks one of them with the seed.

One family uses live web search:

- `web` (depth 2 to 4) asks a question with a stable answer, e.g., "Who wrote the novel Pride and Prejudice?". The model searches the web, opens one of the results, saves a note and lists the notes. The tools have BFCL's names and arguments (`search_engine_query`, `fetch_url_content`) and call the Brave Search API. Results change over time, so A2E only checks the path: that the model searched with non-empty keywords, opened a link that its own search returned, and made the later calls correctly. It does not check the answer. The model can only open links that a search in the same task returned. Network time counts as tool wait. These tasks need `BRAVE_SEARCH_API_KEY` (section 4) and network access.

The tool descriptions in the prompt are padded with extra tools until they reach `schema_len` tokens, counted with the model's own tokenizer. This is how the study controls schema length. By default the extra tools are real BFCL functions, e.g., `book_flight` or `get_stock_info`, which look like the tools a real assistant would offer. Set `distractors = "synthetic"` on a task to use made-up tools instead. A2E never edits a real BFCL description. Only a made-up tool closes the last few tokens of the gap. Every task gets fresh tools: fresh note files, a fresh copy of its file system, and its own list of links it may open. Tasks cannot see each other's results.

**BFCL.** The model answers test cases from the Berkeley Function Calling Leaderboard, version 4. A2E scores the answers with the official BFCL checker code, which is copied unchanged into `third_party/bfcl_eval/`. The categories in scope are:

- the single turn categories `simple_python`, `multiple`, `parallel` and `parallel_multiple`
- the categories that check whether the model calls at all, `irrelevance`, `live_irrelevance` and `live_relevance`
- the multi-turn categories `multi_turn_base`, `multi_turn_miss_func` and `multi_turn_miss_param`

`n` takes a fixed subsample of each category. The subsample is drawn with the suite seed and spread across the category in proportion to its strata, so the Pi and the Mac always run the same cases. Multi-turn categories need `[server] ctx` of at least 16384 because their tool descriptions are several thousand tokens long.

**Bench.** A2E stops llama-server and runs llama.cpp's own `llama-bench` tool on the model file. It runs one prefill test per value in `n_prompt` and one decode test per value in `n_gen`, and records tokens per second for each. Prefill processes many tokens at once and is mostly limited by how fast the chip can compute. Decode produces one token at a time and reads every weight for each token, so it is mostly limited by memory bandwidth. A2E therefore also reports an estimated memory bandwidth for decode, which is decode tokens per second times the model file size.

When `perf = true` and the device is the Pi, A2E runs each test under Linux `perf record` and stores the functions that used the most CPU time. These are mostly ggml kernels, e.g., `ggml_vec_dot_q4_0_q8_0`, which is the dot product for Q4_0 weights. This answers which operations dominate prefill and which dominate decode. Kernel profiling is not implemented on the Mac. A2E records a note instead.

`perf` needs root, or the kernel setting `kernel.perf_event_paranoid` at 1 or lower. Without it, A2E still stores the throughput numbers and records a note that the profile is empty.

### 6.4 Writing your own suite

Copy a shipped suite, change the name and the values, and check it with `plan`.

```bash
cp configs/suites/pilot.toml configs/suites/my_test.toml
uv run a2e plan my_test --jobs
```

`plan --jobs` prints every job with its key and, for skipped jobs, the reason. A2E rejects a suite with an unknown field, an unknown model, an unknown task family or BFCL category, a depth out of range, or a multi-turn BFCL category with a context below 16384. The error names the problem.

### 6.5 The suite lock file

`configs/suites.lock.json` stores each shipped suite's hash, its number of jobs and a hash of its job keys. The test suite fails if a suite file, or the code that expands suites into jobs, changes without the lock being updated. This makes sure that a suite name means the same jobs in every session and on every machine. After you change a suite on purpose, update the lock:

```bash
uv run a2e suite-lock
uv run a2e suite-lock --check
```

## 7. Running

### 7.1 From the command line

```bash
uv run a2e run smoke
```

A2E detects the device, expands the suite, keeps only this device's jobs, and runs them in an order that loads each model file only once. It prints one line per job and one line per repeat. Useful options are:

- `--limit N` runs at most N jobs and then stops. Run the same command again to continue.
- `--retry-failed` also runs jobs that failed earlier, e.g., after you add a missing model file.
- `--new` starts a fresh experiment instead of continuing the existing one.
- `--device pi5` or `--device mac` overrides device detection.
- `--db path` writes to another database file.

On a Mac, add `sudo -E` to record energy:

```bash
sudo -E uv run a2e run pilot
```

`scripts/run_suite.sh <suite>` runs the doctor, prints the plan and then starts the run.

### 7.2 Stopping and resuming

You can stop a run at any time with Ctrl+C. A2E marks the job it was working on as pending again and marks that job's partial runs as aborted. When you run the same command again, A2E continues the same experiment and starts that job from its first repeat. Aborted runs stay in the database for the record, but the analysis views and the results commands never use them.

The same happens if the machine loses power or the process is killed. On the next start A2E finds the job that was running and resets it.

A2E only continues an experiment if the suite file is unchanged. If you edit the suite, its hash changes and the next run starts a new experiment, so results from different suite versions never mix.

### 7.3 What happens during a job

For each job, A2E does these steps in order:

1. If the job sets a CPU governor, A2E sets it (Pi only, needs root).
2. If the model file or server settings differ from the previous job, A2E stops the old llama-server and starts a new one. It records the load time and then measures idle power for `idle_seconds` while the model is loaded but not working.
3. If `cooldown_c` is set, A2E waits until the chip is cooler than that temperature, up to `cooldown_timeout_s`.
4. For each repeat, A2E clears the prompt cache, starts sampling at `sample_hz`, runs the task, stops sampling, and writes all turns, cases and samples to the database in one transaction.

Bench jobs are different. They stop llama-server, because llama-bench loads its own copy of the model and two copies may not fit in the Pi's memory. They then run llama-bench once per test size and sample the llama-bench process instead.
5. When all repeats are written, the job is marked done.

If a job fails, e.g., because a model file is missing, A2E records the error and continues with the next job. If llama-server crashed, the next job starts a new server.

### 7.4 From the terminal UI

```bash
uv run a2e tui
```

The TUI has five tabs. Press the number keys to switch between them.

1. **Suites** lists the suite files. Select one to see its job counts, model loads and runs.
2. **Run** runs the selected suite. You can set a job limit. The progress bar counts finished jobs and the log shows the same lines as the command line. "Stop after this job" finishes the current job and then stops cleanly.
3. **Results** lists the experiments in the database. Select one to see its summary table. "Write report for selected" saves a markdown report next to the database.
4. **Tests** lists the test groups. Select one and press "Run group" to run it and watch the output.
5. **Doctor** shows the same checks as `a2e doctor`.

Press `q` to quit. The TUI reads the same `configs/local.toml` as the command line.

## 8. Reading the results

All results go into one SQLite database file, `results/a2e.db` by default. You can read it with the `a2e results` commands, the TUI, a notebook, or any SQLite tool. You can read it while a run is writing to it.

### 8.1 Commands

```bash
uv run a2e results list                       # experiments and their job counts
uv run a2e results jobs 1 --status failed     # jobs of experiment 1, with reasons and errors
uv run a2e results summary --exp 1            # one row per configuration
uv run a2e results summary --by device,model,quant --csv out.csv
uv run a2e results stages --exp 1             # share of time in each stage
uv run a2e results resources --exp 1          # power min, median, p95, max, memory and thermal envelope
uv run a2e results task-resources --exp 1     # energy, power and peak memory of every task
uv run a2e results efficiency summary --exp 1 # success rate, joules per task, joules per success, waste share
uv run a2e results efficiency frontier        # the same with the Pareto front marked per device and task kind
uv run a2e results efficiency budget          # tasks completed on a fixed energy budget
uv run a2e results efficiency grammar         # grammar off against on for the same configuration
uv run a2e results efficiency stages          # share of energy in prefill, decode, tool wait and the rest
uv run a2e results efficiency harness         # does the harness change joules per success? ratios, intervals, verdict
uv run a2e results efficiency decomposition   # calls per task, joules per call and success rate behind a harness effect
uv run a2e results efficiency noise           # run-to-run noise of the baseline harness and the margin check
uv run a2e results efficiency variance        # share of variance from device, model, quantization and harness
uv run a2e results plots figures/             # the harness figures as PNG with their tables (needs `uv sync --extra plots`)
uv run a2e results stability --exp 3          # speed drift over long sessions
uv run a2e results bench --exp 5              # llama-bench prefill/decode speed and decode bandwidth
uv run a2e results kernels --exp 5 --top 10   # top perf kernels per test (Pi)
uv run a2e results timeline 42                # power, temperature, memory over one run
uv run a2e results report 1 --out report.md   # a markdown report with all of the above
uv run a2e results export out/ --fmt parquet  # every table and view as files
```

`summary` and `stages` accept `--exp` more than once, so you can put a Pi experiment and a Mac experiment in one table and group by `device`.

### 8.2 What the summary columns mean

Every summary number uses only measured runs of complete jobs. Each number is first computed for each run and then averaged over the measured runs of a configuration. A column ending in `_sd` is the standard deviation across those runs.

- `prefill_ms`, `decode_ms` and `tool_wait_ms` are the mean time per model call, in milliseconds.
- `prefill_tok_s` is the number of prompt tokens the model processed per second of prefill. Tokens served from the prefix cache are not counted, because the model did not process them.
- `decode_tok_s` is the number of generated tokens per second of decode.
- `cache_hit` is the share of prompt tokens served from the prefix cache.
- `prefill_share`, `decode_share` and `tool_share` are the share of total turn time spent in each stage. `overhead_share` is the share spent in parsing, validation, the chat template and the harness.
- `j_per_turn` is the energy of one model call in joules, with the idle baseline subtracted.
- `j_per_prompt_token` is prefill energy divided by processed prompt tokens.
- `j_per_gen_token` is decode energy divided by generated tokens.
- `tokens_per_joule` is generated tokens divided by decode energy. This is the same number as tokens per second per watt.
- `j_per_success` is the energy of all turns in a run divided by the number of cases that passed.
- `success` is the share of scored cases that passed.
- `peak_rss_mb`, `max_temp_c` and `throttled_turns` are the highest memory use of the server process, the highest chip temperature and the number of turns during which the Pi reported throttling.

[DATA_DICTIONARY.md](DATA_DICTIONARY.md) gives the exact formula for each column and explains how energy is split between stages.

### 8.3 Using SQL or pandas directly

Three views in the database make analysis easier. They contain only complete runs, and they add the main settings as columns:

- `v_runs` adds `model`, `quant`, `kv_quant`, `grammar`, `prefix_cache`, `decode_len`, `schema_len`, `family`, `depth`, `n_args`, `category` and `governor` to each run.
- `v_turns` adds those settings, the run phase and the device to each turn.
- `v_cases` does the same for each case.

```python
from a2e.store import Store
st = Store("results/a2e.db")
df = st.df("SELECT model, quant, avg(decode_ms) AS decode_ms FROM v_turns WHERE phase = 'measured' "
           "AND task_kind = 'custom' GROUP BY model, quant")
```

### 8.4 Energy numbers

Energy numbers are software estimates. They are meant for comparing configurations on the same device, not for absolute claims.

- On the Pi, A2E reads the voltage and current of each power rail from the power management chip with `vcgencmd pmic_read_adc` and adds them up. This leaves out the power supply's own losses and anything plugged into the USB ports.
- On the Mac, A2E reads the combined CPU, GPU and Neural Engine power from `powermetrics`. This leaves out memory, the display and the power supply.

Every run stores its `power_scope` so the meaning travels with the numbers. Do not compare joules directly between the Pi and the Mac. Compare the patterns, e.g., whether a longer schema costs more energy per call on both devices.

### 8.5 BFCL raw outputs and rescoring

For BFCL jobs, A2E also writes the model's raw text for every case to `results/raw/exp<N>.jsonl`. You can score these files again without the model, e.g., after updating the vendored checker:

```bash
uv run a2e bfcl rescore --raw results/raw/exp1.jsonl
```

## 9. Tests

The tests are grouped in `configs/tests.toml`, so a group name means the same tests in every session.

```bash
uv run a2e test --list
uv run a2e test quick                  # unit and integration tests, about a minute
uv run a2e test sweep                  # large sweeps, a few minutes
uv run a2e test bfcl -x                # extra arguments go to pytest
scripts/run_tests.sh runner
```

The groups are:

- `quick` runs unit and integration tests. Run it after every change.
- `unit` runs pure logic tests.
- `integration` starts real processes. It uses a fake llama-server that answers over HTTP like the real one, so it needs no model.
- `sweep` runs every BFCL ground truth answer through the official checker, every custom chain, and the full suite grid.
- `bfcl` runs everything about the BFCL integration.
- `runner` runs the suite, store, runner and results tests.
- `hardware` runs on a real device with a real llama-server and model. It skips itself unless `A2E_LLAMA_SERVER` and `A2E_GGUF` are set.
- `all` runs everything.

The hardware group is the first place where A2E meets a real llama-server. It checks the parts of the llama-server interface that the fake server copies, e.g., the names of the timing fields in a reply. Run it once on each device before a long run:

```bash
export A2E_LLAMA_SERVER=~/a2e-deps/llama.cpp/build/bin/llama-server
export A2E_GGUF=~/models/qwen3-4b/qwen3-4b-q4_0.gguf
export A2E_CTX=16384
uv run a2e test hardware -v
```

If a hardware test fails, its message names the part of the interface that did not match. [tests/README.md](../tests/README.md) describes every test file.

## 10. Troubleshooting

**"cannot tell whether this is a pi5 or a mac".** Set `device` under `[machine]` in `configs/local.toml`, or pass `--device`.

**A job failed with "model file missing".** Check the path in the error, fix the file name or set the exact path in `configs/local.toml`, and run again with `--retry-failed`.

**A job was skipped with "core tools need ... tokens".** The task's tool descriptions do not fit in the requested schema length, e.g., the retrieval family at 256 tokens. This is expected in the factorial suite.

**No energy columns in the summary.** On the Mac, run with `sudo -E`. On the Pi, check that `vcgencmd pmic_read_adc` prints rail readings. Check `power = "auto"` in `configs/local.toml`, and look for the `WARNING: no energy will be recorded` line at the start of the run. Use `power = "required"` to make such a run fail early. `a2e results report` prints the power scope that was recorded.

**llama-server does not start.** A2E writes the server's own log to `results/work/exp<N>/llama-server-<model>-<quant>.log`. Older llama.cpp builds may not accept `--flash-attn on`. In that case set `flash_attn` in the suite's `[server]` table to the value your build accepts.

**Multi-turn BFCL jobs fail with a server error.** The prompt grew past the context size. Raise `[server] ctx`. On the Pi, use `kv_quant = "q8_0"` for large contexts so the KV cache fits in memory.

**Qwen3 answers are empty or cut off.** Qwen3 can spend its whole decode budget on its thinking text before it writes a tool call. A2E strips thinking text before parsing, so a cut-off answer counts as no call. Use a larger `decode_len` if this happens often. The pilot shows how often it happens.

**The kernel profile is empty.** Run the kernels suite as root, or lower `kernel.perf_event_paranoid` with `sudo sysctl kernel.perf_event_paranoid=1`. The run's `server_json` stores the note perf gave.

**Web jobs fail with "web search needs a Brave API key".** Export `BRAVE_SEARCH_API_KEY` in the shell that runs A2E and run again with `--retry-failed`. If searches return errors, `cases.extra_json` holds the first few error messages, e.g., a 429 when the quota is used up.

**The suite lock check fails.** A suite file or the expansion code changed. If the change was intended, run `uv run a2e suite-lock`.

## 11. Where things are

```text
configs/models.toml           models in the study (committed)
configs/local.toml            this machine's paths and facts (not committed)
configs/suites/*.toml         suites
configs/suites.lock.json      expected jobs per suite
configs/tests.toml            test groups
results/a2e.db                results database (not committed)
results/raw/exp<N>.jsonl      BFCL raw model outputs
results/work/exp<N>/          llama-server logs and per-task working files
third_party/bfcl_eval/        official BFCL checker and data, unchanged (see NOTICE)
src/a2e/                      the code
tests/                        the tests
docs/                         these documents
REFERENCES.md                 every source, tool and dataset the project uses
```
