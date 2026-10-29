# Data dictionary

All results live in one SQLite file, `results/a2e.db` by default. This document describes every table, every column, the analysis views, and how each derived number is computed. The schema is defined in `src/a2e/store.py` and the derived numbers in `src/a2e/results.py`.

## 1. Why SQLite

One SQLite file is enough for this study, and it keeps the setup simple:

- It needs no server, so it works the same on the Pi and the Mac.
- One process writes (the runner) and any number of processes can read at the same time (the TUI, `a2e results`, a notebook). The database runs in WAL mode, which allows this.
- Each run is written in one transaction. A crash in the middle of a run leaves nothing half written.
- The file can be copied between machines. You can merge Pi and Mac results by reading both files, or by exporting both to parquet.
- `a2e results export` writes every table and view as parquet or csv for pandas, R or a spreadsheet.

The expected size is small. A turn row is a few hundred bytes and a sample row about 60 bytes. As an example, the factorial suite has about 648 runnable jobs per device. Each job has 4 runs of 5 chains, and a chain takes about 4 model calls, so one device writes about 52,000 turn rows, which is roughly 20 MB. At 2 samples per second, each hour of measuring adds about 7,200 sample rows, which is under 1 MB.

A server database would only be needed if many machines had to write to one shared store at the same time. That is not the case here, because each device writes its own file.

## 2. Schema versions

The schema version is stored in SQLite's `user_version`. `Store` upgrades an older file on open by applying the entries of `MIGRATIONS` in order, and refuses to open a file that is newer than the code. The current version is 2. Version 1 created the experiment, job, run, turn, case and sample tables and the views. Version 2 added the `bench` and `kernels` tables. A change to the schema must add a new entry to `MIGRATIONS` and must not edit an old one.

The connection settings are `foreign_keys = ON`, `busy_timeout = 10000` ms, `journal_mode = WAL` and `synchronous = NORMAL`.

## 3. How the tables relate

```text
experiments 1──n jobs 1──n runs 1──n turns
                              │──n cases
                              │──n samples
                              │──n bench     (bench jobs)
                              └──n kernels   (bench jobs with perf, Pi)
```

An experiment is one suite on one device. A job is one configuration. A run is one repeat of a job. Turns, cases and samples belong to a run.

## 4. Tables

### 4.1 `experiments`

| Column | Type | Meaning |
|---|---|---|
| `exp_id` | integer | Key. |
| `name` | text | `<suite name>@<device>`. |
| `suite_hash` | text | Hash of the parsed suite. The runner continues an experiment only if the name and hash match. |
| `suite_json` | text | The whole parsed suite as JSON. |
| `env_json` | text | Snapshot of the machine when the experiment started (section 4.7). |
| `git_commit` | text | A2E commit. |
| `git_dirty` | integer | 1 if tracked files had uncommitted changes. |
| `status` | text | `running`, `complete` or `interrupted`. |
| `started_at`, `finished_at` | real | Unix time in seconds. |
| `notes` | text | Free text, not set by A2E. |

### 4.2 `jobs`

| Column | Type | Meaning |
|---|---|---|
| `job_id` | integer | Key. |
| `exp_id` | integer | Experiment. |
| `job_key` | text | Hash of the job settings. Unique within an experiment. |
| `kind` | text | `plain`, `custom` or `bfcl`. |
| `config_json` | text | The job settings (see CONFIG_REFERENCE.md section 1.9). |
| `device`, `model`, `quant`, `kv_quant` | text | Copied from the settings for quick filtering. |
| `status` | text | `pending`, `running`, `done`, `failed` or `skipped`. |
| `reason` | text | Why the job was skipped or failed. |
| `error` | text | Python traceback of a failure. |
| `attempts` | integer | How many times the job was started. |
| `started_at`, `finished_at` | real | Unix time. |

A job becomes `done` only after all of its runs are written. A job that was `running` when the process stopped is set back to `pending` on the next start.

### 4.3 `runs`

| Column | Type | Meaning |
|---|---|---|
| `run_id` | integer | Key. |
| `exp_id`, `job_id` | integer | Experiment and job. Empty for ad hoc runs from `a2e bfcl run`. |
| `config_json` | text | The job settings. |
| `device` | text | `pi5` or `mac`. |
| `phase` | text | `warmup` or `measured`. Ad hoc runs use other values. |
| `repeat_idx` | integer | 0 for the first repeat. |
| `task_kind` | text | `plain`, `custom` or `bfcl`. |
| `status` | text | `running`, `complete` or `aborted`. Only `complete` runs reach the views. |
| `started_at`, `finished_at` | real | Unix time. |
| `idle_power_w` | real | Mean idle power measured after the model load, in watts. Subtracted from all energy numbers of the run. |
| `power_scope` | text | What the power readings cover, e.g., "sum of Pi 5 PMIC rail V*I; excludes PSU losses and peripherals (estimate)". `none` when no power was recorded. |
| `load_time_s` | real | Seconds from starting llama-server to its health check passing. |
| `server_json` | text | Server arguments, the model file fingerprint (size, modification time, SHA-256 of the first MiB), cooldown wait, the actual governor, the cooler setting and notes about the instruments. |

Every run of a job that did not finish is marked `aborted`, including repeats that had completed, because a job's runs only count as a set.

### 4.4 `turns`

One row per model call.

| Column | Unit | Meaning |
|---|---|---|
| `turn_id`, `run_id` | | Key and run. |
| `task_id` | | Plain prompt id, custom task id or BFCL case id. |
| `step` | | Index of the call within the task, from 0. |
| `user_turn` | | BFCL multi-turn: the index of the user turn this call belongs to. |
| `prefill_ms` | ms | Prompt processing time reported by llama-server (`timings.prompt_ms`). Covers only tokens not served from the cache. |
| `decode_ms` | ms | Generation time reported by llama-server (`timings.predicted_ms`). |
| `parse_ms` | ms | Time to read tool calls out of the model's text. |
| `validate_ms` | ms | Time to check the calls against the tool schemas (custom) or to build safe call strings (BFCL multi-turn). |
| `tool_wait_ms` | ms | Time spent running the tools. |
| `template_ms` | ms | Time for llama-server's `/apply-template` call, which formats the messages with the model's chat template. |
| `harness_ms` | ms | Turn time not covered by any other column: `wall_ms − template_ms − model call time − parse_ms − validate_ms − tool_wait_ms`, floored at 0. |
| `wall_ms` | ms | Whole turn, measured by A2E from before the template call to after the tools. |
| `prompt_tokens` | tokens | Whole prompt size: processed tokens plus cached tokens (`prompt_n + cache_n`). |
| `cached_tokens` | tokens | Tokens served from the prefix cache (`cache_n`). |
| `gen_tokens` | tokens | Generated tokens (`predicted_n`). |
| `n_calls` | | Tool calls parsed from this turn. |
| `energy_j` | J | Energy over the whole turn window, idle subtracted. |
| `energy_prefill_j`, `energy_decode_j`, `energy_tool_j` | J | Energy over each stage window, idle subtracted. |
| `avg_power_w` | W | Mean power over the turn window, idle not subtracted. |
| `peak_rss_mb` | MiB | Highest resident memory of llama-server and its child processes during the turn. |
| `avg_cpu_pct` | % | Mean CPU use of the llama-server process (100% is one core). |
| `max_temp_c` | °C | Highest chip temperature (Pi). |
| `mean_freq_mhz` | MHz | Mean ARM clock (Pi). |
| `throttled` | 0 or 1 | 1 if the Pi reported a current throttling condition in any sample of the turn. |
| `correct` | 0 or 1 | Custom chains: whether this call matched the expected call for its step. |
| `error` | | Error text when the model call itself failed. |

### 4.5 `cases`

One row per scored task in a run.

| Column | Meaning |
|---|---|
| `case_row_id`, `run_id` | Key and run. |
| `case_id` | Task or BFCL case id. |
| `category` | `plain`, `custom:<family>` or the BFCL category. |
| `valid` | 1 passed, 0 failed, empty for plain prompts (not scored). |
| `error_type` | Why it failed, e.g., BFCL's `simple_function_checker:wrong_func_name`, `multi_turn:instance_state_mismatch`, A2E's `a2e:no_call:no_json`, or for custom chains `parse_error`, `max_turns`, `server_error`, `wrong_step` or `wrong_state` (every call was right but the final file system differs from the expected one). |
| `status` | How the task ended: `done`, `final_answer`, `parse_error`, `max_turns`, `force_terminated` or `server_error`. |
| `n_user_turns` | User turns in the case (1 except BFCL multi-turn). |
| `n_model_calls` | Model calls the task used. |
| `schema_tokens` | Size of the tool descriptions in model tokens. |
| `wall_ms` | Sum of the turns' `wall_ms`. |
| `extra_json` | Custom: `n_correct`, `n_steps`, `finished`, `token_count`, `tools` (`builtin`, `bfcl` or `web`), `distractors` (`bfcl` or `synthetic`). File-system families add `scenario` (the curated BFCL file system id), and `fs_organize` adds `state_ok`. `web` adds `question`, `web_returned_urls` and up to five `web_errors`. BFCL multi-turn: `irrelevance_valid`, `unseen_calls`. `token_count` is `server` when token counts came from llama-server's tokenizer, or `approx_chars_div_4` when that was not available. |

### 4.6 `samples`

| Column | Unit | Meaning |
|---|---|---|
| `run_id` | | Run. |
| `t` | s | Seconds since the run started. |
| `rss_mb` | MiB | Resident memory of llama-server and its children. |
| `cpu_pct` | % | CPU use of the llama-server process. |
| `temp_c` | °C | Chip temperature (Pi, `vcgencmd measure_temp`). |
| `freq_mhz` | MHz | ARM clock (Pi, `vcgencmd measure_clock arm`). |
| `power_w` | W | Power reading (see section 5). Empty when no reading. |
| `throttled` | 0 or 1 | Pi `get_throttled` current-condition bits (under-voltage, frequency capped, throttled, soft temperature limit). |
| `pressure` | text | Mac thermal pressure level from `powermetrics`, e.g., `Nominal`. |

### 4.7 `bench`

One row per llama-bench test in a bench run.

| Column | Unit | Meaning |
|---|---|---|
| `bench_id`, `run_id` | | Key and run. |
| `test` | | `pp` for prefill (prompt processing) or `tg` for decode (text generation). |
| `n_prompt`, `n_gen` | tokens | Test size. A prefill test has `n_gen = 0` and a decode test has `n_prompt = 0`. |
| `n_threads` | | Threads llama-bench used. |
| `avg_tok_s`, `sd_tok_s` | tokens/s | Mean and standard deviation over llama-bench's repetitions (`avg_ts`, `stddev_ts`). |
| `avg_ms` | ms | Mean time per repetition (`avg_ns` / 1e6). |
| `raw_json` | | The full llama-bench result object. |

### 4.8 `kernels`

One row per function in the perf profile of a bench test (Pi only).

| Column | Unit | Meaning |
|---|---|---|
| `run_id` | | Run. |
| `test` | | `pp` or `tg`. |
| `n_tokens` | tokens | The test size. |
| `rank` | | 1 for the function with the most samples. The top 15 are kept. |
| `pct` | % | Share of CPU samples spent in this function. |
| `symbol` | | Function name, e.g., `ggml_vec_dot_q4_0_q8_0`. |
| `dso` | | Library or binary the function is in, e.g., `libggml-cpu.so`. `[kernel.kallsyms]` means the operating system kernel. |

### 4.9 `env_json` fields

`captured_at`, `hostname`, `os`, `machine`, `hardware` (e.g., "Raspberry Pi 5 Model B Rev 1.0" or the Mac CPU name), `device_detected`, `device_configured`, `cpu_count`, `ram_gb`, `python`, `a2e_commit`, `a2e_dirty`, `llama_server`, `llama_server_version`, `governor`, `cooler`, `power_setting`, `vcgencmd`, `powermetrics` and `is_root`.

## 5. How energy is computed

1. During a run, a background thread reads the power source `sample_hz` times per second and stores `(t, watts)` with the same clock A2E uses for stage timing (`time.perf_counter`).
2. For each turn, A2E knows the start and end of each stage window:
   - the prefill window starts when the template call returns and lasts `prefill_ms`
   - the decode window runs from the end of prefill to when the model call returns
   - the parse, validate and tool windows follow in order
3. A2E treats the power trace as straight lines between samples and integrates it over each window. Where a window starts or ends between two samples, the value at the edge is interpolated. Outside the first and last sample, the nearest reading is held. This means a stage shorter than the sampling interval still gets energy.
4. The idle power measured after the model load is subtracted, so the numbers are the energy above idle.

The prefill and decode windows are approximate. llama-server reports how long each took but not when each started, and the model call also includes the HTTP request. The turn's total energy (`energy_j`) is exact for its window. The prefill, decode and tool energies are close but may not add up to the total, because parse, validate, template and harness time are not in any of the three.

Power sources:

- **Pi 5.** `vcgencmd pmic_read_adc` lists current and voltage per rail. A2E pairs each rail's current with its voltage, multiplies them and adds up all rails. The `EXT5V` input rail has only a voltage reading and is skipped. The total leaves out power supply losses and USB devices.
- **Mac.** `powermetrics --samplers cpu_power,gpu_power,thermal` runs in the background at 500 ms intervals and A2E keeps the last "Combined Power (CPU + GPU + ANE)" value. If that line is missing, CPU plus GPU power is used. The total leaves out memory, the display and the power supply.

## 6. Analysis views

The views drop runs whose status is not `complete`.

`v_runs` is `runs` plus these columns read from `config_json`: `model`, `quant`, `kv_quant`, `grammar`, `prefix_cache`, `decode_len`, `schema_len`, `family`, `depth`, `n_args`, `category` and `governor`. Booleans read from JSON appear as 1 or 0.

`v_turns` is `turns` plus `exp_id`, `job_id`, `phase`, `repeat_idx`, `task_kind`, `device`, `idle_power_w` and the `v_runs` setting columns.

`v_cases` is `cases` plus the same run columns, except `idle_power_w`.

## 7. Derived numbers in `a2e results`

The results commands use measured runs only. Each number is computed per run first and then averaged over the measured runs of a configuration. `_sd` columns are the standard deviation across those runs, with one degree of freedom removed, so they describe the spread between repeats.

Within one run, with sums over the run's turns:

| Number | Formula |
|---|---|
| `prefill_ms`, `decode_ms`, `tool_wait_ms`, `wall_ms` | mean over turns |
| `prefill_tok_s` | Σ(prompt_tokens − cached_tokens) / Σ prefill_ms × 1000 |
| `decode_tok_s` | Σ gen_tokens / Σ decode_ms × 1000 |
| `cache_hit` | Σ cached_tokens / Σ prompt_tokens |
| `prefill_share`, `decode_share`, `tool_share` | Σ stage_ms / Σ wall_ms |
| `overhead_share` | Σ(parse_ms + validate_ms + template_ms + harness_ms) / Σ wall_ms |
| `j_per_turn` | mean energy_j |
| `j_per_prompt_token` | Σ energy_prefill_j / Σ(prompt_tokens − cached_tokens) |
| `j_per_gen_token` | Σ energy_decode_j / Σ gen_tokens |
| `tokens_per_joule` | Σ gen_tokens / Σ energy_decode_j, the same as decode tokens per second per watt |
| `avg_power_w` | mean avg_power_w |
| `peak_rss_mb`, `max_temp_c` | maximum over turns |
| `throttled_turns` | number of turns with throttled = 1 |
| `success` | mean of `valid` over scored cases |
| `n_valid`, `n_scored` | passed cases, scored cases |
| `j_per_success` | Σ energy_j over all turns / n_valid |

`a2e results stages` divides each stage's total time by the total turn time. The stages are `prefill`, `decode`, `parse`, `validate`, `tool_wait`, `template` and `harness`. `unattributed` is the rest, which is time inside the HTTP call that llama-server did not report as prefill or decode.

`a2e results resources` reports the power, memory, CPU and thermal envelope of each configuration from the 2 Hz `samples` table. The samples of a run are taken only while the run is active, so model load and the idle baseline are not included. Each statistic is computed per run and then combined over the measured runs of a configuration:

| Column | Meaning | Combined over runs by |
|---|---|---|
| `samples`, `duration_s` | Number of samples and the time between the first and last | mean |
| `power_min_w`, `power_max_w` | Lowest and highest power reading | lowest, highest |
| `power_median_w`, `power_mean_w`, `power_p95_w` | Median, mean and 95th percentile of the power readings. The median is the sustained level. | mean |
| `energy_j` | Sum of `energy_j` over the run's turns, with the idle baseline subtracted | mean |
| `rss_min_mb`, `rss_mean_mb`, `rss_max_mb` | Resident memory of llama-server and its children | lowest, mean, highest |
| `cpu_mean_pct`, `cpu_max_pct` | CPU use of the llama-server process | mean, highest |
| `temp_max_c` | Highest chip temperature (Pi only; empty on the Mac) | highest |
| `thermal_pressure` | Worst Mac thermal pressure level seen: `Nominal`, `Fair`, `Serious` or `Critical` | worst |
| `power_scope` | What the power readings include, as stored with the run | |

Power and energy columns are empty, never zero, when the run had no power source. On the Mac the chip temperature is not available. `powermetrics` gives only the thermal pressure level. The power readings are CPU, GPU and Neural Engine package power and leave out memory, the display and the power supply. Memory is the resident size of the server process, so it leaves out other processes and the operating system.

`a2e results task-resources` gives one row per task (case). `turns` is the number of model calls in the task, `energy_j` is the sum over those calls, `avg_power_w` is the mean power of the calls weighted by their duration, and `peak_rss_mb` and `max_temp_c` are the highest values over the calls. A task's turns are matched to its case by `task_id` equal to `case_id` inside one run.

`a2e results bench` averages each llama-bench test over measured runs per configuration (`tok_s`, `tok_s_sd`). For decode tests it adds `est_bandwidth_gbs`, which is `tok_s` × model file size in GB. Decode reads every weight once per generated token, so this is an estimate of the memory bandwidth decode uses. Comparing it with the device's memory bandwidth shows how close decode is to the memory limit. Prefill numbers have no bandwidth column, because prefill is mostly limited by compute.

`a2e results kernels` averages each function's share over measured runs and ranks the functions per configuration and test.

`a2e results efficiency` measures how much energy an agent spends per completed task. It uses the scored tasks of measured runs, with each task's energy from `task-resources`. Everything is computed per run and then averaged over the measured runs of a configuration. Rows without energy stay empty.

| Column | Meaning |
|---|---|
| `tasks`, `successes` | Scored tasks in a run, and how many passed |
| `success_rate` | `successes` / `tasks` |
| `energy_j` | Energy over all scored tasks of the run, failures included |
| `j_per_task` | `energy_j` / `tasks` |
| `j_per_success` | `energy_j` / `successes`. Empty when nothing passed. A failed attempt counts against the configuration. |
| `waste_share` | Energy of failed tasks / `energy_j` |
| `_sd` columns | Standard deviation across the measured runs |
| `on_front` | (`frontier` view) True when no other row on the same device, task kind and category has lower or equal `j_per_task` and higher or equal `success_rate`, with one strict |
| `budget_j`, `tasks_done`, `successes` | (`budget` view) Tasks the budget buys when the suite runs in order. A task that would overshoot is not counted. Without a given budget list, ten values up to the median full-suite energy of each device are used. |
| `*_delta`, `*_ratio` | (`grammar` view) Grammar on minus off, and on divided by off. A `j_per_gen_token_ratio` above 1 means grammar made each generated token cost more. |
| `prefill_share`, `decode_share`, `tool_wait_share`, `other_share` | (`stages` view) Shares of attributed energy. `other_share` is parsing, validation, templating and harness time. |

`a2e results stability` takes each job's tasks in order and computes decode tokens per second per task. It reports the mean of the first 3 tasks, the mean of the last 3, and the first task (from the fourth on) whose speed is below 90% of the first 3. This marks where thermal throttling starts to slow decoding. It also reports the highest temperature and the number of tasks during which the Pi reported throttling.
