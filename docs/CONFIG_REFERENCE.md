# Configuration reference

A2E reads five kinds of configuration file. This document lists every field, its default and what it does. The defaults below are the values in the code (`src/a2e/suite.py`, `src/a2e/settings.py`, `src/a2e/testgroups.py`).

| File | Committed | Purpose |
|---|---|---|
| `configs/suites/<name>.toml` | yes | What to measure |
| `configs/models.toml` | yes | Which models exist and where their files are expected |
| `configs/local.toml` | no | Paths and facts about one machine |
| `configs/tests.toml` | yes | Named test groups |
| `configs/suites.lock.json` | yes | The expected jobs of each shipped suite |

All TOML files are read with Python's `tomllib`. A field that takes a list also accepts a single value, e.g., `depth = 3` means `depth = [3]`.

## 1. Suite files

A2E rejects a suite file that contains a field not listed here. The error message names the field.

### 1.1 `[suite]`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `name` | string | required | The suite's name. The experiment for a device is called `<name>@<device>`. |
| `description` | string | `""` | Free text shown by `a2e suites`, `a2e plan` and the TUI. |
| `seed` | integer | `0` | Seed for the custom task generator, the plain prompt choice, the schema padding and the BFCL subsample. |

### 1.2 `[repeats]`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `warmup` | integer, 0 or more | `1` | Runs per job that are recorded with `phase = 'warmup'` and left out of every summary. |
| `measured` | integer, 1 or more | `3` | Runs per job that are analysed (`phase = 'measured'`). |

The study protocol is 1 warm-up run followed by 3 measured runs. The prefix cache is cleared at the start of every run, so every repeat starts from the same state.

### 1.3 `[server]`

These settings are passed to llama-server. Jobs that differ in any of them use a separate server process.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `ctx` | integer, 512 or more | `8192` | Context size in tokens (`-c`). BFCL multi-turn categories need 16384 or more. |
| `flash_attn` | string | `"on"` | Value for `--flash-attn`. Newer llama.cpp builds take `on`, `off` or `auto`. KV cache quantization needs flash attention. |
| `threads` | integer, 0 or more | `0` | Value for `-t`. 0 means use `[machine] threads` from `local.toml`, or llama.cpp's own default if that is unset. |
| `thinking` | string | `"default"` | `"off"` starts llama-server with `--chat-template-kwargs '{"enable_thinking": false}'` and also sends that variable with every chat template request, so reasoning models such as Qwen3 answer directly instead of spending the token budget on a thinking block. Chat templates without that variable (Gemma 3, for one) are not affected. Only `"off"` is stored in a job's settings, so suites written before this field keep their job keys. A suite uses one value, and a different value starts a separate server. |

A2E always passes `--jinja`, and passes `-ngl 99` on the Mac (all layers on the GPU) and `-ngl 0` on the Pi (CPU only). It adds `--cache-type-k` and `--cache-type-v` when `kv_quant` is not `f16`.

### 1.4 `[run]`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `idle_seconds` | number, 0 or more | `20.0` | After each model load, A2E measures power for this long with the model loaded and idle. The mean is subtracted from every energy number. 0 turns it off. |
| `sample_hz` | number above 0 | `2.0` | How many resource samples A2E takes per second during a run. |
| `cooldown_c` | number, 0 or more | `0.0` | Before each job, wait until the chip is at or below this temperature in °C. 0 turns it off. Ignored when no temperature reading is available. |
| `cooldown_timeout_s` | number, 0 or more | `600.0` | Longest wait for the cooldown, in seconds. |
| `max_turns` | integer, 1 or more | `8` | Custom chains: the most model calls per task before A2E gives up. |
| `max_steps` | integer, 1 or more | `20` | BFCL multi-turn: the most model calls per user turn. Upstream BFCL uses 20. Reaching it ends the case as force terminated. |

### 1.5 `[matrix]`

Every combination of these four lists is one model load.

| Field | Type | Default | Allowed values |
|---|---|---|---|
| `device` | list | required | `pi5`, `mac` |
| `model` | list | required | names from `configs/models.toml` |
| `quant` | list | required | `q4_0`, `q8_0`, `q3_k_m` |
| `kv_quant` | list | `["f16"]` | `f16`, `q8_0` |

A2E only runs the jobs whose device matches the machine it runs on. The other device's jobs are not loaded into this machine's experiment.

### 1.6 `[axes]`

These are the default values of the settings a task can vary. Each task uses only some of them (see 1.7).

| Field | Type | Default | Meaning |
|---|---|---|---|
| `grammar` | list of booleans | `[false]` | When true, A2E sends a JSON schema of valid tool calls and llama-server restricts the output to it. |
| `prefix_cache` | list of booleans | `[true]` | When true, llama-server reuses the cached start of the previous prompt (`cache_prompt`). |
| `decode_len` | list of integers, 16 or more | `[256]` | Most tokens the model may generate per call (`n_predict`). Plain jobs generate exactly this many. |
| `schema_len` | list of integers, 16 or more | `[1024]` | Custom tasks: target size of the tool descriptions in tokens. |
| `governor` | list of strings | `[""]` | Pi CPU frequency governor for the job, e.g., `performance`. `""` leaves it unchanged. Needs root. |

### 1.7 `[[tasks]]`

Each entry adds one kind of task to every model load in the matrix.

| Field | Applies to | Type | Default | Meaning |
|---|---|---|---|---|
| `kind` | all | string | required | `plain`, `custom`, `bfcl` or `bench` |
| `n` | all | integer, 1 or more | plain 6, custom 5, BFCL all cases | Prompts, chains or BFCL cases per run |
| `family` | custom | list | required for custom | `calc`, `notes`, `retrieval`, `fs_search`, `fs_organize`, `math`, `fs_math`, `web` |
| `depth` | custom | list | required for custom | Tool calls per chain. `calc`, `notes` and `math` take 1 to 5. `retrieval`, `fs_search` and `fs_organize` take 2 to 5. `fs_math` takes 3 to 5. `web` takes 2 to 4. |
| `n_args` | custom | list | `[2]` | Arguments per note, 1 to 5. Only changes the `notes` family. |
| `distractors` | custom | string | `"bfcl"` | Extra tools that pad the tool list to `schema_len`. `bfcl` uses real BFCL function documents (130 functions from 9 BFCL classes). `synthetic` uses made-up tools. |
| `category` | bfcl | list | required for bfcl | BFCL categories (see below) |
| `n_prompt` | bench | list | empty | Prompt sizes for llama-bench prefill tests (`-p N -n 0`). |
| `n_gen` | bench | list | empty | Token counts for llama-bench decode tests (`-p 0 -n N`). A bench task needs at least one of `n_prompt` and `n_gen`. |
| `perf` | bench | boolean | `false` | Also record the top kernels with Linux perf. Pi only. |
| `reps` | bench | integer, 1 or more | `5` | llama-bench repetitions per test (`-r`). |
| `grammar`, `prefix_cache`, `decode_len`, `schema_len`, `governor` | see below | list | the `[axes]` value | Override the suite axis for this task only |

The axes each kind uses are:

| Kind | Axes used |
|---|---|
| `plain` | `prefix_cache`, `decode_len`, `governor` |
| `custom` | `grammar`, `prefix_cache`, `decode_len`, `schema_len`, `governor` |
| `bfcl` | `grammar`, `prefix_cache`, `decode_len`, `governor` |
| `bench` | `governor` |

An axis a kind does not use is not part of its jobs, so a plain job never runs twice just because the suite lists two grammar values.

The BFCL categories in scope are `simple_python`, `multiple`, `parallel`, `parallel_multiple`, `irrelevance`, `live_irrelevance`, `live_relevance`, `multi_turn_base`, `multi_turn_miss_func` and `multi_turn_miss_param`.

### 1.7a `[[harness]]`

The harness is the code around the model: how the tool list is written, what the model sees of earlier tool results, what happens when a reply cannot be read, and when the loop gives up. A suite can list several harness settings. Every custom and BFCL job then runs once for each one, on the same tasks, so the effect of each setting can be compared. Plain and bench jobs do not call tools and ignore the section. With no `[[harness]]` entry, only the default harness runs.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `name` | string | required | A label for the setting. Names must be unique. |
| `prompt_format` | string | `json_schema` | How the tool list is shown. `json_schema` is one JSON schema per line. `compact` is one line per tool with typed argument names, and a `?` after an optional argument. `natural` is plain sentences. The model still replies with a JSON array, and the parser does not change. |
| `context` | string | `full` | What the model sees of earlier tool results. `full` shows all of them. `mask` replaces every tool result except the latest 2 with a short stub. `truncate` cuts each tool result to 400 characters. The stored history is never changed, only the messages sent to the model. |
| `recovery` | string | `none` | What happens when a reply cannot be read as tool calls. `none` ends the task (custom) or the user turn (BFCL multi-turn). `feedback` keeps the bad reply and adds a message saying what was wrong. `retry` drops the bad reply and asks again with a one-line reminder that is not kept, at most 2 times in a row. BFCL single-turn cases get at most 2 more tries in both modes. A call that parses but fails validation gets an error result back in every mode. |
| `stopping` | string | `fixed` | When to give up. `fixed` uses the turn limits of `[run]`. `tight` caps the model calls at what a correct run needs plus 2 (a custom chain needs one call per step plus `final_answer`; a BFCL user turn needs one call per ground truth call plus the closing reply). `early_stop` ends the task after 2 identical calls in a row, or 2 failed turns in a row (an unreadable reply, a failed validation or a tool error). |

A harness entry with every field at its default is the baseline. Only the fields that differ from the default are written into a job's settings, so a job under the baseline harness has the same key as it had before this section existed. Two entries cannot have the same settings. The constants (2 stub-free results, 400 characters, 2 retries, 2 extra turns, 2 repeats) are in `src/a2e/agent/harness.py`.

### 1.8 Validation rules

A2E refuses a suite and prints the reason when:

- a field is not in the tables above
- a value has the wrong type or is out of range
- a custom task has no `family` or no `depth`, or names an unknown family
- a depth is outside the family's range
- a BFCL task has no `category`, or names a category outside the scope
- `family` or `depth` is set on a task that is not custom, or `category` on a task that is not BFCL
- a bench task has neither `n_prompt` nor `n_gen`, or a size below 1
- `n_prompt`, `n_gen` or `perf` is set on a task that is not bench
- a BFCL multi-turn category is used with `[server] ctx` below 16384
- `[matrix] model` names a model that is not in `configs/models.toml`
- a `[[harness]]` entry has an unknown value, an empty or repeated name, or the same settings as another entry

### 1.9 How a suite becomes jobs

For each combination in `[matrix]`, for each task, A2E takes every combination of the axes that task uses and every combination of the task's own lists (`family` × `depth` × `n_args`, or each `category`). Each result is one job.

A job's settings are stored as JSON with these fields:

- the suite fields `seed`, `warmup`, `measured`, `ctx`, `flash_attn` and `threads`
- the matrix fields `device`, `model`, `quant` and `kv_quant`
- the task kind, `kind`
- the axes the kind uses, e.g., `grammar` and `decode_len`
- the task's own fields, which are `n`, `family`, `depth` and `n_args` for custom tasks, `n` and `category` for BFCL tasks, and `n_prompt`, `n_gen`, `perf` and `reps` for bench tasks

The job key is the first 16 hex characters of the SHA-256 hash of this JSON with sorted keys. Two task entries that produce the same job produce one job.

A job is skipped, with the reason stored, when:

- the device is `pi5`, the quantization is `q8_0` and the model has more than 4 billion parameters ("not run: memory")
- the job sets a governor and the device is not `pi5` ("not run: governor settings apply to the Pi only")
- at run time, a custom task's tools do not fit in `schema_len` tokens ("not run: core tools need ...")

Jobs run in an order that groups them by device, model, quantization, KV cache type, context, flash attention and threads. Each model file therefore loads once per device. Within a model load, plain jobs run first, then custom jobs, then BFCL jobs, then bench jobs. Bench jobs stop llama-server before they start.

A bench job passes these llama-bench options: `-m`, `-p`, `-n`, `-r`, `-o json`, `-ngl 99` on the Mac or `-ngl 0` on the Pi, `-fa 1` when `flash_attn` is `on` (otherwise `-fa 0`), `-ctk` and `-ctv` when `kv_quant` is not `f16`, and `-t` when threads are set. With perf, the command is `perf record -F 999 -o <file> -- llama-bench ...`, followed by `perf report --stdio --no-children --sort dso,sym --percent-limit 0.5`.

The suite hash is the SHA-256 of the whole parsed suite, so comments and formatting do not change it. The experiment for a device is found by the suite name, the device and this hash.

## 2. `configs/models.toml`

```toml
[models.qwen3-4b]
params_b = 4.0
notes = "Qwen3 4B. Has a thinking mode; see docs for the decode budget."
```

| Field | Type | Default | Meaning |
|---|---|---|---|
| `params_b` | number | required | Parameter count in billions. Used for the Pi memory rule. |
| `file` | string | `"{models_dir}/{name}/{name}-{quant}.gguf"` | Template for the GGUF path. Placeholders are `{models_dir}`, `{name}`, `{quant}` (`q4_0`) and `{QUANT}` (`Q4_0`). |
| `files` | table | empty | Exact path per quantization, e.g., `q4_0 = "/data/Qwen3-4B-Q4_0.gguf"`. Wins over `file`. |
| `source` | string | `""` | Where the GGUF came from. Fill it in when you download a model. |
| `notes` | string | `""` | Free text. |

The committed registry has `qwen3-4b` (4.0), `qwen3-8b` (8.0), `xlam2-3b` (3.0), `xlam2-8b` (8.0) and `phi4-mini` (3.8). The Z.ai GLM model is not chosen yet, and the file has a commented placeholder for it.

## 3. `configs/local.toml`

Copy `configs/local.example.toml`. This file is merged over `models.toml`, so it can also override any model field.

### 3.1 `[paths]`

| Field | Default | Meaning |
|---|---|---|
| `llama_server` | none | Path to the llama-server binary. Required for runs. |
| `llama_bench` | `llama-bench` next to `llama_server` | Path to llama-bench. Only bench jobs need it. |
| `perf` | `perf` on the PATH | Linux perf binary for kernel profiles on the Pi. |
| `models_dir` | `<repo>/models` | Folder used by the model file templates. |
| `db` | `<repo>/results/a2e.db` | Results database. |
| `raw_dir` | `<repo>/results/raw` | BFCL raw outputs, one file per experiment. |
| `work_dir` | `<repo>/results/work` | llama-server logs and per-task working files. |

A leading `~` is expanded. A relative path is read from the repository folder.

### 3.2 `[machine]`

| Field | Default | Meaning |
|---|---|---|
| `device` | detected | `pi5` or `mac`. Set it when detection fails. Detection reads `/proc/device-tree/model` on Linux and checks for macOS on arm64. |
| `cooler` | unset | Whether the Pi's active cooler is fitted. Stored with every run. |
| `power` | `"auto"` | `auto` records energy where possible and prints a warning when it cannot (on the Mac, energy needs `sudo -E`). `none` turns energy off. `required` stops the run with an error when energy cannot be recorded. |
| `threads` | unset | llama-server `-t` when the suite's `threads` is 0. |

### 3.3 `[web]`

Settings for the `web` task family, which uses the Brave Search API.

| Field | Default | Meaning |
|---|---|---|
| `api_key_env` | `"BRAVE_SEARCH_API_KEY"` | Name of the environment variable that holds the key. The key is never read from a file. |
| `base_url` | `"https://api.search.brave.com"` | API address. Tests point it at a local fake. |
| `timeout_s` | `15.0` | Timeout per search or page fetch, in seconds. |
| `max_page_chars` | `4000` | Fetched pages are cut to this many characters before the model sees them. |
| `max_results` | `10` | Most results one search can return, whatever the model asks for. |

### 3.4 Per-model overrides

```toml
[models.qwen3-4b.files]
q4_0 = "~/models/Qwen3-4B-Q4_0.gguf"
q8_0 = "~/models/Qwen3-4B-Q8_0.gguf"
```

## 4. Environment variables

These win over both files.

| Variable | Overrides |
|---|---|
| `A2E_LLAMA_SERVER` | `[paths] llama_server` |
| `A2E_MODELS_DIR` | `[paths] models_dir` |
| `A2E_DB` | `[paths] db` |
| `A2E_DEVICE` | `[machine] device` |
| `A2E_CONFIG_DIR` | the `configs` folder |
| `BRAVE_SEARCH_API_KEY` | The Brave Search API key for the `web` family. It is not an override. `[web] api_key_env` sets the variable's name. |

The hardware test group also reads these:

| Variable | Meaning |
|---|---|
| `A2E_LLAMA_SERVER` | llama-server binary for the tests |
| `A2E_GGUF` | one model file for the tests |
| `A2E_DEVICE` | `pi5` or `mac` (detected when unset) |
| `A2E_CTX` | context size, default 4096. Multi-turn hardware tests need 16384. |
| `A2E_THREADS` | llama-server `-t` |
| `A2E_FLASH_ATTN` | value for `--flash-attn`, default `on` |
| `A2E_LLAMA_BENCH` | llama-bench binary for the bench hardware tests, default next to `A2E_LLAMA_SERVER` |

## 5. `configs/tests.toml`

```toml
[groups.quick]
description = "Unit and integration tests. No hardware. About a minute. Run this constantly."
markers = "unit or integration"
```

| Field | Default | Meaning |
|---|---|---|
| `description` | `""` | Shown by `a2e test --list` and the TUI. |
| `markers` | none | pytest `-m` expression. Every test is marked with the folder it is in: `unit`, `integration`, `sweep` or `hardware`. |
| `keyword` | none | pytest `-k` expression. |
| `paths` | empty | Test files, relative to the repository. |
| `env` | empty | Variables the group needs. A2E prints a note when one is missing. |

`a2e test <group> [pytest args]` runs `python -m pytest -ra` with these options from the repository folder. Extra arguments are added at the end.

## 6. `configs/suites.lock.json`

One entry per shipped suite:

| Field | Meaning |
|---|---|
| `suite_hash` | The suite hash (section 1.9). |
| `jobs` | Number of jobs for both devices, skipped ones included. |
| `job_keys_sha` | First 16 hex characters of the SHA-256 of all job keys joined in run order. |

`a2e suite-lock` rewrites the file and `a2e suite-lock --check` compares it with the current suites. The lock is computed from `configs/models.toml` alone, without `local.toml`, so it is the same on every machine. The test `tests/unit/test_suite.py` fails when the lock does not match.
