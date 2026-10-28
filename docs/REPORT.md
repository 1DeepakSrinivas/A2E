# A2E methods and design report

Profiling tool-calling workloads of small language models on a Raspberry Pi 5 and an Apple Silicon Mac.

Date of this version: 28 October 2026.

## About this report

This report describes what the A2E study measures, how the measurement system works, and why each choice was made. It is organized in two parts. The first part covers the functional areas, which are the things the system does. The second part covers the non-functional areas, which are properties the system must have, e.g., reproducibility and data integrity. A decisions log and a list of open items follow.

The report contains no measurement results. At the time of writing, the system has been built and tested against simulated servers in a development container. It has not yet been run on a Pi, on a Mac, or with a real model. Section 12 lists what must be checked on real hardware first. Results will come from the results database through `a2e results report` (section 7.4).

Every source the study uses is listed in [REFERENCES.md](../REFERENCES.md) with its verification status. Field-level detail is in [CONFIG_REFERENCE.md](CONFIG_REFERENCE.md) and [DATA_DICTIONARY.md](DATA_DICTIONARY.md).

## 1. Background and aims

### 1.1 Background

An earlier project (Project 1) profiles small language models on edge hardware for plain text generation. It measures latency, throughput, energy, memory and thermals across model, quantization and sequence length. This study extends that work to agentic workloads, where the model calls tools instead of only generating text.

Tool calling changes the cost of running a model in three ways:

- The prompt grows, because it must include a description of every tool. This makes prefill longer.
- The output must follow a fixed format so a program can read and run it. If decoding is constrained by a grammar, each generated token costs more.
- An agent calls the model several times in a loop, and waits for tools to run between calls. This adds tool wait time and makes the prompt grow with every step.

A single latency or throughput number hides these three costs. The study measures them separately.

### 1.2 Motivation

The proposal gives three reasons for the study:

- Tool calling is now the most common way small models are used on devices, but most edge profiling work covers only plain generation.
- Schema prefill, grammar-constrained decoding and tool wait are separate costs that one latency number hides.
- Few studies look at how quantization and KV cache settings affect whether the model picks the right tool.

### 1.3 Research questions

1. How do model family, model size and quantization affect latency and energy for tool calling compared with plain generation?
2. Where are the bottlenecks? The candidates are schema prefill, constrained decoding, output validation and tool execution.
3. How do schema length, KV cache settings and serving settings, e.g., prefix caching, affect performance and energy over multi-turn agent sessions?
4. How do the results differ between the two devices?

The study is comparative. Its purpose is to find patterns and real bottlenecks across similar configurations on the two devices, not to publish single headline numbers.

### 1.4 Analysis checklist from the proposal

- Normalize latency and energy per token and per successful tool call.
- Separate memory-bound and compute-bound behaviour, and give the share of turn time spent on schema prefill.
- Measure the effect of quantization on tool selection accuracy, not only on text quality.
- Record thermal throttling in long agent sessions and other knee points.
- Give deployment recommendations for on-device agents.

Section 7.4 maps each item to the result table that answers it.

## 2. Scope

### 2.1 In scope

| Area | Scope |
|---|---|
| Devices | Raspberry Pi 5 with 8 GB (CPU only). Apple Silicon Mac (Metal GPU). Both are measured. |
| Runtime | llama.cpp with GGUF model files, driven through `llama-server` over HTTP, and `llama-bench` for microbenchmarks. One llama.cpp commit for the whole study. |
| Models | Qwen3-4B, Qwen3-8B, xLAM-2 3B, xLAM-2 8B, Phi-4-mini 3.8B. One Z.ai GLM small model is planned but not chosen. |
| Quantization | Q8_0 and Q4_0 weights. KV cache at f16 or q8_0. |
| Sequence lengths | Tool schema of 256, 1024 or 2048 tokens. Decode length of 128 or 512 tokens. Agent chains of 2 to 5 tool calls. |
| Workloads | Plain generation, custom tool chains on real local tools, BFCL v4 single-turn, relevance and multi-turn categories, llama-bench prefill and decode tests. |
| Measurements | Stage timing, token counts, prefix cache use, energy, memory, temperature, clock, throttling, CPU use, accuracy, top kernels (Pi). |

### 2.2 Out of scope, and why

| Item | Reason |
|---|---|
| Android phone, MLC LLM, Android Battery Historian | The user narrowed the devices to the Pi and the Mac. |
| Other runtimes (MLC, Ollama, MLX, vLLM) | The user chose llama.cpp only, with Metal on the Mac. |
| BFCL Java and JavaScript categories | The study is about tool calling in general. The Python categories cover it, and the other languages need tree-sitter parsers. |
| BFCL `multi_turn_long_context`, web search and memory categories | Not selected. Their backends need extra packages (BeautifulSoup, requests, rank_bm25, faiss). |
| API-Bank, ToolBench, tau-bench | To be decided after the pilot (section 6.1). |
| Kernel profiles on the Mac | Linux perf does not exist on macOS. Xcode Instruments would be needed. |
| Several requests in parallel (batching) | Not built. Every request is sent alone. See section 13. |
| Text quality of plain generation | Plain generation is the cost baseline. Accuracy is measured on tool calls. |

## Part A. Functional areas

## 3. Experiment definition and configuration

### 3.1 What it does

A suite file in `configs/suites/` describes a set of measurements. It lists devices, models, quantizations and KV cache types (the matrix), the values of the settings that tasks can vary (the axes), and one or more tasks. A2E expands a suite into jobs. A job is one model file on one device with one server configuration, one task and one value for every axis the task uses.

The shipped suites are:

| Suite | Purpose (study design step) | Runnable jobs | Skipped | Runs | Model loads |
|---|---|---|---|---|---|
| `smoke` | Check a new setup | 8 | 0 | 8 | 2 |
| `pilot` | Pilot (step 1) | 108 | 0 | 432 | 4 |
| `baseline` | Baseline, plain vs tool calling (A) | 108 | 12 | 432 | 18 |
| `factorial` | Factorial sweep on custom chains (B) | 1,296 | 144 | 5,184 | 36 |
| `chains` | Chain depth 2 to 5 and argument count (B) | 720 | 80 | 2,880 | 18 |
| `bfcl` | Public benchmark accuracy and cost (B) | 720 | 80 | 1,440 | 36 |
| `cache_ablation` | Prefix cache and KV cache over multi-turn sessions (question 3) | 120 | 0 | 480 | 12 |
| `kernels` | Operator-level bottleneck analysis (C) | 36 | 4 | 36 | 0 |
| `stability` | Thermal throttling and governor variation (D) | 8 | 4 | 24 | 4 |

Counts cover both devices. A run is one repeat of one job. Bench jobs use llama-bench and load no server.

### 3.2 How it works

- **Validation.** Suite files are parsed with strict Pydantic models. An unknown field, an out-of-range value, an unknown family or category, a depth outside the family's range, or a multi-turn BFCL category with a context below 16384 stops A2E with a message that names the problem.
- **Axes per task kind.** Plain tasks use prefix cache, decode length and governor. Custom tasks also use grammar and schema length. BFCL tasks use grammar, prefix cache, decode length and governor. Bench tasks use governor only. An axis a kind does not use is not part of its jobs, so a setting that cannot change a task's result never multiplies its jobs.
- **Job keys.** A job's settings are written as canonical JSON with sorted keys. The key is the first 16 hex characters of its SHA-256 hash.
- **Order.** Jobs are sorted by device, model, quantization, KV cache type, context, flash attention and threads, so each model file loads once per device. Within a load the order is plain, custom, BFCL, bench.
- **Feasibility.** A job is skipped with a stored reason when it would run Q8_0 on the Pi for a model above 4 billion parameters, or set a CPU governor on the Mac. A custom job whose tools do not fit the schema length is skipped when it runs.
- **Machine settings.** `configs/models.toml` (committed) lists the models and their parameter counts. `configs/local.toml` (not committed) holds one machine's paths and facts. Environment variables override both.

### 3.3 Why

- A declarative file can be reviewed, versioned and run again. The same file always gives the same jobs, so "run the pilot" means the same thing in every session.
- Expanding only the axes a task uses keeps the suites small enough to run. The factorial suite alone is 1,296 jobs.
- Grouping jobs by model file avoids reloading a model several gigabytes in size between jobs.
- The Pi memory rule follows from the Pi's 8 GB. An 8B model at Q8_0 is about 8.5 GB of weights, so it cannot load. Recording the skip with a reason keeps the factorial grid complete and honest.

## 4. Model serving

### 4.1 What it does and how

A2E starts `llama-server` for each model load with these arguments:

- `-m <file>`, `--port <free port>`, `-c <ctx>`
- `--jinja`, so the server can apply the model's own chat template
- `--flash-attn <value>`, `on` by default
- `-ngl 99` on the Mac, which puts all layers on the GPU, and `-ngl 0` on the Pi, which keeps everything on the CPU
- `--cache-type-k` and `--cache-type-v` set to `q8_0` when the KV cache is quantized
- `-t <threads>` when threads are set

A2E waits for `/health` (up to 900 s) and records the load time. It keeps the server for every following job with the same server settings and stops it when the settings change. The server's log is written to `results/work/exp<N>/llama-server-<model>-<quant>.log`.

Requests go to `/completion` with `temperature = 0` and `seed = 0`, which is greedy decoding. Each request sets `n_predict` (the decode length), `cache_prompt` (the prefix cache axis), and optionally `json_schema` (the grammar axis) or `ignore_eos` (plain generation).

### 4.2 Why

- A separate server process lets A2E measure the process's memory from outside and restart cleanly after a crash. The same HTTP interface works on both devices.
- llama-server reports prefill time, decode time, processed tokens and cached tokens for every request in a `timings` block. These are measured inside the server, so HTTP overhead does not distort them.
- Greedy decoding makes outputs repeatable, so repeated runs differ only in timing and energy, not in what the model did.
- KV cache quantization in llama.cpp needs flash attention, so flash attention is on by default.

## 5. Workloads

### 5.1 Plain generation

The model writes text for a prompt with no tools. Twelve fixed prompts are in `src/a2e/data/plain_prompts.json`. Each run picks `n` of them with the suite seed. Each request sets `ignore_eos`, so the model generates exactly `decode_len` tokens.

Plain generation is the baseline for research question 1. Fixing the output length makes plain and tool-calling numbers comparable per token, because otherwise a model that stops early would look faster.

### 5.2 Custom tool chains

**Tools.** The tools are real and run locally:

| Tool | What it does |
|---|---|
| `calculator(expression)` | Evaluates arithmetic with a safe parser. Only numbers, `+ - * / // % **` and parentheses are allowed, exponents are limited to 64 and expressions to 200 characters. |
| `note_write(title, body, tags, priority, due)` | Writes a note file as JSON. |
| `note_read(title)`, `note_list()` | Read one note, list all titles. |
| `search_docs(query, top_k)` | Keyword search over a document collection on disk, returning the top `top_k` (default 3) documents. |
| `doc_read(doc_id)` | Reads a document. |
| `set_reminder(text, minutes)` | Appends a reminder to a file. |
| `final_answer(answer)` | Ends the task. |

The document collection has 30 synthetic settlement reports with a name, population, founding year, area and main export. It is generated from the seed, so it is the same on every machine.

**Families.** Three families control chain depth and argument count:

- `calc` takes depth 1 to 5. The model makes one calculator call per expression.
- `notes` takes depth 1 to 5. The model writes depth minus 1 notes and then lists them. `n_args` from 1 to 5 sets how many fields each note has.
- `retrieval` takes depth 2 to 5. The model searches, reads the best document, doubles its population with the calculator, saves a note and lists the notes, in that order, up to the depth. Later steps depend on earlier tool output, e.g., the document id comes from the search result.

**Schema length.** Each family has a fixed set of core tools. A2E adds seeded distractor tools with plausible names and parameters until the rendered tool list is within 5% of `schema_len` tokens, counted with the model's own tokenizer through `/tokenize`. The last distractor's description is lengthened or shortened to close the gap. The tool order is shuffled with the seed so the needed tools are not always first. The same core tools, target and seed always give the same text, so the prompt prefix is identical across tasks and the prefix cache can work. If the core tools alone are larger than the target, the job is skipped.

**Loop.** The model is called, its calls are validated against each tool's JSON schema, valid calls run, and the results go back to the model in a `<tool_response>` message. Invalid calls and tool errors are reported back to the model as text so it can recover. Tool output is cut at 1,000 characters. The loop ends at `final_answer`, at a parse error, at a server error, or after `max_turns` (8) model calls. Each task runs in its own folder, so one task cannot see another task's notes.

**Scoring.** Step k of a task is scored against the first call in model turn k. The call must have the expected tool name, and the expected arguments must match after normalization: strings ignore case and extra spaces, numbers compare by value, and calculator expressions ignore all spaces. For dependent steps, the expected arguments are computed from the real output of the model's earlier calls. A task succeeds when every step is correct and the model called `final_answer`.

**Why.** Real local tools make tool wait a real cost, with real file and CPU work, instead of a fixed sleep. Controlled families let the study vary schema size, argument count and chain depth one at a time, which the public benchmark cannot do. The document collection, notes and arithmetic need no network, so tool time does not depend on a remote service.

### 5.3 BFCL v4

**Source.** The Berkeley Function Calling Leaderboard evaluator and data come from the official repository, `ShishirPatil/gorilla`, folder `berkeley-function-call-leaderboard`, at commit `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8` (23 March 2026), under the Apache 2.0 license. The leaderboard web page and the Hugging Face copy of the data were blocked by the network policy of the build environment. The GitHub repository holds the same data.

**Vendoring.** 50 files are copied byte for byte into `third_party/bfcl_eval/` by `scripts/sync_bfcl.sh`, which refuses to run unless the clone is at the pinned commit. A SHA-256 manifest records every file. The only file that is not from upstream is `constants/model_config.py`. Upstream's version of that module imports every supported LLM SDK, but the checker only uses it to decide whether to replace dots in function names, so the stub always answers "no".

**Categories and sizes.**

| Category | Cases | What it tests |
|---|---|---|
| `simple_python` | 400 | One function, one call |
| `multiple` | 200 | Several functions, pick one |
| `parallel` | 200 | One function, several calls at once |
| `parallel_multiple` | 200 | Several functions, several calls |
| `irrelevance` | 240 | No function fits, so no call is correct |
| `live_irrelevance` | 884 | The same with user-contributed questions |
| `live_relevance` | 16 | A call is expected |
| `multi_turn_base` | 200 | Conversations with stateful tools |
| `multi_turn_miss_func` | 200 | A needed function is revealed only in a later turn |
| `multi_turn_miss_param` | 200 | A turn lacks a needed argument, so the model should not call |

**Subsampling.** `n` cases are taken per category. Cases are grouped into strata: the number of turns for multi-turn categories, and the number of functions (1, 2 to 3, 4 or more) for the others. The `n` cases are spread over the strata in proportion to their size by largest remainder, and drawn within each stratum with a seed built from the suite seed, the category and the stratum. The Pi and the Mac therefore run the same case ids.

**Prompt.** The same fixed format as the custom chains, with BFCL wording. The model must answer with a JSON array of calls, or an empty array when no function fits or a required argument is missing. Function documents are converted from BFCL's types to JSON Schema (`dict` to object, `float` to number, `tuple` to array, `any` to no type).

**Single turn.** One model call per case. Scoring uses the official `ast_checker` for the four AST categories. For relevance categories, BFCL's rule is used: irrelevance passes when no call can be read from the answer, and relevance passes when a non-empty call can be read.

**Multi-turn.** For each user turn, A2E calls the model until it answers without a call, up to 20 calls per turn as upstream does. Reaching 20 ends the case as force terminated. Calls run on BFCL's own Python backends (file system, math, messaging, posting, tickets, trading, travel, vehicle control). Scoring uses the official `multi_turn_checker`, which runs the model's calls and the ground truth calls on fresh copies of the backends and compares their state and their outputs after every turn.

On a `miss_func` reveal turn, upstream's prompting mode puts the held-out function documents into a user message together with the sentence "I have updated some more functions you can choose from. What about now?". A2E does the same, so the prompt only grows by appending and the prefix cache keeps working. As upstream allows, a model may call any method of the involved classes, including one that has not been revealed yet. A2E lists such calls in `unseen_calls`.

**Validation of the integration.** Before trusting any model score, the whole integration was checked with a replayed ground truth. For every in-scope case (1,000 single-turn AST cases, the relevance sets and 600 multi-turn cases), the ground truth answer was turned into model output, sent through the same code path, and scored by the official checkers. All of them pass. Every single-turn case was also scored with deliberately wrong answers (wrong function name, wrong value, missing call, extra call, empty answer), and all were rejected. In the first 60 cases of each multi-turn category, removing one ground truth call made at least 90% of the cases fail.

**Upstream data issues found.** The validation found four issues in the upstream data at the pinned commit:

- In `parallel_multiple_12` and `parallel_multiple_26`, the ground truth lists optional parameters (`permeability`, `type`) that the function documents do not declare. A call that includes them is rejected.
- In `multi_turn_miss_func_49`, the ground truth for turn 1 calls `tail`, which is held out until turn 3. Only a call to a function the model has not seen can pass that turn.
- `multi_turn_irrelevance_checker` is defined upstream but not called by upstream's own runner. A2E reports it as a diagnostic (`irrelevance_valid`) and uses the official `multi_turn_checker` for the score.
- Four of the 884 `live_irrelevance` cases list no functions.

**Rescoring.** For BFCL jobs, A2E stores each case's raw model output. `a2e bfcl rescore` replays the outputs through the harness and the checkers without the model.

**Why.** BFCL is the public reference for function calling, and the official checker makes the accuracy numbers mean what BFCL means. Vendoring unchanged files keeps the scoring identical to upstream while avoiding a dependency tree that would not install well on a Pi. Building the call strings for upstream's `eval()` from validated values keeps model text out of the Python interpreter.

### 5.4 Operator-level bottleneck analysis

A bench job stops llama-server and runs `llama-bench` on the model file, once per prefill size (`-p N -n 0`) and once per decode size (`-p 0 -n N`), with `reps` repetitions each. The shipped `kernels` suite uses prefill sizes 256, 1024 and 2048, which are the schema lengths, and decode sizes 128 and 512. A2E stores tokens per second with its standard deviation.

On the Pi, with `perf = true`, each test runs under `perf record -F 999`. `perf report` then lists the share of CPU samples per function, and A2E keeps the top 15. These functions are mostly ggml kernels, e.g., the matrix and dot product routines for each weight format.

Prefill processes many tokens in one batch and is mostly limited by compute. Decode produces one token at a time and reads every weight once per token, so it is mostly limited by memory bandwidth. A2E reports an estimated decode bandwidth, which is decode tokens per second times the model file size. Comparing this with the device's memory bandwidth shows how close decode is to the memory limit.

**Why.** The study design asks for operator-level breakdowns that identify the top kernels, and the analysis checklist asks to separate memory-bound and compute-bound behaviour. llama-server does not report per-operator times, so a separate benchmark tool is needed. llama-bench measures prefill and decode on their own without HTTP or agent overhead. Stopping the server first keeps two copies of the model out of the Pi's memory.

## 6. Study protocol

### 6.1 Steps

1. **Pilot** (`pilot`). Qwen3-4B Q4_0 on both devices, with KV cache f16 and q8_0. It runs every task kind and a slice of every BFCL category, plus one bench job with kernel profiling. The pilot decides the BFCL subsample sizes, the Pi context size, the cooldown target, whether to fit the Pi cooler and which CPU governor to use as the default. It also shows whether API-Bank, ToolBench and tau-bench are affordable.
2. **Baseline** (`baseline`). Every model at Q4_0 and Q8_0 on the standard prompt set. 1 warm-up run and 3 measured runs per job, with plain generation next to custom chains and BFCL single-turn.
3. **Factorial sweep** (`factorial`, `chains`, `bfcl`). Model × quantization × KV cache × schema length × decode length × grammar × device on custom chains, chain depth 2 to 5 and argument count 1, 3 and 5, and the BFCL categories in scope.
4. **Bottleneck analysis** (`kernels`, and the stage breakdown of every run). Stage shares per configuration, llama-bench prefill and decode, top kernels on the Pi.
5. **Stability** (`stability`). 40 depth-5 retrieval chains per run, 3 runs back to back with no cooldown, under the `ondemand` and `performance` governors on the Pi. This looks for thermal throttling and the point where decode speed drops.
6. **Prefix and KV cache over multi-turn sessions** (`cache_ablation`). Prefix cache on and off, KV cache f16 and q8_0, depth-5 chains and BFCL multi-turn.

### 6.2 Controls inside every job

- **Warm-up and repeats.** 1 warm-up run, recorded but not analysed, then 3 measured runs (`bfcl` uses 1 measured run because greedy answers do not change between repeats; `kernels` uses 1 because llama-bench repeats internally).
- **Cold prefix cache per repeat.** A2E erases the server's cache slot at the start of every run, so every repeat starts from the same state.
- **Idle baseline.** After each model load, A2E measures power for 30 s (5 s in `smoke`) with the model loaded and idle, and subtracts the mean from all energy numbers. Bench jobs load no server and do not use it.
- **Cooldown.** Most suites wait until the chip is at or below 55 °C before each job, for up to 900 s, so a job does not inherit heat from the previous one. `stability` turns this off on purpose.
- **Fixed seed.** Every random choice uses the suite seed (0 in all shipped suites).

## 7. Measurement and metrics

### 7.1 Timing

Each model call is timed with six stamps from `time.perf_counter`: before the chat template call, after it, after the model call returns, after parsing, after validation, and after the tools run. These give the template, parse, validate and tool times and the whole turn time. Prefill and decode times come from llama-server. Harness time is the rest of the turn. The time inside the HTTP call that llama-server does not report as prefill or decode is shown as "unattributed" in the stage breakdown.

### 7.2 Tokens and prefix cache

From each reply, A2E stores the processed prompt tokens, the cached prompt tokens (whose sum is the prompt size) and the generated tokens. The cache hit rate is cached tokens over prompt tokens.

### 7.3 Energy, memory, thermals and utilization

A background thread samples, 2 times per second by default:

- the resident memory of llama-server and its child processes
- the CPU use of the llama-server process
- on the Pi, the temperature, the ARM clock and the current throttling flags from `vcgencmd`
- on the Mac, the thermal pressure level from `powermetrics`
- the power reading

On the Pi, power is the sum of voltage times current over all power management chip rails from `vcgencmd pmic_read_adc`, skipping the `EXT5V` input rail. It leaves out power supply losses and USB devices. On the Mac, power is the "Combined Power (CPU + GPU + ANE)" from `powermetrics` at 500 ms intervals, which needs root. It leaves out memory, the display and the power supply. Every run stores this scope next to its numbers.

Energy for a window is the integral of the power trace over that window, with the trace treated as straight lines between samples and interpolated at the window's edges, minus the idle baseline. A2E computes it for the whole turn and for the prefill, decode and tool windows. The prefill and decode windows are placed using the durations llama-server reports, because the server does not report when each started.

### 7.4 Derived numbers and how they answer the questions

All summaries use measured runs of complete jobs. Each number is computed per run, then averaged over the measured runs, with the standard deviation across runs alongside. The formulas are in [DATA_DICTIONARY.md](DATA_DICTIONARY.md) section 7.

| Question or checklist item | Result table and columns |
|---|---|
| Latency and energy, tool calling vs plain (Q1) | `summary` by model and quant with `task_kind`: `prefill_ms`, `decode_ms`, `j_per_turn`, `j_per_gen_token` |
| Where the bottlenecks are (Q2) | `stages`: share of turn time in prefill, decode, parse, validate, tool wait, template, harness |
| Schema prefill share of turn time | `summary`: `prefill_share`, by `schema_len` |
| Constrained decoding cost | `summary` by `grammar`: `decode_ms`, `decode_tok_s`, and `success` |
| Schema length, KV cache, prefix cache over sessions (Q3) | `summary` by `schema_len`, `kv_quant`, `prefix_cache`: `cache_hit`, `prefill_ms`, `j_per_turn` |
| Differences between devices (Q4) | any table grouped by `device`, comparing patterns |
| Per token and per successful call | `j_per_prompt_token`, `j_per_gen_token`, `tokens_per_joule`, `j_per_success` |
| Memory-bound vs compute-bound | `bench`: prefill vs decode tokens per second, `est_bandwidth_gbs`; `kernels`: top kernels per phase |
| Quantization and tool selection accuracy | `summary` by `quant` and `kv_quant`: `success` for BFCL and custom chains |
| Thermal throttling and knee points | `stability`: start and end decode speed, `knee_task`, `max_temp_c`, `throttled_tasks`; `timeline` per run |
| Deployment recommendations | Drawn from all of the above in the final analysis |

### 7.5 Correctness

Custom chains are scored per step (section 5.2), and a task succeeds when every step is right and the model finished. BFCL uses the official checkers (section 5.3).

## 8. Prompting and output handling

### 8.1 Prompt format

All models get one fixed tool prompt. The system message lists the tools as compact JSON inside `<tools>` tags and asks for a JSON array of calls of the form `[{"name": ..., "arguments": {...}}]`. Custom chains end with `final_answer`. BFCL answers end with an empty array. Tool results go back as a user message inside `<tool_response>` tags, one JSON line per call. The role markup around the messages comes from each model's own chat template through llama-server's `/apply-template`. The prompt only grows by appending messages, so earlier text never changes and the prefix cache can reuse it.

The user chose this over two alternatives. One was rendering each model's native tool template. The other was llama-server's native tool calling on `/v1/chat/completions`. The fixed format keeps the prompt identical across models, which isolates the model as the variable. The cost is that accuracy may differ from each model's published numbers, which use the model's own format.

### 8.2 Reading tool calls

The parser removes thinking blocks first, and an unfinished thinking block removes everything after it. It then reads each `<tool_call>` block, or else the first JSON array or object in the text, skipping prose and code fences before it. It accepts `parameters` instead of `arguments` and arguments given as a JSON string. A broken JSON value, e.g., one cut off by the decode limit, is reported as `malformed_json`. The parser does not look inside it for a smaller value.

### 8.3 Grammar mode

With grammar on, A2E sends a JSON schema of valid answers as `json_schema` and llama-server turns it into a grammar. The schema is an array of calls, each of which is one of the tools, with the tool name fixed and the arguments matching that tool's parameter schema. Custom chains allow 1 to 4 calls per answer. BFCL allows 0 to 8, so the model can still decline. The tool schemas only use JSON Schema keywords that the converter is expected to support (`type`, `description`, `properties`, `required`, `enum`, `items`, `default`, `minimum`, `maximum`). Tests check this for every vendored BFCL function document.

The user chose llama-server's converter over a converter written for A2E, because it is less code and the schemas can be checked offline with a standard JSON Schema validator.

## 9. Data storage and results

### 9.1 Storage

All results go into one SQLite file per machine. The schema has `experiments`, `jobs`, `runs`, `turns`, `cases`, `samples`, `bench` and `kernels`, and three analysis views that keep only complete runs. The schema is versioned with SQLite's `user_version` and upgraded by appending migrations. The current version is 2. The database runs in WAL mode, so the TUI and the results commands can read while a run writes.

The user asked for a simple SQLite approach if it could work without much overhead. It does, for these reasons:

- Each device writes its own file from a single process.
- One run is one transaction.
- The expected size is tens of megabytes per device.
- Pandas reads it directly, and `a2e results export` writes parquet or csv.

A database server would only be needed if several machines wrote to one store at the same time, which this study does not do.

### 9.2 Results tools

`a2e results` lists experiments and jobs, prints per-configuration summaries, stage shares, stability drift, bench and kernel tables and per-run timelines, writes a markdown report per experiment, and exports every table and view. The TUI shows the same summaries.

## 10. Operation

- `a2e doctor` checks the machine before a run.
- `a2e plan <suite>` shows what a suite will run without running it.
- `a2e run <suite>` runs or resumes it. `scripts/run_suite.sh` runs the doctor, the plan and the run in one step.
- `a2e tui` offers the same through a terminal UI with tabs for suites, running, results, tests and the doctor.
- `a2e test <group>` runs a named test group from `configs/tests.toml`.

## Part B. Non-functional areas

## 11. Reproducibility and traceability

The study must be able to say exactly what produced each number, and running a suite again must run the same things.

- **Pinned inputs.** One llama.cpp commit for the whole study, passed to the setup scripts (`LLAMA_COMMIT`). The BFCL checker and data are pinned to one commit and checked by a manifest.
- **Deterministic jobs.** Suites expand to the same ordered job list with the same keys on every machine. The expansion uses only the committed model registry, never machine settings.
- **Suite lock.** `configs/suites.lock.json` stores each suite's hash, job count and a hash of its job keys. A test fails when a suite or the expansion code changes without the lock being updated.
- **Experiment identity.** An experiment is identified by suite name, device and suite hash. An edited suite starts a new experiment, so results of different suite versions never mix.
- **Environment snapshot.** Each experiment stores the hostname, OS, hardware model, RAM, Python version, the A2E commit and whether the working tree had uncommitted changes, the llama-server version, the CPU governor, the cooler setting and the available power tools.
- **Model fingerprint.** Each run stores the model file's size, modification time and the SHA-256 of its first MiB. A full hash of a multi-gigabyte file takes minutes on a Pi.
- **Fixed randomness.** Greedy decoding with seed 0, and every random choice in A2E derived from the suite seed.
- **Raw outputs.** BFCL raw model outputs are stored and can be rescored.

## 12. Measurement validity and known limitations

These limits apply to the numbers the system produces. Readers of the results need them.

**Not yet checked on real hardware.** These interfaces were written from knowledge of the tools and tested only against simulated versions:

- llama-server's `/completion` `timings` field names (`prompt_n`, `prompt_ms`, `cache_n`, `predicted_n`, `predicted_ms`)
- the `/apply-template`, `/tokenize` and `/slots/0?action=erase` endpoints
- the `json_schema` request field and the `ignore_eos` option
- the `--flash-attn on` argument form
- `vcgencmd pmic_read_adc` rail names and output format
- `powermetrics` output lines
- llama-bench JSON field names and its `-fa`, `-ctk` and `-ctv` options
- the line format of `perf report`

The hardware test group checks each of these on a real device, and each failure message names the assumption that broke. These tests must pass on both devices before a study run.

**Energy is a software estimate.** The Pi and Mac sources cover different parts of the system (section 7.3), so energy is compared within a device and as patterns across devices, not as absolute joules. Whether summing the Pi rails double-counts any rail is not yet confirmed. The Pi hardware test checks that the sum rises under CPU load.

**Stage windows are approximate.** Prefill and decode energy use windows placed from durations. The whole-turn energy is exact for its window. At 2 samples per second, a single short stage relies on interpolation between two samples. Averages over many turns are more reliable than single turns.

**Harness overhead is measured, not removed.** Template, parse, validation and harness time are recorded per turn, so their share is visible in the stage breakdown.

**Prompt format.** The fixed prompt makes models comparable with each other, but their accuracy is not comparable with BFCL leaderboard numbers, which use other prompt modes.

**Reasoning models.** Qwen3 can spend its whole decode budget on thinking text before it writes a call. Such answers count as no call. The pilot measures how often this happens at each decode length.

**Thermal state.** The cooldown target reduces carry-over heat between jobs, but room temperature and airflow are not controlled. The cooler setting and the actual governor are stored with every run.

**Background load.** A2E does not stop other processes on the machine. Run the study on an otherwise idle machine.

**Kernel profiles.** Pi only, and only when perf is permitted.

## 13. Reliability and data integrity

- **One transaction per run.** A run's turns, cases and samples are written together, after the run finishes. A crash never leaves half a run.
- **Runs count as a set.** A job is done only when all its runs are written. When a job starts again, or stops before finishing, all its earlier runs are marked aborted, including runs that had completed. The views ignore aborted runs.
- **Crash recovery.** On start, a job found in the running state goes back to pending and its runs are marked aborted.
- **Resume.** Running the same suite again continues the same experiment from the first unfinished job. `--limit` stops after a number of jobs. The TUI's stop button lets the current job finish and then stops. Ctrl+C stops at once, and the interrupted job runs again from its first repeat on resume.
- **Failure isolation.** A failing job, e.g., a missing model file, records its error and the suite continues. If llama-server died, the next job starts a new one. `--retry-failed` runs failed jobs again.
- **Concurrent reading.** WAL mode lets the TUI and the results commands read while a run writes.

## 14. Security

- **BFCL multi-turn.** Upstream's executor runs `eval()` on call strings. A2E builds those strings only from function names in the case's own whitelist and `repr()` of JSON values that have passed type checks. Non-finite numbers, non-string keys, unknown types, nesting deeper than 20 levels and call strings over 20,000 characters are refused. Tests run known injection payloads and check that nothing executes.
- **Calculator.** A parser that allows only numbers, arithmetic operators and parentheses. No names, calls or attributes.
- **Notes.** Note titles are turned into safe file names, so a title cannot write outside the notes folder.
- **No network tools.** No tool reaches the network.
- **Root.** Root is needed only for `powermetrics` on the Mac, for setting the CPU governor on the Pi, and for perf if the kernel setting does not allow it.

## 15. Performance and overhead of the harness

- Sampling runs on its own thread at 2 samples per second by default and reads `vcgencmd` or keeps the last `powermetrics` value.
- Writes happen once per run in one transaction, which keeps SD card writes low on the Pi.
- The harness's own time per turn is recorded, so its cost is part of the data, not hidden.

## 16. Portability

The system runs on Linux on the Pi and macOS on the Mac, with Python 3.11 or newer and uv. It needs no Docker. Device detection reads `/proc/device-tree/model` on Linux and checks for macOS on arm64. Optional parts install as extras: `bfcl` adds `mpmath` for the BFCL math backend, and `tui` adds Textual.

## 17. Maintainability and testing

The test suite has 617 tests in four groups:

| Group | Tests | What it covers |
|---|---|---|
| unit | 322 | Parsers, configuration, suite expansion, scoring, store, energy maths, BFCL conversion and safety, vendored file integrity |
| integration | 141 | Real processes against simulated llama-server, llama-bench and perf: server lifecycle, HTTP client, sampler, agent loop, BFCL runners, the runner end to end, results, CLI, TUI |
| sweep | 117 | BFCL ground truth through the official checkers for every in-scope case, every custom chain, the full suite grid |
| hardware | 37 | Real device and real llama-server. These skip themselves elsewhere. |

In the build environment, all non-hardware tests pass and the hardware tests are skipped. While the tests were being written they found real bugs, e.g., a calculator parameter name that broke every call and energy that was zero for stages shorter than the sampling interval. All of these were fixed before this report.

Other measures that keep the system maintainable:

- Schema migrations are appended, never edited.
- The vendored checker is regenerated by a script and checked by a manifest.
- Test groups are defined in one file.
- The simulated servers are in `tests/fakes/` and a complete machine setup for tests is in `tests/rig.py`.

## 18. Usability

- One command checks the machine (`a2e doctor`) and one shows a plan without running it (`a2e plan`).
- Errors in suites name the field and the reason.
- Job failures and skips store a readable reason.
- The TUI gives the same functions without remembering commands.
- The user guide, configuration reference and data dictionary document every command, field and column.

## 19. Licensing and attribution

The vendored BFCL files are Apache 2.0. Their license and a NOTICE with the source and commit are kept in `third_party/bfcl_eval/`. Every external source, tool and dataset is listed in `REFERENCES.md` with how it was checked.

## Part C. Decisions and open items

## 20. Decisions log

| Decision | Chosen | Other options considered | Reason | Made by |
|---|---|---|---|---|
| Devices | Pi 5 8 GB and Apple Silicon Mac, both measured | Adding an Android phone | Focus on Linux and macOS | User |
| Runtime | llama.cpp only, Metal on the Mac | MLC LLM, Ollama, MLX | One runtime keeps the comparison about hardware and models | User |
| Server interface | llama-server over HTTP | Python bindings, the CLI tools | Process isolation, server-side timings, same on both devices | User (recommended) |
| Energy | Software estimates with stated scope | USB power meter | No meter available. The power source interface allows adding one later. | User |
| Models | Qwen3 4B/8B, xLAM-2 3B/8B, Phi-4-mini, one Z.ai GLM | | Tool-calling models of the 4B to 8B class | User |
| Pi memory rule | 8B models at Q4_0 only on the Pi | Skip silently, record as out of memory | Q8_0 8B does not fit in 8 GB | User |
| Quantization formats | Q8_0 and Q4_0 | Q4_K_M, adding Q5 or Q6 | llama.cpp's basic 8-bit and 4-bit formats, chosen to stand for INT8 and INT4 | User |
| Benchmarks | BFCL v4 plus custom chains. Others after the pilot. | API-Bank, ToolBench, tau-bench now | Cost unknown before the pilot | User |
| Tool execution | Real local tools | Mock tools with set delays | Realistic tool wait | User |
| Constrained decoding | Grammar on vs off as an axis | Always on, native templates only | Isolates the cost and accuracy effect of the grammar | User |
| Prompt format | One fixed format for all models | Native templates, server tool calling | Same prompt for every model | User |
| Grammar implementation | `json_schema` converted by llama-server | A converter written for A2E | Less code, checkable offline | User (recommended) |
| BFCL source | Official GitHub repository, pinned | Leaderboard page, Hugging Face (both blocked) | Same data, reachable | User asked for the official source |
| BFCL scoring | Vendor the checker, stub one module | Score offline with the full package, reimplement | Official scoring with light dependencies | User |
| BFCL scope | Single-turn, multi-turn (base, miss_func, miss_param), relevance, fixed-seed subsample | Adding long_context, web search, memory | Covers the proposal's categories | User |
| Storage | One SQLite file per machine | A database server | Simple, enough for one writer per device | User asked for SQLite if simple |
| Test running | CLI, scripts and a TUI, with groups in a config file | | Same tests in every session | User |
| Operator analysis | llama-bench, plus perf on the Pi | Profiling inside llama-server | llama-server has no per-operator timing | Design |

## 21. Open items

1. **Run the hardware tests on both devices.** Section 12 lists what they check.
2. **Pin the llama.cpp commit** and record it in `REFERENCES.md`.
3. **Download the models**, fill in `source` for each in `configs/models.toml`, and add them to `REFERENCES.md`.
4. **Choose the Z.ai GLM model**, add it to the registry and the suites, and update the lock.
5. **Run the pilot**, then set the BFCL subsample sizes, the Pi context size, the cooldown target, the cooler and the default governor from its results.
6. **Decide on API-Bank, ToolBench and tau-bench** using the pilot's timings.
7. **Batching and parallel requests** are not measured. If needed, add a server setting for parallel slots and a task kind that sends requests at the same time.
8. **Allocator logs** are not parsed. llama-server's load-time memory report is kept in its log file. Peak memory comes from the process's resident set size.
9. **Confirm the citations** marked "from memory" in `REFERENCES.md`, e.g., the BFCL paper.
10. **Kernel profiles on the Mac** would need Xcode Instruments.
