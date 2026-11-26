# Settings

Every value that changes how a study runs is a setting. This page lists them all. The defaults are the values the harness used before settings existed, so a run with no changes behaves as it always did.

## Where settings come from

Settings are TOML. They are read in layers, and a later layer wins over an earlier one. Each key is merged on its own, so a layer only needs the keys it changes. Lists replace the earlier list. The one exception is `[[models]]`: a later file adds its models to the earlier ones, and an entry with the same `id` replaces the earlier entry. That is how `a2e models add` can write to `configs/local.toml` without hiding the models in `configs/models.toml`.

1. Built-in defaults.
2. `configs/models.toml`. It holds the model list and the pinned llama.cpp commit.
3. `configs/settings.toml`. It holds every default and is the file most people edit.
4. `configs/hosts/<user>@<host>.toml`. Use it for a device, such as the Pi or a Mac. `A2E_HOST_ID` names the host.
5. `configs/local.toml`. It is for this checkout only, and Git ignores it. `a2e config init` creates it.
6. A model entry. A `[models.settings.llama]` or `[models.settings.sampling]` table inside a `[[models]]` entry changes those two sections for that model only.
7. A suite file. A suite may set `contexts`, `warmup`, `repeats`, `seed` and a `[budget]` table.
8. Environment variables named `A2E_<SECTION>_<KEY>`. The section `limits` and the key `memory_mb` give `A2E_LIMITS_MEMORY_MB`. Lists are written as TOML, for example `[1024, 2048]`. Older names still work: `A2E_MODELS_DIR`, `A2E_OUTPUT_DIR`, `A2E_LLAMA_SERVER`, `A2E_SANDBOX` and `A2E_SANDBOX_TMP`.
9. The `--set section.key=value` option, which you can repeat. For example, `a2e run core --set limits.output_tokens=512`.

Layers 8 and 9 also win over a suite file. A key that no layer sets keeps its default. A key that does not exist is an error, so a typo cannot pass silently.

Three commands help:

- `a2e config show` prints the final settings.
- `a2e config show --source` also prints which layer set each key.
- `a2e config check` loads every layer and reports the first error.

Two variables belong to Hugging Face and not to the harness: `HF_TOKEN` for gated repositories, and `HF_ENDPOINT` for a mirror.

Secrets are not settings. `WORKER_URL`, `WRITE_TOKEN` and `BRAVE_SEARCH_API_KEY` stay in `.env`.

## Settings and the study id

The final settings are part of the study definition, and the study id is a hash of that definition. A changed setting gives a new study id, so results made under different settings are never mixed. Settings that change speed or memory, such as `flash_attn`, `cache_type_k`, `cache_type_v`, `mlock`, `mmap` and `threads`, change what a result means. Label those runs clearly.

## limits

Budgets for a study. A suite may override the ones that match its own budget keys.

| Key | Default | Meaning |
|---|---|---|
| `memory_mb` | 768 | Memory limit for the harness and the server together. A run that goes over it is stopped. |
| `reserve_mb` | 128 | Free memory that must remain. |
| `context` | `[1024]` | Context sizes to test. Each must be 256 or more. A suite calls this `contexts`. |
| `input_tokens` | 0 | Longest prompt in tokens. 0 means only the context size limits it. |
| `output_tokens` | 256 | Tokens one model call may generate. A suite calls this `generated_tokens`. |
| `model_calls` | 8 | Model calls allowed for one task, from 1 to 128. |
| `call_seconds` | 300 | Time limit for one model call. |
| `task_seconds` | 900 | Time limit for one task. |
| `tool_seconds` | 20 | Time limit for one tool call. |
| `tool_output_bytes` | 8192 | Largest tool output kept, from 256 to 65536. |
| `stall_attempts` | 3 | Attempts in a row with no finished model call before the server is restarted once. If it stalls that many times again, the rest of the model and context group is skipped. 0 never skips. |

## sampling

| Key | Default | Meaning |
|---|---|---|
| `temperature` | 0.0 | Sampling temperature. 0 picks the most likely token. |
| `top_p` | 1.0 | Nucleus sampling. 1.0 is off. |
| `top_k` | 0 | Keep only this many tokens. 0 is off. |
| `min_p` | 0.0 | Drop tokens below this share of the best token. 0 is off. |
| `seed` | 0 | Seed sent with every call. |
| `repeats` | 1 | Measured attempts for each case. |
| `warmup` | 0 | Attempts for each case that are run and not counted. |

## llama

One key for each llama-server flag the harness sets. The pinned commit is the one value that belongs in `configs/models.toml`.

| Key | Default | Flag | Meaning |
|---|---|---|---|
| `commit` | none | none | The full 40 character llama.cpp commit. The server must report its first 8 characters. |
| `server` | `deps/llama.cpp/build-a2e/bin/llama-server` | none | Path to the server program. |
| `host` | `127.0.0.1` | `--host` | Address the server listens on. |
| `port` | 8197 | `--port` | Port the server listens on. |
| `threads` | 4 | `--threads` | CPU threads. |
| `gpu_layers` | 0 | `--n-gpu-layers` | Layers placed on the GPU. |
| `parallel` | 1 | `--parallel` | Request slots. The harness uses slot 0 only. |
| `batch_size` | 128 | `--batch-size` | Logical batch size. |
| `ubatch_size` | 64 | `--ubatch-size` | Physical batch size. |
| `cache_ram` | 0 | `--cache-ram` | MiB of host memory for the prompt cache. The server default grows memory between tasks. |
| `jinja` | true | `--jinja` | Use the model's chat template engine. |
| `special` | true | `--special` | Show special tokens in output. |
| `context_shift` | false | `--no-context-shift` when false | Let the server drop old tokens when the context fills. |
| `flash_attn` | `auto` | `--flash-attn` | `auto` leaves the server default. `on` and `off` pass the flag. |
| `cache_type_k` | `f16` | `--cache-type-k` | Key cache type. Passed only when it is not `f16`. |
| `cache_type_v` | `f16` | `--cache-type-v` | Value cache type. Passed only when it is not `f16`. |
| `mlock` | false | `--mlock` | Keep the model in memory. |
| `mmap` | true | `--no-mmap` when false | Map the model file instead of reading it. |
| `chat_template` | empty | `--chat-template` | Template name. Empty uses the template inside the model file. |
| `extra_args` | `[]` | as written | Flags added at the end of the command, for anything not listed above. |
| `startup_seconds` | 300 | none | Time allowed for the server to load the model. |

A flag that is at the server default is not passed, so the command line for the defaults is the same as it was before these keys existed.

## host

| Key | Default | Meaning |
|---|---|---|
| `device` | `auto` | Device name recorded with results. `auto` detects it. |
| `power` | `auto` | `auto` measures power when it can, `none` skips it, and `required` fails the run without a reading. |
| `sample_ms` | 500 | Milliseconds between memory, thermal and power samples. At least 100. |
| `thermal_path` | `/sys/class/thermal/thermal_zone0/temp` | File that holds the temperature on Linux. |
| `models_dir` | `models` | Folder with the model files. |
| `output_dir` | `results` | Folder for results and logs. |
| `sandbox_dir` | `sandbox` | Folder with the file sandbox. |
| `sandbox_tmp` | empty | Folder for task workspaces. Empty uses a `workspace` folder in each run folder. |

## advanced

Rare limits. Change them only when a run hits one.

| Key | Default | Meaning |
|---|---|---|
| `inference_timeout_s` | 300 | Time limit for one request to the server. |
| `settle_timeout_s` | 60 | Longest wait for a cancelled call to leave its slot before the server restarts. |
| `slot_erase_timeout_s` | 10 | Time limit for clearing the prompt cache slot. |
| `upload_timeout_s` | 30 | Time limit for one upload request. |
| `upload_body_bytes` | 2000000 | Largest upload body. |
| `max_samples` | 20000 | Most resource samples kept for one measurement. |
| `web_timeout_s` | 12 | Time limit for a live web fetch. |
| `dns_timeout_s` | 3 | Time limit for a name lookup. |
| `download_timeout_s` | 1800 | Time limit for one model download. |
| `download_max_bytes` | 8589934592 | Largest model download. |

## Examples

Use a longer output budget for one run:

```bash
a2e run core --set limits.output_tokens=512
```

Give the Pi two threads and a smaller memory limit in `configs/hosts/<user>@<host>.toml`:

```toml
[llama]
threads = 2

[limits]
memory_mb = 700
```

Pass an extra flag to the server for one model:

```toml
[[models]]
id = "qwen3-0.6b"
# other keys as before

[models.settings.llama]
extra_args = ["--verbose-prompt"]
```
