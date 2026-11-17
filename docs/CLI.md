# A2E commands

This page lists every command of the `a2e` program. Run `a2e --help` or `a2e <command> --help` to see the same information with the exact flags of your build. The setup steps are in the [README](../README.md), and the settings are in the [guide](GUIDE.md).

Commands that print data write JSON to stdout. Progress, warnings and prompts go to stderr or to the terminal, so you can pipe the output of a command into another tool. Three options work with every command. `--config <path>` names the model registry, and the default is `configs/models.toml`. `--local` keeps results on the device and sends nothing to the Worker, the same as setting `A2E_LOCAL=1`. `--set section.key=value` overrides one setting for that command, and you can repeat it. `docs/CONFIG.md` lists the settings.

## Menu

Run `a2e` with no command on a terminal to open a menu. The menu offers to start a run, check what a suite would do, show the status of a run, attach to a run, follow a log, stop a run, run `doctor`, and send waiting results. Starting a run asks for the suites, models, variants, whether to keep results on the device, and whether to retry failed attempts. It then shows how many attempts the choice makes and asks before it starts. Esc goes back one step, and Ctrl-C leaves the menu. The menu remembers your last choices in `results/.last-run.toml`. Without a terminal, `a2e` prints the help and exits with code 2, so scripts never wait for input.

## Run

`a2e run [suite]` runs one suite in the current terminal. A bare name such as `smoke` means `configs/suites/smoke.toml`, and a path to a suite file also works. Running the same command again skips attempts that are already recorded.

- `--model`, `--context` and `--variant` take comma separated values and run only that part of the suite. An unknown name stops the command and suggests the closest known name.
- `--retry-failed` runs again the attempts that did not succeed.
- `--allow-missing-model` lets a run continue when a model file is missing. Without it the run stops before it starts.
- `--dry-run` prints the runs and checks that the model files and llama-server exist, and then stops without starting a model.
- `--tools <file>` names the tool definitions, and the default is `configs/tools/standard.toml`.

```bash
a2e run smoke --dry-run
a2e run core --model qwen3-0.6b --variant guided --local
```

On a terminal, a progress bar on stderr shows the attempt count, the model, the elapsed time and an estimate of the time left. It redraws once a second. When stderr is piped or `NO_COLOR` is set, the bar is off and each finished attempt prints one line.

`a2e plan [suite]` lists the runs a suite would make without starting a model. It takes `--tools`, `--model`, `--context` and `--variant`.

## Sessions in tmux

These commands run suites in a detached tmux session, so a run continues after you close the terminal. Only one session can run at a time.

- `a2e start <suites>` starts one suite in the session `a2e-<suite>`, or several comma separated suites one after another in `a2e-queue`. It takes the same options as `a2e run`, except `--dry-run`. The output is also written to a log in `results/logs`.
- `a2e attach <name>` shows the session. Press Ctrl-b and then d to detach.
- `a2e status <name>` says whether the session is running and what waits to upload.
- `a2e logs <name>` follows the newest log of that session.
- `a2e stop <name>` sends Ctrl-C to the session. Start it again later to resume.

```bash
a2e start core,harness --local
a2e stop queue
```

## Check and set up

- `a2e doctor` shows the host, the machine, the memory headroom of each model and the settings. With `--probe` it also loads each model once and checks its output format.
- `a2e models fetch` downloads the pinned models and checks their hashes.
- `a2e cases validate [path]` checks every case against the tool definitions and rubrics, and `a2e cases list [path] [--ids]` counts cases by category and split.
- `a2e config show [--source]` prints the final settings, and with `--source` also the layer that set each key. `a2e config check` loads every layer and reports the first error. `a2e config init` creates `configs/local.toml` from the example.
- `a2e tools validate [file]` checks that every tool definition has a handler and a valid schema.
- `a2e sandbox init [--seed N] [--out dir] [--force]` builds the file sandbox, `a2e sandbox verify [--dir dir]` checks it against its manifest, and `a2e sandbox info [--dir dir]` prints the manifest.
- `a2e upload [--status]` sends waiting results to the Worker, or shows what is waiting.
- `a2e completions <shell>` prints completions for bash, zsh, fish, elvish or PowerShell.

To install completions for zsh, write the output to a file in a folder on your `fpath`, for example `a2e completions zsh > ~/.zfunc/_a2e`. For bash, write it to `~/.local/share/bash-completion/completions/a2e`.

## Environment variables

| Variable | What it does |
|---|---|
| `A2E_LOCAL` | Keeps results on the device when set to `1` |
| `A2E_HOST_ID` | Picks the host file `configs/hosts/<id>.toml` instead of the detected user and host |
| `A2E_<SECTION>_<KEY>` | Overrides one setting, for example `A2E_LIMITS_MEMORY_MB`. See `docs/CONFIG.md` |
| `A2E_MODELS_DIR` | Overrides `host.models_dir` |
| `A2E_LLAMA_SERVER` | Overrides `llama.server` |
| `A2E_OUTPUT_DIR` | Overrides `host.output_dir` |
| `A2E_SANDBOX` | Folder of the file sandbox; overrides `host.sandbox_dir`, which is `sandbox` by default |
| `A2E_RESULTS_ENV` | Path of an `.env` file that holds `WORKER_URL` and `WRITE_TOKEN` |
| `WORKER_URL`, `WRITE_TOKEN` | Where results are sent and the token that allows it |
| `BRAVE_SEARCH_API_KEY` | Key for the live search suite |
| `NO_COLOR` | Turns the progress bar off |

## Exit codes

A command exits with 0 when it succeeds and 1 when it fails, and it prints `error:` and a reason on stderr. `a2e` with no command and no terminal exits with 2. A run stopped with Ctrl-C exits after it has written what it finished, and a queue exits with 130 in that case.
