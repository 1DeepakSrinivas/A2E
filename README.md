# A2E

A2E measures how small language models use tools under time, memory, and call budgets. It is a Rust program that targets the Raspberry Pi 3B+ and Apple Silicon Macs and uses llama.cpp for inference. It tests three models: FunctionGemma 270M, Qwen3-0.6B and LFM2.5-1.2B.

## Get started

You need a Raspberry Pi 3B+ with 64-bit Raspberry Pi OS, or an Apple Silicon Mac, and a normal user account (not root). Follow these four steps in the project checkout.

**1. Set up the machine.** This installs everything, builds the program, downloads the models and checks that all of it works.

```bash
./setup.sh
```

On the Pi this takes a long time, because llama.cpp builds with one job. If it stops part way, fix the problem and run `./setup.sh` again. Steps that are done are quick to repeat.

**2. Turn swap off (Pi only).** The Pi baseline refuses to run with swap on.

```bash
sudo dphys-swapfile swapoff
sudo systemctl disable dphys-swapfile
```

**3. Check what a run would do.** The `smoke` suite is the quickest, so try it first. The dry run lists the runs and reports any model file or llama-server that is missing.

```bash
target/release/a2e run smoke --dry-run
```

**4. Start a run.** Runs take hours on the Pi, so they go in tmux, which keeps them going if you close the terminal or lose the SSH link. Run `a2e` with no arguments on a terminal to pick the suite, models and variants from a menu. Press Esc to go back one step.

```bash
target/release/a2e
```

Or start a suite directly. Add `--local` to keep the results on the device instead of sending them to the cloud, and add `--model`, `--context` or `--variant` to run only part of a suite.

```bash
target/release/a2e start smoke --local
target/release/a2e start core --model qwen3-0.6b --variant guided --local
```

Then watch the run and read its state.

```bash
target/release/a2e attach smoke    # watch it live; detach with Ctrl-b then d
target/release/a2e status smoke    # is it running, and what is waiting to upload
target/release/a2e stop smoke      # stop it; run the same start command later to resume
```

Results are saved under `results`. Judging, scoring and reports live in the A2E-Results project ([github.com/1DeepakSrinivas/A2E-Results](https://github.com/1DeepakSrinivas/A2E-Results)), which is the submodule `a2e-results`. To send results there, copy `.env.example` to `.env` and fill in the Worker settings. See the runbook for the details. Every command and flag is listed in [docs/CLI.md](docs/CLI.md).

## Switch hosts, models and suites

- A host is a file named `configs/hosts/<user>@<hostname>.toml`. The program loads the one that matches the current user and host, or the one named by `A2E_HOST_ID`. It prints a warning when none matches.
- A model is an entry in `configs/models.toml`. A suite with no `models` line runs every model in that file, and `--model` narrows the run to some of them.
- A variant is a block in `configs/variants.toml`. A suite lists the ones it wants in `use_variants`, and `--variant` narrows the run.
- Settings that only apply to one machine go in `configs/local.toml`, which Git ignores.

## Suites

| Suite | What it runs | Rough size |
|---|---|---|
| `smoke` | A few quick cases, once. Use it to check the setup. | Short |
| `files_dev` | The file cases, dev split, once. | Short |
| `files_heldout` | The file cases, held-out split, 1 warmup and 3 repeats. | Long |
| `core` | All held-out cases in autonomous and guided mode, 1 warmup and 3 repeats. | Hours |
| `harness` | Autonomous mode against guided, filter, recent context, constrained decoding and repair. | Hours |
| `test_filesystem` | Dev-split sandbox and file cases (search, create, move, delete) in autonomous and guided mode, plus filter, recent context, constrained decoding and repair. About 1,800 runs, so too long for the Pi; use the two `test_pi` suites there. | Hours |
| `test_pi_100` | 16 dev-split filesystem cases in autonomous and guided mode, one repeat, at most 96 runs. Sized for the Pi. | Short |
| `test_pi_ablation` | 6 of the `test_pi_100` cases in autonomous mode plus filter, recent context, constrained decoding and repair, so every variant covers the same cases. At most 90 runs. Sized for the Pi. | Short |
| `test_pi_guided` | 10 of the `test_pi_100` cases in autonomous, guided and guided_stable mode, to separate what guidance gains from what its prompt cache loss costs. 90 runs. Sized for the Pi. | Short |
| `test_compute` | Dev-split arithmetic, units, time and text cases in autonomous and guided mode. | Short |
| `test_data_web` | Dev-split database, key-value, device and fixture web cases in autonomous and guided mode. | Short |
| `web_live` | Live web search cases. Needs `BRAVE_SEARCH_API_KEY` in `.env`. | Short |

## Which document to read

| I want to | Read |
|---|---|
| Set up a device, run in tmux, resume, upload, or fix a problem | [docs/RUNBOOK.md](docs/RUNBOOK.md) |
| Change settings, add a host, model, suite, tool, check or model format | [docs/GUIDE.md](docs/GUIDE.md) |
| Look up a command or flag | [docs/CLI.md](docs/CLI.md) |
| Know how the study measures things | [docs/REPORT.md](docs/REPORT.md) |
| See the research question and hypotheses | [docs/PREREGISTRATION.md](docs/PREREGISTRATION.md) |
| Check a source, tool or dataset | [REFERENCES.md](REFERENCES.md) |
