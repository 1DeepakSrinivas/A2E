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

**3. Start a run.** Runs take hours on the Pi, so they go in tmux, which keeps them going if you close the terminal or lose the SSH link. Run the script with no arguments to pick suites from a menu.

```bash
scripts/run.sh
```

Or start one suite directly. The `smoke` suite is the quickest, so try it first. Add `--local` to keep the results on the device instead of sending them to the cloud.

```bash
scripts/run.sh start smoke --local
```

**4. Watch it and read the results.**

```bash
scripts/run.sh attach smoke    # watch it live; detach with Ctrl-b then d
scripts/run.sh status smoke    # is it running, and what is waiting to upload
scripts/run.sh stop smoke      # stop it; run the same start command later to resume
```

Results are saved under `results`. Judging, scoring and reports live in the A2E-Results project ([github.com/1DeepakSrinivas/A2E-Results](https://github.com/1DeepakSrinivas/A2E-Results)), which is the submodule `a2e-results`. To send results there, copy `.env.example` to `.env` and fill in the Worker settings. See the runbook for the details.

## Suites

| Suite | What it runs | Rough size |
|---|---|---|
| `smoke` | A few quick cases, once. Use it to check the setup. | Short |
| `files_dev` | The file cases, dev split, once. | Short |
| `files_heldout` | The file cases, held-out split, 1 warmup and 3 repeats. | Long |
| `core` | All held-out cases in autonomous and guided mode, 1 warmup and 3 repeats. | Hours |
| `harness` | Autonomous mode against guided, filter, recent context, constrained decoding and repair. | Hours |
| `test_filesystem` | Dev-split sandbox and file cases (search, create, move, delete) in autonomous and guided mode, plus filter, recent context, constrained decoding and repair. | Hours |
| `test_compute` | Dev-split arithmetic, units, time and text cases in autonomous and guided mode. | Short |
| `test_data_web` | Dev-split database, key-value, device and fixture web cases in autonomous and guided mode. | Short |
| `web_live` | Live web search cases. Needs `BRAVE_SEARCH_API_KEY` in `.env`. | Short |

## Which document to read

| I want to | Read |
|---|---|
| Set up a device, run in tmux, resume, upload, or fix a problem | [docs/RUNBOOK.md](docs/RUNBOOK.md) |
| Change settings, write cases or tools, or understand the commands | [docs/GUIDE.md](docs/GUIDE.md) |
| Know how the study measures things | [docs/REPORT.md](docs/REPORT.md) |
| See the research question and hypotheses | [docs/PREREGISTRATION.md](docs/PREREGISTRATION.md) |
| Check a source, tool or dataset | [REFERENCES.md](REFERENCES.md) |
