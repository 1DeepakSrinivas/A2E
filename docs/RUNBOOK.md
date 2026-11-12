# A2E runbook

This runbook covers setting up a device, running suites in tmux, resuming and uploading, and fixing the problems seen in practice. The guide in [GUIDE.md](GUIDE.md) explains the settings, cases and tools.

The short path is three commands: `./setup.sh`, then `scripts/run.sh` to pick suites, then `scripts/run.sh attach <suite>` to watch. The sections below explain each step and what to do when one fails.

1. [Set up a device](#1-set-up-a-device)
2. [Run a suite in tmux](#2-run-a-suite-in-tmux)
3. [Resume, retry and upload](#3-resume-retry-and-upload)
4. [Build elsewhere and copy to a Pi](#4-build-elsewhere-and-copy-to-a-pi)
5. [Troubleshooting](#5-troubleshooting)

## 1. Set up a device

Run this from the project checkout as your normal user, on a Raspberry Pi 3B+ with 64-bit Raspberry Pi OS or on an Apple Silicon Mac.

```bash
./setup.sh
```

Setup installs the packages and Rust, builds llama.cpp at the commit pinned in `configs/models.toml`, builds `target/release/a2e`, downloads the three models and checks their hashes, generates the file sandbox, and runs the checks. Each step is a script that can be run alone, so a failed step can be repeated without the others. Pass `--skip-llama` or `--skip-models` to leave those steps out.

On the Pi, the llama.cpp build with one job takes a long time. A 1 GB device may need temporary swap to build. Turn swap off afterwards, because the Pi baseline refuses to run with swap on.

```bash
sudo dphys-swapfile swapoff
sudo systemctl disable dphys-swapfile
```

To send results to the cloud, copy `.env.example` to `.env` and set `WORKER_URL` and `WRITE_TOKEN`, or set `A2E_RESULTS_ENV` to the a2e-results `.env` file. Without them, results wait in `results/outbox`.

## 2. Run a suite in tmux

A full suite takes hours on the Pi, so run it in tmux. The session keeps running after you close the terminal or lose the SSH connection.

Run `scripts/run.sh` with no arguments for a menu. It lists the suites in `configs/suites`, lets you pick one or several by number (or `a` for all), asks whether to upload to the Worker and whether to retry failed attempts, and then starts the session. Several suites run one after another in a session named `a2e-queue`. The same menu attaches to, shows the log of, or stops a run. The commands below do the same without the menu.

```bash
scripts/run.sh start smoke        # start a detached session named a2e-smoke
scripts/run.sh attach smoke       # watch it; detach with Ctrl-b d
scripts/run.sh status smoke       # session state and the upload queue
scripts/run.sh logs smoke         # follow the newest log in results/logs
scripts/run.sh stop smoke         # send Ctrl-C; the session closes when the runner exits
```

`start` checks that the sandbox verifies and that the suite plans, and then starts the run. Only one run can hold the runtime lock, so it refuses to start while another `a2e-` session exists. Extra arguments go to `a2e run`, for example `scripts/run.sh start smoke --retry-failed`. Give several suites as `start smoke,files_dev`; they run in order in the session `a2e-queue`.

The repository `.env` normally holds the real Worker credentials, so a normal start uploads results to the live Worker. Add `--local` to keep a test run off the Worker. Results then stay in `results/outbox`.

## 3. Resume, retry and upload

- Running the same command again skips attempts that are already recorded.
- `--retry-failed` repeats unsuccessful attempts, including attempts that failed because of the setup.
- `a2e upload --status` shows what is waiting, and `a2e upload` sends it. A request the Worker refuses for good moves to `results/outbox/rejected`.
- Records from a failed setup, for example a wrong llama.cpp build, stay in the outbox as results. Check `a2e upload --status` before you upload them.

## 4. Build elsewhere and copy to a Pi

Use this route only when the Pi cannot build, for example because it runs out of memory. Build on a Mac with Docker for `linux/arm64`, using an image such as `rust:1-bookworm`. Build the `a2e` binary and then `llama-server` from the pinned commit, with the same CMake options as `scripts/build_llama.sh`.

- A source copy without `.git` has no commit in its version string. Make the build report the first 8 characters of the pin, then check `llama-server --version` on the Pi.
- Copy `a2e`, the two llama.cpp binaries, `configs`, `.env` and the models, and do not copy `results`. Generate the sandbox on the Pi with `./a2e sandbox init`, or copy it with `scripts/sync_sandbox.sh <user@host>`.

## 5. Troubleshooting

**`llama_commit_mismatch` on every attempt.** The runner needs the first 8 characters of the pinned commit in the output of `llama-server --version`. A binary built from a git clone with the default 7 character hash, or from a copy without `.git` (where the hash is `unknown`), fails this check. `scripts/build_llama.sh` sets `core.abbrev 8` and checks the version after the build. Copy the right binary to the machine that runs it, make it executable, run `llama-server --version`, and then use `--retry-failed`.

**Pi baseline refuses to run.** The message says swap must be disabled. Turn swap off as in section 1.

**`Undervoltage detected!` in `dmesg`, or `vcgencmd get_throttled` is not `0x0`.** The power supply is too weak, and the CPU is throttled, so the timings are not usable. Use a supply rated for the Pi 3B+ and a short cable. A USB drive on the same supply adds load.

**exFAT drive.** exFAT has no execute bit, so run `chmod +x` on the binaries after copying. A Mac adds `._*` files, and a case file with that name breaks parsing with a UTF-8 error. Remove them with `find . -name '._*' -delete` or `dot_clean`. The device name can change between mounts (for example `sda1` and `sdb1`), so run `lsblk -f` before mounting. Mount with `sudo mount -o uid=$(id -u),gid=$(id -g) /dev/sdb1 /mnt/usb`.

**The sandbox is missing.** `run` and `plan` only check the sandbox, and they stop with an error when it is absent. Run `a2e sandbox init` and `a2e sandbox verify`. Stop any run first, because the files change under it.

**Another run holds the lock.** Look for an `a2e-` tmux session with `tmux ls`. Attach to it, or stop it with `scripts/run.sh stop <suite>`.
