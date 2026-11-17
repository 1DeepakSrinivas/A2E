# A2E runbook

This runbook covers setting up a device, running suites in tmux, resuming and uploading, and fixing the problems seen in practice. The guide in [GUIDE.md](GUIDE.md) explains the settings, cases and tools.

The short path is three commands: `./setup.sh`, then `target/release/a2e` to pick suites from a menu, then `target/release/a2e attach <suite>` to watch. The sections below explain each step and what to do when one fails.

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

Setup installs the packages and Rust, builds llama.cpp at the commit pinned in `configs/models.toml`, builds `target/release/a2e`, downloads the three models and checks their hashes, generates the file sandbox, and runs the checks. Every step can be repeated, so a failed step can be fixed and setup run again. Pass `--skip-llama` or `--skip-models` to leave those steps out. The Rust version comes from `rust-version` in `Cargo.toml`.

On the Pi, the llama.cpp build with one job takes a long time. A 1 GB device may need temporary swap to build. Turn swap off afterwards, because the Pi baseline refuses to run with swap on.

```bash
sudo dphys-swapfile swapoff
sudo systemctl disable dphys-swapfile
```

The 1.2B model only fits on the 1 GB Pi when almost nothing else holds memory. Its 4-bit weights are about 700 MiB and are mapped from the file, so they must stay in the page cache. Boot to the console and give the GPU the minimum, then reboot.

```bash
sudo systemctl set-default multi-user.target
```

```bash
echo 'gpu_mem=16' | sudo tee -a /boot/firmware/config.txt
```

After the reboot, `a2e doctor` shows `memory.headroom_mib` for each model: the available memory minus the weights and `reserve_mb`. That headroom must still hold the context cache and buffers, which is about 60 MiB for `lfm2.5-1.2b-instruct-qad` and about 280 MiB for `qwen3-0.6b` at context 2048. A model with less headroom than that will page from storage and time out.

To send results to the cloud, copy `.env.example` to `.env` and set `WORKER_URL` and `WRITE_TOKEN`, or set `A2E_RESULTS_ENV` to the a2e-results `.env` file. Without them, results wait in `results/outbox`.

## 2. Run a suite in tmux

A full suite takes hours on the Pi, so run it in tmux. The session keeps running after you close the terminal or lose the SSH connection.

Run `a2e` with no arguments on a terminal for a menu. It lists the suites in `configs/suites`, then asks for the models and variants, whether to keep results on the device, and whether to retry failed attempts. It then shows how many attempts the choice makes and asks before it starts the session. Esc goes back one step, and the menu remembers your last choices in `results/.last-run.toml`. Several suites run one after another in a session named `a2e-queue`. The same menu attaches to, shows the log of, or stops a run. The commands below do the same without the menu.

```bash
a2e start smoke        # start a detached session named a2e-smoke
a2e attach smoke       # watch it; detach with Ctrl-b d
a2e status smoke       # session state and the upload queue
a2e logs smoke         # follow the newest log in results/logs
a2e stop smoke         # send Ctrl-C; the session closes when the runner exits
```

`start` checks that the sandbox verifies and that each suite plans, and then starts the run. Only one run can hold the runtime lock, so it refuses to start while another `a2e-` session exists. The options of `a2e run` work here too, for example `a2e start smoke --retry-failed` or `a2e start core --model qwen3-0.6b --variant guided`. Give several suites as `start smoke,files_dev`, and they run in order in the session `a2e-queue`. To see the runs and check the files first, use `a2e run <suite> --dry-run`.

The repository `.env` normally holds the real Worker credentials, so a normal start uploads results to the live Worker. Add `--local` to keep a test run off the Worker. Results then stay in `results/outbox`.

## 3. Resume, retry and upload

- Running the same command again skips attempts that are already recorded.
- `--retry-failed` repeats unsuccessful attempts, including attempts that failed because of the setup.
- `a2e --local run <suite>` keeps the results on the device and sends nothing to the Worker, the same as `A2E_LOCAL=1`. Use it for trial runs, because the repository `.env` normally points at the production Worker. Every run prints where its results go before it starts.
- `a2e upload --status` shows what is waiting, and `a2e upload` sends it. A request the Worker refuses for good moves to `results/outbox/rejected`.
- Records from a failed setup, for example a wrong llama.cpp build, stay in the outbox as results. Check `a2e upload --status` before you upload them.

## 4. Build elsewhere and copy to a Pi

Use this route only when the Pi cannot build, for example because it runs out of memory. Build on a Mac with Docker for `linux/arm64`, using an image such as `rust:1-bookworm`. Build the `a2e` binary and then `llama-server` from the pinned commit, with the same CMake options as the `build_llama` function in `setup.sh`.

- A source copy without `.git` has no commit in its version string. Make the build report the first 8 characters of the pin, then check `llama-server --version` on the Pi.
- Copy `a2e`, the two llama.cpp binaries, `configs`, `.env` and the models, and do not copy `results`. Generate the sandbox on the Pi with `./a2e sandbox init`. The same seed gives the same tree, and `./a2e sandbox verify` checks it against its manifest.

## 5. Troubleshooting

**`llama_commit_mismatch` on every attempt.** The runner needs the first 8 characters of the pinned commit in the output of `llama-server --version`. A binary built from a git clone with the default 7 character hash, or from a copy without `.git` (where the hash is `unknown`), fails this check. `setup.sh` sets `core.abbrev 8` and checks the version after the build. Copy the right binary to the machine that runs it, make it executable, run `llama-server --version`, and then use `--retry-failed`.

**A run stops with "model file ... is missing".** The model is not in the models folder. Run `a2e models fetch`, or pass `--allow-missing-model` when you want the run to continue and record that model's attempts as failures.

**Pi baseline refuses to run.** The message says swap must be disabled. Turn swap off as in section 1.

**`Undervoltage detected!` in `dmesg`, or `vcgencmd get_throttled` is not `0x0`.** The power supply is too weak, and the CPU is throttled, so the timings are not usable. Use a supply rated for the Pi 3B+ and a short cable. A USB drive on the same supply adds load.

`a2e doctor` lists this under `warnings`, and `a2e run` prints the same warning before it starts. The flags `0x50005` mean under-voltage and throttling both now and since boot. Reboot after fixing the supply, because the "since boot" bits only clear then.

**Most attempts end in `operation timed out`, with many major page faults.** The model does not fit in memory. llama.cpp maps the weights from the file, so when the weights, the context cache and the rest of the system need more than the physical memory, the kernel drops and re-reads weight pages on every token. On the 1 GB Pi 3B+ this happens with `lfm2.5-1.2b-instruct-qad` (weights larger than the free memory) and is marginal with `qwen3-0.6b` at context 2048. The runner warns before loading when the weights plus `reserve_mb` exceed the available memory. Close other programs, boot to the console instead of the desktop, or leave the model out of the suite on that device.

**`model_stalled: skipped after 3 attempts with no completed model call`.** Three attempts in a row for one model and context ended before any model call returned, so the runner skipped the rest of that group instead of waiting out every timeout. The skipped attempts are recorded as excluded, not as model failures. Fix the cause, then rerun with `--retry-failed`. Set `stall_attempts = 0` under `[budget]` in the suite to turn this off.

**`cache_reset_failed`.** The server defers a cache erase while its slot is busy, which happened after a call timed out. The runner now waits up to 60 seconds for the slot and restarts the server when it stays busy, so this should only appear when the restart itself fails.

**`memory_limit_mb` is larger than the device memory.** The memory guard compares the two processes with this limit, so a limit above the physical memory can never trip. `a2e doctor` warns about it. The default is 768.

**exFAT drive.** exFAT has no execute bit, so run `chmod +x` on the binaries after copying. A Mac adds `._*` files, and a case file with that name breaks parsing with a UTF-8 error. Remove them with `find . -name '._*' -delete` or `dot_clean`. The device name can change between mounts (for example `sda1` and `sdb1`), so run `lsblk -f` before mounting. Mount with `sudo mount -o uid=$(id -u),gid=$(id -g) /dev/sdb1 /mnt/usb`.

**The sandbox is missing.** `run` and `plan` only check the sandbox, and they stop with an error when it is absent. Run `a2e sandbox init` and `a2e sandbox verify`. Stop any run first, because the files change under it.

**Another run holds the lock.** Look for an `a2e-` tmux session with `tmux ls`. Attach to it, or stop it with `a2e stop <suite>`.
