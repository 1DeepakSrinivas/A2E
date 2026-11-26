# A2E on the Raspberry Pi: first run and common problems

This page explains how to get A2E running on a Raspberry Pi 3B+ with 1 GB of memory, and how to fix the problems we have already hit. Nothing here needs a build on the Pi, because the Pi is too small to build llama.cpp in a reasonable time.

## What you need

- A Raspberry Pi 3B+ running a 64 bit operating system. Run `uname -m` on the Pi. It must print `aarch64`. If it prints `armv7l`, the binaries will not run.
- A bundle folder with these parts.
  - `a2e`, the runner, built for aarch64 Linux.
  - `deps/llama.cpp/build-a2e/bin/llama-server` and `llama-bench`.
  - `configs/`, copied from the same commit of `main` that the binary was built from.
  - `models/`, with the three `.gguf` files.
  - `.env`, with `WORKER_URL`, `WRITE_TOKEN` and `BRAVE_SEARCH_API_KEY`. The Pi does not need `READ_TOKEN`.
- Do not copy the `results` folder or the `sandbox` folder. The Pi makes its own sandbox.

## Build the binaries on the Mac

Build inside a Debian bookworm container for linux/arm64. On an Apple Silicon Mac this container runs at full speed, so no cross compiler is needed.

- Build `a2e` with `cargo build --release --locked`.
- Check that `deps/llama.cpp` is at the commit pinned in `configs/models.toml`.
- Configure llama.cpp with `-DGGML_NATIVE=OFF -DCMAKE_C_FLAGS=-mcpu=cortex-a53 -DCMAKE_CXX_FLAGS=-mcpu=cortex-a53 -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_TESTS=OFF`.
- Build only the targets `llama-server` and `llama-bench`.

The flag `GGML_NATIVE=OFF` is needed because a native build would use the instructions of the Mac processor, and the program could crash on the Pi.

## Copy the files to the Pi

You can use a USB drive or rsync over SSH. Both give the same result.

```
rsync -av --exclude '._*' --exclude results --exclude sandbox A2E-pi/ pi@raspberrypi:~/A2E/
```

The `--exclude '._*'` part is explained under the problem about macOS sidecar files below.

## First run on the Pi

Run these commands from `~/A2E`, in this order.

```
chmod +x a2e deps/llama.cpp/build-a2e/bin/*
./deps/llama.cpp/build-a2e/bin/llama-server --version
./a2e --version
./a2e sandbox init
./a2e sandbox verify
./a2e --local run smoke
```

- The `llama-server --version` output must contain the first 8 characters of the pinned commit, which is `6666a733` at the time of writing.
- The `--local` flag keeps the results on the Pi. Use it for every test run so nothing is sent to the live Worker.
- If `smoke` passes, you can start the real suites.

## Common problems

### error: unknown field `llama`, expected one of `runtime`, `roles`, `models`

**Cause.** An old `a2e` binary is reading the new config files. The new layout puts a `[llama]` section in `configs/models.toml`, and the old binary does not know that section.

**How we found it.** The `a2e` in `~/A2E` was the new build, but `scripts/run.sh` runs `target/release/a2e`. That file was an older build from a previous day, so the script never used the new binary.

**Fix.** Put the new binary where the script looks for it.

```
cd ~/A2E
mv target/release/a2e target/release/a2e.old
cp a2e target/release/a2e
./target/release/a2e --version
```

To check that two copies are the same build, compare `md5sum` on the Pi with `md5` on the Mac. The file size should also match.

### llama_commit_mismatch

**Cause.** The runner refuses to start when `llama-server --version` does not contain the pinned commit from `configs/models.toml`.

**Fix.** Rebuild llama.cpp at the pinned commit and copy the new `llama-server` and `llama-bench` over the old ones. Do not change the pin just to make the check pass.

### Exec format error, or Illegal instruction

**Cause.** An Exec format error means the binary was built for the wrong kind of processor, e.g., a Mac binary was copied by mistake. An Illegal instruction error means llama.cpp was built with `GGML_NATIVE=ON`, so it used instructions the Pi does not have.

**Fix.** Check `uname -m` on the Pi, then rebuild in the arm64 container with the flags listed above.

### Permission denied when you start a binary

**Cause.** Files copied from an exFAT USB drive lose their executable bit.

**Fix.** Run `chmod +x a2e deps/llama.cpp/build-a2e/bin/*`.

### Files that start with ._ appear next to everything

**Cause.** macOS writes a small hidden sidecar file next to every file it copies to an exFAT drive. Inside `configs/cases/` these look like real case files, and the runner may try to read them.

**Fix.** Delete them on the Pi with `find ~/A2E -name '._*' -delete`. When you copy from the Mac, add `--exclude '._*'` to rsync, and check the drive again after any later copy from the Mac.

### Results went to the live Worker

**Cause.** The `.env` file holds real Worker credentials, and a normal run uploads to production.

**Fix.** Add `--local` to every test run. With `scripts/run.sh`, answer "No" to the question about sending results to the Worker, and the results stay in `results/outbox`. Only answer "Yes" when you want the data in the live results.

### The runner cannot find configs, models or the sandbox

**Cause.** The runner looks for `configs/`, `models/`, `sandbox/`, `results/` and `.env` in the folder you start it from.

**Fix.** Always `cd ~/A2E` before you run `./a2e`.

### Another a2e session is running

**Cause.** Only one run can hold the runtime lock at a time, and `scripts/run.sh` enforces this.

**Fix.** Use the Attach option to see the running session, or the Stop option to end it, then start the new run.

### Out of memory, or the Pi becomes very slow

**Cause.** The Pi has 1 GB of memory. The memory limit is set to 1024 MiB and swap is kept off on purpose, so a run that goes over the limit fails instead of slowing down and giving unfair timings.

**Fix.** Close other programs on the Pi, and do not build anything on it while a run is going.

### Old files on the USB drive

**Cause.** The USB drive may still hold an older bundle with outdated binaries and configs.

**Fix.** Copy from the newest bundle only, and delete the old one once you no longer need it.

## Quick checklist before every run

- You are in `~/A2E`.
- `./a2e --version` works, and `target/release/a2e` is the same build if you use `scripts/run.sh`.
- `llama-server --version` shows the pinned commit.
- `configs/` comes from the same commit as the binary.
- No `._*` files exist.
- You used `--local`, unless you want to upload on purpose.
