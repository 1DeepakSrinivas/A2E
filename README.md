# A2E

A2E measures how small language models use tools under time, memory, and call budgets. The native Rust runtime targets Raspberry Pi 3B+ and Apple Silicon Mac and uses llama.cpp for inference. It is the supported runtime. The Python runtime, which ran BFCL v4 on Raspberry Pi 5 and Apple Silicon Mac, is retired and its files are frozen.

## Native quick start

On the Pi 3B+ with 64-bit Raspberry Pi OS, run this from the project checkout.

```bash
./setup-pi.sh
```

The script builds the pinned llama.cpp checkout under `deps/llama.cpp`,
builds the Rust CLI, and downloads and verifies the three GGUF files under
`models/native`. After setup, disable swap for the Pi baseline and run
`target/release/a2e-native run`.

See the [native guide](docs/NATIVE.md) for Mac setup, writing cases and tools, and
sending results to the cloud. Judging, scoring and reports live in the A2E-Results
project, which is the submodule `third_party/a2e-results`.

## Python and BFCL quick start (retired)

These commands still work on the frozen Python files, but they are no longer maintained.

```bash
scripts/setup_pi.sh                            # or scripts/setup_mac.sh on the Mac
uv sync --extra bfcl --extra tui
cp configs/local.example.toml configs/local.toml   # then set llama_server and models_dir
uv run a2e doctor                                 # check the machine
uv run a2e plan smoke                             # see what a suite runs
uv run a2e run smoke                              # run it (Ctrl+C to stop, run again to resume)
uv run a2e results summary                        # read the results
uv run a2e tui                                    # or do all of this from the terminal UI
uv run a2e test quick                             # run the tests
```

On a Mac, run with `sudo -E` to record energy.

## Documents

- [docs/USER_GUIDE.md](docs/USER_GUIDE.md) explains how to install, configure, run, stop and resume, and read results.
- [docs/CONFIG_REFERENCE.md](docs/CONFIG_REFERENCE.md) lists every field of every configuration file.
- [docs/DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md) lists every table and column of the results database and how each number is computed.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) describes the code and how data moves through it.
- [docs/REPORT.md](docs/REPORT.md) is the methods report. It explains what is measured, how it is measured, and why each choice was made.
- [REFERENCES.md](REFERENCES.md) lists every source, tool and dataset, with how each was checked.
- [tests/README.md](tests/README.md) describes the tests.

## Status

The native smoke suite has run against all three models on an Apple Silicon Mac. Run the Pi smoke suite after installation to check its memory and latency limits. The Python path also has a separate `uv run a2e test hardware` group.
