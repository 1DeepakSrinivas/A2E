# A2E

A2E measures what it costs to run small language models as tool-calling agents on a Raspberry Pi 5 and an Apple Silicon Mac. It runs models with llama.cpp and records, for every model call, the time spent reading the prompt (prefill), writing the answer (decode), parsing and checking tool calls, and running tools. It also records tokens, prefix cache use, energy, memory, temperature and throttling, and whether the model called the right tools. Accuracy on public data comes from the official Berkeley Function Calling Leaderboard (BFCL v4) checker.

## Quick start

```bash
LLAMA_COMMIT=<sha> scripts/setup_pi.sh          # or scripts/setup_mac.sh on the Mac
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

The system is built and its tests pass against simulated servers. It has not yet been run on real hardware. Run `uv run a2e test hardware` on each device first (see the user guide, section 9).
