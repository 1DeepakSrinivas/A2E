# A2E

A2E measures how small language models use tools under time, memory, and call budgets. It is a Rust program that targets Raspberry Pi 3B+ and Apple Silicon Mac and uses llama.cpp for inference.

## Quick start

On the Pi 3B+ with 64-bit Raspberry Pi OS, run this from the project checkout.

```bash
./setup-pi.sh
```

The script builds the pinned llama.cpp checkout under `deps/llama.cpp`, builds the Rust CLI, and downloads and verifies the three GGUF files under `models`. After setup, disable swap for the Pi baseline and run `target/release/a2e run`.

See the [guide](docs/GUIDE.md) for Mac setup, writing cases and tools, and sending results to the cloud. Judging, scoring and reports live in the A2E-Results project ([github.com/1DeepakSrinivas/A2E-Results](https://github.com/1DeepakSrinivas/A2E-Results)), which is the submodule `a2e-results`.

## Documents

- [docs/GUIDE.md](docs/GUIDE.md) explains how to install, configure, run, resume, write cases and tools, and send results to the cloud.
- [docs/REPORT.md](docs/REPORT.md) is the methods report.
- [docs/PREREGISTRATION.md](docs/PREREGISTRATION.md) states the research question and hypotheses.
- [REFERENCES.md](REFERENCES.md) lists every source, tool and dataset, with how each was checked.
