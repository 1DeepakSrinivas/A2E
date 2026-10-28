# A2E test suites

```bash
scripts/run_tests.sh quick        # unit + integration (~30 s), run this constantly
scripts/run_tests.sh sweep        # full grid + baseline-protocol mini sweeps
scripts/run_tests.sh hardware     # real device / real llama-server (self-skips without opt-in)
scripts/run_tests.sh all
scripts/run_tests.sh unit -k energy -x     # extra args go to pytest
```

| Suite | Needs | Covers |
|---|---|---|
| `unit` | nothing | config validation and Pi memory rule, energy integration (edges, ramps, short windows), `vcgencmd`/`powermetrics`/PMIC parsers, `timings` parsing, llama-server arg building, store |
| `integration` | nothing (uses `tests/fakes/fake_llama_server.py`) | process lifecycle (slow health, early exit, timeout, restart), HTTP client (prefix cache, grammar overhead, 500s, bad JSON, missing timings, timeouts, refused connection), sampler (RSS incl. children, dying process, flaky power), stage energy end to end, Parquet export |
| `sweep` | nothing | all 480 grid cells (uniqueness, axis coverage, feasibility, grammar on/off pairing, server args for every feasible cell); 1 warm-up + 3 measured protocol; mini sweep over schema length x decode length x grammar x prefix cache |
| `hardware` | a real device | Pi: real `vcgencmd` readings and PMIC power tracking CPU load. Mac (sudo): real `powermetrics`. Either: real llama-server boot, timing field names, prefix cache, `/slots` erase, grammar output, KV-cache quant, prefill vs decode speed |

## Running the hardware suite

```bash
export A2E_LLAMA_SERVER=$HOME/a2e-deps/llama.cpp/build/bin/llama-server
export A2E_GGUF=/path/to/model-q4_0.gguf
export A2E_DEVICE=pi5            # or mac (inferred from the OS when unset)
scripts/run_tests.sh hardware -v
sudo -E env "PATH=$PATH" scripts/run_tests.sh hardware -k powermetrics   # Mac power test needs root
```

If a hardware test fails, the assertion message names the assumption that broke. The likely ones: llama-server's
`timings` field names, the `/slots/0?action=erase` endpoint, the `--flash-attn on` flag form (set `A2E_FLASH_ATTN=1`
or `auto` for older builds), `pmic_read_adc` rail names, and `powermetrics` output format.

The fake server's behaviour (cache semantics, timings shape) was written from memory of llama.cpp, so the
`hardware` suite is the check that those assumptions hold.
