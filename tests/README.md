# Tests

The tests are run by group. The groups are defined in `configs/tests.toml`, so a group name means the same tests in every session.

```bash
uv run a2e test --list
uv run a2e test quick          # unit + integration, about a minute
uv run a2e test sweep          # large sweeps, a few minutes
uv run a2e test all
scripts/run_tests.sh runner -x # extra arguments go to pytest
```

Every test is marked with the folder it is in.

| Folder (marker) | Needs | What it covers |
|---|---|---|
| `unit` | nothing | configuration and suite expansion, store and migrations, settings and doctor, energy integration, the vcgencmd, powermetrics and PMIC parsers, the tool-call parser, grammar schemas, schema padding, tools and their safety, scoring and task generation, BFCL conversion, call-string safety, subsampling and vendored-file integrity, llama-bench and perf parsing |
| `integration` | nothing | real processes against the simulated llama-server (`tests/fakes/fake_llama_server.py`), llama-bench and perf: server lifecycle, HTTP client, sampler, agent loop, BFCL runners and rescoring, the suite runner end to end (resume, crash recovery, failures, skips, stop, energy), results and reports, the CLI and the TUI |
| `sweep` | `uv sync --extra bfcl` | every BFCL ground truth answer through the official checker and every corruption of it, every custom chain, the full suite grid, the baseline protocol |
| `hardware` | a real device | real llama-server, llama-bench, perf, vcgencmd and powermetrics. Skips itself unless set up (below). |

Helpers:

- `tests/rig.py` builds a whole machine setup in a temporary folder. It has fake binaries, dummy model files, settings, a small suite and constant fake power.
- `tests/stubs.py` is an in-process client with scripted model output.
- `tests/oracle.py` and `tests/bfcl_oracle.py` turn expected answers into model output. `oracle_outputs_any` works for every custom family.
- `tests/fakes/fake_brave.py` is a local stand-in for the Brave Search API and the pages its results point to.

The full evaluation suite is checked by `tests/unit/test_full_eval_suite.py` (50 tasks per configuration, coverage of every category and horizon) and `tests/integration/test_full_eval_runner.py` (one model and quant run end to end, then the report script).

The harness statistics are tested in `tests/unit/test_harness_effect.py` (equivalence verdicts against known effects, decomposition identity, noise floor, variance shares) and the figures in `tests/unit/test_harness_plots.py` (needs `uv sync --extra plots`).

The tests for harness variants are `tests/unit/test_harness_variants.py` (prompt formats, context views, budgets, stop rules, suite expansion and job-key stability), `tests/unit/test_store_v3.py` (the migration), `tests/integration/test_harness_loop.py`, `tests/integration/test_harness_bfcl.py` and `tests/integration/test_harness_runner.py` (the loop, BFCL runners and a whole runner pass under each variant).

The tests for the BFCL tool families and web search are `tests/unit/test_agent_tools.py`, `tests/sweep/test_agent_tools_oracle.py` (every family, every depth, many seeds) and `tests/integration/test_agent_tools_runner.py`. `tests/hardware/test_real_web.py` checks the real Brave API when `BRAVE_SEARCH_API_KEY` is set.

## Running the hardware group

```bash
export A2E_LLAMA_SERVER=~/a2e-deps/llama.cpp/build/bin/llama-server
export A2E_GGUF=~/models/qwen3-4b-instruct-2507/qwen3-4b-instruct-2507-q5_k_m.gguf
export A2E_DEVICE=pi5        # or mac (detected when unset)
export A2E_CTX=16384         # multi-turn BFCL tests need this
uv run a2e test hardware -v
sudo -E env "PATH=$PATH" uv run a2e test hardware -k powermetrics   # the Mac power test needs root
```

The simulated servers were written from knowledge of the real tools. The hardware group is where those assumptions are checked, e.g., the field names in llama-server's `timings` block, the `/slots` erase endpoint, `pmic_read_adc` rail names, `powermetrics` output, llama-bench JSON fields and the perf report format. A failing hardware test names the assumption that broke.
