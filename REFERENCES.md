# References

Everything this project relies on or cites: models, tools, system interfaces and papers. Add an entry when something is first used.

Status key:
- Verified: fetched or inspected, and the detail below was checked then.
- From memory: stated from model knowledge and not checked. Confirm it before citing.
- From summary: found by search and only the summary was read. Read the full source before citing.
- Pending: planned but not yet chosen or used.

---

## 1. Models

The models and exact files are set in `configs/models.toml`. Each entry records the source repository, the revision and the SHA-256 of the GGUF file. `scripts/download_models.sh` downloads and verifies them.

| Model | Quant | Source repository |
|---|---|---|
| FunctionGemma 270M | Q4_K_M | https://huggingface.co/unsloth/functiongemma-270m-it-GGUF |
| Qwen3-0.6B | Q4_K_M | https://huggingface.co/unsloth/Qwen3-0.6B-GGUF |
| LFM2.5-1.2B-Instruct | QAD_Q4_0 | https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF |

Status: Verified for file names and hashes by the download script. License and tool-call format of each model are pending verification. The tool-call format of each model is handled in `src/adapter.rs`.

---

## 2. Runtime and system interfaces

### llama.cpp and llama-server
- **Repo:** https://github.com/ggml-org/llama.cpp, pinned by `llama_commit` in `configs/models.toml` and built by `scripts/build_llama.sh`.
- **Endpoints used:** `/health`, `/completion`, `/slots`, `/apply-template`.
- **Flags used:** `--jinja`, `--special`, `--no-context-shift`, `--slots`, `--slot-save-path`, `--ctx-size`, `--threads`, `--n-gpu-layers`, `--parallel`, `--batch-size`, `--ubatch-size`.
- **Status:** From memory. The assumptions are exercised against a real build by the smoke suite.
- **Where used:** `src/inference.rs`.

### Joular power model for the Pi 3B+
- **Source:** https://github.com/joular/powermodels, file `Raspberry Pi/rbp3b+1.3/rbp3b+1.3.json`, read 2026-11-10. The data is under the Open Database License (ODbL) 1.0, so keep this attribution.
- **Method:** the model comes from Noureddine and Kanso, "Automated Power Modeling of Computing Devices: Implementation and Use Case for Raspberry Pis". The title is from a search summary and is not checked against the paper.
- **Use:** a degree 9 polynomial of CPU usage (0 to 1) with an intercept of 2.484 W. It gives 2.48 W at idle and 5.27 W at full load.
- **Status:** Verified for the coefficients, which were copied from the file. The model was fitted on a 32-bit OS, A2E uses a 64-bit OS, and the error on this unit is unknown.
- **Where used:** `src/measure.rs`. Values are labeled `modeled` in `power_scope`.

### Raspberry Pi thermal data
- **Sources:** `/sys/class/thermal/thermal_zone0/temp` and `vcgencmd get_throttled`.
- **Status:** From memory. Output format unverified on hardware.
- **Where used:** `src/measure.rs`.

### Apple powermetrics
- **Source:** macOS `man powermetrics`. It needs root, which `scripts/setup_mac_telemetry.sh` arranges.
- **Command used:** `powermetrics --samplers cpu_power,gpu_power -i 500`, reading the CPU, GPU and ANE power lines.
- **Status:** From memory. Output format unverified on other macOS versions.
- **Caveat:** the reading covers CPU, GPU and ANE and excludes DRAM, the display and power-supply losses.
- **Where used:** `src/measure.rs`.

### Brave Search API
- **Endpoint used:** `GET https://api.search.brave.com/res/v1/web/search` with the header `X-Subscription-Token`, read from `BRAVE_SEARCH_API_KEY`.
- **Status:** From memory. Results change over time.
- **Where used:** `src/tools/web.rs` and the `web_live` suite.

### SQLite
- **Use:** an in-memory database per attempt for the database tools, seeded from `configs/fixtures`, and the local run store. The crate bundles SQLite through `rusqlite`.
- **Status:** Verified by the tests.
- **Where used:** `src/tools/db.rs` and `src/store/mod.rs`.

### JSON Schema
- **Use:** validation of tool arguments with the `jsonschema` crate.
- **Where used:** `src/config.rs` and `src/runner.rs`.

---

## 3. Software dependencies

Rust 1.96 or newer. The crates and their version ranges are in `Cargo.toml`, and exact versions are in `Cargo.lock`. Status: Verified (built and tested in CI).

---

## 4. Related work for the success-per-joule study

All entries below were found by web search on 2026-10-29 and only their summaries were read. Status: **From
summary**. Read each paper in full before citing it.

- **TinyLLM** (arXiv 2511.22138): small models on agentic tasks on edge devices. Closest prior work.
  Check whether it reports energy.
- **SWEnergy** (ICSE 2026 AGENT workshop): energy of agent frameworks with small models on SWE-bench Verified Mini.
- **Token Arena** (arXiv 2605.00300): joules per correct answer for inference endpoints.
- **Towards Sustainable Web Agents** (arXiv 2502.17903): argues for energy metrics for agents.
- **Where Does the Energy Go? Profiling LLM Agent Inference on Blackwell GPUs** (arXiv 2609.29707): context growth
  and tool idle time on GPUs.
- **Agentic AI Workload Characteristics** (arXiv 2605.26297) and **TokenPowerBench** (arXiv 2512.03024).
- **AI Agents That Matter** (arXiv 2407.01502, https://agents.cs.princeton.edu/): cost-controlled agent
  evaluation. **What Moves the Pareto Frontier in Tool-Using Agents?** (ACL SRW 2026): token-budget Pareto study.
- **Sustainable LLM Inference for Edge AI** (arXiv 2504.03360): accuracy against energy per response on edge devices.
- **Flat Score, Amplified Failures** (arXiv 2607.27275) and **Which Decisions Low-Bit Quantization Breaks**
  (arXiv 2608.06564): quantization damage to agents and tool calling.
- **Constrained Decoding Eliminates Structural Failures in Small LLMs** (arXiv 2609.23742), **Alignment Tax of
  Constrained Decoding** (arXiv 2604.06066), **JSONSchemaBench** (arXiv 2501.10868): accuracy effects of
  constrained decoding.
- **GreenBench** (arXiv 2608.28667): notes that `powermetrics` excludes DRAM.
- **RPi5-power** (https://github.com/jfikar/RPi5-power): community Pi 5 PMIC calibration against a USB-C meter.

### Harness effects (searched 2026-10-29; status From summary)
- **Stop Comparing LLM Agents Without Disclosing the Harness** (arXiv 2605.23950, ICML 2026): harness choice moves
  scores; proposes a variance-decomposition protocol.
- **AgencyBench** (arXiv 2601.11044): a model gains 20.5% in its native harness.
- **SWE-Serve** (arXiv 2609.26777) and **From Question Answering to Task Completion** (survey, arXiv 2606.20683):
  model-specific harnesses and richer scaffolds did not always help.
- **The Scaffold Effect in Coding Agents** (arXiv 2607.22585): up to 40x tokens per solved task with small pass-rate
  differences. **BankerToolBench** (arXiv 2604.11304): modest accuracy effect, large runtime effect.
- **SWEnergy** (arXiv 2512.09543): framework design drove up to 9.4x energy with small models.
- **The Complexity Trap** (arXiv 2508.21433): observation masking as good as LLM summarization. **Less Context,
  Better Agents** (arXiv 2606.10209): pruning plus a running summary cut tokens about 60%.
- **Small Reasoning Models are Instruction Followers in Function Calling** (arXiv 2608.22472): rigid schemas and
  small-model function calling; contested.
---

## 5. Measurement caveats to cite alongside results

- Mac power is a software estimate from `powermetrics`. Pi power is modeled from CPU usage and is not measured.
- The Pi and Mac use different scopes, so any cross-device comparison must say so.
- All models get tool instructions in their own call format through the adapters, so accuracy can differ from each model's published numbers.

---

## 6. Workspace and file-system tool design

Found by web search on 2026-11-10. Only search summaries were read. Status: **From summary**. Read each source in
full before citing it.

- **SWE-agent** (arXiv 2405.15793, https://princeton-nlp.github.io/SWE-agent/background/aci): the agent-computer
  interface. A file viewer window of about 100 lines worked best, search results are summarised, and the agent gets
  clear feedback on bad commands. The "withhold results above 50 matches" detail comes from one secondary source.
- **Terminal-Bench** (arXiv 2601.11868): each task is a prepared container with an instruction and tests that pass only
  if the final state is right.
- **xAFS** (https://huggingface.co/datasets/supermemory/xAFS): agentic retrieval over file systems from 5 to about
  10,000 files. **CoDA-Bench** (secondary report only): over 1,000 data files with distractors similar to the target.
