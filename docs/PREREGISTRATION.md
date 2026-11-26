# Pre-registration: what a successful tool call costs on small devices

Status: draft, written before the full sweep is run. Commit this file, then run the sweep, and record any later change as a dated amendment at the bottom. Do not edit the hypotheses after seeing the data.

## Question

Three small models (FunctionGemma 270M, Qwen3-0.6B and LFM2.5-1.2B-Instruct) run on a Raspberry Pi 3B+ and on an Apple Silicon Mac. Within a fixed budget of calls, tokens and time, which model completes the most tool-calling tasks, and what does each success cost in time, memory and, on the Mac, energy?

The study also asks whether the harness changes the answer. The harness is the code around the model that builds the prompt, reads the reply and runs the tools.

The main claim being tested is that the most accurate configuration is often not the cheapest per success. A failed attempt still costs time and energy, so a model that is a little less accurate but much faster per attempt can win.

## Why this is worth asking

The searches behind this plan (see `REFERENCES.md`, section 4) found per-token and per-response energy on edge devices, agent energy profiling on data-center GPUs, and cost-aware agent evaluation in dollars or tokens. They did not find a cost-per-success result for multi-turn tool-calling agents on devices with 1 GB of memory. Treat that as a claim to re-check before submission, because the searches only read summaries.

## Measures

The runner records each attempt. The A2E-Results project computes the measures from the held-out attempts, with warm-up repeats excluded.

| Measure | Definition |
|---|---|
| `success_rate` | Attempts that passed, divided by attempts scored. |
| `seconds_per_success` | Wall time of all attempts, failures included, divided by the number that passed. Empty when none passed. |
| `waste_share` | Share of wall time that went into attempts that failed. |
| `peak_rss` | Highest resident memory of the inference process plus the harness, from the samples. |
| `cpu_seconds_per_success` | Both devices. CPU time of the inference and harness processes over all attempts, failures included, divided by the number that passed. |
| `joules_per_success` | Energy of all attempts, failures included, divided by the number that passed. Measured on the Mac and modeled on the Pi, and the two are never pooled. |
| `parse_failure_share` | Share of failed attempts whose failure class is `parse_failure`. |

On the Mac, energy is the software estimate from `powermetrics`, which covers CPU, GPU and ANE and leaves out memory, the display and the power supply. The Pi 3B+ has no power sensor and the study uses no external meter. On the Pi, energy is modeled by putting whole-system CPU usage into the published Joular `rbp3b+1.3` polynomial, and it is labeled `modeled`. The model was fitted by its authors on another unit with a 32-bit OS, so its error on this unit is unknown. Pi energy is reported as an estimate with that statement, and Pi cost comparisons rest on `cpu_seconds_per_success` and `seconds_per_success`, which are measured.

## Factors

- Device: Raspberry Pi 3B+ (1 GB) and Apple Silicon Mac.
- Settings: the harness reads its settings from layered files (see `docs/CONFIG.md`). Their defaults equal the values used before the settings files existed, so a default run is unchanged. The final settings are now part of the study definition, so the study ids differ from studies run before this change even when the effective values are the same.
- Model: the three registered models, each at the quant in `configs/models.toml`.
- Variant: autonomous and guided in the `core` suite, and autonomous, guided, filter, recent, constrained and repair in the `harness` suite.
- Category: the eight case categories, held-out split.

## Hypotheses

Each can be shown wrong by the data.

- **H1, the harness changes the cost of a success.** For at least one variant of the `harness` suite, `seconds_per_success` differs from the autonomous variant by more than the margin set below. Refuted if every variant passes the equivalence test.
- **H2, guided mode helps the smallest model most.** Guided mode raises `success_rate` on the `chain` category more for FunctionGemma 270M than for LFM2.5-1.2B. Refuted if the gain is equal or larger for LFM2.5-1.2B.
- **H3, constrained decoding removes parse failures.** The constrained variant has a lower `parse_failure_share` than the autonomous variant for every model. Refuted if any model does not improve.
- **H4, the harness is a small share of time.** Time outside the model, the tools and parsing is under 5% of wall time on both devices. Refuted if either device is at 5% or more.
- **H5, time and energy rank the models alike.** On the Mac, `joules_per_success` ranks the three models in the same order as `seconds_per_success`. Refuted if the order differs.
- **H6, CPU time stands in for energy on the Pi.** On the Pi, `cpu_seconds_per_success` ranks the three models in the same order as modeled `joules_per_success`. Refuted if the order differs. The model uses CPU usage alone, so agreement shows only that the two views are consistent, not that the energy is right.

## Decision rules for the harness

1. For each variant against the autonomous baseline, take the ratio of `seconds_per_success` over the same cases, as the geometric mean over cells. A cell is one device, model and category, so every cell counts the same. The 90% interval comes from resampling cases 2,000 times.
2. The margin is 10%, so the bounds are 1 / 1.10 and 1.10.
   - A variant is equivalent when its whole interval is inside the bounds.
   - A variant differs when its whole interval is outside the bounds, or when it never succeeds where the baseline does.
   - Otherwise it is inconclusive.
3. The study concludes that the model is the limit only if every variant is equivalent. It concludes that the harness matters if any variant differs. Anything else is inconclusive, and the response is more repeats, not a verdict.
4. The baseline runs with 3 repeats, so its own run-to-run spread is known. The margin must be at least twice the median standard deviation of log `seconds_per_success` across baseline cells. If it is not, report both the 10% verdict and the verdict with the margin set to twice the noise.

## Analysis plan

1. Configurations with no energy readings are dropped from every energy result and listed. Measured Mac energy and modeled Pi energy are never shown on one axis without the label.
2. Success rate against `seconds_per_success` is drawn per device, with the spread over repeats as error bars. A configuration is on the front only if no other has lower or equal time and higher or equal success, with one strict.
3. Variant effects are paired. They compare the same model, device and case, with only the variant changed.
4. Greedy decoding gives the same answer on each repeat, so repeats are used for time, memory and temperature, not for success. Report the intervals, and do not claim a difference smaller than the repeat-to-repeat spread.
5. No result is dropped because it looks odd. Runs that fail or throttle are reported as such.

## Validation of the energy numbers

No external meter is used, so the energy numbers cannot be checked against a reference. Both are reported as software estimates, and the Pi value has no known error. The only checks are internal.

- On the Pi, confirm that the modeled power at idle is close to the model's intercept of 2.48 W and that it rises with CPU usage.
- On the Mac, confirm that `powermetrics` power at idle is lower than during a prefill-heavy and a decode-heavy workload.

## Known threats

- The energy numbers are software estimates with no external check. The Pi model uses CPU usage alone, so it misses the effect of memory traffic and of the frequency governor, and llama.cpp is memory-bound.
- Each model uses its own call format through its adapter, so accuracy may differ from the numbers its authors publish.
- Three measured repeats limit the statistics. Increase repeats where intervals overlap on a claim that matters.
- The case library has 134 cases, so some cells are small and their intervals are wide.
- Related work may already cover part of this. Read TinyLLM (arXiv 2511.22138) and SWEnergy (arXiv 2512.09543) before submission.

## Amendments

### 2026-11-26, before the first full run

Changes to how attempts are scored, from the harness fixes merged in A2E #28:

- A call that does not fit its tool schema no longer ends the attempt. It is fed back like a tool error and still counted in `argument_errors`. The `argument_error` class now means the model stopped right after such a call.
- An attempt that hits a limit (context budget, timeout, output limit) is scored on the state it reached instead of having no checks.
- Number and whitespace formatting no longer fail a correct answer, and the Qwen and generic parsers read common JSON slips (such as arguments sent as a string) instead of failing the attempt.
- The filter variant shows four tools instead of two. Two hid a tool on the correct path in 14 of the 19 `harness` cases, so those cases could not be solved in that variant.

Changes in this amendment's pull request:

- Every suite runs at a context of 2048 tokens instead of 1024. At 1024, with 256 kept for output, many `core` and `harness` cases could not fit their own correct path, so they failed for every model. A test now replays each case's correct path at its suite's context in every variant.
- Distractor tools are chosen per case from a hash of the case id, instead of being the same first tools in name order for every case. Every variant and repeat of a case sees the same distractors, so comparisons stay paired.
- In autonomous mode, a call to a tool that is not listed gets "unknown tool X; available tools: ..." instead of "tool is not available in this stage". Guided modes keep the stage message. The feedback is the same for every model.
- The three live web summary cases check the answer for the facts their judge looks for, instead of accepting any text.
- The three Pi test suites are one `test_pi` suite, with every variant on the same 16 cases.

How the measures and the decision rules are computed:

- An attempt is scored when it passed or failed. Pending attempts (no verdict yet) and excluded attempts (interrupted, or the harness or host failed) are left out of every measure, including the wall time in `seconds_per_success` and `waste_share`.
- `cpu_seconds_per_success` and `joules_per_success` are computed over the scored attempts that have the reading: their summed readings divided by the successes among them. The share of scored attempts with a reading is reported beside each value. Before, one missing reading made the value empty for the whole configuration.
- A cell for the H1 test is one device, model, context and category. Every suite runs at one context, so this is the same as the cells above. A cell where neither the variant nor autonomous succeeds on the shared cases has no ratio and is left out. The interval is the 5th and 95th percentile of the resampled geometric means.
- When the noise check fails, the second margin is exp(2 × the median SD of log `seconds_per_success`) − 1, so it is twice the noise on the log scale.
- Judge agreement is also reported: the share of judged attempts whose verdict matches the deterministic checks. The judge only sees attempts whose checks passed, so it is the share the judge also passed. It is not a decision rule.

### 2026-11-26, memory limit

- The memory limit for the harness and the server together is 1024 MiB on both devices instead of 768. At 768, LFM2.5-1.2B was stopped on the Mac before it ran out of memory, because resident memory counts the weight pages that llama.cpp maps from the file. On the 1 GB Pi the new limit is above physical memory, so the check that at least 128 MiB stays free is what stops a run there. A model that does not fit the Pi is reported as not fitting.
- Swap stays off for measured runs. Swap does not lower the resident memory that the limit checks, and paging weights in and out would change the time and energy per success that the study measures.
