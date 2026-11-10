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

None yet.
