# Pre-registration: success per joule for small tool-calling agents on edge hardware

Status: draft, written before the full sweep is run. Commit this file, then run the sweep, and record any later
change as a dated amendment at the bottom. Do not edit the hypotheses after seeing the data.

## Question

When a fixed amount of energy is available, which combination of model, quantization and decoding mode completes
the most tool-calling tasks, on a Raspberry Pi 5 and on an Apple Silicon Mac?

The main claim being tested is that the best configuration by accuracy is often not the best by joules per
successful task. A failed attempt still costs energy, and the cost of a failure grows with the number of turns, so
a model that is a little less accurate but much cheaper per attempt can win.

## Why this is worth asking

The searches behind this plan (see `REFERENCES.md`, section 7) found per-token and per-response energy on edge
devices, agent energy profiling on data-center GPUs, and cost-aware agent evaluation in dollars or tokens. They did
not find a success-versus-joules result for multi-turn tool-calling agents on small devices. Treat that as a claim
to re-check before submission, because the searches only read summaries.

## Measures

All measures come from `a2e results efficiency` (module `src/a2e/efficiency.py`), computed from measured repeats of
complete runs. Warm-up repeats are excluded.

| Measure | Definition |
|---|---|
| `success_rate` | Scored tasks that passed, divided by scored tasks. |
| `j_per_task` | Energy over all scored tasks of a run divided by the number of scored tasks. |
| `j_per_success` | The same energy, failures included, divided by the number that passed. Empty when none passed. |
| `waste_share` | Share of a run's task energy that went into tasks that failed. |
| budget curve | Tasks completed and passed when the suite is worked through in order on a fixed joule budget. |
| stage energy | Share of energy in prefill, decode, tool wait and everything else. |

Energy is the software estimate with the idle baseline subtracted. On the Pi it comes from the PMIC and on the Mac
from `powermetrics`, which leaves out DRAM, the display and the power supply. Every energy result is reported with
its power scope.

## Factors

- Device: Raspberry Pi 5 (8 GB) and Apple Silicon Mac.
- Model: the registered models that fit each device.
- Quantization: q3_k_m, q4_0 and q8_0 (q8_0 on the Pi only for models of 4B parameters or fewer).
- Grammar: off and on, for the same model, quantization and tasks.
- Tasks: curated BFCL categories and the custom local-tool families.

## Hypotheses

Each can be shown wrong by the data.

- **H1, grammar cost is larger on the Pi.** The ratio of joules per generated token with grammar on to grammar off
  is larger on the Pi than on the Mac, for the same model and quantization. Reason: llama.cpp's grammar sampler
  does its work on the CPU for each token, and the Pi has the weaker CPU. Refuted if the Pi ratio is not larger
  with a 95% interval that excludes equality.
- **H2, grammar pays only for weaker models.** Grammar lowers `j_per_success` only for configurations whose
  grammar-off `success_rate` is below a threshold. The threshold is not set in advance. It is estimated from the
  data, and the claim is refuted if grammar lowers `j_per_success` for the strongest models as often as for the
  weakest.
- **H3, quantization acts through failures.** Moving from q8_0 to q4_0 changes `j_per_success` mostly through
  `success_rate` and `waste_share`, not through `j_per_gen_token`. Refuted if the change in joules per generated
  token explains more than half of the change in joules per success.
- **H4, tool wait is a larger share on the Pi.** The tool-wait share of energy is higher on the Pi than on the Mac
  for tasks with real tools. Refuted if it is equal or lower.

## Analysis plan

1. Configurations with no energy readings are dropped from every energy result and listed.
2. The Pareto front of `success_rate` against `j_per_task` is drawn per device and task kind, with repeat-to-repeat
   standard deviation as error bars. A configuration is on the front only if no other has lower or equal energy and
   higher or equal success, with one strict.
3. Budget curves use the same budget grid for every configuration on a device.
4. Grammar effects are paired: the same model, quantization, device and task, grammar off against on.
5. With 3 measured repeats per configuration, intervals are wide. Report them, and do not claim a difference
   smaller than the repeat-to-repeat spread.
6. No result is dropped because it looks odd. Runs that fail or throttle are reported as such.

## Validation of the energy numbers

Before the main sweep, compare the software energy with an external USB power meter on the Pi and a wall meter on
the Mac, over at least 3 workloads (idle, prefill-heavy, decode-heavy) and a subset of agent runs. Report the
ratio and its spread. If the software estimate is off by more than 15% in either direction, report all energy
results with a correction and the uncorrected values side by side.

## Known threats

- The energy numbers are software estimates. This is why the validation step comes first.
- The fixed prompt format may lower accuracy compared with each model's own published numbers.
- Three measured repeats limit the statistics. Increase repeats where intervals overlap on a claim that matters.
- Related work may already cover part of this. Read TinyLLM (arXiv 2511.22138) and SWEnergy before submission.

## Amendments

None yet.
