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

## Amendment 1 (2026-10-29): the harness question

Added before any sweep was run. The study now also asks whether the harness, the code around the model, changes
success per joule. If it does, the answer is shown with figures. If it does not, the study says the system is
LLM-bound. Both answers come out of the rules below, so the conclusion is set by the data and not by the author.

### What the harness is

The harness is what the agent loop does around each model call. Four choices are varied, each with a default
that is the loop as it was before this amendment:

| Choice | Levels | What changes |
|---|---|---|
| `prompt_format` | `json_schema` (default), `compact`, `natural` | How the tool list is written in the prompt. The reply format and the parser stay the same. |
| `context` | `full` (default), `mask`, `truncate` | What the model sees of earlier tool results. `mask` keeps the latest 2 and replaces older ones with a stub. `truncate` cuts each to 400 characters. |
| `recovery` | `none` (default), `feedback`, `retry` | What happens to a reply that cannot be read. `feedback` adds an error message and keeps the bad reply. `retry` drops it and asks again with a one-line reminder, at most 2 times in a row. |
| `stopping` | `fixed` (default), `tight`, `early_stop` | `tight` caps the calls at what a correct run needs plus 2. `early_stop` ends the task after 2 identical calls or 2 failed turns in a row. |

The suite `harness_ablation` runs one choice at a time against the baseline (8 variants), plus two combinations:
`lean` (compact, mask, feedback, early_stop) and `heavy` (natural and retry). Every harness runs the same tasks.
The tool-list padding (`schema_len`) is always measured with the JSON rendering, so all formats carry the same
tools.

### Hypotheses

- **H5.** For at least one lever, the harness changes joules per success by more than the margin. Refuted if every
  variant passes the equivalence test below.
- **H6.** Where a harness changes joules per success, most of the change comes from the number of model calls per
  task and not from the energy per call. Refuted if the share of the change from calls per task is under half.
- **H7.** The harness's own code (templating, parsing, validating) uses under 5% of turn energy on both devices.
  Refuted if either device is at 5% or more.

### Decision rules

1. **Primary: equivalence.** For each harness against the baseline, take the ratio of joules per success, task by
   task on the same tasks, as the geometric mean over cells (each cell is one device, model, quantization, task
   family or category, and server setting, so every cell counts the same). The 90% interval comes from resampling
   tasks 2,000 times, which is two one-sided tests at 5%. The bounds are 1 / 1.10 and 1.10, a margin of 10%.
   - A variant is **equivalent** when its whole interval is inside the bounds.
   - A variant **differs** when its whole interval is outside the bounds, or when it never succeeds where the
     baseline does.
   - Otherwise it is **inconclusive**.
   - The study concludes **LLM-bound** only if every variant is equivalent. It concludes **the harness matters** if
     any variant differs. Anything else is **inconclusive**, and the response is more repeats, not a verdict.
2. **Supporting: variance share.** The share of the variance in log joules per success that each of device, model,
   quantization and harness explains on its own (omega squared), after the mean of each task cell is removed, with
   a 90% bootstrap interval over runs. A harness share under 10% supports LLM-bound.
3. **Noise floor.** The baseline harness is measured with 3 repeats, so its own run-to-run spread is known. The
   margin must be at least twice the median standard deviation of log joules per success across baseline cells. If
   it is not, `a2e results efficiency noise` prints a warning, the verdict is re-read with `--margin` set to at least twice the noise, and both verdicts are reported.
4. **Decomposition.** Joules per success equals calls per task times joules per call divided by success rate. The
   change in each term is reported for every variant, and it is the basis for H6.

### Commands

```bash
uv run a2e results efficiency harness         # ratios, intervals, verdicts and the conclusion
uv run a2e results efficiency decomposition   # calls per task, joules per call, success rate
uv run a2e results efficiency noise           # baseline noise and the margin check
uv run a2e results efficiency variance        # variance shares
uv run a2e results plots figures/             # the six figures with their tables
```

### Limits

- The margin of 10% is set before the sweep. It can be changed only through a dated amendment.
- Prompt variants can change accuracy for reasons unrelated to energy. That is part of the result, and the
  full prompt text of each variant is in `src/a2e/agent/prompt.py`.
- With 3 measured repeats per job, intervals are wide for small cells. The `inconclusive` verdict is how that
  shows up.
- The tasks come from the custom local-tool families and curated BFCL categories. A harness result does not carry
  over to other tasks without a new run.

## Amendment 2 (2026-10-29): thinking is switched off in `full_eval`

Added after the first `full_eval` run on the Mac, before any pre-registered sweep. With the 256-token `decode_len`,
Qwen3 used the whole budget on its thinking block when grammar was off. Replies were cut off before the tool call, so
custom chains scored 0% for a reason that has nothing to do with tool calling. Gemma 3 has no thinking block and was
not affected. The `[server] thinking = "off"` setting turns thinking off for every model that has the template
variable. `full_eval` now uses it, and runs record it (`thinking` in the run's settings, `default` for older runs).

- Results from runs made before this amendment, and runs with `thinking = "default"`, are not comparable with runs
  that have `thinking = "off"` for models with a thinking mode. The report and the efficiency views keep them apart.
- `harness_ablation` is unchanged and still uses the model's default behavior. If thinking should be off there too,
  that is a further dated amendment before its sweep, because it changes the baseline every harness is compared with.
- The fixed 10% margin and all decision rules of Amendment 1 are unchanged.
