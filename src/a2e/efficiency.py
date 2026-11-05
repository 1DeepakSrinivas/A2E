"""Success per joule: how much energy an agent spends to complete a task, failed attempts included.

Everything here is computed from measured repeats of complete runs, on top of `results.case_resources` (one row
per task with its energy and pass/fail). Energy is the software estimate with the idle baseline subtracted, so the
numbers inherit that caveat. Rows without energy are left empty, never filled with zeros.

Definitions
- `j_per_task`: total joules over every scored task of a run, divided by the number of scored tasks.
- `j_per_success`: total joules over every scored task, failures included, divided by the number that passed.
  It is empty when nothing passed. A failed attempt still cost energy, so it counts against the configuration.
- `waste_share`: the share of a run's task energy that went into tasks that failed.
- `budget_curve`: how many tasks a configuration completes when it is given a fixed energy budget and works
  through the suite in order.
- `grammar_effect`: the same configuration with grammar off and on, side by side.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from a2e import results as R
from a2e.store import Store

EFFICIENCY_BY = ["device", "model", "quant", "kv_quant", "task_kind", "family", "depth", "category", "harness",
                 "thinking", "grammar"]
PAIR_BY = ["device", "model", "quant", "kv_quant", "task_kind", "family", "depth", "category", "harness", "thinking"]
# Everything that defines the task and the server setting except the harness. Two runs with the same cell are the
# same work done under different harnesses, so they can be compared task by task.
CELL_FIELDS = ["device", "model", "quant", "kv_quant", "task_kind", "family", "depth", "n_args", "category",
               "grammar", "prefix_cache", "decode_len", "schema_len", "thinking"]
# The task part of the cell. Variance shares are computed after removing the mean of each task cell, so that
# "easy and hard tasks use different energy" is not counted as an effect of the model or the harness.
TASK_FIELDS = ["task_kind", "family", "depth", "n_args", "category", "grammar", "prefix_cache", "decode_len",
               "schema_len", "kv_quant"]


def _keep(by: list[str], df: pd.DataFrame) -> list[str]:
    return [b for b in by if b in df.columns and df[b].notna().any()]


def case_energy(store: Store, exp_id=None, phase: str = "measured") -> pd.DataFrame:
    """One row per scored task with its energy, in the order the run worked through the tasks."""
    cr = R.case_resources(store, exp_id, phase)
    if cr.empty:
        return cr
    order = R.case_frame(store, exp_id, phase)[["run_id", "case_id", "case_row_id"]]
    out = cr.merge(order, on=["run_id", "case_id"], how="left")
    out = out[out["valid"].notna()].copy()
    out["valid"] = out["valid"].astype(int)
    return out.sort_values(["run_id", "case_row_id"]).reset_index(drop=True)


def _run_row(g: pd.DataFrame) -> dict:
    n, ok = len(g), int(g["valid"].sum())
    e = g["energy_j"]
    total = e.sum(min_count=1)
    failed = g.loc[g["valid"] == 0, "energy_j"].sum(min_count=1)
    return {"tasks": n, "successes": ok, "success_rate": ok / n if n else float("nan"),
            "energy_j": total,
            "j_per_task": total / n if n and pd.notna(total) else float("nan"),
            "j_per_success": total / ok if ok and pd.notna(total) else float("nan"),
            "waste_share": failed / total if pd.notna(total) and total and pd.notna(failed) else float("nan")}


def efficiency(store: Store, exp_id=None, by: list[str] | None = None, phase: str = "measured") -> pd.DataFrame:
    """Success rate, joules per task and joules per success for each configuration, averaged over its runs."""
    ce = case_energy(store, exp_id, phase)
    if ce.empty:
        return pd.DataFrame()
    by = _keep(by or EFFICIENCY_BY, ce)
    rows = [{**dict(zip(by + ["run_id"], keys)), **_run_row(g)}
            for keys, g in ce.groupby(by + ["run_id"], dropna=False)]
    r = pd.DataFrame(rows)
    metrics = ["tasks", "successes", "success_rate", "energy_j", "j_per_task", "j_per_success", "waste_share"]
    grouped = r.groupby(by, dropna=False)
    out = grouped[metrics].mean()
    out["success_rate_sd"] = grouped["success_rate"].std(ddof=1)
    out["j_per_success_sd"] = grouped["j_per_success"].std(ddof=1)
    out.insert(0, "runs", grouped.size())
    return out.reset_index()


def pareto_front(df: pd.DataFrame, cost: str = "j_per_task", value: str = "success_rate",
                 within: list[str] | None = None) -> pd.DataFrame:
    """Add `on_front`: True when no other row in the same `within` group has lower or equal cost and higher or
    equal value with at least one strict. Rows with no cost or value are never on the front."""
    out = df.copy()
    out["on_front"] = False
    within = [w for w in (within or ["device", "task_kind", "category"]) if w in out.columns]
    parts = [g for _, g in out.groupby(within, dropna=False)] if within else [out]
    for g in parts:
        g = g[g[cost].notna() & g[value].notna()]
        for i, a in g.iterrows():
            dominated = ((g[cost] <= a[cost]) & (g[value] >= a[value]) &
                         ((g[cost] < a[cost]) | (g[value] > a[value]))).any()
            out.loc[i, "on_front"] = not dominated
    return out


def budget_curve(store: Store, exp_id=None, budgets: list[float] | None = None, by: list[str] | None = None,
                 phase: str = "measured", points: int = 10) -> pd.DataFrame:
    """Tasks completed at each energy budget when a configuration works through the suite in order.

    For each run the tasks are taken in the order they ran. The budget buys tasks until it runs out; a task that
    would overshoot is not counted. Results are averaged over the runs of a configuration. Without `budgets`, a
    grid of `points` values up to the middle (median) full-suite energy of each device is used, so configurations
    on one device share the same budgets.
    """
    ce = case_energy(store, exp_id, phase)
    ce = ce[ce["energy_j"].notna()] if not ce.empty else ce
    if ce.empty:
        return pd.DataFrame()
    by = _keep(by or EFFICIENCY_BY, ce)
    totals = ce.groupby(["device", "run_id"])["energy_j"].sum().groupby("device").median()
    rows = []
    for keys, g in ce.groupby(by + ["run_id"], dropna=False):
        meta = dict(zip(by + ["run_id"], keys))
        cum_e, cum_ok = g["energy_j"].cumsum().to_numpy(), g["valid"].cumsum().to_numpy()
        dev = g["device"].iloc[0]
        grid = budgets if budgets else list(np.linspace(0, totals[dev], points + 1)[1:])
        for b in grid:
            n = int(np.searchsorted(cum_e, b, side="right"))
            rows.append({**meta, "budget_j": float(b), "tasks_done": n, "successes": int(cum_ok[n - 1]) if n else 0})
    r = pd.DataFrame(rows)
    out = r.groupby(by + ["budget_j"], dropna=False)[["tasks_done", "successes"]].mean()
    out.insert(0, "runs", r.groupby(by + ["budget_j"], dropna=False).size())
    return out.reset_index()


def grammar_effect(store: Store, exp_id=None, phase: str = "measured") -> pd.DataFrame:
    """Grammar off and on for the same configuration, side by side.

    `*_delta` is on minus off. `*_ratio` is on divided by off. A ratio above 1 for `j_per_gen_token` means the
    grammar made each generated token cost more energy. Only configurations measured both ways appear.
    """
    e = efficiency(store, exp_id, PAIR_BY + ["grammar"], phase)
    s = R.summary(store, exp_id, PAIR_BY + ["grammar"], phase)
    if e.empty or s.empty:
        return pd.DataFrame()
    cols = [c for c in ("decode_tok_s", "j_per_gen_token", "decode_ms") if c in s.columns]
    keys = [k for k in PAIR_BY + ["grammar"] if k in e.columns and k in s.columns]
    m = e.merge(s[keys + cols], on=keys, how="left")
    pair = _keep(PAIR_BY, m)
    off, on = m[m["grammar"] == False], m[m["grammar"] == True]  # noqa: E712
    if off.empty or on.empty:
        return pd.DataFrame()
    j = off.merge(on, on=pair, suffixes=("_off", "_on"))
    out = j[pair].copy()
    for c in ("success_rate", "j_per_task", "j_per_success", "waste_share", "decode_tok_s", "j_per_gen_token"):
        if f"{c}_off" in j and f"{c}_on" in j:
            out[f"{c}_off"], out[f"{c}_on"] = j[f"{c}_off"], j[f"{c}_on"]
    out["success_delta"] = j["success_rate_on"] - j["success_rate_off"]
    for c in ("j_per_task", "j_per_success", "decode_tok_s", "j_per_gen_token"):
        if f"{c}_off" in j and f"{c}_on" in j:
            out[f"{c}_ratio"] = j[f"{c}_on"] / j[f"{c}_off"]
    return out


def stage_energy(store: Store, exp_id=None, by: list[str] | None = None, phase: str = "measured") -> pd.DataFrame:
    """Share of energy spent in prefill, decode, tool wait and the rest (parsing, templating, harness), per
    configuration. The shares add to one for the energy that was attributed. `harness_code_share` is the part of
    `other_share` that is the harness's own code (templating, parsing, validating), when it was recorded."""
    t = R.turn_frame(store, exp_id, phase)
    t = t[t["energy_j"].notna()] if not t.empty else t
    if t.empty:
        return pd.DataFrame()
    by = _keep(by or EFFICIENCY_BY, t)
    g = t.groupby(by, dropna=False)[["energy_j", "energy_prefill_j", "energy_decode_j", "energy_tool_j"]].sum()
    out = pd.DataFrame(index=g.index)
    for name, col in (("prefill", "energy_prefill_j"), ("decode", "energy_decode_j"), ("tool_wait", "energy_tool_j")):
        out[f"{name}_share"] = g[col] / g["energy_j"]
    out["other_share"] = 1 - out[["prefill_share", "decode_share", "tool_wait_share"]].sum(axis=1)
    if "energy_harness_j" in t.columns and t["energy_harness_j"].notna().any():
        out["harness_code_share"] = t.groupby(by, dropna=False)["energy_harness_j"].sum() / g["energy_j"]
    out.insert(0, "energy_j", g["energy_j"])
    return out.reset_index()


# =============================================================================================== harness effect
# Does the harness change joules per success? The questions are answered with a rule fixed in advance
# (docs/PREREGISTRATION.md): an equivalence test against the baseline harness, a decomposition of any difference,
# the run-to-run noise of the baseline, and the share of variance each factor explains.
BASELINE = "baseline"
DEFAULT_MARGIN = 0.10  # equivalence bounds are 1 / (1 + margin) and 1 + margin, as ratios of joules per success


def _key(df: pd.DataFrame, fields: list[str]) -> pd.Series:
    """A text key of the given fields; empty values show as "-" so they can be compared and joined."""
    return df[fields].apply(lambda r: "|".join("-" if pd.isna(v) else str(v) for v in r), axis=1)


def task_table(store: Store, exp_id=None, phase: str = "measured") -> pd.DataFrame:
    """One row per task per harness per cell, averaged over the repeats: energy, success and model calls.

    Rows without energy or without a score are dropped. `cell` is a text key of everything except the harness.
    """
    ce = case_energy(store, exp_id, phase)
    if ce.empty:
        return ce
    ce = ce[ce["energy_j"].notna()].copy()
    if ce.empty:
        return ce
    if "harness" not in ce.columns:
        ce["harness"] = BASELINE
    cell = [c for c in CELL_FIELDS if c in ce.columns]
    ce["cell"] = _key(ce, cell)
    keys = ["cell", "harness", "device", "model", "quant", "case_id"]
    return ce.groupby(keys, dropna=False).agg(energy_j=("energy_j", "mean"), success=("valid", "mean"),
                                              turns=("turns", "mean"), repeats=("run_id", "nunique")).reset_index()


def _bootstrap_log_ratio(cells: list[tuple], boot: int, rng: np.random.Generator) -> tuple[float, np.ndarray, int]:
    """Pooled log ratio of joules per success (variant over baseline), its bootstrap draws, and the number of
    cells that could not be compared. Each cell is (E_variant, S_variant, E_baseline, S_baseline), one entry per
    task; the bootstrap resamples tasks within a cell, the same tasks for both harnesses. The pooled value is the
    mean of the cells' log ratios, so every cell counts the same."""
    point, draws, used = [], [], 0
    for ev, sv, eb, sb in cells:
        if sv.sum() <= 0 or sb.sum() <= 0:
            continue
        used += 1
        point.append(np.log((ev.sum() / sv.sum()) / (eb.sum() / sb.sum())))
        idx = rng.integers(0, len(ev), size=(boot, len(ev)))
        s_v, s_b = sv[idx].sum(1), sb[idx].sum(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            d = np.log((ev[idx].sum(1) / s_v) / (eb[idx].sum(1) / s_b))
        draws.append(np.where(np.isfinite(d), d, np.nan))  # a resample with no success on one side is left out
    if not point:
        return float("nan"), np.full(boot, np.nan), len(cells)
    draws = np.array(draws)  # (cells, boot); a resample with no success in a cell is nan and ignored
    with np.errstate(all="ignore"):
        pooled = np.nanmean(draws, axis=0)
    return float(np.mean(point)), pooled, len(cells) - used


def harness_effect(store: Store, exp_id=None, baseline: str = BASELINE, margin: float = DEFAULT_MARGIN,
                   level: float = 0.90, boot: int = 2000, seed: int = 0, by: list[str] | None = None,
                   phase: str = "measured") -> pd.DataFrame:
    """Joules per success of every harness against the baseline, with an equivalence verdict.

    For each harness and group (`by`, default one group), the ratio is the geometric mean over cells of
    (variant joules per success) / (baseline joules per success), compared task by task. The interval is a
    bootstrap percentile interval over tasks at `level` (90% gives the two one-sided 5% tests of an equivalence
    test). Verdicts: `equivalent` when the whole interval is inside the bounds, `differs` when it is wholly
    outside them, otherwise `inconclusive`. Only tasks run under both harnesses are compared.
    """
    t = task_table(store, exp_id, phase)
    if t.empty or baseline not in set(t["harness"]):
        return pd.DataFrame()
    lo_b, hi_b = 1 / (1 + margin), 1 + margin
    rng = np.random.default_rng(seed)
    alpha = (1 - level) / 2
    by = [b for b in (by or []) if b in t.columns]
    rows = []
    for gkey, g in (t.groupby(by, dropna=False) if by else [((), t)]):
        gkey = gkey if isinstance(gkey, tuple) else (gkey,)
        base = g[g["harness"] == baseline].set_index(["cell", "case_id"])
        for h in sorted(set(g["harness"]) - {baseline}):
            var = g[g["harness"] == h].set_index(["cell", "case_id"])
            common = base.index.intersection(var.index)
            if not len(common):
                continue
            cells = []
            first = common.get_level_values(0)
            for cell in first.unique():
                ix = common[first == cell]
                cells.append((var.loc[ix, "energy_j"].to_numpy(), var.loc[ix, "success"].to_numpy(),
                              base.loc[ix, "energy_j"].to_numpy(), base.loc[ix, "success"].to_numpy()))
            point, draws, undefined = _bootstrap_log_ratio(cells, boot, rng)
            ok = draws[~np.isnan(draws)]
            lo, hi = (np.exp(np.quantile(ok, [alpha, 1 - alpha])) if len(ok) else (np.nan, np.nan))
            ratio = float(np.exp(point)) if not np.isnan(point) else float("nan")
            lost = sum(1 for _, sv, _, sb in cells if sv.sum() <= 0 < sb.sum())  # the baseline passed, the variant never
            verdict = ("inconclusive" if np.isnan(lo) else "equivalent" if lo >= lo_b and hi <= hi_b
                       else "differs" if lo > hi_b or hi < lo_b else "inconclusive")
            if lost == len(cells):  # the variant never succeeds where the baseline does: infinitely worse
                ratio, lo, hi, verdict = float("inf"), float("inf"), float("inf"), "differs"
            rows.append({**dict(zip(by, gkey)), "harness": h, "cells": len(cells), "cells_undefined": undefined,
                         "cells_lost": lost, "tasks": int(len(common)), "ratio": ratio, "lo": float(lo), "hi": float(hi),
                         "lower_bound": lo_b, "upper_bound": hi_b, "verdict": verdict})
    return pd.DataFrame(rows)


def conclusion(effect: pd.DataFrame) -> str:
    """The study's verdict under the pre-registered rule. Judged per group when the table was split by group."""
    if effect.empty:
        return "no comparison: need the baseline harness and at least one other harness measured on the same tasks"
    v = set(effect["verdict"])
    if "differs" in v:
        n = int((effect["verdict"] == "differs").sum())
        return f"harness matters: {n} of {len(effect)} comparisons are wholly outside the equivalence bounds"
    if v == {"equivalent"}:
        return "LLM-bound: every harness is equivalent to the baseline within the margin"
    return "inconclusive: some intervals straddle a bound; run more repeats before concluding"


def decompose(store: Store, exp_id=None, baseline: str = BASELINE, by: list[str] | None = None,
              phase: str = "measured") -> pd.DataFrame:
    """Why a harness changes joules per success: joules per success = model calls per task x joules per call /
    success rate. For each harness against the baseline, per cell, the log ratio of each term, averaged over the
    cells. `d_log_total` equals `d_log_turns + d_log_j_per_turn - d_log_success` (to rounding) and is the log of the
    ratio of the ratios of sums, so it can differ a little from `harness_effect`, which compares task by task.
    `turns_share` is the share of the absolute change that comes from the number of calls."""
    t = task_table(store, exp_id, phase)
    if t.empty or baseline not in set(t["harness"]):
        return pd.DataFrame()
    t = t.assign(n=1)
    agg = t.groupby(["cell", "harness", "device"], dropna=False).agg(
        tasks=("n", "sum"), energy=("energy_j", "sum"), turns=("turns", "sum"), ok=("success", "sum")).reset_index()
    agg = agg[agg["ok"] > 0]
    agg["turns_per_task"] = agg["turns"] / agg["tasks"]
    agg["j_per_turn"] = agg["energy"] / agg["turns"]
    agg["success_rate"] = agg["ok"] / agg["tasks"]
    agg["j_per_success"] = agg["energy"] / agg["ok"]
    b = agg[agg["harness"] == baseline].set_index(["cell", "device"])
    v = agg[agg["harness"] != baseline].set_index(["cell", "device"], append=False)
    v = v.reset_index().merge(b[["turns_per_task", "j_per_turn", "success_rate", "j_per_success"]].reset_index(),
                              on=["cell", "device"], suffixes=("", "_base"))
    v["d_log_turns"] = np.log(v["turns_per_task"] / v["turns_per_task_base"])
    v["d_log_j_per_turn"] = np.log(v["j_per_turn"] / v["j_per_turn_base"])
    v["d_log_success"] = np.log(v["success_rate"] / v["success_rate_base"])
    v["d_log_total"] = np.log(v["j_per_success"] / v["j_per_success_base"])
    by = [x for x in (by or []) if x in v.columns]
    terms = ["d_log_turns", "d_log_j_per_turn", "d_log_success", "d_log_total"]
    out = v.groupby(by + ["harness"], dropna=False)[terms].mean()
    out.insert(0, "cells", v.groupby(by + ["harness"], dropna=False).size())
    mag = out[["d_log_turns", "d_log_j_per_turn", "d_log_success"]].abs()
    out["turns_share"] = mag["d_log_turns"] / mag.sum(axis=1).replace(0, np.nan)
    return out.reset_index()


def _run_table(store: Store, exp_id=None, phase: str = "measured") -> pd.DataFrame:
    """One row per run (one repeat of one job): joules per success over its scored tasks."""
    ce = case_energy(store, exp_id, phase)
    if ce.empty:
        return ce
    ce = ce[ce["energy_j"].notna()].copy()
    if ce.empty:
        return ce
    if "harness" not in ce.columns:
        ce["harness"] = BASELINE
    cell = [c for c in CELL_FIELDS if c in ce.columns]
    ce["cell"] = _key(ce, cell)
    fields = [c for c in CELL_FIELDS + ["harness"] if c in ce.columns]
    g = ce.groupby(["run_id", "cell"] + fields, dropna=False).agg(
        energy=("energy_j", "sum"), ok=("valid", "sum"), tasks=("valid", "size")).reset_index()
    g["j_per_success"] = np.where(g["ok"] > 0, g["energy"] / g["ok"].replace(0, np.nan), np.nan)
    return g.merge(ce[["run_id", "repeat_idx"]].drop_duplicates(), on="run_id", how="left")


def noise_floor(store: Store, exp_id=None, baseline: str = BASELINE, phase: str = "measured") -> pd.DataFrame:
    """Run-to-run noise of the baseline harness: for each cell, the standard deviation of log joules per success
    over its measured repeats, and the largest gap between two repeats (as a ratio)."""
    r = _run_table(store, exp_id, phase)
    if r.empty:
        return r
    r = r[(r["harness"] == baseline) & r["j_per_success"].notna()]
    rows = []
    for cell, g in r.groupby("cell"):
        if len(g) < 2:
            continue
        logs = np.log(g["j_per_success"].to_numpy())
        rows.append({"cell": cell, "repeats": len(g), "sd_log": float(np.std(logs, ddof=1)),
                     "max_ratio": float(np.exp(logs.max() - logs.min()))})
    return pd.DataFrame(rows)


def margin_check(noise: pd.DataFrame, margin: float = DEFAULT_MARGIN) -> dict:
    """Is the margin wide enough for the noise? It should be at least twice the typical (median) run-to-run
    standard deviation of log joules per success."""
    if noise.empty:
        return {"cells": 0, "median_sd_log": float("nan"), "margin_log": float(np.log1p(margin)), "ok": None}
    med = float(noise["sd_log"].median())
    return {"cells": len(noise), "median_sd_log": med, "margin_log": float(np.log1p(margin)),
            "ok": bool(np.log1p(margin) >= 2 * med)}


def _omega2(y: np.ndarray, groups: np.ndarray) -> tuple[float, float]:
    """(eta squared, omega squared) of a one-way grouping."""
    levels, inv = np.unique(groups, return_inverse=True)
    n, k = len(y), len(levels)
    if k < 2 or n <= k:
        return float("nan"), float("nan")
    grand = y.mean()
    ss_t = float(((y - grand) ** 2).sum())
    if ss_t == 0:
        return float("nan"), float("nan")
    means = np.array([y[inv == i].mean() for i in range(k)])
    counts = np.bincount(inv)
    ss_b = float((counts * (means - grand) ** 2).sum())
    ms_w = (ss_t - ss_b) / (n - k)
    return ss_b / ss_t, max((ss_b - (k - 1) * ms_w) / (ss_t + ms_w), 0.0)


def variance_share(store: Store, exp_id=None, factors: tuple[str, ...] = ("device", "model", "quant", "harness"),
                   boot: int = 1000, seed: int = 0, level: float = 0.90, phase: str = "measured") -> pd.DataFrame:
    """Share of the variance in log joules per success that each factor explains on its own.

    One observation is one run. The mean of its task cell (task, grammar, schema length and so on) is removed
    first, so differences between easy and hard tasks do not count as an effect of any factor. `omega2` is the
    bias-corrected share, with a bootstrap interval over runs. A factor with one level has no share.
    """
    r = _run_table(store, exp_id, phase)
    if r.empty:
        return r
    r = r[r["j_per_success"].notna()].copy()
    task = [c for c in TASK_FIELDS if c in r.columns]
    r["task_cell"] = _key(r, task)
    r["y"] = np.log(r["j_per_success"])
    r["y"] -= r.groupby("task_cell")["y"].transform("mean")
    y = r["y"].to_numpy()
    rng = np.random.default_rng(seed)
    alpha = (1 - level) / 2
    rows = []
    for f in factors:
        if f not in r.columns:
            continue
        grp = r[f].astype(str).to_numpy()
        eta, om = _omega2(y, grp)
        draws = []
        for _ in range(boot):
            i = rng.integers(0, len(y), len(y))
            draws.append(_omega2(y[i], grp[i])[1])
        d = np.array(draws)
        d = d[~np.isnan(d)]
        lo, hi = (np.quantile(d, [alpha, 1 - alpha]) if len(d) else (np.nan, np.nan))
        rows.append({"factor": f, "levels": int(len(set(grp))), "runs": len(y), "eta2": eta, "omega2": om,
                     "omega2_lo": float(lo), "omega2_hi": float(hi)})
    return pd.DataFrame(rows)
