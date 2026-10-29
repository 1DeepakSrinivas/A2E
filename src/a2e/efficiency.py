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

EFFICIENCY_BY = ["device", "model", "quant", "kv_quant", "task_kind", "category", "grammar"]
PAIR_BY = ["device", "model", "quant", "kv_quant", "task_kind", "category"]


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
    by = _keep(by or ["device", "model", "quant", "kv_quant", "task_kind", "category", "grammar"], ce)
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
    configuration. The shares add to one for the energy that was attributed."""
    t = R.turn_frame(store, exp_id, phase)
    t = t[t["energy_j"].notna()] if not t.empty else t
    if t.empty:
        return pd.DataFrame()
    by = _keep(by or ["device", "model", "quant", "kv_quant", "task_kind", "category", "grammar"], t)
    g = t.groupby(by, dropna=False)[["energy_j", "energy_prefill_j", "energy_decode_j", "energy_tool_j"]].sum()
    out = pd.DataFrame(index=g.index)
    for name, col in (("prefill", "energy_prefill_j"), ("decode", "energy_decode_j"), ("tool_wait", "energy_tool_j")):
        out[f"{name}_share"] = g[col] / g["energy_j"]
    out["other_share"] = 1 - out[["prefill_share", "decode_share", "tool_wait_share"]].sum(axis=1)
    out.insert(0, "energy_j", g["energy_j"])
    return out.reset_index()
