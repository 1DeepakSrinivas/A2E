"""Read results back out of the store: summaries per configuration, stage breakdowns, accuracy, energy,
stability drift, and a markdown report. Only measured repeats of complete runs are used (the v_* views
already drop aborted runs, and warm-up runs are filtered here).

How numbers are aggregated: each metric is first computed per run (one repeat of one job), then averaged
across the measured runs of a configuration. The `_sd` columns are the standard deviation across those runs,
so they show repeat-to-repeat variation, not turn-to-turn variation.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone

import pandas as pd

from a2e.store import Store

CONFIG_FIELDS = ["device", "model", "quant", "kv_quant", "task_kind", "family", "depth", "n_args", "category",
                 "grammar", "prefix_cache", "decode_len", "schema_len", "governor", "harness", "thinking"]
DEFAULT_BY = ["device", "model", "quant", "kv_quant", "task_kind", "family", "category", "grammar", "prefix_cache",
              "decode_len", "schema_len", "harness", "thinking"]
STAGES = ["prefill_ms", "decode_ms", "parse_ms", "validate_ms", "tool_wait_ms", "template_ms", "harness_ms"]


def _ts(x) -> str:
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else \
        datetime.fromtimestamp(x, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _where(exp_id, phase, alias=""):
    p = f"{alias}." if alias else ""
    clauses, params = [], []
    if exp_id is not None:
        ids = exp_id if isinstance(exp_id, (list, tuple)) else [exp_id]
        clauses.append(f"{p}exp_id IN ({','.join('?' * len(ids))})")
        params += list(ids)
    if phase:
        clauses.append(f"{p}phase = ?")
        params.append(phase)
    return (" WHERE " + " AND ".join(clauses)) if clauses else "", params


def experiments(store: Store) -> pd.DataFrame:
    exps = store.df("SELECT exp_id, name, status, started_at, finished_at, suite_hash, git_commit FROM experiments")
    if exps.empty:
        return exps
    jobs = store.df("SELECT exp_id, status, count(*) AS n FROM jobs GROUP BY exp_id, status")
    piv = jobs.pivot(index="exp_id", columns="status", values="n").fillna(0).astype(int) if not jobs.empty else None
    out = exps.set_index("exp_id")
    if piv is not None:
        out = out.join(piv, how="left")
    out["started"] = out["started_at"].map(_ts)
    out["git_commit"] = out["git_commit"].fillna("").str[:10]
    return out.drop(columns=["started_at", "finished_at"]).reset_index()


def jobs(store: Store, exp_id: int) -> pd.DataFrame:
    return store.df("SELECT job_id, status, kind, model, quant, kv_quant, attempts, reason, config_json FROM jobs "
                    "WHERE exp_id = ? ORDER BY job_id", [exp_id])


def turn_frame(store: Store, exp_id=None, phase: str | None = "measured") -> pd.DataFrame:
    where, params = _where(exp_id, phase)
    df = store.df(f"SELECT * FROM v_turns{where}", params)
    if not df.empty:
        df["fresh_tokens"] = (df["prompt_tokens"].fillna(0) - df["cached_tokens"].fillna(0)).clip(lower=0)
    return df


def case_frame(store: Store, exp_id=None, phase: str | None = "measured") -> pd.DataFrame:
    where, params = _where(exp_id, phase)
    return store.df(f"SELECT * FROM v_cases{where}", params)


def _div(a, b):
    return a / b if b else float("nan")


def _per_run(df: pd.DataFrame) -> dict:
    s = lambda c: df[c].fillna(0).sum()  # noqa: E731
    wall = s("wall_ms")
    has_e = df["energy_j"].notna().any()
    out = {
        "turns": len(df),
        "prefill_ms": df["prefill_ms"].mean(), "decode_ms": df["decode_ms"].mean(),
        "tool_wait_ms": df["tool_wait_ms"].mean(), "wall_ms": df["wall_ms"].mean(),
        "prefill_tok_s": _div(s("fresh_tokens"), s("prefill_ms") / 1000),
        "decode_tok_s": _div(s("gen_tokens"), s("decode_ms") / 1000),
        "cache_hit": _div(s("cached_tokens"), s("prompt_tokens")),
        "prefill_share": _div(s("prefill_ms"), wall), "decode_share": _div(s("decode_ms"), wall),
        "tool_share": _div(s("tool_wait_ms"), wall),
        "overhead_share": _div(s("parse_ms") + s("validate_ms") + s("template_ms") + s("harness_ms"), wall),
        "peak_rss_mb": df["peak_rss_mb"].max(), "max_temp_c": df["max_temp_c"].max(),
        "throttled_turns": df["throttled"].fillna(0).sum(),
    }
    if has_e:
        out.update({
            "j_per_turn": df["energy_j"].mean(),
            "j_per_gen_token": _div(s("energy_decode_j"), s("gen_tokens")),
            "j_per_prompt_token": _div(s("energy_prefill_j"), s("fresh_tokens")),
            "tokens_per_joule": _div(s("gen_tokens"), s("energy_decode_j")),  # = decode tokens/s per watt
            "avg_power_w": df["avg_power_w"].mean(),
        })
    return out


def summary(store: Store, exp_id=None, by: list[str] | None = None, phase: str = "measured") -> pd.DataFrame:
    """One row per configuration (the `by` fields): latency, throughput, stage shares, cache, energy, accuracy."""
    by = by or DEFAULT_BY
    t = turn_frame(store, exp_id, phase)
    if t.empty:
        return pd.DataFrame()
    runs = []
    for keys, g in t.groupby(by + ["run_id"], dropna=False):
        runs.append({**dict(zip(by + ["run_id"], keys)), **_per_run(g)})
    r = pd.DataFrame(runs)
    c = case_frame(store, exp_id, phase)
    if not c.empty:
        scored = c[c["valid"].notna()]
        acc = scored.groupby("run_id")["valid"].agg(["mean", "sum", "count"]).rename(
            columns={"mean": "success", "sum": "n_valid", "count": "n_scored"})
        r = r.merge(acc, left_on="run_id", right_index=True, how="left")
        if "j_per_turn" in r:
            e = t.groupby("run_id")["energy_j"].sum()
            r["j_per_success"] = r.apply(lambda x: _div(e.get(x["run_id"], float("nan")), x.get("n_valid") or 0),
                                         axis=1)
    metrics = [m for m in r.columns if m not in by + ["run_id"]]
    grouped = r.groupby(by, dropna=False)
    out = grouped[metrics].mean()
    sd = grouped[[m for m in ("decode_tok_s", "prefill_ms", "success", "j_per_gen_token") if m in r]].std(ddof=1)
    out = out.join(sd.add_suffix("_sd"))
    out.insert(0, "runs", grouped.size())
    out = out.reset_index()
    keep = [b for b in by if out[b].notna().any()]  # drop config columns that are empty for every row
    return out[keep + [c for c in out.columns if c not in by]]


def stages(store: Store, exp_id=None, by: list[str] | None = None, phase: str = "measured") -> pd.DataFrame:
    """Share of total turn time spent in each stage, per configuration."""
    by = by or ["device", "model", "quant", "task_kind", "category", "grammar"]
    t = turn_frame(store, exp_id, phase)
    if t.empty:
        return pd.DataFrame()
    g = t.groupby(by, dropna=False)[STAGES + ["wall_ms"]].sum()
    shares = g[STAGES].div(g["wall_ms"], axis=0)
    shares.columns = [c.replace("_ms", "") for c in shares.columns]
    # the rest is time inside the HTTP call that the server does not report as prefill or decode
    shares["unattributed"] = 1 - shares.sum(axis=1)
    shares["wall_s"] = g["wall_ms"] / 1000
    out = shares.reset_index()
    keep = [b for b in by if out[b].notna().any()]
    return out[keep + [c for c in out.columns if c not in by]]


def stability(store: Store, exp_id=None, window: int = 3, drop: float = 0.9) -> pd.DataFrame:
    """Per job: decode speed and temperature over successive tasks, and the first task where decode speed
    falls below `drop` x the mean of the first `window` tasks (a simple knee marker for thermal throttling)."""
    t = turn_frame(store, exp_id, None)
    if t.empty:
        return pd.DataFrame()
    rows = []
    for job_id, g in t.sort_values("turn_id").groupby("job_id"):
        per_task = g.groupby("task_id", sort=False).agg(gen=("gen_tokens", "sum"), dec=("decode_ms", "sum"),
                                                         temp=("max_temp_c", "max"), thr=("throttled", "max"))
        speed = (per_task["gen"] / (per_task["dec"] / 1000)).tolist()
        base = sum(speed[:window]) / max(len(speed[:window]), 1)
        knee = next((i for i, v in enumerate(speed) if i >= window and v < drop * base), None)
        rows.append({"job_id": job_id, "model": g["model"].iloc[0], "quant": g["quant"].iloc[0],
                     "governor": g["governor"].iloc[0], "tasks": len(speed), "start_tok_s": base,
                     "end_tok_s": sum(speed[-window:]) / max(len(speed[-window:]), 1),
                     "knee_task": knee, "max_temp_c": per_task["temp"].max(),
                     "throttled_tasks": int(per_task["thr"].fillna(0).sum())})
    return pd.DataFrame(rows)


def timeline(store: Store, run_id: int) -> pd.DataFrame:
    return store.df("SELECT * FROM samples WHERE run_id = ? ORDER BY t", [run_id])


PRESSURE_ORDER = ["Nominal", "Fair", "Serious", "Critical"]  # Mac thermal pressure levels, mildest first


def _worst_pressure(levels) -> str | None:
    seen = {x for x in levels if isinstance(x, str)}
    ranked = [p for p in PRESSURE_ORDER if p in seen]
    return ranked[-1] if ranked else (sorted(seen)[-1] if seen else None)


def _sample_stats(g: pd.DataFrame, energy_j: float | None) -> dict:
    """Envelope of one run's samples. Samples are taken only while the run is active, not during model load or idle."""
    out = {"samples": len(g), "duration_s": g["t"].max() - g["t"].min() if len(g) > 1 else 0.0,
           "rss_min_mb": g["rss_mb"].min(), "rss_mean_mb": g["rss_mb"].mean(), "rss_max_mb": g["rss_mb"].max(),
           "cpu_mean_pct": g["cpu_pct"].mean(), "cpu_max_pct": g["cpu_pct"].max()}
    p = g["power_w"].dropna()
    out.update({"power_min_w": p.min(), "power_median_w": p.median(), "power_mean_w": p.mean(),
                "power_p95_w": p.quantile(0.95) if len(p) else float("nan"), "power_max_w": p.max(),
                "energy_j": energy_j})
    out["temp_max_c"] = g["temp_c"].max()
    out["thermal_pressure"] = _worst_pressure(g["pressure"])
    return out


def resources(store: Store, exp_id=None, by: list[str] | None = None, phase: str = "measured") -> pd.DataFrame:
    """Power, memory, CPU and thermal envelope per configuration, from the 2 Hz samples of each measured run.

    Each statistic is computed per run, then combined over the runs of a configuration: `_min` columns take the
    lowest value, `_max` columns the highest, and the others (median, mean, p95, energy) are averaged.
    Power and energy are empty when the run had no power source; they are never filled with zeros.
    """
    by = by or DEFAULT_BY
    where, params = _where(exp_id, phase, "r")
    cols = ", ".join(f"r.{c}" for c in dict.fromkeys(by + ["power_scope"]))
    df = store.df(f"SELECT s.run_id, s.t, s.rss_mb, s.cpu_pct, s.temp_c, s.power_w, s.pressure, {cols} "
                  f"FROM samples s JOIN v_runs r ON r.run_id = s.run_id{where}", params)
    if df.empty:
        return pd.DataFrame()
    energy = turn_frame(store, exp_id, phase).groupby("run_id")["energy_j"].sum(min_count=1)
    rows = []
    for keys, g in df.groupby(by + ["power_scope", "run_id"], dropna=False):
        meta = dict(zip(by + ["power_scope", "run_id"], keys))
        e = energy.get(meta["run_id"])
        rows.append({**meta, **_sample_stats(g, None if e is None or pd.isna(e) else float(e))})
    r = pd.DataFrame(rows)
    stats = [c for c in r.columns if c not in by + ["power_scope", "run_id", "thermal_pressure"]]
    agg = {c: "min" if c.endswith("_min_w") or c.endswith("_min_mb") else
           "max" if c.endswith("_max_w") or c.endswith("_max_mb") or c.endswith("_max_pct") or c == "temp_max_c"
           else "mean" for c in stats}
    grouped = r.groupby(by + ["power_scope"], dropna=False)
    out = grouped.agg({**agg, "thermal_pressure": _worst_pressure}).join(grouped.size().rename("runs"))
    out = out.reset_index()
    keep = [b for b in by if out[b].notna().any()]
    first = keep + ["power_scope", "runs"]
    return out[first + [c for c in out.columns if c not in first]]


def case_resources(store: Store, exp_id=None, phase: str = "measured") -> pd.DataFrame:
    """One row per task (case): energy, power, memory and time, summed or maximised over the task's model calls."""
    t = turn_frame(store, exp_id, phase)
    c = case_frame(store, exp_id, phase)
    if t.empty or c.empty:
        return pd.DataFrame()

    def agg(g: pd.DataFrame) -> pd.Series:
        w = g.loc[g["avg_power_w"].notna(), "wall_ms"]
        power = (g.loc[w.index, "avg_power_w"] * w).sum() / w.sum() if w.sum() else float("nan")
        return pd.Series({"turns": len(g), "energy_j": g["energy_j"].sum(min_count=1), "avg_power_w": power,
                          "peak_rss_mb": g["peak_rss_mb"].max(), "max_temp_c": g["max_temp_c"].max()})

    per_task = t.groupby(["run_id", "task_id"]).apply(agg, include_groups=False).reset_index()
    out = c.merge(per_task, left_on=["run_id", "case_id"], right_on=["run_id", "task_id"], how="left")
    keep = ["exp_id", "run_id", "repeat_idx", "device", "model", "quant", "kv_quant", "task_kind", "family", "depth",
            "n_args", "grammar", "prefix_cache", "decode_len", "schema_len", "harness", "thinking", "category", "case_id",
            "valid", "turns", "wall_ms", "energy_j", "avg_power_w", "peak_rss_mb", "max_temp_c"]
    out = out[[k for k in keep if k in out.columns]]
    return out.sort_values(["exp_id", "run_id", "case_id"]).reset_index(drop=True)


def failures(store: Store, exp_id=None, by: list[str] | None = None, phase: str = "measured") -> pd.DataFrame:
    """Why tasks failed: the number of failed tasks per reason, with the number of tasks run. `error_type` is the
    official BFCL checker's error type for BFCL, or for a custom chain `wrong_step`, `wrong_state`, `parse_error`,
    `max_turns`, `early_stop` or `server_error`. Tasks that passed are not listed."""
    by = by or ["device", "model", "quant", "grammar", "category"]
    where, params = _where(exp_id, phase)
    cols = ", ".join(dict.fromkeys(by))
    df = store.df(f"SELECT {cols}, coalesce(error_type, 'unknown') AS reason, count(*) AS tasks FROM v_cases"
                  f"{where}{' AND' if where else ' WHERE'} valid = 0 GROUP BY {cols}, reason", params)
    if df.empty:
        return df
    ran = store.df(f"SELECT {cols}, count(*) AS tasks_run FROM v_cases{where}{' AND' if where else ' WHERE'} "
                   f"valid IS NOT NULL GROUP BY {cols}", params)
    out = df.merge(ran, on=list(dict.fromkeys(by)), how="left")
    return out.sort_values(list(dict.fromkeys(by)) + ["tasks"], ascending=[True] * len(dict.fromkeys(by)) + [False]
                           ).reset_index(drop=True)


def bench_summary(store: Store, exp_id=None, phase: str = "measured") -> pd.DataFrame:
    """llama-bench throughput per configuration and test size. For decode (tg) the estimated memory bandwidth is
    tokens/s x model file size: every weight is read once per generated token, so decode speed is bounded by it."""
    where, params = _where(exp_id, phase, "v")
    df = store.df("SELECT b.test, b.n_prompt, b.n_gen, b.avg_tok_s, b.sd_tok_s, v.device, v.model, v.quant, "
                  f"v.kv_quant, v.governor, v.server_json FROM bench b JOIN v_runs v ON v.run_id = b.run_id{where}",
                  params)
    if df.empty:
        return df
    df["model_gb"] = df["server_json"].map(lambda j: (json.loads(j or "{}").get("model_file") or {}).get("size", 0) / 1e9)
    df["tokens"] = df["n_prompt"].where(df["test"] == "pp", df["n_gen"])
    keys = ["device", "model", "quant", "kv_quant", "governor", "test", "tokens"]
    out = df.groupby(keys, dropna=False).agg(runs=("avg_tok_s", "size"), tok_s=("avg_tok_s", "mean"),
                                             tok_s_sd=("avg_tok_s", "std"), model_gb=("model_gb", "max")).reset_index()
    out["est_bandwidth_gbs"] = (out["tok_s"] * out["model_gb"]).where(out["test"] == "tg")
    keep = [k for k in keys if out[k].notna().any() and (out[k] != "").any()]
    return out[keep + ["runs", "tok_s", "tok_s_sd", "model_gb", "est_bandwidth_gbs"]]


def kernels(store: Store, exp_id=None, top: int = 10, phase: str = "measured") -> pd.DataFrame:
    """Top kernels (perf symbols) per configuration and test, averaged over measured runs."""
    where, params = _where(exp_id, phase, "v")
    df = store.df("SELECT k.test, k.n_tokens, k.symbol, k.dso, k.pct, v.device, v.model, v.quant, v.kv_quant "
                  f"FROM kernels k JOIN v_runs v ON v.run_id = k.run_id{where}", params)
    if df.empty:
        return df
    keys = ["device", "model", "quant", "kv_quant", "test", "n_tokens"]
    g = df.groupby(keys + ["symbol", "dso"], dropna=False)["pct"].mean().reset_index()
    g["rank"] = g.groupby(keys, dropna=False)["pct"].rank(ascending=False, method="first").astype(int)
    return g[g["rank"] <= top].sort_values(keys + ["rank"]).reset_index(drop=True)


# ------------------------------------------------------------------------------------------- report
def plain_number(v: float, digits: int = 3) -> str:
    """A float with `digits` significant digits and no scientific notation: 1.23, 12.3, 123, 1234, 0.0000188.
    Very small or large values (outside 1e-8 to 1e15) and infinities fall back to the short forms."""
    if v == 0:
        return "0"
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    a = abs(v)
    if a < 1e-8 or a >= 1e15:
        return f"{v:.{digits - 1}e}"
    if a >= 10 ** (digits - 1):
        return f"{v:.0f}"
    dec = digits - 1 - math.floor(math.log10(a))
    text = f"{v:.{dec}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def md_table(df: pd.DataFrame, digits: int = 3, max_rows: int = 200) -> str:
    if df is None or df.empty:
        return "_No rows._\n"

    def fmt(v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return ""
        if isinstance(v, float):
            return plain_number(v, digits)
        return str(v).replace("|", "/")

    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.head(max_rows).iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in cols) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_{len(df) - max_rows} more rows not shown; use `a2e results export`._")
    return "\n".join(lines) + "\n"


def cache_reset_warning(store: Store, exp_id: int) -> str:
    """A warning when a measured run may have started with a warm prefix cache. Empty when every run's cache was
    erased. A run stores how its cache was reset: `erase` (fine), `flush` (the erase failed, an unrelated prompt
    replaced the cached prefix), `failed`, or nothing at all (a run made before the reset was checked)."""
    rows = store.query(
        "SELECT coalesce(json_extract(server_json, '$.cache_reset'), 'not recorded') AS how, count(*) AS runs "
        "FROM v_runs WHERE exp_id = ? AND phase = 'measured' AND task_kind != 'bench' AND coalesce(prefix_cache, 1) = 1 GROUP BY how", [exp_id])
    bad = [r for r in rows if r["how"] not in ("erase", "unsupported")]
    if not bad:
        return ""
    detail = "; ".join(f"{r['runs']} run(s): {r['how']}" for r in bad)
    return ("## Warning: prefix cache\n\n"
            "Some runs may have started with prompts left in the prefix cache by an earlier job, which makes prefill "
            "and energy look lower than they are. Affected: " + detail + ". `flush` means the server's cache erase "
            "failed and an unrelated prompt was used instead; `not recorded` means the run was made before this check "
            "existed. Compare prefill, cache hit and energy of such runs with care.\n")


def report_markdown(store: Store, exp_id: int) -> str:
    exp = store.query("SELECT * FROM experiments WHERE exp_id = ?", [exp_id])
    if not exp:
        raise KeyError(f"no experiment {exp_id}")
    e = exp[0]
    env = json.loads(e["env_json"] or "{}")
    parts = [f"# Results for experiment {exp_id}: {e['name']}\n",
             "This report is generated from the results database. It contains measured repeats only.\n"]
    warn = cache_reset_warning(store, exp_id)
    if warn:
        parts.append(warn)
    parts += ["## Experiment\n",
             md_table(pd.DataFrame([{
                 "status": e["status"], "started": _ts(e["started_at"]), "finished": _ts(e["finished_at"]),
                 "suite hash": e["suite_hash"], "a2e commit": (e["git_commit"] or "")[:12],
                 "uncommitted changes": bool(e["git_dirty"]), "hardware": env.get("hardware"),
                 "ram GB": env.get("ram_gb"), "llama-server": env.get("llama_server_version", "")[:60],
                 "governor": env.get("governor"), "cooler": env.get("cooler")}])),
             "## Jobs\n"]
    j = store.df("SELECT status, kind, count(*) AS jobs FROM jobs WHERE exp_id = ? GROUP BY status, kind", [exp_id])
    parts.append(md_table(j))
    sk = store.df("SELECT reason, count(*) AS jobs FROM jobs WHERE exp_id = ? AND status IN ('skipped','failed') "
                  "GROUP BY reason", [exp_id])
    if not sk.empty:
        parts += ["Reasons for skipped or failed jobs:\n", md_table(sk)]
    b = bench_summary(store, exp_id)
    if not b.empty:
        parts += ["## Prefill and decode microbenchmarks (llama-bench)\n",
                  "`tok_s` is tokens per second. For decode (tg), `est_bandwidth_gbs` is tokens per second times "
                  "the model file size, an estimate of the memory bandwidth decode uses.\n", md_table(b)]
        k = kernels(store, exp_id, top=5)
        if not k.empty:
            parts += ["## Top kernels (perf, Pi)\n", "Share of CPU samples per function, top 5 per test.\n",
                      md_table(k)]
    s = summary(store, exp_id)
    if s.empty:
        parts.append("\nNo measured agent or plain runs.\n")
        return "\n".join(parts)
    id_cols = [c for c in DEFAULT_BY if c in s.columns]
    parts += ["## Latency and throughput\n",
              "Times are mean milliseconds per model call. Token rates are per second. `_sd` is the standard "
              "deviation across measured repeats.\n",
              md_table(s[id_cols + [c for c in ("runs", "turns", "prefill_ms", "decode_ms", "tool_wait_ms",
                                                "prefill_tok_s", "decode_tok_s", "decode_tok_s_sd", "cache_hit")
                                    if c in s]]),
              "## Where the time goes\n", "Share of total turn time spent in each stage.\n",
              md_table(stages(store, exp_id))]
    if "success" in s:
        parts += ["## Accuracy\n", "Success is the share of scored cases that passed (BFCL official checker, or "
                  "every step correct for custom chains).\n",
                  md_table(s[id_cols + [c for c in ("n_scored", "success", "success_sd") if c in s]])]
        f = failures(store, exp_id)
        if not f.empty:
            parts += ["## Why tasks failed\n",
                      "Failed tasks per reason. For BFCL the reason is the official checker's error type; for custom "
                      "chains it is `wrong_step`, `wrong_state`, `parse_error` (the reply could not be read), "
                      "`max_turns`, `early_stop` or `server_error`. `tasks_run` is all scored tasks in the group.\n",
                      md_table(f)]
    if "j_per_turn" in s:
        scope = store.df("SELECT DISTINCT power_scope FROM runs WHERE exp_id = ?", [exp_id])
        parts += ["## Energy\n",
                  "Energy is a software estimate with the idle baseline subtracted. Power scope: "
                  + "; ".join(str(x) for x in scope["power_scope"]) + "\n",
                  md_table(s[id_cols + [c for c in ("j_per_turn", "j_per_prompt_token", "j_per_gen_token",
                                                    "tokens_per_joule", "j_per_success", "avg_power_w") if c in s]])]
    if "j_per_turn" not in s:
        scopes = store.df("SELECT DISTINCT power_scope FROM runs WHERE exp_id = ?", [exp_id])
        parts += ["## Energy\n",
                  "No energy was recorded for this experiment (power scope: "
                  + "; ".join(str(x) for x in scopes["power_scope"]) + "). On the Mac `powermetrics` needs root; "
                  "run with `sudo -E env \"PATH=$PATH\" .venv/bin/a2e run <suite>` and set `power = \"required\"` in "
                  "`configs/local.toml` so a run without a power source fails at the start. Energy columns, "
                  "joules per success and the efficiency views are empty without it.\n"]
    parts += ["## Memory and thermals\n",
              md_table(s[id_cols + [c for c in ("peak_rss_mb", "max_temp_c", "throttled_turns") if c in s]])]
    res = resources(store, exp_id)
    if not res.empty:
        parts += ["## Resource envelope\n",
                  "Power, memory and CPU statistics over the 2 Hz samples taken while each run was active. `_min` "
                  "and `_max` columns are the lowest and highest value over the runs of a configuration; the others "
                  "are averages. Power is empty when no power source was available. `thermal_pressure` is the "
                  "worst level seen (Mac). The power scope says what the watts include.\n",
                  md_table(res)]
    return "\n".join(parts)
