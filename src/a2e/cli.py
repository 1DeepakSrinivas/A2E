"""Command line interface. Run `a2e --help` or see docs/USER_GUIDE.md."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import typer

from a2e.settings import CONFIG_DIR, REPO, load_settings

app = typer.Typer(help="A2E: profiling tool-calling LLM workloads on Raspberry Pi 5 and Apple Silicon Macs.",
                  no_args_is_help=True)
results_app = typer.Typer(help="Read results from the database.", no_args_is_help=True)
bfcl_app = typer.Typer(help="BFCL v4 (vendored official checker) without a suite.", no_args_is_help=True)
app.add_typer(results_app, name="results")
app.add_typer(bfcl_app, name="bfcl")

LOCK = CONFIG_DIR / "suites.lock.json"


def _settings(db: Path | None = None, device: str | None = None):
    s = load_settings()
    if db:
        s.db = str(db)
    if device:
        s.device = device
    return s


def _store(db: Path | None):
    from a2e.store import Store
    return Store(db or _settings().db)


def _print_df(df, empty: str = "No rows.") -> None:
    import pandas as pd
    if df is None or df.empty:
        typer.echo(empty)
        return
    with pd.option_context("display.max_columns", None, "display.width", 200, "display.max_colwidth", 60,
                           "display.float_format", lambda v: f"{v:.4g}"):
        typer.echo(df.to_string(index=False))


def format_event(e: dict) -> str | None:
    ev = e.get("event")
    if ev == "plan":
        return f"experiment {e['exp_id']}: {e['todo']} jobs to run now; job states {e['counts']}"
    if ev == "new_experiment":
        return f"new experiment {e['exp_id']}"
    if ev == "resume":
        return f"resuming experiment {e['exp_id']}"
    if ev == "instruments":
        msg = f"instruments: power scope '{e['scope']}'; " + "; ".join(e["notes"])
        if e["scope"] == "none":
            msg = f"WARNING: no energy will be recorded for this run. {msg}"
        return msg
    if ev == "server_start":
        return f"  loading {e['model']} {e['quant']} kv={e['kv_quant']} ..."
    if ev == "server_ready":
        return f"  model loaded in {e['load_time_s']} s; idle power {e['idle_power_w']}"
    if ev == "cooldown":
        return f"  waited {e['waited_s']:.0f} s to cool down (now {e['temp_c']} C)"
    if ev == "job_start":
        return f"[{e['i']}/{e['n']}] job {e['job_id']}: {e['label']}"
    if ev == "run_done":
        sc = f", {e['valid']}/{e['scored']} correct" if e["scored"] else ""
        return f"    repeat {e['repeat']} ({e['phase']}): {e['units']} tasks{sc}"
    if ev == "job_failed":
        return f"    FAILED: {e['error']}"
    if ev == "job_skipped":
        return f"    skipped: {e['reason']}"
    if ev == "finished":
        return f"finished experiment {e['exp_id']}: {e['counts']}" + (" (stopped early)" if e["stopped"] else "")
    return None


# ------------------------------------------------------------------------------------------- setup
@app.command()
def doctor(models: str = typer.Option("", help="comma-separated model names to check (default: all)")) -> None:
    """Check this machine: llama-server, model files, power and thermal tools, database, extras."""
    from a2e.doctor import run_checks
    checks = run_checks(_settings(), [m for m in models.split(",") if m] or None)
    for c in checks:
        mark = "ok  " if c.ok else ("FAIL" if c.critical else "warn")
        typer.echo(f"{mark}  {c.name:42s} {c.detail}")
    if any(c.critical and not c.ok for c in checks):
        raise typer.Exit(1)


# ------------------------------------------------------------------------------------------- suites
@app.command()
def suites() -> None:
    """List suite files with their size."""
    from a2e.suite import expand, load_suite, plan_summary
    s = _settings()
    for p in sorted((CONFIG_DIR / "suites").glob("*.toml")):
        try:
            suite = load_suite(p)
            plan = plan_summary(expand(suite, s))
            typer.echo(f"{p.stem:16s} {plan['runnable']:5d} jobs ({plan['skipped']} skipped), "
                       f"{plan['server_loads']} model loads  {suite.suite.description}")
        except Exception as e:  # noqa: BLE001
            typer.echo(f"{p.stem:16s} ERROR {e}")


@app.command()
def plan(suite: str, device: str = typer.Option(None, help="pi5 or mac; default: all devices"),
         jobs: bool = typer.Option(False, "--jobs", help="list every job")) -> None:
    """Show what a suite will run: job counts per kind and device, model loads, runs, skipped jobs."""
    from a2e.runner import job_label
    from a2e.suite import expand, load_suite, plan_summary
    s = load_suite(suite)
    all_jobs = [j for j in expand(s, _settings()) if not device or j["device"] == device]
    p = plan_summary(all_jobs)
    typer.echo(f"suite {s.suite.name} (hash {s.hash}): {s.suite.description}")
    typer.echo(json.dumps(p, indent=2))
    lock = json.loads(LOCK.read_text()).get(s.suite.name) if LOCK.exists() else None
    if lock and lock["suite_hash"] != s.hash:
        typer.echo("note: this suite differs from configs/suites.lock.json (run `a2e suite-lock` if intended)")
    if jobs:
        for j in all_jobs:
            typer.echo(f"{j['job_key']}  {j['device']:4s} {job_label(j)}" + (f"  [{j['skip_reason']}]"
                                                                               if j["skip_reason"] else ""))


@app.command()
def run(suite: str,
        device: str = typer.Option(None, help="override device detection (pi5 or mac)"),
        db: Path = typer.Option(None, help="results database (default from configs/local.toml)"),
        new: bool = typer.Option(False, help="start a new experiment instead of resuming"),
        limit: int = typer.Option(None, help="run at most this many jobs, then stop (resume later)"),
        retry_failed: bool = typer.Option(False, help="also re-run jobs that failed before")) -> None:
    """Run (or resume) a suite on this machine and store everything in the results database."""
    from a2e.hardware import PowerUnavailable
    from a2e.runner import Runner
    from a2e.suite import ConfigError
    try:
        r = Runner(suite, _settings(db, device), log=lambda e: (m := format_event(e)) and typer.echo(m))
    except (ConfigError, RuntimeError, KeyError) as e:
        typer.echo(f"error: {e}")
        raise typer.Exit(2)
    try:
        out = r.run(new=new, limit=limit, retry_failed=retry_failed)
    except PowerUnavailable as e:
        typer.echo(f"error: {e}")
        raise typer.Exit(2)
    except KeyboardInterrupt:
        typer.echo("interrupted; run the same command again to resume")
        raise typer.Exit(130)
    raise typer.Exit(0 if not out["counts"].get("failed") else 1)


@app.command("suite-lock")
def suite_lock(check: bool = typer.Option(False, help="only verify; exit 1 if any suite changed")) -> None:
    """Record (or verify) each suite's hash and job list in configs/suites.lock.json."""
    from a2e.suite import expand, load_suite, lock_entry
    s = load_settings(local=REPO / "configs" / "__no_local__.toml")  # machine-independent
    current = {}
    for p in sorted((CONFIG_DIR / "suites").glob("*.toml")):
        suite = load_suite(p)
        current[suite.suite.name] = lock_entry(suite, expand(suite, s))
    if check:
        old = json.loads(LOCK.read_text()) if LOCK.exists() else {}
        changed = sorted(k for k in set(old) | set(current) if old.get(k) != current.get(k))
        typer.echo("suites match the lock file" if not changed else f"changed suites: {', '.join(changed)}")
        raise typer.Exit(1 if changed else 0)
    LOCK.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
    typer.echo(f"wrote {LOCK} ({len(current)} suites)")


# ------------------------------------------------------------------------------------------- results
@results_app.command("list")
def results_list(db: Path = typer.Option(None)) -> None:
    """Experiments in the database with their job counts."""
    from a2e import results as R
    _print_df(R.experiments(_store(db)), "No experiments yet.")


@results_app.command("jobs")
def results_jobs(exp: int, db: Path = typer.Option(None), status: str = typer.Option(None)) -> None:
    """Jobs of one experiment (optionally only one status)."""
    from a2e import results as R
    df = R.jobs(_store(db), exp).drop(columns=["config_json"])
    _print_df(df if not status else df[df.status == status])


@results_app.command("summary")
def results_summary(exp: list[int] = typer.Option(None, help="experiment id(s); default all"),
                    by: str = typer.Option("", help="comma-separated grouping fields"),
                    db: Path = typer.Option(None), csv: Path = typer.Option(None, help="also write to this csv")) -> None:
    """Latency, throughput, stage shares, cache hits, energy and accuracy per configuration."""
    from a2e import results as R
    df = R.summary(_store(db), exp or None, [b for b in by.split(",") if b] or None)
    _print_df(df, "No measured runs.")
    if csv is not None and not df.empty:
        df.to_csv(csv, index=False)


@results_app.command("stages")
def results_stages(exp: list[int] = typer.Option(None), db: Path = typer.Option(None),
                   by: str = typer.Option("")) -> None:
    """Share of turn time in prefill, decode, parse, validation, tools, template and harness."""
    from a2e import results as R
    _print_df(R.stages(_store(db), exp or None, [b for b in by.split(",") if b] or None))


@results_app.command("resources")
def results_resources(exp: list[int] = typer.Option(None, help="experiment id(s); default all"),
                      by: str = typer.Option("", help="comma-separated grouping fields"),
                      db: Path = typer.Option(None), csv: Path = typer.Option(None, help="also write to this csv")) -> None:
    """Power (min, median, p95, max), memory, CPU and thermal pressure per configuration."""
    from a2e import results as R
    df = R.resources(_store(db), exp or None, [b for b in by.split(",") if b] or None)
    _print_df(df, "No measured runs.")
    if csv is not None and not df.empty:
        df.to_csv(csv, index=False)


@results_app.command("task-resources")
def results_task_resources(exp: list[int] = typer.Option(None, help="experiment id(s); default all"),
                           db: Path = typer.Option(None),
                           csv: Path = typer.Option(None, help="also write to this csv")) -> None:
    """Energy, power, peak memory and time of every task."""
    from a2e import results as R
    df = R.case_resources(_store(db), exp or None)
    _print_df(df, "No measured runs.")
    if csv is not None and not df.empty:
        df.to_csv(csv, index=False)


@results_app.command("stability")
def results_stability(exp: list[int] = typer.Option(None), db: Path = typer.Option(None)) -> None:
    """Decode speed drift over long sessions and the first task where it drops (thermal knee)."""
    from a2e import results as R
    _print_df(R.stability(_store(db), exp or None))


@results_app.command("bench")
def results_bench(exp: list[int] = typer.Option(None), db: Path = typer.Option(None)) -> None:
    """llama-bench prefill/decode throughput and estimated decode memory bandwidth."""
    from a2e import results as R
    _print_df(R.bench_summary(_store(db), exp or None), "No bench runs.")


@results_app.command("kernels")
def results_kernels(exp: list[int] = typer.Option(None), top: int = 10, db: Path = typer.Option(None)) -> None:
    """Top kernels from perf (Pi) per configuration and test."""
    from a2e import results as R
    _print_df(R.kernels(_store(db), exp or None, top), "No kernel profiles.")


@results_app.command("timeline")
def results_timeline(run_id: int, db: Path = typer.Option(None), csv: Path = typer.Option(None)) -> None:
    """Resource samples of one run (power, temperature, frequency, memory, throttling) over time."""
    from a2e import results as R
    df = R.timeline(_store(db), run_id)
    _print_df(df)
    if csv is not None:
        df.to_csv(csv, index=False)


@results_app.command("report")
def results_report(exp: int, db: Path = typer.Option(None), out: Path = typer.Option(None)) -> None:
    """Markdown report for one experiment (printed, or written to --out)."""
    from a2e import results as R
    md = R.report_markdown(_store(db), exp)
    if out:
        out.write_text(md)
        typer.echo(f"wrote {out}")
    else:
        typer.echo(md)


@results_app.command("export")
def results_export(out: Path, db: Path = typer.Option(None), fmt: str = typer.Option("parquet")) -> None:
    """Write every table and analysis view as parquet or csv files (for pandas, R, spreadsheets)."""
    for f in _store(db).export(out, fmt):
        typer.echo(str(f))


# ------------------------------------------------------------------------------------------- tests & tui
@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def test(ctx: typer.Context, group: str = typer.Argument("quick"),
         list_groups: bool = typer.Option(False, "--list", help="show the groups")) -> None:
    """Run a named test group from configs/tests.toml. Extra arguments go to pytest."""
    from a2e.testgroups import load_groups
    groups = load_groups()
    if list_groups:
        for g in groups.values():
            typer.echo(f"{g.name:12s} {g.description}")
        return
    if group not in groups:
        typer.echo(f"unknown group {group!r}; choose from {', '.join(groups)}")
        raise typer.Exit(2)
    g = groups[group]
    if g.missing_env():
        typer.echo(f"note: {', '.join(g.missing_env())} not set, so these tests will skip")
    raise typer.Exit(subprocess.call(g.command(ctx.args), cwd=REPO))


@app.command()
def tui() -> None:
    """Open the terminal UI (needs `uv sync --extra tui`)."""
    from a2e.tui import A2EApp
    A2EApp().run()


# ------------------------------------------------------------------------------------------- bfcl (no suite)
@bfcl_app.command("list")
def bfcl_list() -> None:
    """Categories in scope with their case counts."""
    from a2e.bfcl.load import ALL_CATEGORIES, load_cases
    for c in ALL_CATEGORIES:
        typer.echo(f"{c:24s} {len(load_cases(c)):5d}")


@bfcl_app.command("run")
def bfcl_run(
    url: str = typer.Option("http://127.0.0.1:8080", help="running llama-server"),
    category: list[str] = typer.Option(..., help="repeat for several categories"),
    model: str = typer.Option(...), quant: str = typer.Option(...), device: str = typer.Option(...),
    kv_quant: str = "f16", n: int = typer.Option(None, help="cases per category (fixed-seed stratified); all if unset"),
    seed: int = 0, grammar: bool = False, no_prefix_cache: bool = False, n_predict: int = 256,
    db: Path = Path("a2e.db"), raw: Path = Path("bfcl_raw.jsonl"),
) -> None:
    """Run BFCL categories against an already running llama-server (quick checks; suites are preferred)."""
    from a2e.bfcl.runner import run_category, safe_token_counter
    from a2e.client import Client
    from a2e.config import RunConfig
    from a2e.store import Store
    cfg = RunConfig(model=model, quant=quant, kv_quant=kv_quant, device=device, grammar=grammar,
                    prefix_cache=not no_prefix_cache)
    client, store = Client(url), Store(db)
    count = safe_token_counter(client)
    for cat in category:
        run_id = store.insert("runs", {"config_json": cfg.model_dump_json(), "device": device,
                                       "phase": f"bfcl:{cat}", "started_at": time.time()})
        outs = run_category(client, cat, store, run_id, n=n, seed=seed, raw_path=raw, grammar=grammar,
                            cache_prompt=not no_prefix_cache, n_predict=n_predict, count_tokens=count)
        ok = sum(o.result.valid for o in outs)
        typer.echo(f"{cat}: {ok}/{len(outs)} valid ({100 * ok / max(len(outs), 1):.1f}%)  run_id={run_id}")


@bfcl_app.command("rescore")
def bfcl_rescore(raw: Path = Path("bfcl_raw.jsonl")) -> None:
    """Re-score recorded raw outputs with the vendored checkers (no model needed)."""
    from collections import defaultdict

    from a2e.bfcl.runner import rescore
    tally = defaultdict(lambda: [0, 0])
    for r in rescore(raw):
        tally[r.category][0] += r.valid
        tally[r.category][1] += 1
    for cat, (ok, total) in tally.items():
        typer.echo(f"{cat}: {ok}/{total} valid ({100 * ok / total:.1f}%)")


if __name__ == "__main__":
    app()
