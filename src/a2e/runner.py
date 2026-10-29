"""Suite executor.

`run_suite` expands a suite into jobs, records them in the store, and runs every pending job for this device:

  experiment  one per (suite name, device, suite hash); re-running the same suite resumes it
  job         one measured configuration (see a2e.suite); status pending -> running -> done | failed | skipped
  run         one repeat of a job (warm-up or measured); every turn, case and resource sample hangs off a run

Crash safety: a job only becomes `done` after all its runs are written. If the process dies, the next start
marks that job's partial runs `aborted` (the analysis views ignore them) and runs the job again from the start.
"""
from __future__ import annotations

import json
import socket
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from a2e import env as envmod
from a2e.bench import bench_args, perf_available, run_bench
from a2e.agent.harness import Harness
from a2e.agent.loop import AgentLoop, TurnRecord
from a2e.agent.schema_gen import SchemaTooLarge, pad_tools
from a2e.client import CacheResetError, Client
from a2e.config import RunConfig
from a2e.hardware import Instruments, make_instruments, read_governor, set_governor, wait_for_cooldown
from a2e.metrics.power_base import integrate_energy
from a2e.metrics.sampler import Sampler, measure_idle_power
from a2e.plain import plain_prompts, run_plain
from a2e.server import THINKING_OFF, LlamaServer, build_args
from a2e.settings import Settings, resolve_device
from a2e.store import Store
from a2e.suite import Suite, expand, load_suite, server_key
from a2e.tasks.custom import FAMILY_TOOLS, generate
from a2e.tools.builtin import build_registry, make_corpus

Log = Callable[[dict], None]


class Stop(Exception):
    """Raised by an executor when its ctx carries a stop event that is set. The runner itself only checks its stop
    event between jobs, so the TUI's stop button lets the current job finish; Ctrl+C stops at once."""


@dataclass
class Unit:
    """One task's worth of results inside a run: its model calls and its case-level score."""
    task_id: str
    category: str
    turns: list[TurnRecord]
    valid: bool | None = None
    error_type: str | None = None
    status: str = "done"
    n_user_turns: int = 1
    schema_tokens: int | None = None
    extra: dict = field(default_factory=dict)
    user_turns: list[int] | None = None
    correct: list[bool | None] | None = None  # per turn, where the task defines it
    raw: str | None = None  # BFCL raw-output line for rescoring


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _safe_counter(client):
    try:
        client.tokenize("probe")
        return client.tokenize, "server"
    except Exception:  # noqa: BLE001
        return (lambda t: max(len(t) // 4, 1)), "approx_chars_div_4"


# ------------------------------------------------------------------------------------------- task kinds
def exec_plain(client, job: dict, ctx: dict) -> list[Unit]:
    out = []
    for tid, prompt in plain_prompts(job["n"], job["seed"]):
        _check_stop(ctx)
        turn = run_plain(client, prompt, job["decode_len"], job["prefix_cache"])
        out.append(Unit(tid, "plain", [turn], status="server_error" if turn.error else "done"))
    return out


def exec_custom(client, job: dict, ctx: dict) -> list[Unit]:
    from a2e.tasks.custom import FAMILY_ENV
    from a2e.tasks.envs import build_task_registry, family_core_specs
    from a2e.tools.bfcl_tools import distractor_pool
    base = Path(ctx["work_dir"])
    fam = job["family"]
    meta = make_corpus(base / "corpus_meta", n=30, seed=job["seed"]) if fam == "retrieval" else None
    count, how = _safe_counter(client)
    pool = list(distractor_pool()) if job.get("distractors", "bfcl") == "bfcl" else None
    specs, n_tok = pad_tools(family_core_specs(fam), job["schema_len"], count, seed=job["seed"], pool=pool)
    out = []
    for i, task in enumerate(generate(fam, job["n"], job["depth"], job["n_args"], job["seed"], meta)):
        _check_stop(ctx)
        # a fresh tool state per task (notes folder, BFCL instances, web URL list), so tasks cannot see each other
        reg = build_task_registry(task, base / f"task{i}", job["seed"], ctx.get("web"))
        trace = AgentLoop(client, reg, specs, grammar=job["grammar"], n_predict=job["decode_len"],
                          cache_prompt=job["prefix_cache"], harness=Harness.from_job(job)
                          ).run(task.prompt, max_turns=ctx["max_turns"], expected_calls=len(task.steps) + 1)
        sc = trace.score(task.steps)
        if task.state_check is not None:
            sc.state_ok = bool(task.state_check(reg))
        correct = [sc.steps[k].ok if k < len(sc.steps) else None for k in range(len(trace.turns))]
        extra = {"n_correct": sc.n_correct, "n_steps": len(task.steps), "finished": sc.finished,
                 "token_count": how, "tools": FAMILY_ENV[fam], "distractors": job.get("distractors", "bfcl")}
        if task.state_check is not None:
            extra["state_ok"] = sc.state_ok
        if task.env.get("scenario"):
            extra["scenario"] = task.env["scenario"]
        if getattr(reg, "web", None) is not None:
            extra.update({"web_returned_urls": len(reg.web.returned), "web_errors": reg.web.errors[:5],
                          "question": task.env.get("question")})
        status = trace.status
        err = None if sc.success else ("wrong_state" if sc.state_ok is False and status == "final_answer"
                                       else status if status != "final_answer" else "wrong_step")
        out.append(Unit(task.task_id, f"custom:{fam}", trace.turns, sc.success, err, status, 1, n_tok, extra,
                        correct=correct))
    return out


def exec_bfcl(client, job: dict, ctx: dict) -> list[Unit]:
    from a2e.bfcl.load import load_subset
    from a2e.bfcl.runner import raw_record, run_case
    count, how = _safe_counter(client)
    out = []
    for case in load_subset(job["category"], job["n"], job["seed"]):
        _check_stop(ctx)
        o = run_case(client, case, grammar=job["grammar"], cache_prompt=job["prefix_cache"],
                     n_predict=job["decode_len"], max_steps=ctx["max_steps"], count_tokens=count,
                     harness=Harness.from_job(job))
        extra = {k: v for k, v in o.result.extra.items() if k != "schema_tokens"}
        out.append(Unit(case.id, case.category, o.turns, o.result.valid, o.result.error_type, o.status,
                        len(case.turns), o.result.extra.get("schema_tokens"), {**extra, "token_count": how},
                        user_turns=o.user_turn_of or None, raw=raw_record(o, -1, Harness.from_job(job))))
    return out


EXECUTORS = {"plain": exec_plain, "custom": exec_custom, "bfcl": exec_bfcl}


def _check_stop(ctx: dict) -> None:
    ev = ctx.get("stop")
    if ev is not None and ev.is_set():
        raise Stop()


# ------------------------------------------------------------------------------------------- metrics
def _window_samples(samples: list[tuple], t0: float, t1: float) -> list[tuple]:
    inside = [s for s in samples if t0 <= s[0] <= t1]
    if inside:
        return inside
    before = [s for s in samples if s[0] <= t1]
    return before[-1:]  # a short turn between two samples: use the latest reading


def turn_row(turn: TurnRecord, run_id: int, unit: Unit, step: int, trace: list[tuple[float, float]],
             samples: list[tuple], idle_w: float | None) -> dict:
    row = turn.row(run_id, unit.task_id)
    row.update({"step": step, "template_ms": turn.template_ms, "n_calls": len(turn.calls), "error": turn.error,
                "user_turn": unit.user_turns[step] if unit.user_turns and step < len(unit.user_turns) else None})
    if unit.correct and step < len(unit.correct) and unit.correct[step] is not None:
        row["correct"] = int(unit.correct[step])
    w = turn.windows
    if not w:
        return row
    start = w["prefill"][0] - turn.template_ms / 1000
    end = w["tool"][1]
    idle = idle_w or 0.0
    if trace and len(trace) >= 2:
        row["energy_j"] = integrate_energy(trace, start, end, idle)
        row["energy_prefill_j"] = integrate_energy(trace, *w["prefill"], idle)
        row["energy_decode_j"] = integrate_energy(trace, *w["decode"], idle)
        row["energy_tool_j"] = integrate_energy(trace, *w["tool"], idle)
        # the harness's own code: templating, parsing and validating (the model and the tools are separate)
        row["energy_harness_j"] = (integrate_energy(trace, start, w["prefill"][0], idle)
                                   + integrate_energy(trace, *w["parse"], idle)
                                   + integrate_energy(trace, *w["validate"], idle))
        if end > start:
            row["avg_power_w"] = integrate_energy(trace, start, end) / (end - start)
    ws = _window_samples(samples, start, end)
    if ws:
        rss = [s[1] for s in ws if s[1] is not None]
        cpu = [s[2] for s in ws if s[2] is not None]
        temp = [s[3] for s in ws if s[3] is not None]
        freq = [s[4] for s in ws if s[4] is not None]
        thr = [s[6] for s in ws if len(s) > 6 and s[6] is not None]
        row.update({"peak_rss_mb": max(rss) if rss else None, "avg_cpu_pct": sum(cpu) / len(cpu) if cpu else None,
                    "max_temp_c": max(temp) if temp else None,
                    "mean_freq_mhz": sum(freq) / len(freq) if freq else None,
                    "throttled": int(any(thr)) if thr else None})
    return row


# ------------------------------------------------------------------------------------------- runner
@dataclass
class RunnerHooks:
    """Injection points for tests: replace the server, the client or the instruments."""
    start_server: Callable[[dict, Settings], tuple[object, str, int | None, float, dict]] | None = None
    make_client: Callable[[str], object] | None = None
    instruments: Callable[[str, Settings], Instruments] | None = None


class Runner:
    def __init__(self, suite: Suite | str | Path, settings: Settings, *, device: str | None = None,
                 store: Store | None = None, log: Log | None = None, stop: threading.Event | None = None,
                 hooks: RunnerHooks | None = None):
        self.suite = suite if isinstance(suite, Suite) else load_suite(suite)
        self.settings = settings
        self.device = device or resolve_device(settings)
        self.store = store or Store(settings.db)
        self.log = log or (lambda e: None)
        self.stop = stop or threading.Event()
        self.hooks = hooks or RunnerHooks()
        self.jobs = [j for j in expand(self.suite, settings) if j["device"] == self.device]
        self.exp_id: int | None = None
        self._server = None
        self._server_key = None
        self._server_info: dict = {}

    # ---- experiment & jobs
    @property
    def name(self) -> str:
        return f"{self.suite.suite.name}@{self.device}"

    def open_experiment(self, new: bool = False) -> int:
        st = self.store
        row = None if new else st.query(
            "SELECT exp_id FROM experiments WHERE name = ? AND suite_hash = ? ORDER BY exp_id DESC LIMIT 1",
            [self.name, self.suite.hash])
        if row:
            self.exp_id = row[0]["exp_id"]
            st.update("experiments", "exp_id", self.exp_id, {"status": "running", "finished_at": None})
            self.log({"event": "resume", "exp_id": self.exp_id})
        else:
            env = envmod.capture(self.settings)
            self.exp_id = st.insert("experiments", {
                "name": self.name, "suite_hash": self.suite.hash, "suite_json": json.dumps(self.suite.canonical()),
                "env_json": json.dumps(env), "git_commit": env["a2e_commit"], "git_dirty": env["a2e_dirty"],
                "status": "running", "started_at": time.time()})
            self.log({"event": "new_experiment", "exp_id": self.exp_id})
        known = {r["job_key"] for r in st.query("SELECT job_key FROM jobs WHERE exp_id = ?", [self.exp_id])}
        with st.transaction():
            for j in self.jobs:
                if j["job_key"] in known:
                    continue
                cfg = {k: v for k, v in j.items() if k not in ("job_key", "skip_reason")}
                st.insert("jobs", {"exp_id": self.exp_id, "job_key": j["job_key"], "kind": j["kind"],
                                   "config_json": json.dumps(cfg, sort_keys=True), "device": j["device"],
                                   "model": j["model"], "quant": j["quant"], "kv_quant": j["kv_quant"],
                                   "status": "skipped" if j["skip_reason"] else "pending",
                                   "reason": j["skip_reason"]})
            # recover from a crash: a job's runs only count once the whole job is done, so every run of an
            # unfinished job (even a completed repeat) is aborted, and the job goes back to pending
            st.execute("UPDATE jobs SET status = 'pending' WHERE status = 'running' AND exp_id = ?", [self.exp_id])
            st.execute("UPDATE runs SET status = 'aborted' WHERE exp_id = ? AND status != 'aborted' AND job_id IN "
                       "(SELECT job_id FROM jobs WHERE exp_id = ? AND status != 'done')", [self.exp_id, self.exp_id])
        return self.exp_id

    def pending_jobs(self, retry_failed: bool = False) -> list[dict]:
        states = ("pending", "failed") if retry_failed else ("pending",)
        rows = self.store.query(f"SELECT job_id, job_key FROM jobs WHERE exp_id = ? AND status IN "
                                f"({','.join('?' * len(states))})", [self.exp_id, *states])
        ids = {r["job_key"]: r["job_id"] for r in rows}
        return [{**j, "job_id": ids[j["job_key"]]} for j in self.jobs if j["job_key"] in ids]

    def counts(self) -> dict:
        rows = self.store.query("SELECT status, count(*) AS n FROM jobs WHERE exp_id = ? GROUP BY status",
                                [self.exp_id])
        return {r["status"]: r["n"] for r in rows}

    # ---- server
    def _start_server(self, job: dict):
        if self.hooks.start_server:
            return self.hooks.start_server(job, self.settings)
        model_path = self.settings.model_path(job["model"], job["quant"])
        if not model_path.exists():
            raise FileNotFoundError(f"model file missing: {model_path}")
        if not self.settings.llama_server:
            raise RuntimeError("llama_server is not set in configs/local.toml")
        port = _free_port()
        cfg = RunConfig(model=job["model"], quant=job["quant"], kv_quant=job["kv_quant"], device=job["device"])
        log_dir = Path(self.settings.work_dir) / f"exp{self.exp_id}"
        log_dir.mkdir(parents=True, exist_ok=True)
        slot_dir = log_dir / "slots"
        slot_dir.mkdir(exist_ok=True)
        args = build_args(cfg, self.settings.llama_server, str(model_path), port=port, ctx=job["ctx"],
                          threads=job["threads"] or self.settings.threads, flash_attn=job["flash_attn"],
                          slot_save_path=str(slot_dir), thinking=job.get("thinking", "default"))
        srv = LlamaServer(args, port=port, log_path=log_dir / f"llama-server-{job['model']}-{job['quant']}.log")
        load_s = srv.start(timeout=900)
        info = {"args": args, "model_file": envmod.file_fingerprint(model_path)}
        return srv, srv.url, srv.pid, load_s, info

    def _ensure_server(self, job: dict, inst: Instruments):
        key = server_key(job)
        if key == self._server_key and self._server is not None:
            return
        self._stop_server()
        self.log({"event": "server_start", "model": job["model"], "quant": job["quant"], "kv_quant": job["kv_quant"]})
        srv, url, pid, load_s, info = self._start_server(job)
        self._server, self._server_key = srv, key
        client = (self.hooks.make_client or Client)(url)
        if job.get("thinking") == "off":
            client.template_kwargs = dict(THINKING_OFF)
        idle = None
        if inst.power and self.suite.run.idle_seconds > 0:
            idle = measure_idle_power(inst.power, self.suite.run.idle_seconds)
        self._server_info = {"url": url, "pid": pid, "load_time_s": load_s, "idle_power_w": idle, "client": client,
                             **info}
        self.log({"event": "server_ready", "load_time_s": round(load_s, 2), "idle_power_w": idle})

    def _stop_server(self) -> None:
        if self._server is not None and hasattr(self._server, "stop"):
            self._server.stop()
        self._server, self._server_key = None, None

    # ---- one job
    def run_job(self, job: dict, inst: Instruments) -> None:
        st = self.store
        st.update("jobs", "job_id", job["job_id"], {"status": "running", "started_at": time.time(), "error": None,
                                                     "attempts": (st.scalar("SELECT attempts FROM jobs WHERE job_id=?",
                                                                            [job["job_id"]]) or 0) + 1})
        # runs left by an earlier, unfinished attempt of this job never count
        st.execute("UPDATE runs SET status = 'aborted' WHERE job_id = ? AND status != 'aborted'", [job["job_id"]])
        if job.get("governor"):
            set_governor(job["governor"])
        self._ensure_server(job, inst)
        info = self._server_info
        client = info["client"]
        waited, temp = wait_for_cooldown(inst.thermal, self.suite.run.cooldown_c, self.suite.run.cooldown_timeout_s)
        if waited:
            self.log({"event": "cooldown", "waited_s": waited, "temp_c": temp})
        cfg_json = json.dumps({k: v for k, v in job.items() if k not in ("job_id", "skip_reason")}, sort_keys=True)
        server_json = json.dumps({k: v for k, v in info.items() if k not in ("client",)} |
                                 {"cooldown_wait_s": waited, "governor_actual": read_governor(),
                                  "cooler": self.settings.cooler, "instrument_notes": inst.notes}, default=str)
        total = job["warmup"] + job["measured"]
        for r in range(total):
            phase = "warmup" if r < job["warmup"] else "measured"
            self.log({"event": "run_start", "job_id": job["job_id"], "repeat": r, "phase": phase})
            # every repeat starts with a cold prefix cache, and how that was done is stored with the run
            reset = getattr(client, "reset_cache", None)
            try:
                how = (reset() or "unknown") if reset else "unsupported"  # stubs may have no such method
            except CacheResetError as e:
                how = "failed"
                self.log({"event": "cache_reset", "how": how, "error": str(e)})
                if job.get("prefix_cache", True):
                    raise RuntimeError(f"no cold prefix cache: {e}") from e
            if how not in ("erase", "unknown", "unsupported"):
                self.log({"event": "cache_reset", "how": how})
            run_id = st.insert("runs", {
                "exp_id": self.exp_id, "job_id": job["job_id"], "config_json": cfg_json, "device": job["device"],
                "phase": phase, "repeat_idx": r, "task_kind": job["kind"], "status": "running",
                "started_at": time.time(), "idle_power_w": info["idle_power_w"], "power_scope": inst.scope,
                "load_time_s": info["load_time_s"],
                "server_json": json.dumps({**json.loads(server_json), "cache_reset": how})})
            sampler = Sampler(info["pid"], inst.power, inst.thermal, hz=self.suite.run.sample_hz) if info["pid"] \
                else None
            t_run0 = time.perf_counter()
            if sampler:
                sampler.start()
            work = Path(self.settings.work_dir) / f"exp{self.exp_id}" / job["job_key"] / f"r{r}"
            try:
                units = EXECUTORS[job["kind"]](client, job, {"work_dir": work, "web": self.settings.web,
                                                             "max_turns": self.suite.run.max_turns,
                                                             "max_steps": self.suite.run.max_steps})
            finally:
                samples = sampler.stop() if sampler else []
            self._write_run(run_id, units, samples, t_run0, info["idle_power_w"])
            ok = [u.valid for u in units if u.valid is not None]
            self.log({"event": "run_done", "job_id": job["job_id"], "repeat": r, "phase": phase, "units": len(units),
                      "valid": sum(ok), "scored": len(ok)})
        st.update("jobs", "job_id", job["job_id"], {"status": "done", "finished_at": time.time()})

    def run_bench_job(self, job: dict, inst: Instruments) -> None:
        """llama-bench prefill/decode tests (and perf top kernels on the Pi). No llama-server involved."""
        st = self.store
        st.update("jobs", "job_id", job["job_id"], {"status": "running", "started_at": time.time(), "error": None,
                                                     "attempts": (st.scalar("SELECT attempts FROM jobs WHERE job_id=?",
                                                                            [job["job_id"]]) or 0) + 1})
        st.execute("UPDATE runs SET status = 'aborted' WHERE job_id = ? AND status != 'aborted'", [job["job_id"]])
        if job.get("governor"):
            set_governor(job["governor"])
        self._stop_server()  # llama-bench loads its own copy of the model; two copies may not fit on the Pi
        binary = self.settings.bench_binary()
        if not binary or not Path(binary).exists():
            raise FileNotFoundError("llama-bench not found: set [paths] llama_bench in configs/local.toml")
        model_path = self.settings.model_path(job["model"], job["quant"])
        if not model_path.exists():
            raise FileNotFoundError(f"model file missing: {model_path}")
        notes = list(inst.notes)
        perf = None
        if job["perf"]:
            if job["device"] != "pi5":
                notes.append("kernel profiling is only implemented on the Pi (Linux perf)")
            else:
                perf = perf_available(self.settings.perf)
                if not perf:
                    notes.append("perf not found: no kernel profile")
        waited, temp = wait_for_cooldown(inst.thermal, self.suite.run.cooldown_c, self.suite.run.cooldown_timeout_s)
        cfg_json = json.dumps({k: v for k, v in job.items() if k not in ("job_id", "skip_reason")}, sort_keys=True)
        tests = [("pp", n, 0) for n in job["n_prompt"]] + [("tg", 0, n) for n in job["n_gen"]]
        for r in range(job["warmup"] + job["measured"]):
            phase = "warmup" if r < job["warmup"] else "measured"
            server_json = json.dumps({"bench": binary, "model_file": envmod.file_fingerprint(model_path),
                                      "cooldown_wait_s": waited, "governor_actual": read_governor(),
                                      "cooler": self.settings.cooler, "instrument_notes": notes}, default=str)
            run_id = st.insert("runs", {
                "exp_id": self.exp_id, "job_id": job["job_id"], "config_json": cfg_json, "device": job["device"],
                "phase": phase, "repeat_idx": r, "task_kind": "bench", "status": "running", "started_at": time.time(),
                "power_scope": inst.scope, "server_json": server_json})
            t_run0 = time.perf_counter()
            bench_rows, kernel_rows, samples = [], [], []
            for test, p, g in tests:
                holder: dict = {}

                def on_start(pid, holder=holder):
                    holder["s"] = Sampler(pid, inst.power, inst.thermal, hz=self.suite.run.sample_hz).start()

                args = bench_args(binary, str(model_path), n_prompt=p, n_gen=g, reps=job["reps"],
                                  device=job["device"], kv_quant=job["kv_quant"], flash_attn=job["flash_attn"],
                                  threads=job["threads"] or self.settings.threads)
                try:
                    results, kernels, note = run_bench(args, test, p, g, perf=perf, on_start=on_start)
                finally:
                    if "s" in holder:
                        samples += holder["s"].stop()
                if note:
                    notes.append(note)
                bench_rows += [{"run_id": run_id, "test": b.test, "n_prompt": b.n_prompt, "n_gen": b.n_gen,
                                "n_threads": b.n_threads, "avg_tok_s": b.avg_tok_s, "sd_tok_s": b.sd_tok_s,
                                "avg_ms": b.avg_ms, "raw_json": json.dumps(b.raw)} for b in results]
                kernel_rows += [{"run_id": run_id, "test": test, "n_tokens": p or g, **k} for k in kernels]
            with st.transaction():
                st.insert_many("bench", bench_rows)
                st.insert_many("kernels", kernel_rows)
                st.insert_many("samples", [{"run_id": run_id, "t": x[0] - t_run0, "rss_mb": x[1], "cpu_pct": x[2],
                                            "temp_c": x[3], "freq_mhz": x[4], "power_w": x[5],
                                            "throttled": None if x[6] is None else int(x[6]), "pressure": x[7]}
                                           for x in samples])
                st.update("runs", "run_id", run_id, {"status": "complete", "finished_at": time.time(),
                                                     "server_json": json.dumps({**json.loads(server_json),
                                                                                "instrument_notes": notes})})
            self.log({"event": "run_done", "job_id": job["job_id"], "repeat": r, "phase": phase,
                      "units": len(bench_rows), "valid": 0, "scored": 0})
        st.update("jobs", "job_id", job["job_id"], {"status": "done", "finished_at": time.time()})

    def _write_run(self, run_id: int, units: list[Unit], samples: list[tuple], t_run0: float,
                   idle_w: float | None) -> None:
        st = self.store
        trace = [(s[0], s[5]) for s in samples if s[5] is not None]
        raw_lines = []
        with st.transaction():
            for u in units:
                for step, turn in enumerate(u.turns):
                    st.insert("turns", turn_row(turn, run_id, u, step, trace, samples, idle_w))
                st.insert("cases", {"run_id": run_id, "case_id": u.task_id, "category": u.category,
                                    "valid": None if u.valid is None else int(u.valid), "error_type": u.error_type,
                                    "status": u.status, "n_user_turns": u.n_user_turns, "n_model_calls": len(u.turns),
                                    "schema_tokens": u.schema_tokens, "wall_ms": sum(t.wall_ms for t in u.turns),
                                    "extra_json": json.dumps(u.extra)})
                if u.raw:
                    rec = json.loads(u.raw)
                    rec["run_id"] = run_id
                    raw_lines.append(json.dumps(rec))
            st.insert_many("samples", [{"run_id": run_id, "t": s[0] - t_run0, "rss_mb": s[1], "cpu_pct": s[2],
                                        "temp_c": s[3], "freq_mhz": s[4], "power_w": s[5],
                                        "throttled": None if len(s) < 7 or s[6] is None else int(s[6]),
                                        "pressure": s[7] if len(s) > 7 else None} for s in samples])
            st.update("runs", "run_id", run_id, {"status": "complete", "finished_at": time.time()})
        if raw_lines:
            raw = Path(self.settings.raw_dir) / f"exp{self.exp_id}.jsonl"
            raw.parent.mkdir(parents=True, exist_ok=True)
            with open(raw, "a") as f:
                f.write("\n".join(raw_lines) + "\n")

    # ---- the whole suite
    def run(self, *, new: bool = False, limit: int | None = None, retry_failed: bool = False) -> dict:
        self.open_experiment(new=new)
        todo = self.pending_jobs(retry_failed)
        if limit is not None:
            todo = todo[:limit]
        self.log({"event": "plan", "exp_id": self.exp_id, "todo": len(todo), "counts": self.counts()})
        inst = (self.hooks.instruments or make_instruments)(self.device, self.settings)
        if inst.notes:
            self.log({"event": "instruments", "scope": inst.scope, "notes": inst.notes})
        original_gov = read_governor() if self.device == "pi5" else None
        stopped = False
        try:
            for i, job in enumerate(todo, 1):
                if self.stop.is_set():
                    stopped = True
                    break
                self.log({"event": "job_start", "i": i, "n": len(todo), "job_id": job["job_id"], "kind": job["kind"],
                          "label": job_label(job)})
                try:
                    (self.run_bench_job if job["kind"] == "bench" else self.run_job)(job, inst)
                    self.log({"event": "job_done", "i": i, "n": len(todo), "job_id": job["job_id"]})
                except (Stop, KeyboardInterrupt):
                    self._abort_job(job, "pending", "stopped by user")
                    stopped = True
                    break
                except SchemaTooLarge as e:
                    self._abort_job(job, "skipped", f"not run: {e}")
                    self.log({"event": "job_skipped", "job_id": job["job_id"], "reason": str(e)})
                except Exception as e:  # noqa: BLE001 - one bad job must not stop the suite
                    self._abort_job(job, "failed", f"{type(e).__name__}: {e}", traceback.format_exc(limit=8))
                    self.log({"event": "job_failed", "job_id": job["job_id"], "error": f"{type(e).__name__}: {e}"})
                    if self._server is not None and getattr(self._server, "proc", None) is not None \
                            and self._server.proc.poll() is not None:
                        self._stop_server()  # server died: the next job starts a fresh one
        finally:
            self._stop_server()
            inst.close()
            if original_gov and read_governor() != original_gov:
                try:
                    set_governor(original_gov)
                except RuntimeError:
                    pass
            counts = self.counts()
            done = not counts.get("pending")
            self.store.update("experiments", "exp_id", self.exp_id,
                              {"status": "complete" if done and not stopped else "interrupted",
                               "finished_at": time.time()})
            self.store.checkpoint()
        self.log({"event": "finished", "exp_id": self.exp_id, "counts": counts, "stopped": stopped})
        return {"exp_id": self.exp_id, "counts": counts, "stopped": stopped}

    def _abort_job(self, job: dict, status: str, reason: str, error: str | None = None) -> None:
        st = self.store
        st.execute("UPDATE runs SET status = 'aborted' WHERE job_id = ? AND status != 'aborted'", [job["job_id"]])
        st.update("jobs", "job_id", job["job_id"], {"status": status, "reason": reason, "error": error,
                                                     "finished_at": time.time()})


def job_label(job: dict) -> str:
    parts = [job["model"], job["quant"], f"kv={job['kv_quant']}", job["kind"]]
    for k in ("family", "depth", "n_args", "category", "n_prompt", "n_gen", "perf", "schema_len", "decode_len",
              "grammar", "prefix_cache", "governor"):
        if k in job and job[k] not in ("", None):
            parts.append(f"{k}={job[k]}")
    return " ".join(str(p) for p in parts)
