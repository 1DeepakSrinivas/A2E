import json
import threading

import pytest

from a2e import runner as runner_mod
from a2e.runner import Runner, RunnerHooks, job_label
from a2e.store import Store
from tests.rig import SMALL, instruments, make_settings, write_suite

pytest.importorskip("mpmath")


def make(tmp_path, body=SMALL, device="mac", events=None, w=5.0, **kw):
    s = make_settings(tmp_path, device=device)
    log = (lambda e: events.append(e)) if events is not None else None
    return Runner(write_suite(tmp_path, body), s, store=Store(s.db), log=log,
                  hooks=RunnerHooks(instruments=instruments(w)), **kw)


def test_end_to_end_small_suite_writes_everything(tmp_path):
    events = []
    r = make(tmp_path, events=events)
    out = r.run()
    st = r.store
    assert out["counts"] == {"done": 4} and not out["stopped"]
    assert st.scalar("SELECT status FROM experiments") == "complete"
    assert st.scalar("SELECT count(*) FROM jobs WHERE device = 'pi5'") == 0  # other device's jobs are not loaded
    assert st.scalar("SELECT count(*) FROM runs WHERE status = 'complete'") == 4 * 3
    assert st.scalar("SELECT count(*) FROM v_runs WHERE phase = 'measured'") == 4 * 2
    assert sum(e["event"] == "server_start" for e in events) == 1  # one model load for all four jobs
    # every turn has timings, energy and resource stats
    bad = st.scalar("SELECT count(*) FROM turns WHERE prefill_ms IS NULL OR energy_j IS NULL OR peak_rss_mb IS NULL "
                    "OR max_temp_c IS NULL")
    assert bad == 0 and st.scalar("SELECT count(*) FROM turns") > 0
    assert st.scalar("SELECT count(*) FROM samples") > 0
    assert st.scalar("SELECT count(DISTINCT category) FROM cases") == 4  # plain, custom:calc, two BFCL categories
    raw = (tmp_path / "raw" / f"exp{out['exp_id']}.jsonl").read_text().splitlines()
    assert len(raw) == (2 + 2) * 3 and all(json.loads(x)["run_id"] > 0 for x in raw)
    env = json.loads(st.scalar("SELECT env_json FROM experiments"))
    assert "hostname" in env and st.scalar("SELECT suite_hash FROM experiments") == r.suite.hash


def test_energy_is_power_times_time_without_idle_baseline(tmp_path):
    r = make(tmp_path, w=5.0)
    r.run()
    rows = r.store.query("SELECT wall_ms, template_ms, energy_j, energy_prefill_j, energy_decode_j, energy_tool_j, "
                         "avg_power_w FROM turns WHERE energy_j IS NOT NULL")
    for x in rows:
        assert x["energy_j"] == pytest.approx(5.0 * x["wall_ms"] / 1000, rel=0.05, abs=2e-3)
        assert x["avg_power_w"] == pytest.approx(5.0, rel=0.01)
        parts = x["energy_prefill_j"] + x["energy_decode_j"] + x["energy_tool_j"]
        assert parts <= x["energy_j"] + 1e-6


def test_idle_baseline_is_subtracted(tmp_path):
    body = SMALL.replace("idle_seconds = 0", "idle_seconds = 0.2")
    r = make(tmp_path, body=body, w=5.0)
    r.run()
    assert r.store.scalar("SELECT min(idle_power_w) FROM runs") == pytest.approx(5.0)
    assert abs(r.store.scalar("SELECT max(abs(energy_j)) FROM turns")) < 1e-6  # constant power = idle: net zero


def test_resume_after_limit_continues_same_experiment(tmp_path):
    r = make(tmp_path)
    first = r.run(limit=2)
    assert first["counts"] == {"done": 2, "pending": 2}
    assert r.store.scalar("SELECT status FROM experiments") == "interrupted"
    r2 = Runner(r.suite, r.settings, store=r.store, hooks=r.hooks)
    second = r2.run()
    assert second["exp_id"] == first["exp_id"] and second["counts"] == {"done": 4}
    assert r.store.scalar("SELECT count(*) FROM jobs") == 4 and r.store.scalar("SELECT count(*) FROM experiments") == 1
    assert r.store.scalar("SELECT count(*) FROM runs") == 12


def test_crash_mid_job_aborts_partial_runs_and_reruns_the_job(tmp_path, monkeypatch):
    r = make(tmp_path)
    real = runner_mod.EXECUTORS["custom"]
    calls = {"n": 0}

    def boom(client, job, ctx):
        calls["n"] += 1
        if calls["n"] == 2:  # second repeat of the custom job: like Ctrl+C mid-job
            raise KeyboardInterrupt
        return real(client, job, ctx)

    monkeypatch.setitem(runner_mod.EXECUTORS, "custom", boom)
    out = r.run()
    assert out["stopped"] and r.store.scalar("SELECT status FROM experiments") == "interrupted"
    assert r.store.scalar("SELECT count(*) FROM runs WHERE status = 'aborted'") >= 1
    custom = r.store.query("SELECT status FROM jobs WHERE kind = 'custom'")[0]["status"]
    assert custom == "pending"
    monkeypatch.setitem(runner_mod.EXECUTORS, "custom", real)
    out2 = Runner(r.suite, r.settings, store=r.store, hooks=r.hooks).run()
    assert out2["counts"] == {"done": 4}
    # aborted runs stay in the raw table for the record but never reach the analysis views
    assert r.store.scalar("SELECT count(*) FROM v_runs WHERE job_id IN (SELECT job_id FROM jobs WHERE kind='custom')") == 3


def test_process_death_leaves_running_rows_that_the_next_start_recovers(tmp_path):
    r = make(tmp_path)
    r.open_experiment()
    job = r.pending_jobs()[0]
    st = r.store
    st.update("jobs", "job_id", job["job_id"], {"status": "running"})
    st.insert("runs", {"exp_id": r.exp_id, "job_id": job["job_id"], "config_json": "{}", "device": "mac",
                       "phase": "warmup", "status": "running", "started_at": 0.0})
    Runner(r.suite, r.settings, store=st, hooks=r.hooks).open_experiment()
    assert st.scalar("SELECT status FROM jobs WHERE job_id = ?", [job["job_id"]]) == "pending"
    assert st.scalar("SELECT status FROM runs") == "aborted"


def test_missing_model_fails_only_its_jobs_and_retry_failed_reruns_them(tmp_path):
    body = SMALL.replace('model = ["qwen3-4b"]', 'model = ["qwen3-4b", "gemma3-4b"]')
    r = make(tmp_path, body=body)
    missing = tmp_path / "models" / "gemma3-4b" / "gemma3-4b-q4_0.gguf"
    missing.unlink()
    out = r.run()
    assert out["counts"] == {"done": 4, "failed": 4}
    err = r.store.query("SELECT DISTINCT reason FROM jobs WHERE status = 'failed'")
    assert "model file missing" in err[0]["reason"]
    missing.write_bytes(b"GGUF")
    out2 = Runner(r.suite, r.settings, store=r.store, hooks=r.hooks).run(retry_failed=True)
    assert out2["counts"] == {"done": 8}
    assert r.store.scalar("SELECT max(attempts) FROM jobs") == 2


def test_pi_memory_rule_and_mac_governor_are_skipped_with_reasons(tmp_path):
    body = SMALL.replace('model = ["qwen3-4b"]', 'model = ["big-8b"]').replace('quant = ["q4_0"]',
                                                                               'quant = ["q8_0"]')
    r = make(tmp_path, body=body, device="pi5")
    out = r.run()
    assert out["counts"] == {"skipped": 4}
    assert "not run: memory" in r.store.scalar("SELECT reason FROM jobs LIMIT 1")
    body2 = SMALL + '\n[[tasks]]\nkind = "plain"\nn = 1\ngovernor = ["performance"]\n'
    r2 = make(tmp_path / "b", body=body2) if (tmp_path / "b").mkdir() is None else None
    out2 = r2.run()
    assert out2["counts"]["skipped"] == 1 and "governor" in r2.store.scalar(
        "SELECT reason FROM jobs WHERE status='skipped'")


def test_schema_too_large_skips_the_job(tmp_path):
    body = SMALL.replace("schema_len = [512]", "schema_len = [16]").replace('family = "calc"', 'family = "retrieval"')
    r = make(tmp_path, body=body)
    out = r.run()
    assert out["counts"].get("skipped") == 1
    assert "core tools need" in r.store.scalar("SELECT reason FROM jobs WHERE status = 'skipped'")


def test_stop_event_stops_between_jobs(tmp_path):
    stop = threading.Event()
    events = []

    def log(e):
        events.append(e)
        if e["event"] == "job_done":
            stop.set()

    s = make_settings(tmp_path)
    r = Runner(write_suite(tmp_path, SMALL), s, store=Store(s.db), log=log, stop=stop,
               hooks=RunnerHooks(instruments=instruments()))
    out = r.run()
    assert out["stopped"] and out["counts"]["done"] == 1 and out["counts"]["pending"] == 3


def test_new_flag_and_edited_suite_start_new_experiments(tmp_path):
    r = make(tmp_path)
    r.run(limit=1)
    e2 = Runner(r.suite, r.settings, store=r.store, hooks=r.hooks).run(new=True, limit=0)["exp_id"]
    edited = write_suite(tmp_path, SMALL.replace("measured = 2", "measured = 3"))
    e3 = Runner(edited, r.settings, store=r.store, hooks=r.hooks).run(limit=0)["exp_id"]
    assert len({1, e2, e3}) == 3


def test_jobs_with_different_server_settings_load_the_model_again(tmp_path):
    events = []
    body = SMALL.replace('quant = ["q4_0"]', 'quant = ["q4_0", "q8_0"]')
    r = make(tmp_path, body=body, events=events)
    r.run()
    starts = [(e["quant"]) for e in events if e["event"] == "server_start"]
    assert starts == ["q4_0", "q8_0"]  # grouped: each model file loads once


def test_custom_turn_correctness_is_recorded(tmp_path):
    r = make(tmp_path)
    r.run()
    # the fake model never makes a valid call, so every scored step is wrong and every task fails
    assert r.store.scalar("SELECT count(*) FROM v_cases WHERE category = 'custom:calc' AND valid = 0") == 2 * 3
    assert r.store.scalar("SELECT count(*) FROM turns WHERE task_id LIKE 'calc-%' AND correct = 0") > 0


def test_job_label_is_readable():
    j = {"model": "qwen3-4b", "quant": "q4_0", "kv_quant": "f16", "kind": "custom", "family": "calc", "depth": 3,
         "grammar": True, "governor": ""}
    assert job_label(j) == "qwen3-4b q4_0 kv=f16 custom family=calc depth=3 grammar=True"
