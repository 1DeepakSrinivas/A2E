import json
import os

import pytest

from a2e import results as R
from a2e.runner import Runner, RunnerHooks
from a2e.store import Store
from tests.rig import instruments, make_settings, write_suite

pytest.importorskip("mpmath")

BENCH = """
[repeats]
warmup = 0
measured = 2
[run]
idle_seconds = 0
sample_hz = 50
[matrix]
device = ["pi5", "mac"]
model = ["qwen3-4b"]
quant = ["q4_0"]
[axes]
decode_len = [16]
[[tasks]]
kind = "plain"
n = 1
[[tasks]]
kind = "bench"
n_prompt = [64, 128]
n_gen = [32]
perf = true
reps = 2
"""


def make(tmp_path, device, events=None):
    s = make_settings(tmp_path, device=device)
    return Runner(write_suite(tmp_path, BENCH), s, store=Store(s.db), log=(events.append if events is not None else None),
                  hooks=RunnerHooks(instruments=instruments()))


def test_bench_on_pi_records_throughput_kernels_and_samples(tmp_path):
    events = []
    r = make(tmp_path, "pi5", events)
    out = r.run()
    st = r.store
    assert out["counts"] == {"done": 2}
    assert st.scalar("SELECT count(*) FROM bench") == 3 * 2  # 2 prefill sizes + 1 decode size, 2 measured runs
    assert st.scalar("SELECT count(*) FROM bench WHERE test = 'pp' AND avg_tok_s = 400") == 4
    assert st.scalar("SELECT count(*) FROM kernels WHERE rank = 1 AND symbol = 'ggml_vec_dot_q4_0_q8_0'") == 3 * 2
    run = st.scalar("SELECT run_id FROM runs WHERE task_kind = 'bench' LIMIT 1")
    assert st.scalar("SELECT max(rss_mb) FROM samples WHERE run_id = ?", [run]) > 40  # sampled the bench process
    # the plain job used llama-server; it was stopped before the bench job (one start, then no server)
    assert [e["event"] for e in events].count("server_start") == 1


def test_bench_on_mac_notes_that_kernel_profiles_are_pi_only(tmp_path):
    r = make(tmp_path, "mac")
    r.run()
    st = r.store
    assert st.scalar("SELECT count(*) FROM kernels") == 0 and st.scalar("SELECT count(*) FROM bench") == 6
    notes = json.loads(st.scalar("SELECT server_json FROM runs WHERE task_kind = 'bench' LIMIT 1"))["instrument_notes"]
    assert any("only implemented on the Pi" in n for n in notes)


def test_missing_bench_binary_or_failing_bench_fails_the_job(tmp_path, monkeypatch):
    r = make(tmp_path, "pi5")
    (tmp_path / "llama-bench").unlink()
    out = r.run()
    assert out["counts"] == {"done": 1, "failed": 1}
    assert "llama-bench not found" in r.store.scalar("SELECT reason FROM jobs WHERE status = 'failed'")
    r2 = make(tmp_path / "x", "pi5") if (tmp_path / "x").mkdir() is None else None
    monkeypatch.setenv("FAKE_BENCH_FAIL", "1")
    out2 = r2.run()
    assert out2["counts"]["failed"] == 1 and "llama-bench failed" in r2.store.scalar(
        "SELECT reason FROM jobs WHERE status = 'failed'")


def test_bench_results_and_report(tmp_path):
    r = make(tmp_path, "pi5")
    r.run()
    b = R.bench_summary(r.store)
    assert set(b["test"]) == {"pp", "tg"} and set(b["tokens"]) == {64, 128, 32}
    tg = b[b.test == "tg"].iloc[0]
    assert tg["tok_s"] == 20 and tg["est_bandwidth_gbs"] == pytest.approx(20 * tg["model_gb"])
    assert b[b.test == "pp"]["est_bandwidth_gbs"].isna().all()
    k = R.kernels(r.store, top=2)
    assert len(k) == 3 * 2 and set(k["rank"]) == {1, 2}
    md = R.report_markdown(r.store, 1)
    assert "## Prefill and decode microbenchmarks" in md and "## Top kernels" in md
