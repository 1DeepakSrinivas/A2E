"""Baseline protocol (1 warm-up + 3 measured) and a mini sweep over seqlen x grammar x cache,
run end to end against the fake server. `run_baseline` is a stand-in for the Phase 5 runner."""
import itertools
import json
import statistics
import time

import pytest

from a2e.client import Client
from a2e.config import RunConfig
from a2e.metrics.power_base import PowerSource, integrate_energy
from a2e.metrics.sampler import Sampler
from a2e.store import Store


class Watts(PowerSource):
    name, scope = "const", "test"

    def read_watts(self):
        return 5.0


def schema_prompt(schema_len: int, question: str) -> str:
    return "tool_schema " * schema_len + question


def run_baseline(client, store, sampler, cfg: RunConfig, n_questions=3, warmup=1, measured=3):
    """Warm-up runs are stored with phase='warmup' and excluded from stats; the cache is reset per run."""
    n_predict = cfg.decode_len
    grammar = 'root ::= "x"+' if cfg.grammar else None
    for i in range(warmup + measured):
        phase = "warmup" if i < warmup else "measured"
        rid = store.insert("runs", {"config_json": cfg.model_dump_json(), "device": cfg.device,
                                    "phase": phase, "started_at": time.time()})
        client.reset_cache()
        for q in range(n_questions):
            t0 = time.perf_counter()
            r = client.complete(schema_prompt(cfg.schema_len, f"q{q}"), n_predict, grammar=grammar,
                                cache_prompt=cfg.prefix_cache)
            t1 = time.perf_counter()
            store.insert("turns", {
                "run_id": rid, "task_id": f"t{q}", "step": 0, "prefill_ms": r.prefill_ms,
                "decode_ms": r.decode_ms, "wall_ms": r.wall_ms, "prompt_tokens": r.prompt_tokens,
                "cached_tokens": r.cached_tokens, "gen_tokens": r.gen_tokens,
                "energy_j": integrate_energy(sampler.power_trace(), t0, t1)})


@pytest.fixture
def rig(fake_server, tmp_path):
    srv = fake_server("--ms-per-prompt-token", "0.05", "--ms-per-gen-token", "0.2")
    sampler = Sampler(srv.pid, Watts(), hz=100).start()
    time.sleep(0.1)
    yield Client(srv.url), Store(tmp_path / "s.db"), sampler
    sampler.stop()


def cfg(**kw):
    return RunConfig(**{"model": "qwen3-4b-instruct-2507", "quant": "q5_k_m", "device": "mac", **kw})


def test_one_warmup_plus_three_measured_runs(rig):
    client, store, sampler = rig
    run_baseline(client, store, sampler, cfg())
    phases = [r[0] for r in store.db.execute("SELECT phase FROM runs ORDER BY run_id")]
    assert phases == ["warmup", "measured", "measured", "measured"]


def test_turn_rows_complete_and_linked(rig):
    client, store, sampler = rig
    run_baseline(client, store, sampler, cfg())
    n = store.db.execute("SELECT count(*) FROM turns").fetchone()[0]
    assert n == 4 * 3
    orphans = store.db.execute("SELECT count(*) FROM turns WHERE run_id NOT IN (SELECT run_id FROM runs)")
    assert orphans.fetchone()[0] == 0
    nulls = store.db.execute("SELECT count(*) FROM turns WHERE prefill_ms IS NULL OR decode_ms IS NULL "
                             "OR energy_j IS NULL OR gen_tokens IS NULL").fetchone()[0]
    assert nulls == 0


def test_config_json_stored_per_run(rig):
    client, store, sampler = rig
    c = cfg(schema_len=256, grammar=True)
    run_baseline(client, store, sampler, c)
    stored = {r[0] for r in store.db.execute("SELECT config_json FROM runs")}
    assert stored == {c.model_dump_json()}
    assert RunConfig.model_validate(json.loads(stored.pop())) == c


def test_schema_prefix_cached_after_first_question_each_run(rig):
    client, store, sampler = rig
    run_baseline(client, store, sampler, cfg(schema_len=256))
    rows = store.db.execute("SELECT run_id, task_id, cached_tokens FROM turns ORDER BY turn_id").fetchall()
    for _, task, cached in rows:
        assert (cached == 0) == (task == "t0")  # cache reset per run, then schema reused for later questions
        assert task == "t0" or cached >= 256


@pytest.mark.parametrize("schema_len", [256, 1024, 2048])
def test_prefill_scales_with_schema_length(rig, schema_len):
    client, store, sampler = rig
    run_baseline(client, store, sampler, cfg(schema_len=schema_len), warmup=0, measured=1, n_questions=1)
    (prefill, prompt) = store.db.execute("SELECT prefill_ms, prompt_tokens FROM turns").fetchone()
    assert prompt == schema_len + 1
    assert prefill == pytest.approx(prompt * 0.05, rel=0.01)


def test_grammar_off_vs_on_ablation(rig):
    client, store, sampler = rig
    for g in (False, True):
        run_baseline(client, store, sampler, cfg(grammar=g, decode_len=512), warmup=0, measured=2, n_questions=1)
    rows = store.db.execute("SELECT config_json, decode_ms FROM runs JOIN turns USING(run_id)").fetchall()
    by = {False: [], True: []}
    for c, d in rows:
        by[json.loads(c)["grammar"]].append(d)
    assert statistics.mean(by[True]) > statistics.mean(by[False])  # constrained decode costs more


def test_prefix_cache_off_removes_reuse(rig):
    client, store, sampler = rig
    run_baseline(client, store, sampler, cfg(prefix_cache=False), warmup=0, measured=1)
    assert store.db.execute("SELECT max(cached_tokens) FROM turns").fetchone()[0] == 0


def test_repeatability_across_measured_runs(rig):
    client, store, sampler = rig
    run_baseline(client, store, sampler, cfg(decode_len=128), n_questions=1, warmup=1, measured=3)
    walls = [r[0] for r in store.db.execute(
        "SELECT wall_ms FROM turns JOIN runs USING(run_id) WHERE phase='measured'")]
    assert len(walls) == 3
    assert statistics.pstdev(walls) / statistics.mean(walls) < 0.5  # loose: catches wild variance only


@pytest.mark.parametrize("schema_len,decode_len,grammar,prefix_cache",
                         list(itertools.product([256, 1024], [128, 512], [False, True], [True, False])))
def test_mini_sweep_cells_all_produce_rows(rig, schema_len, decode_len, grammar, prefix_cache):
    client, store, sampler = rig
    c = cfg(schema_len=schema_len, decode_len=decode_len, grammar=grammar, prefix_cache=prefix_cache)
    run_baseline(client, store, sampler, c, n_questions=2, warmup=1, measured=1)
    gen = [r[0] for r in store.db.execute("SELECT gen_tokens FROM turns")]
    assert gen and all(g == decode_len for g in gen)
    assert all(e > 0 for (e,) in store.db.execute("SELECT energy_j FROM turns"))
