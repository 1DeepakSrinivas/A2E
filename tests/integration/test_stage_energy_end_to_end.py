"""Sampler + client + energy integration + store, the same wiring the runner will use."""
import time

import pytest

from a2e.client import Client
from a2e.metrics.power_base import PowerSource, integrate_energy
from a2e.metrics.sampler import Sampler
from a2e.store import Store


class Watts(PowerSource):
    name, scope = "const", "test"

    def __init__(self, w):
        self.w = w

    def read_watts(self):
        return self.w


def timed_request(client, prompt, n):
    t0 = time.perf_counter()
    r = client.complete(prompt, n)
    return r, t0, time.perf_counter()


def test_stage_energy_matches_power_times_duration(fake_server):
    srv = fake_server()
    s = Sampler(srv.pid, Watts(5.0), hz=50).start()
    time.sleep(0.2)
    r, t0, t1 = timed_request(Client(srv.url), "w " * 50, 100)
    time.sleep(0.2)
    s.stop()
    e = integrate_energy(s.power_trace(), t0, t1)
    assert e == pytest.approx(5.0 * (t1 - t0), rel=0.05)


def test_short_request_between_samples_gets_energy(fake_server):
    srv = fake_server()
    s = Sampler(srv.pid, Watts(5.0), hz=4).start()  # 250 ms sample period
    time.sleep(0.6)
    r, t0, t1 = timed_request(Client(srv.url), "w w", 20)  # ~20 ms request
    time.sleep(0.6)
    s.stop()
    assert integrate_energy(s.power_trace(), t0, t1) > 0


def test_energy_per_token_and_idle_subtraction(fake_server):
    srv = fake_server()
    s = Sampler(srv.pid, Watts(6.0), hz=50).start()
    time.sleep(0.2)
    r, t0, t1 = timed_request(Client(srv.url), "w " * 10, 200)
    time.sleep(0.2)
    s.stop()
    gross = integrate_energy(s.power_trace(), t0, t1)
    net = integrate_energy(s.power_trace(), t0, t1, idle_w=2.0)
    assert net == pytest.approx(gross * (4.0 / 6.0), rel=0.01)
    assert net / r.gen_tokens > 0  # J/token is well defined


def test_turn_row_stored_and_exported(fake_server, tmp_path):
    srv = fake_server()
    store = Store(tmp_path / "r.db")
    rid = store.insert("runs", {"config_json": "{}", "device": "mac", "phase": "measured",
                                "started_at": time.time()})
    r = Client(srv.url).complete("a b c d", 10)
    store.insert("turns", {"run_id": rid, "prefill_ms": r.prefill_ms, "decode_ms": r.decode_ms,
                           "prompt_tokens": r.prompt_tokens, "cached_tokens": r.cached_tokens,
                           "gen_tokens": r.gen_tokens, "wall_ms": r.wall_ms})
    store.export_parquet(tmp_path / "pq")
    import pandas as pd

    df = pd.read_parquet(tmp_path / "pq" / "turns.parquet")
    assert len(df) == 1 and df.loc[0, "gen_tokens"] == 10
    assert (tmp_path / "pq" / "runs.parquet").exists() and (tmp_path / "pq" / "samples.parquet").exists()
