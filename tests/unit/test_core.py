from a2e.config import factorial
from a2e.metrics.power_base import integrate_energy
from a2e.store import Store


def test_factorial_covers_every_cell():
    rows = list(factorial(["qwen3-4b-instruct-2507"], ["q5_k_m", "iq3_m"], ["f16"], [256], [128], [False], ["pi5", "mac"]))
    assert len(rows) == 4 and len({r.model_dump_json() for r in rows}) == 4


def test_energy_integration_constant_power():
    samples = [(t / 10, 5.0) for t in range(0, 21)]
    assert abs(integrate_energy(samples, 0.0, 2.0) - 10.0) < 1e-9
    assert abs(integrate_energy(samples, 0.0, 2.0, idle_w=2.0) - 6.0) < 1e-9


def test_store_roundtrip(tmp_path):
    s = Store(tmp_path / "a.db")
    rid = s.insert("runs", {"config_json": "{}", "device": "mac", "phase": "measured", "started_at": 0.0})
    s.insert("turns", {"run_id": rid, "prefill_ms": 1.0})
    assert s.db.execute("SELECT count(*) FROM turns").fetchone()[0] == 1
