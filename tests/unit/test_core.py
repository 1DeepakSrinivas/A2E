from a2e.config import RunConfig, factorial, feasible
from a2e.metrics.power_base import integrate_energy
from a2e.store import Store


def test_pi_memory_rule():
    assert not feasible(RunConfig(model="qwen3-8b", quant="q8_0", device="pi5"))[0]
    assert feasible(RunConfig(model="qwen3-8b", quant="q4_0", device="pi5"))[0]
    assert feasible(RunConfig(model="qwen3-4b", quant="q8_0", device="pi5"))[0]
    assert feasible(RunConfig(model="qwen3-8b", quant="q8_0", device="mac"))[0]


def test_factorial_marks_infeasible():
    rows = list(factorial(["qwen3-8b"], ["q8_0", "q4_0"], ["f16"], [256], [128], [False], ["pi5", "mac"]))
    assert len(rows) == 4
    assert sum(1 for _, ok, _ in rows if not ok) == 1


def test_energy_integration_constant_power():
    samples = [(t / 10, 5.0) for t in range(0, 21)]
    assert abs(integrate_energy(samples, 0.0, 2.0) - 10.0) < 1e-9
    assert abs(integrate_energy(samples, 0.0, 2.0, idle_w=2.0) - 6.0) < 1e-9


def test_store_roundtrip(tmp_path):
    s = Store(tmp_path / "a.db")
    rid = s.insert("runs", {"config_json": "{}", "device": "mac", "phase": "measured", "started_at": 0.0})
    s.insert("turns", {"run_id": rid, "prefill_ms": 1.0})
    assert s.db.execute("SELECT count(*) FROM turns").fetchone()[0] == 1
