import json
import math

import pytest

from a2e import results as R
from a2e.store import Store

pytest.importorskip("pandas")

CFG = {"model": "m", "quant": "q4_0", "kv_quant": "f16", "grammar": False, "prefix_cache": True,
       "decode_len": 128, "schema_len": 0, "device": "mac"}


def _run(st: Store, exp: int, power: list[float | None], scope: str, rss: list[float], pressure: list[str | None]):
    run = st.insert("runs", {"exp_id": exp, "config_json": json.dumps(CFG), "device": "mac", "phase": "measured",
                             "task_kind": "plain", "status": "complete", "started_at": 0.0, "power_scope": scope})
    st.insert_many("samples", [{"run_id": run, "t": i * 0.5, "rss_mb": rss[i], "cpu_pct": 100.0 + i,
                                "power_w": power[i], "pressure": pressure[i]} for i in range(len(power))])
    for task, energy, p, wall in (("a", 2.0, 4.0, 1000.0), ("a", 3.0, 8.0, 3000.0), ("b", 1.0, 5.0, 500.0)):
        st.insert("turns", {"run_id": run, "task_id": task, "energy_j": energy, "avg_power_w": p, "wall_ms": wall,
                            "peak_rss_mb": rss[0] + (10 if task == "b" else 0)})
    for task in ("a", "b"):
        st.insert("cases", {"run_id": run, "case_id": task, "category": "plain", "valid": 1, "wall_ms": 1.0})
    return run


@pytest.fixture
def st(tmp_path):
    s = Store(tmp_path / "a2e.db")
    s.insert("experiments", {"name": "t", "started_at": 0.0})
    return s


def test_power_and_memory_envelope_is_exact(st):
    _run(st, 1, [2.0, 4.0, 6.0, 8.0, 10.0], "test-scope", [100, 150, 120, 180, 110],
         ["Nominal", "Nominal", "Fair", "Nominal", "Nominal"])
    r = R.resources(st, 1).iloc[0]
    assert (r.power_min_w, r.power_median_w, r.power_mean_w, r.power_max_w) == (2.0, 6.0, 6.0, 10.0)
    assert r.power_p95_w == pytest.approx(9.6)
    assert (r.rss_min_mb, r.rss_max_mb) == (100, 180) and r.rss_mean_mb == pytest.approx(132)
    assert r.cpu_max_pct == 104 and r.samples == 5 and r.duration_s == 2.0
    assert r.energy_j == 6.0  # 2 + 3 + 1 over the run's turns
    assert r.thermal_pressure == "Fair" and r.power_scope == "test-scope" and r.runs == 1


def test_runs_without_power_leave_power_empty_not_zero(st):
    _run(st, 1, [None, None, None], "none", [100, 110, 120], [None, None, None])
    r = R.resources(st, 1).iloc[0]
    assert all(math.isnan(r[c]) for c in ("power_min_w", "power_median_w", "power_p95_w", "power_max_w"))
    assert r.rss_max_mb == 120 and r.power_scope == "none" and r.thermal_pressure is None


def test_min_and_max_combine_over_runs_other_stats_average(st):
    _run(st, 1, [2.0, 4.0], "s", [100, 200], [None, None])
    _run(st, 1, [6.0, 10.0], "s", [300, 400], [None, None])
    r = R.resources(st, 1).iloc[0]
    assert r.runs == 2 and r.power_min_w == 2.0 and r.power_max_w == 10.0
    assert r.power_mean_w == pytest.approx((3.0 + 8.0) / 2)
    assert r.rss_min_mb == 100 and r.rss_max_mb == 400


def test_case_resources_sum_energy_and_weight_power_by_time(st):
    _run(st, 1, [1.0, 2.0], "s", [100, 100], [None, None])
    df = R.case_resources(st, 1).set_index("case_id")
    assert df.loc["a", "turns"] == 2 and df.loc["a", "energy_j"] == 5.0
    assert df.loc["a", "avg_power_w"] == pytest.approx((4.0 * 1000 + 8.0 * 3000) / 4000)
    assert df.loc["a", "peak_rss_mb"] == 100 and df.loc["b", "peak_rss_mb"] == 110


def test_report_includes_resource_envelope(st):
    _run(st, 1, [2.0, 4.0], "s", [100, 200], [None, None])
    st.update("experiments", "exp_id", 1, {"env_json": "{}"})
    st.insert("jobs", {"exp_id": 1, "job_key": "k", "kind": "plain", "config_json": "{}"})
    assert "## Resource envelope" in R.report_markdown(st, 1)


def test_empty_database_gives_empty_frames(st):
    assert R.resources(st).empty and R.case_resources(st).empty
