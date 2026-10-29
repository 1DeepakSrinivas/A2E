"""The harness figures draw from a store and write a PNG with its table beside it."""
import json

import numpy as np
import pytest
from typer.testing import CliRunner

pytest.importorskip("matplotlib", reason="figures need `uv sync --extra plots`")

from a2e import plots  # noqa: E402
from a2e.cli import app  # noqa: E402
from a2e.store import Store  # noqa: E402

HARNESSES = {"baseline": (1.0, 1.0), "compact": (0.95, 1.0), "heavy": (1.4, 1.2)}  # energy, calls


@pytest.fixture
def store(tmp_path):
    st = Store(tmp_path / "a.db")
    rng = np.random.default_rng(0)
    for device, scale in (("mac", 1.0), ("pi5", 3.0)):
        for model in ("a", "b"):
            for name, (em, cm) in HARNESSES.items():
                for rep in range(3):
                    cfg = {"model": model, "quant": "q4_0", "kv_quant": "f16", "family": "math", "depth": 3}
                    if name != "baseline":
                        cfg["harness"] = name
                    run = st.insert("runs", {"config_json": json.dumps(cfg), "device": device, "phase": "measured",
                                             "repeat_idx": rep, "task_kind": "custom", "status": "complete",
                                             "started_at": 0.0})
                    for i in range(6):
                        st.insert("cases", {"run_id": run, "case_id": f"t{i}", "category": "c", "valid": int(i != 5)})
                        for _ in range(max(1, round(2 * cm))):
                            e = 5 * scale * em * float(np.exp(rng.normal(0, 0.03)))
                            st.insert("turns", {"run_id": run, "task_id": f"t{i}", "wall_ms": 5.0, "energy_j": e,
                                                "energy_prefill_j": e * .3, "energy_decode_j": e * .5,
                                                "energy_tool_j": e * .1, "energy_harness_j": e * .02})
    return st


def test_every_figure_is_written_with_its_table(store, tmp_path):
    made = plots.make_figures(store, tmp_path / "figs")
    names = {p.name for p in made}
    assert names == {"harness_forest.png", "harness_decomposition.png", "harness_frontier.png",
                     "harness_variance.png", "harness_stage_energy.png", "harness_noise.png"}
    for p in made:
        assert p.stat().st_size > 5000 and p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        assert p.with_suffix(".csv").stat().st_size > 20


def test_the_forest_table_has_a_verdict_per_harness_and_device(store, tmp_path):
    plots.make_figures(store, tmp_path)
    import pandas as pd
    t = pd.read_csv(tmp_path / "harness_forest.csv")
    assert set(t["harness"]) == {"compact", "heavy"} and set(t["device"]) == {"mac", "pi5"}
    assert (t[t.harness == "heavy"]["verdict"] == "differs").all()
    assert (t[t.harness == "compact"]["verdict"] == "equivalent").all()


def test_nothing_is_drawn_without_data(tmp_path):
    assert plots.make_figures(Store(":memory:"), tmp_path) == []


def test_figures_are_skipped_when_a_part_is_missing(store, tmp_path):
    store.execute("DELETE FROM turns WHERE energy_harness_j IS NOT NULL AND run_id IN "
                  "(SELECT run_id FROM runs WHERE config_json LIKE '%heavy%')")
    assert plots.make_figures(store, tmp_path)  # still draws what it can


def test_cli_views_and_plots(store, tmp_path, monkeypatch):
    monkeypatch.setenv("A2E_DB", store.path)
    runner = CliRunner()
    out = runner.invoke(app, ["results", "efficiency", "harness"]).output
    assert "heavy" in out and "Conclusion" in out and "harness matters" in out
    for view in ("decomposition", "noise", "variance"):
        r = runner.invoke(app, ["results", "efficiency", view])
        assert r.exit_code == 0, r.output
    assert "run-to-run sd" in runner.invoke(app, ["results", "efficiency", "noise"]).output
    r = runner.invoke(app, ["results", "plots", str(tmp_path / "f")])
    assert r.exit_code == 0 and "harness_forest.png" in r.output
