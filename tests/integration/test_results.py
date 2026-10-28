import math

import pandas as pd
import pytest

from a2e import results as R
from a2e.runner import Runner, RunnerHooks
from a2e.store import Store
from tests.rig import SMALL, instruments, make_settings, write_suite

pytest.importorskip("mpmath")


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("res")
    s = make_settings(tmp)
    r = Runner(write_suite(tmp, SMALL), s, store=Store(s.db), hooks=RunnerHooks(instruments=instruments(6.0)))
    out = r.run()
    return r.store, out["exp_id"]


def test_experiments_listing(ran):
    st, exp = ran
    df = R.experiments(st)
    row = df[df.exp_id == exp].iloc[0]
    assert row["status"] == "complete" and row["done"] == 4 and row["name"] == "t@mac"


def test_summary_has_one_row_per_configuration_with_measured_runs_only(ran):
    st, exp = ran
    s = R.summary(st, exp)
    assert len(s) == 4 and set(s["runs"]) == {2}
    assert (s["decode_tok_s"] > 0).all() and s["cache_hit"].between(0, 1).all()
    shares = s[["prefill_share", "decode_share", "tool_share", "overhead_share"]].sum(axis=1)
    assert (shares <= 1.0001).all()
    assert (s["j_per_turn"] > 0).all() and (s["avg_power_w"].round(3) == 6.0).all()
    assert R.summary(st, exp, phase="warmup")["runs"].eq(1).all()


def test_summary_accuracy_columns(ran):
    st, exp = ran
    s = R.summary(st, exp).set_index("category")
    assert s.loc["irrelevance", "success"] == 1.0  # the fake model never calls a tool
    assert s.loc["simple_python", "success"] == 0.0
    assert math.isnan(s.loc["simple_python", "j_per_success"]) or s.loc["simple_python", "j_per_success"] != 0


def test_summary_custom_grouping(ran):
    st, exp = ran
    s = R.summary(st, exp, by=["model"])
    assert len(s) == 1 and s.iloc[0]["runs"] == 8


def test_stage_shares_add_up_to_one(ran):
    st, exp = ran
    df = R.stages(st, exp)
    cols = ["prefill", "decode", "parse", "validate", "tool_wait", "template", "harness", "unattributed"]
    assert df[cols].sum(axis=1).round(6).eq(1).all() and (df[cols[:-1]] >= 0).all().all()


def test_stability_and_timeline(ran):
    st, exp = ran
    stab = R.stability(st, exp)
    assert len(stab) == 4 and (stab["tasks"] >= 1).all()
    run_id = st.scalar("SELECT run_id FROM v_runs LIMIT 1")
    tl = R.timeline(st, run_id)
    assert len(tl) > 0 and tl["t"].is_monotonic_increasing and (tl["t"] >= 0).all()


def test_report_has_all_sections(ran):
    st, exp = ran
    md = R.report_markdown(st, exp)
    for h in ("# Results for experiment", "## Jobs", "## Latency and throughput", "## Where the time goes",
              "## Accuracy", "## Energy", "## Memory and thermals"):
        assert h in md
    assert "test constant" in md  # power scope is stated next to energy numbers
    with pytest.raises(KeyError):
        R.report_markdown(st, 999)


def test_empty_database_is_handled(tmp_path):
    st = Store(tmp_path / "e.db")
    assert R.experiments(st).empty and R.summary(st).empty and R.stages(st).empty and R.stability(st).empty


def test_per_run_math_on_known_numbers():
    df = pd.DataFrame([
        {"prefill_ms": 100.0, "decode_ms": 1000.0, "tool_wait_ms": 50.0, "wall_ms": 1200.0, "parse_ms": 1.0,
         "validate_ms": 1.0, "template_ms": 2.0, "harness_ms": 6.0, "fresh_tokens": 200, "prompt_tokens": 1000,
         "cached_tokens": 800, "gen_tokens": 20, "energy_j": 6.0, "energy_prefill_j": 0.5, "energy_decode_j": 5.0,
         "energy_tool_j": 0.2, "avg_power_w": 5.0, "peak_rss_mb": 900.0, "max_temp_c": 60.0, "throttled": 0},
        {"prefill_ms": 300.0, "decode_ms": 1000.0, "tool_wait_ms": 150.0, "wall_ms": 1500.0, "parse_ms": 1.0,
         "validate_ms": 1.0, "template_ms": 2.0, "harness_ms": 6.0, "fresh_tokens": 600, "prompt_tokens": 1200,
         "cached_tokens": 600, "gen_tokens": 30, "energy_j": 8.0, "energy_prefill_j": 1.5, "energy_decode_j": 5.0,
         "energy_tool_j": 0.6, "avg_power_w": 7.0, "peak_rss_mb": 950.0, "max_temp_c": 65.0, "throttled": 1}])
    m = R._per_run(df)
    assert m["decode_tok_s"] == pytest.approx(50 / 2.0)
    assert m["prefill_tok_s"] == pytest.approx(800 / 0.4)
    assert m["cache_hit"] == pytest.approx(1400 / 2200)
    assert m["prefill_share"] == pytest.approx(400 / 2700)
    assert m["j_per_gen_token"] == pytest.approx(10 / 50) and m["tokens_per_joule"] == pytest.approx(5.0)
    assert m["j_per_prompt_token"] == pytest.approx(2.0 / 800)
    assert m["peak_rss_mb"] == 950 and m["throttled_turns"] == 1


def test_md_table_formatting():
    t = R.md_table(pd.DataFrame([{"a": 1.23456, "b": "x|y", "c": float("nan")}]))
    assert t.splitlines()[0] == "| a | b | c |" and "| 1.23 | x/y |  |" in t
    assert R.md_table(pd.DataFrame()) == "_No rows._\n"
