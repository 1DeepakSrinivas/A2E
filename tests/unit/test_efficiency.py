"""Success per joule, checked against a store filled by hand so every number can be worked out on paper."""
import json
import math

import pandas as pd
import pytest

from a2e import efficiency as E
from a2e.store import Store


def add_run(st, *, model="m", grammar=False, device="mac", phase="measured", cases, status="complete"):
    """cases: list of (valid, energy_j). Each case gets one turn so the energy can be attributed to it."""
    cfg = {"model": model, "quant": "q5_k_m", "kv_quant": "f16", "grammar": grammar, "category": "simple_python"}
    run = st.insert("runs", {"config_json": json.dumps(cfg), "device": device, "phase": phase, "repeat_idx": 0,
                             "task_kind": "bfcl", "status": status, "started_at": 0.0})
    for i, (valid, joules) in enumerate(cases):
        cid = f"c{i}"
        st.insert("cases", {"run_id": run, "case_id": cid, "category": "simple_python", "valid": valid})
        st.insert("turns", {"run_id": run, "task_id": cid, "wall_ms": 100.0, "energy_j": joules,
                            "energy_prefill_j": None if joules is None else joules * 0.25,
                            "energy_decode_j": None if joules is None else joules * 0.5,
                            "energy_tool_j": None if joules is None else joules * 0.125, "gen_tokens": 10, "prompt_tokens": 100,
                            "cached_tokens": 0, "prefill_ms": 25.0, "decode_ms": 50.0, "tool_wait_ms": 0.0})
    return run


@pytest.fixture
def st():
    return Store(":memory:")


def test_failed_tasks_count_against_joules_per_success(st):
    add_run(st, cases=[(1, 10.0), (0, 30.0), (1, 20.0), (0, 40.0)])  # 100 J total, 2 of 4 passed
    row = E.efficiency(st).iloc[0]
    assert row["success_rate"] == 0.5 and row["tasks"] == 4 and row["successes"] == 2
    assert row["energy_j"] == pytest.approx(100.0)
    assert row["j_per_task"] == pytest.approx(25.0) and row["j_per_success"] == pytest.approx(50.0)
    assert row["waste_share"] == pytest.approx(0.7)  # 70 of 100 J went into the two failures


def test_no_successes_gives_no_number_not_infinity(st):
    add_run(st, cases=[(0, 10.0), (0, 10.0)])
    row = E.efficiency(st).iloc[0]
    assert math.isnan(row["j_per_success"]) and row["success_rate"] == 0 and row["waste_share"] == pytest.approx(1.0)


def test_unscored_cases_are_ignored(st):
    add_run(st, cases=[(1, 10.0), (None, 500.0)])  # a case the checker did not score
    row = E.efficiency(st).iloc[0]
    assert row["tasks"] == 1 and row["energy_j"] == pytest.approx(10.0)


def test_runs_without_energy_stay_empty_not_zero(st):
    add_run(st, cases=[(1, None), (0, None)])
    row = E.efficiency(st).iloc[0]
    assert row["success_rate"] == 0.5 and math.isnan(row["j_per_task"]) and math.isnan(row["j_per_success"])


def test_warmup_and_aborted_runs_are_excluded(st):
    add_run(st, cases=[(1, 10.0)])
    add_run(st, phase="warmup", cases=[(0, 999.0)])
    add_run(st, status="aborted", cases=[(0, 999.0)])
    row = E.efficiency(st).iloc[0]
    assert row["runs"] == 1 and row["energy_j"] == pytest.approx(10.0)


def test_repeats_are_averaged_per_configuration(st):
    add_run(st, cases=[(1, 10.0), (1, 10.0)])  # 10 J per success
    add_run(st, cases=[(1, 20.0), (0, 20.0)])  # 40 J per success
    row = E.efficiency(st).iloc[0]
    assert row["runs"] == 2 and row["j_per_success"] == pytest.approx(25.0)
    assert row["j_per_success_sd"] == pytest.approx(math.sqrt(450.0))


def test_pareto_front_marks_only_undominated_rows():
    df = pd.DataFrame({"device": ["mac"] * 4, "j_per_task": [10, 20, 30, 20], "success_rate": [0.5, 0.8, 0.8, 0.6]})
    out = E.pareto_front(df, within=["device"]).set_index(["j_per_task", "success_rate"])["on_front"]
    assert bool(out[(10, 0.5)]) and bool(out[(20, 0.8)])
    assert not bool(out[(30, 0.8)])  # same success as 20 J, costs more
    assert not bool(out[(20, 0.6)])  # same cost as the 0.8 row, worse success


def test_pareto_front_is_per_device_and_skips_missing_values():
    df = pd.DataFrame({"device": ["mac", "pi5", "pi5"], "j_per_task": [50.0, 5.0, float("nan")],
                       "success_rate": [0.1, 0.9, 1.0]})
    out = E.pareto_front(df, within=["device"])
    assert out["on_front"].tolist() == [True, True, False]


def test_budget_curve_counts_tasks_the_budget_can_buy(st):
    add_run(st, cases=[(1, 10.0), (0, 10.0), (1, 10.0), (1, 10.0)])
    out = E.budget_curve(st, budgets=[5.0, 10.0, 25.0, 40.0]).set_index("budget_j")
    assert out.loc[5.0, "tasks_done"] == 0 and out.loc[5.0, "successes"] == 0
    assert out.loc[10.0, "tasks_done"] == 1 and out.loc[10.0, "successes"] == 1
    assert out.loc[25.0, "tasks_done"] == 2 and out.loc[25.0, "successes"] == 1  # third task would overshoot
    assert out.loc[40.0, "tasks_done"] == 4 and out.loc[40.0, "successes"] == 3


def test_budget_curve_default_grid_is_shared_within_a_device(st):
    add_run(st, model="a", cases=[(1, 10.0)] * 4)
    add_run(st, model="b", cases=[(1, 30.0)] * 4)
    out = E.budget_curve(st, points=4)
    grids = out.groupby("model")["budget_j"].apply(lambda s: tuple(round(x, 6) for x in s))
    assert grids["a"] == grids["b"] and len(grids["a"]) == 4
    assert max(grids["a"]) == pytest.approx(80.0)  # median of the two full-suite energies, 40 and 120


def test_grammar_effect_pairs_off_and_on(st):
    add_run(st, grammar=False, cases=[(1, 10.0), (0, 10.0), (0, 10.0), (1, 10.0)])  # 0.5 success, 20 J/success
    add_run(st, grammar=True, cases=[(1, 12.0), (1, 12.0), (0, 12.0), (1, 12.0)])  # 0.75 success, 16 J/success
    add_run(st, model="other", grammar=True, cases=[(1, 1.0)])  # measured one way only
    out = E.grammar_effect(st)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["success_delta"] == pytest.approx(0.25)
    assert row["j_per_task_ratio"] == pytest.approx(1.2)
    assert row["j_per_success_ratio"] == pytest.approx(0.8)


def test_grammar_effect_empty_without_both_modes(st):
    add_run(st, grammar=True, cases=[(1, 1.0)])
    assert E.grammar_effect(st).empty


def test_stage_energy_shares_add_to_one(st):
    add_run(st, cases=[(1, 8.0), (1, 8.0)])
    row = E.stage_energy(st).iloc[0]
    assert row["energy_j"] == pytest.approx(16.0)
    assert (row["prefill_share"], row["decode_share"], row["tool_wait_share"]) == pytest.approx((0.25, 0.5, 0.125))
    assert row["other_share"] == pytest.approx(0.125)


def test_empty_store_gives_empty_frames(st):
    for fn in (E.efficiency, E.budget_curve, E.grammar_effect, E.stage_energy):
        assert fn(st).empty
