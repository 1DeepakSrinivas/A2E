"""Does the harness change joules per success? The statistics, checked on stores filled by hand."""
import json
import math

import numpy as np
import pytest

from a2e import efficiency as E
from a2e.store import Store


def add_run(st, *, harness="baseline", model="m", device="mac", repeat=0, tasks, family="fs_search"):
    """tasks: list of (case_id, valid, [energy of each model call])."""
    cfg = {"model": model, "quant": "q4_0", "kv_quant": "f16", "grammar": False, "family": family, "depth": 3}
    if harness != "baseline":
        cfg["harness"] = harness
    run = st.insert("runs", {"config_json": json.dumps(cfg), "device": device, "phase": "measured",
                             "repeat_idx": repeat, "task_kind": "custom", "status": "complete", "started_at": 0.0})
    for cid, valid, calls in tasks:
        st.insert("cases", {"run_id": run, "case_id": cid, "category": "c", "valid": valid})
        for j in calls:
            st.insert("turns", {"run_id": run, "task_id": cid, "wall_ms": 10.0, "energy_j": j})
    return run


@pytest.fixture
def st():
    return Store(":memory:")


def fill(st, scale=1.0, n=10, harness="variant", valid=1, calls=1, base_energy=10.0, repeat=0, **kw):
    """Baseline and variant over the same n tasks; the variant's energy per call is `scale` times the baseline's."""
    ids = [f"t{i}" for i in range(n)]
    add_run(st, repeat=repeat, tasks=[(c, 1, [base_energy * (1 + i / n)] * calls) for i, c in enumerate(ids)], **kw)
    add_run(st, harness=harness, repeat=repeat,
            tasks=[(c, valid, [base_energy * (1 + i / n) * scale] * calls) for i, c in enumerate(ids)], **kw)


def row(df, h="variant"):
    return df[df["harness"] == h].iloc[0]


# ------------------------------------------------------------------------------------------- equivalence test
def test_a_small_known_effect_is_equivalent_and_a_large_one_differs(st):
    fill(st, scale=1.05, harness="small")
    fill(st, scale=1.30, harness="large")
    out = E.harness_effect(st)
    small, large = row(out, "small"), row(out, "large")
    assert small["ratio"] == pytest.approx(1.05) and small["lo"] == pytest.approx(1.05) and \
        small["verdict"] == "equivalent"
    assert large["ratio"] == pytest.approx(1.30) and large["verdict"] == "differs"
    assert small["lower_bound"] == pytest.approx(1 / 1.1) and small["upper_bound"] == pytest.approx(1.1)


def test_a_variant_that_never_succeeds_where_the_baseline_does_differs(st):
    fill(st, scale=0.5, valid=0)  # half the energy but every task fails
    r = row(E.harness_effect(st))
    assert r["cells_lost"] == 1 and r["verdict"] == "differs" and math.isinf(r["ratio"])


def test_a_baseline_that_never_succeeds_cannot_be_compared(st):
    ids = [f"t{i}" for i in range(5)]
    add_run(st, tasks=[(c, 0, [10.0]) for c in ids])
    add_run(st, harness="variant", tasks=[(c, 1, [10.0]) for c in ids])
    r = row(E.harness_effect(st))
    assert r["cells_undefined"] == 1 and r["verdict"] == "inconclusive" and math.isnan(r["ratio"])


def test_lower_success_at_equal_energy_raises_joules_per_success(st):
    ids = [f"t{i}" for i in range(20)]
    add_run(st, tasks=[(c, 1, [10.0]) for c in ids])
    add_run(st, harness="variant", tasks=[(c, int(i % 2 == 0), [10.0]) for i, c in enumerate(ids)])  # half pass
    r = row(E.harness_effect(st))
    assert r["ratio"] == pytest.approx(2.0) and r["verdict"] == "differs"


def test_a_noisy_effect_near_the_bound_is_inconclusive(st):
    ids = [f"t{i}" for i in range(6)]
    add_run(st, tasks=[(c, 1, [10.0]) for c in ids])
    add_run(st, harness="variant", tasks=[(c, 1, [8.0 if i % 2 == 0 else 14.0]) for i, c in enumerate(ids)])
    r = row(E.harness_effect(st))
    assert r["ratio"] == pytest.approx(1.1) and r["verdict"] == "inconclusive"
    assert r["lo"] < r["lower_bound"] + 1 and r["hi"] > r["upper_bound"]


def test_only_tasks_run_under_both_harnesses_are_compared(st):
    add_run(st, tasks=[("a", 1, [10.0]), ("b", 1, [10.0]), ("only_base", 1, [999.0])])
    add_run(st, harness="variant", tasks=[("a", 1, [10.0]), ("b", 1, [10.0]), ("only_variant", 1, [1.0])])
    r = row(E.harness_effect(st))
    assert r["tasks"] == 2 and r["ratio"] == pytest.approx(1.0)


def test_cells_are_averaged_in_log_space_so_each_counts_equally(st):
    fill(st, scale=2.0, model="a")          # ratio 2
    fill(st, scale=0.5, model="b")          # ratio 0.5
    r = row(E.harness_effect(st))
    assert r["cells"] == 2 and r["ratio"] == pytest.approx(1.0)


def test_grouping_gives_one_verdict_per_device(st):
    fill(st, scale=1.0, device="mac")
    fill(st, scale=1.5, device="pi5")
    out = E.harness_effect(st, by=["device"]).set_index("device")
    assert out.loc["mac", "verdict"] == "equivalent" and out.loc["pi5", "verdict"] == "differs"


def test_missing_baseline_or_energy_gives_an_empty_table(st):
    add_run(st, harness="variant", tasks=[("a", 1, [1.0])])
    assert E.harness_effect(st).empty
    assert E.harness_effect(Store(":memory:")).empty


def test_results_do_not_depend_on_the_seed_when_nothing_varies(st):
    fill(st, scale=1.02)
    assert E.harness_effect(st, seed=1)["ratio"].iloc[0] == E.harness_effect(st, seed=2)["ratio"].iloc[0]


def test_conclusion_follows_the_rule(st):
    fill(st, scale=1.02, harness="a")
    fill(st, scale=1.04, harness="b")
    assert E.conclusion(E.harness_effect(st)).startswith("LLM-bound")
    fill(st, scale=1.6, harness="c")
    assert E.conclusion(E.harness_effect(st)).startswith("harness matters: 1 of 3")
    assert E.conclusion(E.harness_effect(Store(":memory:"))).startswith("no comparison")


def test_conclusion_inconclusive_when_nothing_is_decisive(st):
    ids = [f"t{i}" for i in range(6)]
    add_run(st, tasks=[(c, 1, [10.0]) for c in ids])
    add_run(st, harness="v", tasks=[(c, 1, [8.0 if i % 2 == 0 else 14.0]) for i, c in enumerate(ids)])
    assert E.conclusion(E.harness_effect(st)).startswith("inconclusive")


# ------------------------------------------------------------------------------------------- decomposition
def test_decomposition_terms_add_up_to_the_total(st):
    ids = [f"t{i}" for i in range(10)]
    add_run(st, tasks=[(c, 1, [4.0] * 3) for c in ids])                                   # 3 calls of 4 J, all pass
    add_run(st, harness="variant", tasks=[(c, int(i < 8), [5.0] * 4) for i, c in enumerate(ids)])  # 4 calls of 5 J
    d = row(E.decompose(st))
    assert d["d_log_turns"] == pytest.approx(math.log(4 / 3))
    assert d["d_log_j_per_turn"] == pytest.approx(math.log(5 / 4))
    assert d["d_log_success"] == pytest.approx(math.log(0.8))
    assert d["d_log_total"] == pytest.approx(d["d_log_turns"] + d["d_log_j_per_turn"] - d["d_log_success"])
    assert d["d_log_total"] == pytest.approx(math.log((10 * 20.0) / 8 / (10 * 12.0 / 10)))


def test_a_harness_that_only_adds_calls_is_all_turns(st):
    ids = [f"t{i}" for i in range(8)]
    add_run(st, tasks=[(c, 1, [3.0]) for c in ids])
    add_run(st, harness="variant", tasks=[(c, 1, [3.0, 3.0]) for c in ids])
    d = row(E.decompose(st))
    assert d["d_log_turns"] == pytest.approx(math.log(2)) and d["d_log_j_per_turn"] == pytest.approx(0, abs=1e-12)
    assert d["d_log_success"] == pytest.approx(0) and d["turns_share"] == pytest.approx(1.0)


def test_a_harness_with_no_effect_has_no_turns_share(st):
    fill(st, scale=1.0)
    d = row(E.decompose(st))
    assert d["d_log_total"] == pytest.approx(0) and math.isnan(d["turns_share"])


# ------------------------------------------------------------------------------------------- noise floor
def test_noise_floor_is_the_spread_of_baseline_repeats(st):
    for rep, e in enumerate([10.0, 11.0, 9.0]):
        add_run(st, repeat=rep, tasks=[(f"t{i}", 1, [e]) for i in range(5)])
    n = E.noise_floor(st)
    assert len(n) == 1 and n.iloc[0]["repeats"] == 3
    assert n.iloc[0]["sd_log"] == pytest.approx(np.std(np.log([10.0, 11.0, 9.0]), ddof=1))
    assert n.iloc[0]["max_ratio"] == pytest.approx(11 / 9)


def test_noise_floor_ignores_other_harnesses_and_single_repeats(st):
    add_run(st, tasks=[("a", 1, [1.0])])
    add_run(st, harness="variant", repeat=1, tasks=[("a", 1, [5.0])])
    assert E.noise_floor(st).empty


def test_margin_check_wants_twice_the_noise():
    from pandas import DataFrame
    quiet = DataFrame({"sd_log": [0.01, 0.02, 0.03]})
    loud = DataFrame({"sd_log": [0.06, 0.07, 0.08]})
    assert E.margin_check(quiet, 0.10)["ok"] is True
    assert E.margin_check(loud, 0.10)["ok"] is False
    assert E.margin_check(DataFrame(), 0.10)["ok"] is None


# ------------------------------------------------------------------------------------------- variance share
def test_variance_share_finds_the_factor_that_carries_the_variance(st):
    rng = np.random.default_rng(1)
    for device in ("mac", "pi5"):
        for model in ("a", "b"):
            for harness in ("baseline", "heavy"):
                for rep in range(8):
                    j = 10.0 * (3.0 if harness == "heavy" else 1.0) * float(np.exp(rng.normal(0, 0.02)))
                    add_run(st, harness=harness, model=model, device=device, repeat=rep, tasks=[("t", 1, [j])])
    out = E.variance_share(st, boot=200).set_index("factor")
    assert out.loc["harness", "omega2"] > 0.95
    assert out.loc["model", "omega2"] < 0.05 and out.loc["device", "omega2"] < 0.05
    assert out.loc["harness", "omega2_lo"] > 0.9 and out.loc["harness", "levels"] == 2


def test_variance_share_is_low_for_a_factor_that_changes_nothing(st):
    rng = np.random.default_rng(2)
    for model in ("a", "b"):
        for harness in ("baseline", "lean"):
            for rep in range(10):
                j = 10.0 * (1.5 if model == "b" else 1.0) * float(np.exp(rng.normal(0, 0.05)))
                add_run(st, harness=harness, model=model, repeat=rep, tasks=[("t", 1, [j])])
    out = E.variance_share(st, boot=200).set_index("factor")
    assert out.loc["model", "omega2"] > 0.9 and out.loc["harness", "omega2"] < 0.05


def test_variance_share_removes_the_task_cell_first(st):
    for family, scale in (("easy", 1.0), ("hard", 20.0)):  # hard tasks cost 20x, whatever the harness
        for harness in ("baseline", "lean"):
            for rep in range(6):
                add_run(st, harness=harness, family=family, repeat=rep, tasks=[("t", 1, [10.0 * scale * (1 + rep / 100)])])
    assert E.variance_share(st, boot=100).set_index("factor").loc["harness", "omega2"] < 0.05


def test_variance_share_empty_without_data():
    assert E.variance_share(Store(":memory:")).empty
