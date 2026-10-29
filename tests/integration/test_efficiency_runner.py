"""Success per joule on a real runner pass: grammar off and on against the simulated llama-server."""
import pytest

from a2e import efficiency as E
from a2e.runner import Runner, RunnerHooks
from a2e.store import Store
from tests.rig import SMALL, instruments, make_settings, write_suite

pytest.importorskip("mpmath")


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("eff")
    s = make_settings(tmp)
    body = SMALL.replace("grammar = [true]", "grammar = [false, true]").replace('["mac", "pi5"]', '["mac"]')
    r = Runner(write_suite(tmp, body), s, store=Store(s.db), hooks=RunnerHooks(instruments=instruments(6.0)))
    out = r.run()
    return r.store, out["exp_id"]


def test_every_configuration_has_energy_and_a_success_rate(ran):
    st, exp = ran
    e = E.efficiency(st, exp)
    assert not e.empty and set(e["grammar"]) == {False, True}
    assert (e["j_per_task"] > 0).all() and e["success_rate"].between(0, 1).all()
    assert (e["waste_share"].dropna().between(0, 1)).all()


def test_joules_per_success_matches_the_stored_turn_energy(ran):
    st, exp = ran
    e = E.efficiency(st, exp, by=["model"]).iloc[0]
    total = st.scalar("SELECT sum(t.energy_j) FROM v_turns t WHERE t.exp_id = ? AND t.phase = 'measured' "
                      "AND t.task_id IN (SELECT case_id FROM v_cases c WHERE c.run_id = t.run_id "
                      "AND c.valid IS NOT NULL)", [exp])
    assert e["energy_j"] * e["runs"] == pytest.approx(total)


def test_grammar_effect_pairs_the_two_modes(ran):
    st, exp = ran
    g = E.grammar_effect(st, exp)
    assert not g.empty and g["j_per_task_ratio"].gt(0).all() and "success_delta" in g


def test_budget_curve_is_bounded_and_never_falls(ran):
    st, exp = ran
    b = E.budget_curve(st, exp)
    assert not b.empty and (b["successes"] <= b["tasks_done"] + 1e-9).all()
    keys = [c for c in b.columns if c not in ("runs", "budget_j", "tasks_done", "successes")]
    for _, g in b.groupby(keys, dropna=False):
        g = g.sort_values("budget_j")
        assert g["tasks_done"].is_monotonic_increasing and g["successes"].is_monotonic_increasing
        total = E.efficiency(st, exp, by=[k for k in keys if k in E.EFFICIENCY_BY]).iloc[0]["tasks"]
        assert g["tasks_done"].max() <= total + 1e-9
