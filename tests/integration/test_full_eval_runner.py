"""The shipped full_eval suite end to end against the simulated llama-server, then the report script."""
import os
import subprocess
from pathlib import Path

import pytest

from a2e.runner import Runner, RunnerHooks
from a2e.store import Store
from a2e.suite import load_suite
from tests.rig import instruments, make_settings

pytest.importorskip("mpmath")
REPO = Path(__file__).parents[2]


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("full_eval")
    s = make_settings(tmp, models=("qwen3-4b", "gemma3-4b"), quants=("q4_0", "q8_0"))
    suite = load_suite(REPO / "configs" / "suites" / "full_eval.toml")
    suite.run.idle_seconds = 0
    suite.run.sample_hz = 50
    r = Runner(suite, s, store=Store(s.db), hooks=RunnerHooks(instruments=instruments(6.0)))
    out = r.run(limit=42)  # the first model and quant: 21 jobs with grammar off and 21 with it on
    return r.store, out["exp_id"], s, out


def test_one_model_and_quant_runs_every_job_and_scores_50_tasks_per_grammar_setting(ran):
    st, exp, _, out = ran
    assert out["counts"].get("done") == 42 and not out["counts"].get("failed")
    rows = st.query("SELECT grammar, count(*) AS n FROM v_cases WHERE exp_id = ? GROUP BY grammar", [exp])
    assert {r["grammar"]: r["n"] for r in rows} == {0: 50, 1: 50}


def test_every_task_kind_and_horizon_is_in_the_results(ran):
    st, exp, _, _ = ran
    kinds = {r["task_kind"] for r in st.query("SELECT DISTINCT task_kind FROM v_cases WHERE exp_id = ?", [exp])}
    assert kinds == {"bfcl", "custom"}
    depths = {r["depth"] for r in st.query("SELECT DISTINCT depth FROM v_cases WHERE task_kind = 'custom'")}
    assert depths == {2, 3, 4, 5}


def test_the_report_script_writes_every_file(ran, tmp_path):
    st, exp, s, _ = ran
    out = tmp_path / "report"
    env = {**os.environ, "OUT_DIR": str(out)}
    r = subprocess.run([str(REPO / "scripts" / "report_full_eval.sh"), str(exp), s.db], env=env, cwd=REPO,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr
    names = {p.name for p in out.iterdir()}
    for want in ("report.md", "by_model.csv", "by_category.csv", "by_family_depth.csv", "success_per_joule.csv",
                 "grammar_effect.csv", "energy_by_stage.csv", "pareto.csv", "per_task.csv", "README.txt"):
        assert want in names, want
    text = (out / "report.md").read_text()
    assert "## Accuracy" in text and "## Energy" in text
    assert len((out / "by_category.csv").read_text().splitlines()) > 5
