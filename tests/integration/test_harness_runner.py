"""A whole runner pass over harness variants against the simulated llama-server."""
import json

import pytest

from a2e.runner import Runner, RunnerHooks
from a2e.store import Store
from tests.rig import instruments, make_settings, write_suite

pytest.importorskip("mpmath")

BODY = """
[repeats]
warmup = 0
measured = 1
[run]
idle_seconds = 0
sample_hz = 50
[matrix]
device = ["mac"]
model = ["qwen3-4b-instruct-2507"]
quant = ["q5_k_m"]
[axes]
decode_len = [16]
schema_len = [512]
[[harness]]
name = "baseline"
[[harness]]
name = "mask"
context = "mask"
[[harness]]
name = "lean"
prompt_format = "compact"
context = "mask"
recovery = "retry"
stopping = "early_stop"
[[tasks]]
kind = "custom"
family = "calc"
depth = 2
n = 2
[[tasks]]
kind = "bfcl"
category = ["simple_python", "irrelevance"]
n = 2
"""


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("harness")
    s = make_settings(tmp)
    r = Runner(write_suite(tmp, BODY), s, store=Store(s.db), hooks=RunnerHooks(instruments=instruments(6.0)))
    out = r.run()
    return r.store, out["exp_id"], out


def test_every_variant_ran_and_is_labelled(ran):
    st, exp, out = ran
    assert out["counts"].get("failed", 0) == 0
    seen = {(r["harness"], r["task_kind"]) for r in st.query("SELECT DISTINCT harness, task_kind FROM v_runs")}
    assert seen == {(h, k) for h in ("baseline", "mask", "lean") for k in ("custom", "bfcl")}


def test_non_default_fields_are_stored_in_the_config_and_baseline_stores_none(ran):
    st, exp, _ = ran
    cfgs = {json.loads(r["config_json"]).get("harness", "baseline"): json.loads(r["config_json"])
            for r in st.query("SELECT config_json FROM runs WHERE task_kind = 'custom'")}
    assert not any(f in cfgs["baseline"] for f in ("prompt_format", "context", "recovery", "stopping"))
    assert cfgs["lean"]["prompt_format"] == "compact" and cfgs["lean"]["stopping"] == "early_stop"


def test_harness_energy_is_recorded_and_is_a_part_of_the_turn(ran):
    st, exp, _ = ran
    rows = st.query("SELECT energy_j, energy_harness_j FROM turns WHERE energy_j IS NOT NULL")
    assert rows and all(r["energy_harness_j"] is not None for r in rows)
    assert all(0 <= r["energy_harness_j"] <= r["energy_j"] + 1e-9 for r in rows)
