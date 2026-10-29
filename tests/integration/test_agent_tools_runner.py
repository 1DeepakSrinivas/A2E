import json

import pytest

from a2e.runner import Runner, RunnerHooks
from a2e.settings import WebSettings
from a2e.store import Store
from tests.fakes.fake_brave import KEY, FakeBrave
from tests.rig import instruments, make_settings, write_suite

pytest.importorskip("mpmath")

SUITE = """
[repeats]
warmup = 0
measured = 1
[run]
idle_seconds = 0
sample_hz = 20
[matrix]
device = ["mac"]
model = ["qwen3-4b"]
quant = ["q4_0"]
[axes]
decode_len = [16]
schema_len = [1024]
grammar = [false]
[[tasks]]
kind = "custom"
family = ["fs_search", "fs_organize", "math"]
depth = 2
n = 2
[[tasks]]
kind = "custom"
family = "fs_math"
depth = 3
n = 2
distractors = "synthetic"
[[tasks]]
kind = "custom"
family = "web"
depth = 2
n = 2
"""


@pytest.fixture
def brave(monkeypatch):
    fake = FakeBrave()
    monkeypatch.setenv("A2E_IT_BRAVE", KEY)
    yield fake
    fake.close()


def make(tmp_path, brave, key_env="A2E_IT_BRAVE"):
    s = make_settings(tmp_path)
    s.web = WebSettings(api_key_env=key_env, base_url=brave.base)
    return Runner(write_suite(tmp_path, SUITE), s, store=Store(s.db), hooks=RunnerHooks(instruments=instruments()))


def test_new_families_run_end_to_end_and_record_their_details(tmp_path, brave):
    r = make(tmp_path, brave)
    out = r.run()
    st = r.store
    assert out["counts"] == {"done": 5}
    cats = {c: n for c, n in st.db.execute("SELECT category, count(*) FROM cases GROUP BY category")}
    assert cats == {"custom:fs_search": 2, "custom:fs_organize": 2, "custom:math": 2, "custom:fs_math": 2,
                    "custom:web": 2}
    extras = {c: json.loads(e) for c, e in st.db.execute("SELECT category, extra_json FROM cases")}
    assert extras["custom:fs_search"]["tools"] == "bfcl" and extras["custom:fs_search"]["scenario"].startswith("fs_")
    assert extras["custom:fs_organize"]["state_ok"] is False  # the fake model never acts, so the state differs
    assert extras["custom:fs_math"]["distractors"] == "synthetic"
    assert extras["custom:web"]["tools"] == "web" and "question" in extras["custom:web"]
    # the padded schema really contains BFCL functions from other classes for the default (bfcl) jobs
    run_id = st.scalar("SELECT run_id FROM v_runs WHERE family = 'math'")
    assert st.scalar("SELECT schema_tokens FROM cases WHERE run_id = ?", [run_id]) > 0


def test_web_jobs_fail_clearly_without_a_key(tmp_path, brave, monkeypatch):
    monkeypatch.delenv("A2E_NO_KEY_HERE", raising=False)
    r = make(tmp_path, brave, key_env="A2E_NO_KEY_HERE")
    out = r.run()
    assert out["counts"] == {"done": 4, "failed": 1}
    reason = r.store.scalar("SELECT reason FROM jobs WHERE status = 'failed'")
    assert "A2E_NO_KEY_HERE" in reason
