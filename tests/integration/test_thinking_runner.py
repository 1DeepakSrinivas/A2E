"""The runner with thinking off: the template request carries the variable and the runs say so."""
import json

import pytest

from a2e.client import Client
from a2e.runner import Runner, RunnerHooks
from a2e.store import Store
from tests.rig import SMALL, instruments, make_settings, write_suite

pytest.importorskip("mpmath")
BODY = SMALL.replace('["mac", "pi5"]', '["mac"]')


def run(tmp, thinking):
    s = make_settings(tmp)
    clients = []

    def make(url):
        c = Client(url)
        clients.append(c)
        return c

    server = f'[server]\nthinking = "{thinking}"\n' if thinking else ""
    r = Runner(write_suite(tmp, server + BODY), s, store=Store(s.db),
               hooks=RunnerHooks(instruments=instruments(6.0), make_client=make))
    return r, r.run(), clients


def test_off_sets_the_client_variable_and_is_recorded_on_every_run(tmp_path):
    r, out, clients = run(tmp_path, "off")
    assert out["counts"].get("done") and not out["counts"].get("failed")
    assert clients and all(c.template_kwargs == {"enable_thinking": False} for c in clients)
    rows = r.store.query("SELECT DISTINCT thinking FROM v_runs WHERE task_kind != 'bench'")
    assert [x["thinking"] for x in rows] == ["off"]
    assert all(json.loads(x["config_json"])["thinking"] == "off" for x in r.store.query("SELECT config_json FROM runs"))


def test_the_template_the_model_sees_ends_with_an_empty_think_block(fake_server):
    c = Client(fake_server().url)
    messages = [{"role": "user", "content": "hi"}]
    assert not c.apply_template(messages).endswith("</think>\n\n")
    c.template_kwargs = {"enable_thinking": False}
    assert c.apply_template(messages).endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_default_leaves_the_client_alone(tmp_path):
    r, out, clients = run(tmp_path, None)
    assert out["counts"].get("done") and all(c.template_kwargs is None for c in clients)
    assert [x["thinking"] for x in r.store.query("SELECT DISTINCT thinking FROM v_runs")] == ["default"]
