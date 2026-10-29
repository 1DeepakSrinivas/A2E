"""The thinking switch: suite field, job keys, server arguments and the template request."""
import json
from pathlib import Path

import httpx
import pytest

from a2e.client import Client
from a2e.config import RunConfig
from a2e.server import THINKING_OFF, build_args
from a2e.suite import expand, load_suite, server_key
from tests.rig import make_settings

REPO = Path(__file__).parents[2]
BASE = """
[repeats]
warmup = 0
measured = 1
[run]
idle_seconds = 0
[matrix]
device = ["mac"]
model = ["qwen3-4b-instruct-2507"]
quant = ["q5_k_m"]
[axes]
decode_len = [16]
schema_len = [512]
[[tasks]]
kind = "custom"
family = "calc"
depth = 2
n = 1
"""


def suite(tmp_path, server=""):
    p = tmp_path / "s.toml"
    p.write_text('[suite]\nname = "s"\n' + (f"[server]\n{server}\n" if server else "") + BASE)
    return load_suite(p)


def test_default_leaves_the_hash_and_job_keys_alone(tmp_path):
    s = suite(tmp_path)
    assert s.server.thinking == "default" and "thinking" not in s.canonical()["server"]
    jobs = expand(s, make_settings(tmp_path))
    assert not any("thinking" in j for j in jobs)
    assert expand(suite(tmp_path, 'thinking = "default"'), make_settings(tmp_path)) == jobs


def test_off_is_stored_in_every_job_and_changes_the_keys(tmp_path):
    default = expand(suite(tmp_path), make_settings(tmp_path))
    off = expand(suite(tmp_path, 'thinking = "off"'), make_settings(tmp_path))
    assert all(j["thinking"] == "off" for j in off)
    assert {j["job_key"] for j in off}.isdisjoint({j["job_key"] for j in default})
    assert suite(tmp_path, 'thinking = "off"').canonical()["server"]["thinking"] == "off"


def test_the_server_restarts_when_thinking_differs(tmp_path):
    a = expand(suite(tmp_path), make_settings(tmp_path))[0]
    b = expand(suite(tmp_path, 'thinking = "off"'), make_settings(tmp_path))[0]
    assert server_key(a) != server_key(b) and server_key(a)[-1] == "default" and server_key(b)[-1] == "off"


def test_bad_values_are_rejected(tmp_path):
    with pytest.raises(Exception):
        suite(tmp_path, 'thinking = "maybe"')


def test_build_args_adds_the_template_kwargs_only_when_off():
    cfg = RunConfig(model="qwen3-4b-instruct-2507", quant="q5_k_m", kv_quant="f16", device="mac")
    on = build_args(cfg, "llama-server", "m.gguf")
    off = build_args(cfg, "llama-server", "m.gguf", thinking="off")
    assert "--chat-template-kwargs" not in on and "--jinja" in on
    assert off[off.index("--chat-template-kwargs") + 1] == json.dumps({"enable_thinking": False})
    assert off[: len(on)] == on  # nothing else changes


def test_client_sends_the_template_variables_only_when_set():
    seen = []

    def handler(req):
        seen.append(json.loads(req.read()))
        return httpx.Response(200, json={"prompt": "p"})

    c = Client("http://fake")
    c.http = httpx.Client(base_url="http://fake", transport=httpx.MockTransport(handler))
    c.apply_template([{"role": "user", "content": "hi"}])
    c.template_kwargs = dict(THINKING_OFF)
    c.apply_template([{"role": "user", "content": "hi"}])
    assert "chat_template_kwargs" not in seen[0]
    assert seen[1]["chat_template_kwargs"] == {"enable_thinking": False}


def test_full_eval_turns_thinking_off():
    s = load_suite(REPO / "configs" / "suites" / "full_eval.toml")
    assert s.server.thinking == "off"
