"""Clearing the prefix cache between runs: the erase, the flush fallback, and what happens when both fail."""
import json

import httpx
import pytest

from a2e.client import FLUSH_PROMPT, CacheResetError, Client
from a2e.results import cache_reset_warning
from a2e.store import Store


def client_with(handler) -> Client:
    c = Client("http://fake")
    c.http = httpx.Client(base_url="http://fake", transport=httpx.MockTransport(handler))
    return c


def ok_completion():
    return httpx.Response(200, json={"content": "x", "timings": {"prompt_n": 3, "prompt_ms": 1, "cache_n": 0,
                                                                   "predicted_n": 1, "predicted_ms": 1}})


def test_erase_is_used_when_it_works():
    seen = []

    def handler(req):
        seen.append(req.url.path + "?" + req.url.query.decode())
        return httpx.Response(200, json={"id_slot": 0})

    assert client_with(handler).reset_cache() == "erase"
    assert seen == ["/slots/0?action=erase"]


def test_flush_prompt_replaces_the_cache_when_erase_fails():
    seen = []

    def handler(req):
        seen.append(req.url.path)
        if req.url.path.startswith("/slots"):
            return httpx.Response(501, json={"error": "not supported"})
        body = req.read().decode()
        assert FLUSH_PROMPT in body and '"n_predict":1' in body.replace(" ", "")
        return ok_completion()

    assert client_with(handler).reset_cache() == "flush"
    assert seen == ["/slots/0", "/completion"]


def test_both_failing_raises_with_both_reasons():
    def handler(req):
        return httpx.Response(501 if req.url.path.startswith("/slots") else 500, json={"error": "no"})

    with pytest.raises(CacheResetError) as e:
        client_with(handler).reset_cache()
    assert "erase failed" in str(e.value) and "flush failed" in str(e.value)


def test_a_connection_error_counts_as_a_failed_erase():
    calls = []

    def handler(req):
        calls.append(req.url.path)
        if req.url.path.startswith("/slots"):
            raise httpx.ConnectError("refused")
        return ok_completion()

    assert client_with(handler).reset_cache() == "flush" and calls == ["/slots/0", "/completion"]


def test_report_warns_about_flushed_failed_and_unrecorded_runs():
    st = Store(":memory:")
    st.insert("experiments", {"name": "e", "started_at": 0.0})
    for how in ("erase", "flush", None):
        st.insert("runs", {"exp_id": 1, "config_json": json.dumps({"prefix_cache": True}), "device": "mac",
                           "phase": "measured", "task_kind": "custom", "status": "complete", "started_at": 0.0,
                           "server_json": json.dumps({} if how is None else {"cache_reset": how})})
    w = cache_reset_warning(st, 1)
    assert "1 run(s): flush" in w and "1 run(s): not recorded" in w and "run(s): erase" not in w


def test_no_warning_when_every_cache_was_erased_or_cache_is_off():
    st = Store(":memory:")
    st.insert("experiments", {"name": "e", "started_at": 0.0})
    for cfg, how in (({"prefix_cache": True}, "erase"), ({"prefix_cache": False}, "flush"),
                     ({"prefix_cache": True}, "unsupported")):
        st.insert("runs", {"exp_id": 1, "config_json": json.dumps(cfg), "device": "mac", "phase": "measured",
                           "task_kind": "custom", "status": "complete", "started_at": 0.0,
                           "server_json": json.dumps({"cache_reset": how})})
    st.insert("runs", {"exp_id": 1, "config_json": "{}", "device": "mac", "phase": "measured", "task_kind": "bench",
                       "status": "complete", "started_at": 0.0, "server_json": "{}"})
    assert cache_reset_warning(st, 1) == ""


def test_warm_up_runs_do_not_trigger_the_warning():
    st = Store(":memory:")
    st.insert("experiments", {"name": "e", "started_at": 0.0})
    st.insert("runs", {"exp_id": 1, "config_json": json.dumps({"prefix_cache": True}), "device": "mac",
                       "phase": "warmup", "task_kind": "custom", "status": "complete", "started_at": 0.0,
                       "server_json": json.dumps({"cache_reset": "failed"})})
    assert cache_reset_warning(st, 1) == ""
