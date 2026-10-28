"""Real llama-server smoke + the assumptions the fake server encodes. Works on Pi and Mac."""
import json
import time

import httpx
import pytest

from a2e.client import Client
from a2e.metrics.sampler import Sampler

from .conftest import needs_llama

pytestmark = needs_llama

SCHEMA = ("You can call tools. " * 80)
JSON_GRAMMAR = r'root ::= "{" ws "\"ok\"" ws ":" ws ("true" | "false") ws "}"' + "\nws ::= [ \\t\\n]*"


def test_server_boots_and_reports_timing_fields(real_server):
    srv = real_server()
    r = Client(srv.url).complete("Hello, my name is", 16)
    raw = r.raw["timings"]
    # field names the client depends on; if this fails the llama.cpp build changed its schema
    for k in ("prompt_n", "prompt_ms", "predicted_n", "predicted_ms", "cache_n"):
        assert k in raw, f"timings missing {k}: {sorted(raw)}"
    assert r.gen_tokens > 0 and r.decode_ms > 0 and r.decode_tok_s > 0


def test_prefix_cache_hit_on_repeat(real_server):
    c = Client(real_server().url)
    first = c.complete(SCHEMA + "Question A:", 8)
    second = c.complete(SCHEMA + "Question B:", 8)
    assert first.cached_tokens == 0
    assert second.cached_tokens > 50, "prefix cache did not reuse the schema prefix"
    assert second.prefill_ms < first.prefill_ms


def test_cache_prompt_false_disables_reuse(real_server):
    c = Client(real_server().url)
    c.complete(SCHEMA + "Q:", 4)
    assert c.complete(SCHEMA + "Q:", 4, cache_prompt=False).cached_tokens == 0


def test_slot_erase_endpoint(real_server):
    c = Client(real_server().url)
    c.complete(SCHEMA + "Q:", 4)
    c.reset_cache()
    assert c.complete(SCHEMA + "Q:", 4).cached_tokens == 0, "POST /slots/0?action=erase did not clear cache"


def test_grammar_constrains_output(real_server):
    c = Client(real_server().url)
    r = c.complete("Is the sky blue? Answer as JSON:", 32, grammar=JSON_GRAMMAR)
    assert isinstance(json.loads(r.text)["ok"], bool)


def test_greedy_decoding_is_deterministic(real_server):
    c = Client(real_server().url)
    a = c.complete("The capital of France is", 12, cache_prompt=False).text
    b = c.complete("The capital of France is", 12, cache_prompt=False).text
    assert a == b


def test_rss_reflects_model_size(real_server):
    import os
    srv = real_server()
    size_mb = os.path.getsize(os.environ["A2E_GGUF"]) / 2**20
    s = Sampler(srv.pid, hz=10).start()
    Client(srv.url).complete("warm", 8)
    time.sleep(0.3)
    peak = max(x[1] for x in s.stop())
    # weights are mmapped; on Mac with Metal they may be wired outside RSS, so only bound loosely
    assert peak > 0.2 * size_mb


@pytest.mark.parametrize("kv", ["f16", "q8_0"])
def test_kv_cache_quant_boots_and_generates(real_server, kv):
    srv = real_server(kv_quant=kv)
    assert Client(srv.url).complete("Hello", 8).gen_tokens > 0


@pytest.mark.parametrize("grammar", [False, True])
def test_grammar_overhead_measured_on_real_server(real_server, grammar):
    c = Client(real_server().url)
    c.complete("warm", 4)
    g = JSON_GRAMMAR if grammar else None
    r = c.complete("Is water wet? Answer as JSON:", 24, grammar=g, cache_prompt=False)
    assert r.decode_tok_s > 0  # compare the two params' reported tok/s in the -v output


def test_long_schema_prefill_dominated_by_prompt(real_server):
    c = Client(real_server().url)
    r = c.complete(("tool " * 1500) + "go", 4, cache_prompt=False)
    assert r.prefill_ms > r.decode_ms
    assert r.prefill_tok_s > r.decode_tok_s  # prefill batches; if not, offload/threading is misconfigured


def test_unreachable_after_stop(real_server):
    srv = real_server()
    srv.stop()
    with pytest.raises(httpx.HTTPError):
        httpx.get(f"{srv.url}/health", timeout=1)
