import httpx
import pytest

from a2e.client import Client

PROMPT = "schema " * 200 + "alpha"


def test_basic_completion_has_all_stage_fields(fake_server):
    c = Client(fake_server().url)
    r = c.complete(PROMPT, n_predict=20)
    assert r.gen_tokens == 20 and r.prompt_tokens == 201 and r.cached_tokens == 0
    assert r.prefill_ms > 0 and r.decode_ms > 0 and r.decode_tok_s > 0 and r.prefill_tok_s > 0


def test_wall_covers_server_stage_times(fake_server):
    r = Client(fake_server().url).complete(PROMPT, 50)
    assert r.wall_ms >= (r.prefill_ms + r.decode_ms) * 0.95  # server sleeps for both


def test_prefix_cache_hit_on_shared_schema(fake_server):
    c = Client(fake_server().url)
    first = c.complete(PROMPT, 8)
    second = c.complete("schema " * 200 + "beta", 8)
    assert first.cached_tokens == 0
    assert second.cached_tokens == 200  # whole schema reused
    assert second.prefill_ms < first.prefill_ms


def test_prefix_cache_disabled_per_request(fake_server):
    c = Client(fake_server().url)
    c.complete(PROMPT, 8)
    assert c.complete(PROMPT, 8, cache_prompt=False).cached_tokens == 0


def test_identical_prompt_still_evaluates_one_token(fake_server):
    c = Client(fake_server().url)
    c.complete(PROMPT, 8)
    r = c.complete(PROMPT, 8)
    assert r.cached_tokens == 200 and r.prompt_tokens - r.cached_tokens == 1


def test_reset_cache_clears_prefix(fake_server):
    c = Client(fake_server().url)
    c.complete(PROMPT, 8)
    c.reset_cache()
    assert c.complete(PROMPT, 8).cached_tokens == 0


def test_grammar_adds_decode_overhead(fake_server):
    c = Client(fake_server("--grammar-overhead", "2.0").url)
    plain = c.complete("a b c", 100)
    gram = c.complete("a b c", 100, grammar='root ::= "x"+')
    assert gram.decode_ms == pytest.approx(plain.decode_ms * 2.0, rel=0.01)


def test_slots_endpoint(fake_server):
    assert Client(fake_server().url).slots()[0]["id"] == 0


def test_http_error_propagates(fake_server):
    c = Client(fake_server("--fail-completion", "500").url)
    with pytest.raises(httpx.HTTPStatusError):
        c.complete("hi", 4)


def test_malformed_json_raises(fake_server):
    c = Client(fake_server("--bad-json").url)
    with pytest.raises(ValueError):
        c.complete("hi", 4)


def test_missing_timings_yields_zeroes_not_crash(fake_server):
    r = Client(fake_server("--no-timings").url).complete("hi there", 4)
    assert r.prefill_ms == 0 and r.decode_ms == 0 and r.decode_tok_s == 0 and r.prefill_tok_s == 0


def test_client_timeout(fake_server):
    c = Client(fake_server("--ms-per-gen-token", "500").url, timeout=0.3)
    with pytest.raises(httpx.TimeoutException):
        c.complete("hi", 10)


def test_connection_refused_after_server_stops(fake_server):
    srv = fake_server()
    c = Client(srv.url)
    srv.stop()
    with pytest.raises(httpx.ConnectError):
        c.complete("hi", 4)
