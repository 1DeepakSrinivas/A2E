"""BFCL through a real llama-server. Accuracy is printed, not asserted: these check the plumbing
(json_schema honoured, timings present, multi-turn runs to completion), not model quality.
Multi-turn cases need a large context: set A2E_CTX=16384 or more."""
import os

import pytest

import a2e.bfcl  # noqa: F401
from a2e.bfcl.load import load_subset
from a2e.bfcl.runner import run_case, safe_token_counter
from a2e.client import Client

from .conftest import needs_llama

pytestmark = needs_llama
pytest.importorskip("mpmath", reason="multi-turn backends need `uv sync --extra bfcl`")


def ctx() -> int:
    return int(os.environ.get("A2E_CTX", 4096))


@pytest.mark.parametrize("category", ["simple_python", "multiple", "parallel"])
def test_grammar_on_yields_parseable_calls_for_every_case(real_server, category):
    c = Client(real_server().url)
    outs = [run_case(c, case, grammar=True, n_predict=256) for case in load_subset(category, 4, seed=0)]
    bad = [o.case.id for o in outs if o.turns[0].parse_error]
    assert not bad, f"constrained output failed to parse for {bad}"
    print(category, "valid:", sum(o.result.valid for o in outs), "/", len(outs))


@pytest.mark.parametrize("grammar", [False, True])
def test_single_turn_reports_timings_and_schema_tokens(real_server, grammar):
    c = Client(real_server().url)
    o = run_case(c, load_subset("multiple", 1)[0], grammar=grammar, count_tokens=safe_token_counter(c))
    t = o.turns[0]
    assert t.completion.prefill_ms > 0 and t.completion.decode_ms > 0
    assert o.result.extra["schema_tokens"] and t.completion.prompt_tokens >= o.result.extra["schema_tokens"]


def test_irrelevance_declines_or_not_but_never_crashes(real_server):
    c = Client(real_server().url)
    outs = [run_case(c, case, grammar=False, n_predict=128) for case in load_subset("irrelevance", 4)]
    print("irrelevance valid:", sum(o.result.valid for o in outs), "/", len(outs))
    assert all(o.status == "done" for o in outs)


@pytest.mark.skipif(ctx() < 16384, reason="multi-turn schemas need A2E_CTX >= 16384")
@pytest.mark.parametrize("category", ["multi_turn_base", "multi_turn_miss_func"])
def test_multi_turn_runs_to_completion_with_real_backends(real_server, category):
    c = Client(real_server().url)
    o = run_case(c, load_subset(category, 1, seed=0)[0], grammar=True, n_predict=256, max_steps=6,
                 count_tokens=safe_token_counter(c))
    print(category, o.status, o.result.valid, o.result.error_type, "schema tokens:", o.result.extra["schema_tokens"])
    assert o.status in ("done", "force_terminated") and o.turns
    called = [t for t in o.turns if t.calls]
    if called:
        assert sum(t.tool_wait_ms for t in called) > 0
    assert o.result.extra["schema_tokens"] and o.result.extra["schema_tokens"] > 1000


@pytest.mark.skipif(ctx() < 16384, reason="multi-turn schemas need A2E_CTX >= 16384")
def test_multi_turn_prefix_cache_applies_across_steps(real_server):
    c = Client(real_server().url)
    o = run_case(c, load_subset("multi_turn_base", 1, seed=0)[0], grammar=True, max_steps=6)
    later = [t.completion.cached_tokens / t.completion.prompt_tokens for t in o.turns[1:] if t.completion]
    assert later and max(later) > 0.5, f"cache hit rates after the first call: {later}"
