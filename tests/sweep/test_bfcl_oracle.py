"""The main proof that the vendored official checkers work: replaying ground truth passes every case,
and corrupting it fails. Hundreds of cases per run."""
import json

import pytest

import a2e.bfcl  # noqa: F401  (puts the vendored checker on sys.path)
from a2e.agent.parse import parse_calls
from a2e.bfcl.load import (AST_CATEGORIES, MULTI_TURN_CATEGORIES, RELEVANCE_CATEGORIES, load_cases)
from a2e.bfcl.multi_turn import run_multi_turn, score_trace
from a2e.bfcl.score import score_single
from tests.bfcl_oracle import multi_outputs, perturbations, single_output
from tests.stubs import StubClient

pytest.importorskip("mpmath", reason="multi-turn backends need `uv sync --extra bfcl`")


def parsed(text):
    r = parse_calls(text, allow_empty=True)
    return r.calls, r.error


@pytest.mark.parametrize("category", AST_CATEGORIES)
def test_oracle_passes_every_single_turn_case(category):
    cases = load_cases(category)
    failures = []
    for c in cases:
        res = score_single(c, *parsed(single_output(c)))
        if not res.valid:
            failures.append((c.id, res.error_type, res.detail))
    assert not failures, f"{len(failures)}/{len(cases)} oracle outputs rejected, e.g. {failures[:3]}"


@pytest.mark.parametrize("category", AST_CATEGORIES)
def test_every_perturbation_is_rejected(category):
    leaks = []
    for c in load_cases(category):
        for name, text in perturbations(c).items():
            if score_single(c, *parsed(text)).valid:
                leaks.append((c.id, name))
    assert not leaks, f"{len(leaks)} wrong outputs accepted, e.g. {leaks[:5]}"


@pytest.mark.parametrize("category", list(RELEVANCE_CATEGORIES))
def test_relevance_rule_both_directions(category):
    expect_call = RELEVANCE_CATEGORIES[category]
    for c in load_cases(category):
        no_call = score_single(c, *parsed("[]")).valid
        name = c.functions[0]["name"] if c.functions else "some_function"  # a few live cases list no functions
        a_call = score_single(c, *parsed(json.dumps([{"name": name, "arguments": {}}]))).valid
        assert no_call == (not expect_call) and a_call == expect_call, c.id


@pytest.mark.parametrize("category", MULTI_TURN_CATEGORIES)
def test_oracle_passes_every_multi_turn_case(category):
    cases = load_cases(category)
    failures = []
    for c in cases:
        trace = run_multi_turn(StubClient(multi_outputs(c)), c)
        res = score_trace(c, trace)
        if not res.valid:
            failures.append((c.id, trace.status, res.error_type, str(res.detail)[:120]))
    assert not failures, f"{len(failures)}/{len(cases)} oracle runs rejected, e.g. {failures[:3]}"


@pytest.mark.parametrize("category", MULTI_TURN_CATEGORIES)
def test_dropping_a_call_fails_multi_turn_cases(category):
    """Remove the first ground-truth call of the first non-empty turn: state or responses must differ."""
    accepted, total = 0, 0
    for c in load_cases(category)[:60]:
        outs = multi_outputs(c)
        first = next(i for i, o in enumerate(outs) if o != "[]")
        calls = json.loads(outs[first])
        outs[first] = json.dumps(calls[1:]) if len(calls) > 1 else "[]"
        res = score_trace(c, run_multi_turn(StubClient(outs), c))
        total += 1
        accepted += res.valid
    assert accepted <= total * 0.1, f"{accepted}/{total} cases still passed with a ground-truth call removed"
