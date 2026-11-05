"""BFCL runners under harness variants."""
import pytest

pytest.importorskip("mpmath", reason="multi-turn backends need `uv sync --extra bfcl`")

from a2e.agent.harness import MAX_RETRIES, Harness  # noqa: E402
from a2e.bfcl.load import load_subset  # noqa: E402
from a2e.bfcl.multi_turn import run_multi_turn, score_trace  # noqa: E402
from a2e.bfcl.runner import raw_record, rescore, run_case  # noqa: E402
from a2e.bfcl.single_turn import run_single_turn_turns  # noqa: E402
from tests.bfcl_oracle import multi_outputs, single_output  # noqa: E402
from tests.stubs import StubClient  # noqa: E402

BAD = "no idea"


@pytest.fixture(scope="module")
def simple():
    return load_subset("simple_python", 1, 0)[0]


@pytest.fixture(scope="module")
def base0():
    return load_subset("multi_turn_base", 1, 0)[0]


def test_single_turn_default_is_one_call(simple):
    res, turns = run_single_turn_turns(StubClient([single_output(simple)]), simple)
    assert res.valid and len(turns) == 1


def test_single_turn_prompt_format_changes_the_prompt(simple):
    client = StubClient([single_output(simple)])
    res, _ = run_single_turn_turns(client, simple, harness=Harness(prompt_format="compact"))
    assert "Available tools (name(arguments)" in client.prompts[0] and res.valid


@pytest.mark.parametrize("recovery", ["feedback", "retry"])
def test_single_turn_recovers_from_an_unreadable_reply_and_counts_both_calls(simple, recovery):
    client = StubClient([BAD, single_output(simple)])
    res, turns = run_single_turn_turns(client, simple, harness=Harness(recovery=recovery))
    assert res.valid and len(turns) == 2 and res.extra["recovery_attempts"] == 1
    assert (BAD in client.prompts[1]) == (recovery == "feedback")


def test_single_turn_without_recovery_fails_on_the_unreadable_reply(simple):
    res, turns = run_single_turn_turns(StubClient([BAD, single_output(simple)]), simple)
    assert not res.valid and len(turns) == 1


def test_single_turn_recovery_is_bounded(simple):
    _, turns = run_single_turn_turns(StubClient([BAD] * 10), simple, harness=Harness(recovery="retry"))
    assert len(turns) == 1 + MAX_RETRIES


def test_multi_turn_default_unchanged(base0):
    tr = run_multi_turn(StubClient(multi_outputs(base0)), base0)
    assert score_trace(base0, tr).valid and tr.stop_reason is None


def test_multi_turn_oracle_still_passes_under_non_destructive_variants(base0):
    for h in (Harness(prompt_format="compact"), Harness(prompt_format="natural"), Harness(stopping="tight"),
              Harness(stopping="early_stop")):
        tr = run_multi_turn(StubClient(multi_outputs(base0)), base0, harness=h)
        assert score_trace(base0, tr).valid, h


def test_multi_turn_step_limit_records_why(base0):
    tr = run_multi_turn(StubClient(['[{"name": "ls", "arguments": {}}]']), base0, max_steps=3)
    assert tr.status == "force_terminated" and tr.stop_reason == "step_limit"


def test_multi_turn_early_stop_ends_a_repeating_model_sooner(base0):
    loop = ['[{"name": "ls", "arguments": {}}]']
    full = run_multi_turn(StubClient(loop), base0, max_steps=10)
    early = run_multi_turn(StubClient(loop), base0, max_steps=10, harness=Harness(stopping="early_stop"))
    assert early.status == "force_terminated" and early.stop_reason == "early_stop"
    assert len(early.turns) == 2 < len(full.turns)


def test_multi_turn_tight_stops_before_the_suite_limit(base0):
    loop = ['[{"name": "ls", "arguments": {}}]']
    tight = run_multi_turn(StubClient(loop), base0, max_steps=20, harness=Harness(stopping="tight"))
    assert tight.stop_reason == "step_limit" and len(tight.turns) < 20


@pytest.mark.parametrize("recovery", ["feedback", "retry"])
def test_multi_turn_recovery_gets_another_try(base0, recovery):
    outs = multi_outputs(base0)
    client = StubClient([BAD] + outs)
    tr = run_multi_turn(client, base0, harness=Harness(recovery=recovery))
    assert score_trace(base0, tr).valid and len(tr.turns) == len(outs) + 1


def test_multi_turn_without_recovery_treats_the_unreadable_reply_as_the_end_of_the_user_turn(base0):
    tr = run_multi_turn(StubClient([BAD] + multi_outputs(base0)), base0)
    assert not score_trace(base0, tr).valid


def test_run_case_passes_the_harness_and_marks_early_stops(base0):
    o = run_case(StubClient(['[{"name": "ls", "arguments": {}}]']), base0, max_steps=10,
                 harness=Harness(stopping="early_stop"))
    assert o.status == "force_terminated" and o.result.extra["stop_reason"] == "early_stop"


def test_raw_records_carry_a_non_default_harness_and_rescoring_replays_retries(simple, tmp_path):
    h = Harness(recovery="retry")
    o = run_case(StubClient([BAD, single_output(simple)]), simple, harness=h)
    assert o.result.valid and o.texts == [BAD, single_output(simple)]
    assert "harness" not in raw_record(o, 1) and "harness" not in raw_record(o, 1, Harness())
    path = tmp_path / "raw.jsonl"
    path.write_text(raw_record(o, 1, h) + "\n")
    assert [r.valid for r in rescore(path)] == [True]
