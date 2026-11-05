"""The agent loop under each harness variant, with scripted model output."""
import json

import pytest

from a2e.agent.harness import MASKED, MAX_RETRIES, TIGHT_SLACK, Harness
from a2e.agent.loop import AgentLoop
from tests.oracle import make_env
from tests.stubs import StubClient


def call(name, **args):
    return json.dumps([{"name": name, "arguments": args}])


FINAL = call("final_answer", answer="done")
BAD = "I am not sure what to do."
CALC = call("calculator", expression="1+1")


@pytest.fixture
def reg(tmp_path):
    return make_env(tmp_path)[0]


def run(reg, outputs, harness=None, **kw):
    client = StubClient(outputs)
    return AgentLoop(client, reg, reg.specs(), harness=harness).run("x", **kw), client


def test_default_harness_matches_no_harness(reg):
    a, ca = run(reg, [CALC, FINAL])
    b, cb = run(reg, [CALC, FINAL], Harness())
    assert ca.prompts == cb.prompts and a.status == b.status == "final_answer"


# ---------------------------------------------------------------------------------------------- prompt format
@pytest.mark.parametrize("fmt, marker", [("compact", "Available tools (name(arguments)"),
                                         ("natural", "Available tools:\n<tools>")])
def test_prompt_format_reaches_the_model(reg, fmt, marker):
    trace, client = run(reg, [CALC, FINAL], Harness(prompt_format=fmt))
    assert marker in client.prompts[0] and trace.status == "final_answer"
    assert '"parameters"' not in client.prompts[0]


# ---------------------------------------------------------------------------------------------- recovery
def test_none_ends_on_the_first_unreadable_reply(reg):
    trace, client = run(reg, [BAD, FINAL])
    assert trace.status == "parse_error" and len(client.prompts) == 1


def test_feedback_keeps_the_bad_reply_and_says_what_was_wrong(reg):
    trace, client = run(reg, [BAD, FINAL], Harness(recovery="feedback"))
    assert trace.status == "final_answer" and len(trace.turns) == 2
    assert BAD in client.prompts[1] and "could not be read" in client.prompts[1] and "no_json" in client.prompts[1]


def test_retry_drops_the_bad_reply_and_adds_a_reminder_that_is_not_kept(reg):
    trace, client = run(reg, [BAD, CALC, FINAL], Harness(recovery="retry"))
    assert trace.status == "final_answer" and len(trace.turns) == 3
    assert BAD not in client.prompts[1] and "Reply with only a JSON array" in client.prompts[1]
    assert "Reply with only a JSON array" not in client.prompts[2]  # forgotten once the next reply is readable


def test_retry_gives_up_after_the_limit(reg):
    trace, client = run(reg, [BAD] * 10, Harness(recovery="retry"))
    assert trace.status == "parse_error" and len(trace.turns) == MAX_RETRIES + 1


def test_retry_count_resets_after_a_good_turn(reg):
    outs = [BAD, BAD, CALC, BAD, BAD, FINAL]
    trace, _ = run(reg, outs, Harness(recovery="retry"))
    assert trace.status == "final_answer" and len(trace.turns) == 6


def test_feedback_is_bounded_by_the_turn_limit(reg):
    trace, _ = run(reg, [BAD] * 20, Harness(recovery="feedback"), max_turns=5)
    assert trace.status == "max_turns" and len(trace.turns) == 5


# ---------------------------------------------------------------------------------------------- context
def test_mask_hides_old_tool_results_from_the_model_only(reg):
    outs = [CALC] * 4 + [FINAL]
    trace, client = run(reg, outs, Harness(context="mask"))
    assert trace.status == "final_answer"
    assert MASKED not in client.prompts[2] and MASKED in client.prompts[3]
    assert all(t.results[0].output for t in trace.turns[:4])  # the real results still exist in the trace


def test_truncate_shortens_long_tool_output(reg, tmp_path):
    reg.tools["calculator"].fn = lambda expression: "y" * 2000
    trace, client = run(reg, [CALC, FINAL], Harness(context="truncate"))
    assert "[truncated]" in client.prompts[1] and "y" * 1000 not in client.prompts[1]


# ---------------------------------------------------------------------------------------------- stopping
def test_tight_caps_the_turns_at_what_a_correct_run_needs(reg):
    trace, client = run(reg, [CALC] * 20, Harness(stopping="tight"), max_turns=8, expected_calls=3)
    assert len(trace.turns) == 3 + TIGHT_SLACK and trace.status == "max_turns"


def test_tight_without_a_known_need_uses_the_suite_limit(reg):
    trace, _ = run(reg, [CALC] * 20, Harness(stopping="tight"), max_turns=6)
    assert len(trace.turns) == 6


def test_early_stop_ends_on_a_repeated_call(reg):
    trace, _ = run(reg, [CALC] * 20, Harness(stopping="early_stop"), max_turns=8)
    assert trace.status == "early_stop" and len(trace.turns) == 2


def test_early_stop_ends_on_two_failed_turns(reg):
    bad_args = [call("calculator", wrong=1), call("calculator", other=2)]
    trace, _ = run(reg, bad_args + [FINAL], Harness(stopping="early_stop"))
    assert trace.status == "early_stop" and len(trace.turns) == 2


def test_early_stop_lets_a_varied_run_finish(reg):
    outs = [call("calculator", expression="1+1"), call("calculator", expression="2+2"), FINAL]
    trace, _ = run(reg, outs, Harness(stopping="early_stop"))
    assert trace.status == "final_answer"


def test_fixed_runs_until_the_limit(reg):
    trace, _ = run(reg, [CALC] * 20, max_turns=7)
    assert trace.status == "max_turns" and len(trace.turns) == 7
