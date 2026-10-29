import json

import pytest

import a2e.bfcl  # noqa: F401
from a2e.bfcl import score as bscore
from a2e.bfcl.load import load_cases
from a2e.bfcl.multi_turn import run_multi_turn, score_trace
from a2e.bfcl.single_turn import run_single_turn
from tests.bfcl_oracle import multi_outputs, parse_gt_string, single_output
from tests.stubs import StubClient

pytest.importorskip("mpmath", reason="multi-turn backends need `uv sync --extra bfcl`")


def case_by_id(cat, cid):
    return next(c for c in load_cases(cat) if c.id == cid)


# ---------------------------------------------------------------- single turn
def test_single_turn_pass_records_stage_timings_and_schema_tokens():
    c = case_by_id("simple_python", "simple_python_0")
    client = StubClient([single_output(c)])
    res, turn = run_single_turn(client, c, count_tokens=client.tokenize)
    assert res.valid and res.extra["schema_tokens"] > 10
    assert turn.completion.prefill_ms > 0 and turn.wall_ms > 0
    assert turn.validate_ms == 0 and turn.tool_wait_ms == 0  # single-turn has no tool execution
    assert turn.windows["parse"][0] <= turn.windows["tool"][1]
    assert "calculate_triangle_area" in client.prompts[0] and "Find the area of a triangle" in client.prompts[0]


def test_single_turn_wrong_answer_and_parse_failures_are_scored_not_raised():
    c = case_by_id("simple_python", "simple_python_0")
    bad = json.dumps([{"name": "calculate_triangle_area", "arguments": {"base": 99, "height": 5}}])
    for text, kind in [(bad, "value_error"), ("I do not know", "a2e:no_call:no_json"),
                       ('[{"name": "calculate_triangle_area", "argum', "a2e:no_call:malformed_json"),
                       ("[]", "a2e:no_call:empty"), ("[1, 2]", "a2e:no_call:bad_shape")]:
        res, _ = run_single_turn(StubClient([text]), c)
        assert not res.valid and (res.error_type or "").startswith(kind), (text, res.error_type)


def test_single_turn_non_object_arguments_do_not_crash_the_checker():
    c = case_by_id("simple_python", "simple_python_0")
    res, _ = run_single_turn(StubClient(['[{"name": "calculate_triangle_area", "arguments": [1, 2]}]']), c)
    assert not res.valid and res.error_type.startswith("a2e:checker_exception")


@pytest.mark.parametrize("cat,answer,valid", [("irrelevance", "[]", True), ("irrelevance", "I cannot help", True),
                                              ("irrelevance", '[{"name":"f","arguments":{}}]', False),
                                              ("live_relevance", "[]", False),
                                              ("live_relevance", '[{"name":"f","arguments":{}}]', True)])
def test_single_turn_relevance_rules(cat, answer, valid):
    c = load_cases(cat)[0]
    assert run_single_turn(StubClient([answer]), c)[0].valid is valid


def test_single_turn_grammar_schema_allows_empty_and_pins_function_names():
    c = case_by_id("multiple", "multiple_0")
    client = StubClient([single_output(c)])
    run_single_turn(client, c, grammar=True)
    schema = client.calls[0]["json_schema"]
    assert schema["minItems"] == 0 and schema["maxItems"] == 8
    names = {o["properties"]["name"]["const"] for o in schema["items"]["anyOf"]}
    assert names == {d["name"] for d in c.functions} and len(names) > 1
    assert run_single_turn(StubClient([single_output(c)]), c)[1].completion is not None
    c2 = StubClient([single_output(c)])
    run_single_turn(c2, c, grammar=False)
    assert c2.calls[0]["json_schema"] is None


def test_single_turn_server_error_is_a_scored_failure():
    c = case_by_id("simple_python", "simple_python_0")
    res, turn = run_single_turn(StubClient(["[]"], fail_at=0), c)
    assert not res.valid and res.error_type == "a2e:server_error" and turn.error


def test_single_turn_cache_prompt_flag_passes_through():
    c = case_by_id("simple_python", "simple_python_0")
    client = StubClient(["[]"])
    run_single_turn(client, c, cache_prompt=False, n_predict=77)
    assert client.calls[0]["cache_prompt"] is False and client.calls[0]["n_predict"] == 77


# ---------------------------------------------------------------- multi turn
@pytest.fixture
def base0():
    return case_by_id("multi_turn_base", "multi_turn_base_0")


def test_multi_turn_oracle_trace_structure(base0):
    client = StubClient(multi_outputs(base0))
    tr = run_multi_turn(client, base0)
    assert tr.status == "done" and score_trace(base0, tr).valid
    assert tr.user_turn_of == sorted(tr.user_turn_of) and set(tr.user_turn_of) == set(range(len(base0.turns)))
    assert len(tr.per_turn_steps) == len(base0.turns) and all(len(s) == 1 for s in tr.per_turn_steps)
    called = [t for t in tr.turns if t.calls]
    assert sum(t.tool_wait_ms for t in called) > 0 and all(t.results[0].ok for t in called)
    assert not tr.unseen_calls


def test_multi_turn_prompt_only_grows_so_prefix_cache_applies(base0):
    client = StubClient(multi_outputs(base0))
    run_multi_turn(client, base0)
    for a, b in zip(client.prompts, client.prompts[1:]):
        assert b.startswith(a)


def test_multi_turn_hallucinated_function_rejected_and_fed_back(base0):
    outs = ['[{"name": "definitely_not_a_function", "arguments": {}}]', "[]"] * len(base0.turns)
    client = StubClient(outs)
    tr = run_multi_turn(client, base0)
    first = tr.turns[0]
    assert first.validation_errors[0] and not first.results[0].ok and "invalid call" in client.prompts[1]
    assert all(not s for s in tr.per_turn_steps)  # nothing was executed
    assert not score_trace(base0, tr).valid


def test_multi_turn_injection_attempts_are_inert(base0, tmp_path):
    marker = tmp_path / "pwned"
    evil = [{"name": f"__import__('os').system('touch {marker}')", "arguments": {}},
            {"name": "cd", "arguments": {"folder": f"x'); __import__('os').system('touch {marker}'); ('"}},
            {"name": "cd", "arguments": {f"x=1); __import__('os').system('touch {marker}'); cd(y": 1}}]
    client = StubClient([json.dumps(evil), "[]"] + ["[]"] * 10)
    tr = run_multi_turn(client, base0)
    assert not marker.exists()
    assert [e is not None for e in tr.turns[0].validation_errors] == [True, False, True]
    assert any("__import__" in s for step in tr.per_turn_steps[0] for s in step)  # ran only as a string literal


def test_multi_turn_model_prose_ends_the_turn_and_conversation_continues(base0):
    client = StubClient(["Sorry, I cannot do that."])
    tr = run_multi_turn(client, base0)
    assert len(tr.turns) == len(base0.turns) and all(t.parse_error for t in tr.turns)
    assert all(s == [] for s in tr.per_turn_steps) and not score_trace(base0, tr).valid
    assert "Sorry, I cannot do that." in client.prompts[1]


def test_multi_turn_force_terminated_after_step_limit(base0):
    client = StubClient(['[{"name": "ls", "arguments": {}}]'])
    tr = run_multi_turn(client, base0)
    assert tr.status == "force_terminated" and len(tr.turns) == 20
    res = score_trace(base0, tr)
    assert not res.valid and res.error_type == "multi_turn:force_terminated"


def test_multi_turn_custom_step_limit(base0):
    tr = run_multi_turn(StubClient(['[{"name": "ls", "arguments": {}}]']), base0, max_steps=3)
    assert tr.status == "force_terminated" and len(tr.turns) == 3


def test_multi_turn_server_error_mid_case(base0):
    tr = run_multi_turn(StubClient(multi_outputs(base0), fail_at=2), base0)
    res = score_trace(base0, tr)
    assert tr.status == "server_error" and tr.turns[-1].error
    assert not res.valid and res.error_type == "a2e:server_error"


def test_backend_results_are_visible_to_the_model(base0):
    client = StubClient(multi_outputs(base0))
    tr = run_multi_turn(client, base0)
    first_call = next(i for i, t in enumerate(tr.turns) if t.calls)
    out = tr.turns[first_call].results[0].output
    assert json.dumps(out)[1:-1] in client.prompts[first_call + 1]  # tool output travels JSON-encoded


def test_same_case_run_twice_gets_fresh_state_each_time(base0):
    for _ in range(2):
        tr = run_multi_turn(StubClient(multi_outputs(base0)), base0)
        assert score_trace(base0, tr).valid


def test_scoring_twice_is_stable_and_leaks_no_instances(base0):
    from bfcl_eval.eval_checker.multi_turn_eval import multi_turn_utils as mtu
    tr = run_multi_turn(StubClient(multi_outputs(base0)), base0)
    assert score_trace(base0, tr).valid and score_trace(base0, tr).valid
    assert not [k for k in vars(mtu) if k.startswith("a2e")]


def test_scoring_detects_a_wrong_state(base0):
    outs = multi_outputs(base0)
    first = next(i for i, o in enumerate(outs) if o != "[]")
    calls = json.loads(outs[first])
    calls[-1]["arguments"] = {k: "zzz" for k in calls[-1]["arguments"]} or {"x": 1}
    outs[first] = json.dumps(calls)
    res = score_trace(base0, run_multi_turn(StubClient(outs), base0))
    assert not res.valid and res.error_type.startswith("multi_turn")


# ------------------------------------------------------------ miss_func / miss_param
@pytest.fixture
def miss_func():
    return next(c for c in load_cases("multi_turn_miss_func") if c.held_out)


def test_held_out_docs_are_revealed_by_appending_a_user_message(miss_func):
    t_reveal = next(iter(miss_func.held_out))
    hidden = {d["name"] for d in miss_func.held_out[t_reveal]}
    client = StubClient(multi_outputs(miss_func))
    tr = run_multi_turn(client, miss_func)
    first_idx = tr.user_turn_of.index(t_reveal)
    name = sorted(hidden)[0]
    assert f'"name":"{name}"' not in client.prompts[first_idx - 1]
    assert f'"name":"{name}"' in client.prompts[first_idx]
    assert "I have updated some more functions you can choose from. What about now?" in client.prompts[first_idx]
    for a, b in zip(client.prompts, client.prompts[1:]):  # append-only, so the prefix cache keeps working
        assert b.startswith(a)
    c = tr.turns[first_idx].completion
    assert c.cached_tokens > 0.5 * c.prompt_tokens


def test_calling_a_held_out_function_early_is_allowed_but_flagged():
    case = next(c for c in load_cases("multi_turn_miss_func") if c.id == "multi_turn_miss_func_49")
    tr = run_multi_turn(StubClient(multi_outputs(case)), case)
    assert "tail" in tr.unseen_calls  # upstream's own ground truth needs `tail` before it is revealed
    res = score_trace(case, tr)
    assert res.valid and "tail" in res.extra["unseen_calls"]


def test_grammar_schema_is_rebuilt_when_functions_are_revealed(miss_func):
    t_reveal = next(iter(miss_func.held_out))
    hidden = {d["name"] for d in miss_func.held_out[t_reveal]}
    client = StubClient(multi_outputs(miss_func))
    tr = run_multi_turn(client, miss_func, grammar=True)
    def names(i):
        return {o["properties"]["name"]["const"] for o in client.calls[i]["json_schema"]["items"]["anyOf"]}
    first_idx = tr.user_turn_of.index(t_reveal)
    assert not (hidden & names(first_idx - 1)) and hidden <= names(first_idx)
    assert all(c["json_schema"]["minItems"] == 0 for c in client.calls)


def test_miss_param_empty_ground_truth_turn_must_have_no_call():
    case = next(c for c in load_cases("multi_turn_miss_param") if any(not t for t in c.ground_truth))
    ok = score_trace(case, run_multi_turn(StubClient(multi_outputs(case)), case))
    assert ok.valid and ok.extra["irrelevance_valid"] is True
    # a model that calls something on the turn that should stay empty
    docs = {d["name"]: d for d in case.functions}
    outs = []
    for gt in case.ground_truth:
        outs.append(json.dumps([parse_gt_string(x, docs) for x in gt]) if gt else '[{"name": "ls", "arguments": {}}]')
        outs.append("[]")
    res = score_trace(case, run_multi_turn(StubClient(outs), case))
    assert res.extra["irrelevance_valid"] is False
