import json

import pytest

from a2e.agent.parse import ToolCall
from a2e.tasks.custom import FAMILY_TOOLS, generate
from a2e.tasks.scoring import ExpectedCall, norm, score_call
from a2e.tools.builtin import build_registry, make_corpus


@pytest.mark.parametrize("a,b", [("  Hello   World ", "hello world"), ("5", 5), (5, 5.0), ("5.0", 5),
                                 ("A", "a")])
def test_norm_equal(a, b):
    assert norm(a) == norm(b)


@pytest.mark.parametrize("a,b", [("5", 6), ("abc", "abd"), (True, 1.0 + 1), ("x", None)])
def test_norm_unequal(a, b):
    assert norm(a) != norm(b)


def call(name, **args):
    return ToolCall(name, args)


def test_correct_and_case_space_insensitive():
    e = ExpectedCall("note_write", {"title": "Alpha 1", "priority": 3})
    assert score_call(e, [], call("note_write", title="alpha  1", priority="3")).ok


def test_wrong_tool_and_wrong_arg():
    e = ExpectedCall("calculator", {"expression": "1+1"})
    assert not score_call(e, [], call("note_list")).name_ok
    s = score_call(e, [], call("calculator", expression="1+2"))
    assert s.name_ok and not s.args_ok and "expression" in s.note


def test_extra_args_ignored_and_missing_arg_fails():
    e = ExpectedCall("note_write", {"title": "t"})
    assert score_call(e, [], call("note_write", title="t", body="extra")).ok
    assert not score_call(e, [], call("note_write", body="x")).ok


def test_compare_keys_limits_comparison():
    e = ExpectedCall("note_write", {"title": "t", "body": "specific"}, compare_keys=["title"])
    assert score_call(e, [], call("note_write", title="t", body="whatever")).ok


def test_squash_whitespace_for_expressions():
    e = ExpectedCall("calculator", {"expression": "12 * 7"}, squash_ws=True)
    assert score_call(e, [], call("calculator", expression="12*7")).ok
    assert not ExpectedCall("calculator", {"expression": "12 * 7"}).resolve([]) == {}


def test_dependent_args_use_previous_results_and_unresolvable_is_wrong():
    e = ExpectedCall("doc_read", lambda r: {"doc_id": json.loads(r[0])[0]["doc_id"]})
    assert score_call(e, ['[{"doc_id": "doc_004"}]'], call("doc_read", doc_id="doc_004")).ok
    s = score_call(e, [], call("doc_read", doc_id="doc_004"))
    assert not s.ok and "unresolvable" in s.note
    assert not score_call(e, ["not json"], call("doc_read", doc_id="x")).ok


def test_non_dict_arguments_scored_wrong():
    assert not score_call(ExpectedCall("f", {"a": 1}), [], ToolCall("f", [1])).args_ok


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    return make_corpus(tmp_path_factory.mktemp("c"), n=10, seed=1)


def test_generation_is_deterministic_and_seeded(corpus):
    a = generate("notes", 5, 3, 3, seed=1)
    b = generate("notes", 5, 3, 3, seed=1)
    assert [t.prompt for t in a] == [t.prompt for t in b]
    assert [t.prompt for t in a] != [t.prompt for t in generate("notes", 5, 3, 3, seed=2)]
    assert len({t.task_id for t in a}) == 5 and len({t.prompt for t in a}) > 1


@pytest.mark.parametrize("depth", [1, 2, 3, 4, 5])
def test_calc_and_notes_depth_gives_expected_step_count(depth):
    assert all(len(t.steps) == depth for t in generate("calc", 3, depth))
    assert all(len(t.steps) == depth for t in generate("notes", 3, depth))


@pytest.mark.parametrize("depth", [2, 3, 4, 5])
def test_retrieval_depth(depth, corpus):
    assert all(len(t.steps) == depth for t in generate("retrieval", 3, depth, corpus=corpus))


@pytest.mark.parametrize("n_args", [1, 2, 3, 4, 5])
def test_note_arg_count_controls_expected_args(n_args):
    for t in generate("notes", 4, 3, n_args):
        writes = [s for s in t.steps if s.name == "note_write"]
        assert writes and all(len(s.args) == n_args for s in writes)
        assert all(str(v) in t.prompt for s in writes for v in s.args.values())


def test_tool_sets_exist_and_contain_all_used_tools(tmp_path, corpus):
    names = {s["name"] for s in build_registry(tmp_path).specs()}
    for fam in FAMILY_TOOLS:
        tasks = generate(fam, 2, 3, corpus=corpus)
        for t in tasks:
            assert set(t.tool_names) <= names and "final_answer" in t.tool_names
            assert {s.name for s in t.steps} <= set(t.tool_names)


@pytest.mark.parametrize("kw", [dict(family="calc", depth=0), dict(family="calc", depth=6),
                                dict(family="notes", depth=2, n_args=0), dict(family="notes", depth=2, n_args=6),
                                dict(family="retrieval", depth=1), dict(family="retrieval", depth=3)])
def test_invalid_parameters_raise(kw):
    kw = {"n": 1, "n_args": 2, **kw}  # retrieval depth=3 has no corpus -> raises
    with pytest.raises(ValueError):
        generate(**kw)
