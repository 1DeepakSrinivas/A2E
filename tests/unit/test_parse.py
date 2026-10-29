import json

import pytest

from a2e.agent.parse import parse_calls

C = '{"name": "calculator", "arguments": {"expression": "1+1"}}'


def names(r):
    return [c.name for c in r.calls]


@pytest.mark.parametrize("text", [
    f"[{C}]",
    C,
    f"```json\n[{C}]\n```",
    f"Sure! Here you go: [{C}] hope that helps",
    f"[{C}] trailing garbage {{",
    f"\n\n  [{C}]  \n",
    f"<think>{{\"name\": \"x\"}} hmm [</think>[{C}]",
    f"<tool_call>{C}</tool_call>",
])
def test_single_call_formats(text):
    r = parse_calls(text)
    assert r.error is None and names(r) == ["calculator"] and r.calls[0].arguments == {"expression": "1+1"}


def test_parameters_alias():
    r = parse_calls('{"name": "f", "parameters": {"a": 1}}')
    assert r.calls[0].arguments == {"a": 1}


def test_arguments_as_json_string():
    r = parse_calls(json.dumps([{"name": "f", "arguments": json.dumps({"a": 1})}]))
    assert r.calls[0].arguments == {"a": 1}


def test_missing_arguments_default_empty():
    assert parse_calls('{"name": "note_list"}').calls[0].arguments == {}


def test_parallel_array():
    r = parse_calls(json.dumps([{"name": n, "arguments": {}} for n in "abc"]))
    assert names(r) == ["a", "b", "c"]


def test_multiple_tool_call_tags():
    r = parse_calls('<tool_call>{"name":"a","arguments":{"x":1}}</tool_call>\n<tool_call>{"name":"b","arguments":{}}</tool_call>')
    assert names(r) == ["a", "b"]


def test_unterminated_tool_call_tag_still_parses():
    assert names(parse_calls('<tool_call>{"name":"a","arguments":{}}')) == ["a"]


def test_braces_and_unicode_inside_strings():
    r = parse_calls('[{"name":"note_write","arguments":{"title":"café } ] {","body":"☃"}}]')
    assert r.calls[0].arguments["title"] == "café } ] {"


def test_nested_arguments():
    r = parse_calls('[{"name":"f","arguments":{"a":{"b":[1,2,{"c":3}]}}}]')
    assert r.calls[0].arguments["a"]["b"][2] == {"c": 3}


@pytest.mark.parametrize("text,err", [
    ("", "empty"), ("   \n", "empty"),
    ("<think>still reasoning and ran out of tokens", "empty"),
    ("<think>done</think>", "empty"),
    ("I think the answer is 5.", "no_json"),
    ('[{"name": "a", "arguments": {', "malformed_json"),   # truncated by n_predict
    ('[{"name": "a", "arguments": {"x": 1}}', "malformed_json"),  # outer array not closed: no inner salvage
    ("[1, 2]", "bad_shape"), ('{"foo": 1}', "bad_shape"), ("[]", "bad_shape"), ('[{"name": 5}]', "bad_shape"),
    ("<tool_call>nothing</tool_call>", "no_json"),
])
def test_errors(text, err):
    r = parse_calls(text)
    assert r.error == err and r.calls == []


def test_one_bad_tag_block_fails_the_whole_parse():
    r = parse_calls('<tool_call>{"name":"a","arguments":{}}</tool_call><tool_call>{"x":1}</tool_call>')
    assert r.error == "bad_shape"


def test_non_dict_arguments_kept_for_validation():
    assert parse_calls('{"name":"f","arguments":[1,2]}').calls[0].arguments == [1, 2]
