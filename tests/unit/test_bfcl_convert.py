import ast
import json
import math

import pytest
from jsonschema import Draft202012Validator

import a2e.bfcl  # noqa: F401
from a2e.agent.grammar import SUPPORTED_KEYWORDS
from a2e.agent.parse import ToolCall
from a2e.bfcl.convert import _lit, call_string, schema_from_bfcl, to_ast_output, tool_spec
from a2e.bfcl.load import ALL_CATEGORIES, load_cases


def test_type_mapping_table():
    doc = {"type": "dict", "properties": {"a": {"type": "float"}, "b": {"type": "tuple", "items": {"type": "integer"}},
                                          "c": {"type": "any"}, "d": {"type": "array", "items": {"type": "string"}},
                                          "e": {"type": "boolean"}}, "required": ["a"]}
    s = schema_from_bfcl(doc)
    assert s["type"] == "object" and s["required"] == ["a"]
    p = s["properties"]
    assert p["a"] == {"type": "number"} and p["b"] == {"type": "array", "items": {"type": "integer"}}
    assert p["c"] == {} and p["d"]["items"] == {"type": "string"} and p["e"] == {"type": "boolean"}


def test_unsupported_keywords_dropped_supported_kept():
    s = schema_from_bfcl({"type": "string", "description": "d", "enum": ["x", "y"], "default": "x",
                          "format": "date", "pattern": "^a", "additionalProperties": False, "minLength": 2})
    assert s == {"type": "string", "description": "d", "enum": ["x", "y"], "default": "x"}


def test_tool_spec_drops_response_field():
    spec = tool_spec({"name": "f", "description": "d", "parameters": {"type": "dict", "properties": {}},
                      "response": {"type": "dict"}})
    assert set(spec) == {"name", "description", "parameters"} and spec["parameters"]["type"] == "object"


def _keywords_ok(node, where):
    extra = set(node) - SUPPORTED_KEYWORDS
    assert not extra, (where, extra)
    for k, v in node.get("properties", {}).items():
        _keywords_ok(v, f"{where}.{k}")
    if isinstance(node.get("items"), dict):
        _keywords_ok(node["items"], f"{where}[]")


def test_every_vendored_function_doc_converts_to_valid_supported_json_schema():
    seen = set()
    for cat in ALL_CATEGORIES:
        for case in load_cases(cat):
            for d in case.functions + [x for ds in case.held_out.values() for x in ds]:
                key = json.dumps(d, sort_keys=True)
                if key in seen:
                    continue
                seen.add(key)
                spec = tool_spec(d)
                Draft202012Validator.check_schema(spec["parameters"])
                _keywords_ok(spec["parameters"], d["name"])
    assert len(seen) > 1000


def test_to_ast_output_shape_and_non_dict_rejected():
    assert to_ast_output([ToolCall("f", {"a": 1}), ToolCall("g", {})]) == [{"f": {"a": 1}}, {"g": {}}]
    with pytest.raises(ValueError):
        to_ast_output([ToolCall("f", [1, 2])])


ALLOWED = {"cd", "ls", "mv"}


@pytest.mark.parametrize("args,expected", [
    ({"folder": "document"}, "cd(folder='document')"),
    ({}, "cd()"),
    ({"a": True, "b": None, "c": 3, "d": 2.5}, "cd(a=True, b=None, c=3, d=2.5)"),
    ({"l": [1, "x", [2]], "m": {"k": [1]}}, "cd(l=[1, 'x', [2]], m={'k': [1]})"),
])
def test_call_string_good(args, expected):
    assert call_string("cd", args, ALLOWED) == expected


TRICKY = ["it's", 'say "hi"', "back\\slash", "line1\nline2", "tab\t", "café ☃", "\x00", "'''", '"""', "{}", "%s",
          "f'{1+1}'", "', __import__('os').system('x'), '", ")(", "#", "a" * 5000]


@pytest.mark.parametrize("s", TRICKY)
def test_string_values_round_trip_exactly_through_eval_with_no_builtins(s):
    rec = []
    code = call_string("f", {"v": s, "nested": {"k": [s]}}, {"f"})
    eval(code, {"__builtins__": {}, "f": lambda **kw: rec.append(kw)})  # noqa: S307 - proves what eval would see
    assert rec == [{"v": s, "nested": {"k": [s]}}]


def test_injection_payloads_never_execute(tmp_path):
    marker = tmp_path / "pwned"
    payloads = [f"x'); __import__('os').system('touch {marker}'); ('", f"\"); open('{marker}','w'); (\"",
                f"{{'a': __import__('os').system('touch {marker}')}}"]
    for p in payloads:
        rec = []
        eval(call_string("f", {"v": p}, {"f"}), {"f": lambda **kw: rec.append(kw)})  # noqa: S307
        assert rec == [{"v": p}]
    assert not marker.exists()


@pytest.mark.parametrize("name,args", [
    ("rm", {}), ("__import__('os').system('x')", {}), ("cd; ls", {}), ("cd()", {}), ("", {}), ("cd.__class__", {}),
    ("cd", {"x); import os; (": 1}), ("cd", {"class": 1}), ("cd", {"1a": 1}), ("cd", {"a b": 1}), ("cd", {"": 1}),
    ("cd", [1]), ("cd", "x"), ("cd", None),
    ("cd", {"v": float("nan")}), ("cd", {"v": float("inf")}), ("cd", {"v": -math.inf}),
    ("cd", {"v": object()}), ("cd", {"v": (1, 2)}), ("cd", {"v": {1: 2}}), ("cd", {"v": {1, 2}}), ("cd", {"v": b"x"}),
    ("cd", {"v": lambda: 1}),
])
def test_call_string_rejects_everything_unsafe(name, args):
    with pytest.raises(ValueError):
        call_string(name, args, ALLOWED | {"cd; ls", "cd()", "", "cd.__class__", "__import__('os').system('x')"})


def test_deep_nesting_and_oversized_calls_rejected():
    deep = cur = []
    for _ in range(25):
        nxt = []
        cur.append(nxt)
        cur = nxt
    with pytest.raises(ValueError):
        call_string("cd", {"v": deep}, ALLOWED)
    with pytest.raises(ValueError):
        call_string("cd", {"v": "a" * 30000}, ALLOWED)


def test_name_check_uses_whitelist_even_when_syntactically_valid():
    with pytest.raises(ValueError):
        call_string("rm", {}, ALLOWED)
    assert call_string("rm", {}, ALLOWED | {"rm"}) == "rm()"


def test_lit_literal_eval_round_trip_for_json_values():
    for v in [None, True, False, 0, -5, 3.14, 10**30, "s", [], {}, [1, [2, {"a": None}]], {"k": {"z": [1.5]}}]:
        assert ast.literal_eval(_lit(v)) == v
