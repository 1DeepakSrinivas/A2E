import pytest
from jsonschema import Draft202012Validator

from a2e.agent.grammar import SUPPORTED_KEYWORDS, call_schema
from a2e.tools.builtin import build_registry


@pytest.fixture
def specs(tmp_path):
    return build_registry(tmp_path).specs()


def valid(schema, value):
    return Draft202012Validator(schema).is_valid(value)


def test_schema_is_itself_valid(specs):
    Draft202012Validator.check_schema(call_schema(specs))


def test_accepts_good_calls(specs):
    s = call_schema(specs)
    assert valid(s, [{"name": "calculator", "arguments": {"expression": "1+1"}}])
    assert valid(s, [{"name": "note_write", "arguments": {"title": "t", "priority": 2}},
                     {"name": "note_list", "arguments": {}}])


@pytest.mark.parametrize("bad", [
    [],                                                                     # minItems
    [{"name": "ghost", "arguments": {}}],                                   # unknown tool
    [{"name": "calculator", "arguments": {}}],                              # missing required arg
    [{"name": "calculator", "arguments": {"expression": 3}}],               # wrong type
    [{"name": "calculator"}],                                               # no arguments
    {"name": "calculator", "arguments": {"expression": "1"}},               # not an array
    [{"name": "note_write", "arguments": {"title": "t", "priority": "x"}}],
])
def test_rejects_bad_calls(specs, bad):
    assert not valid(call_schema(specs), bad)


def test_max_calls_bound(specs):
    one = {"name": "note_list", "arguments": {}}
    assert valid(call_schema(specs, max_calls=3), [one] * 3)
    assert not valid(call_schema(specs, max_calls=3), [one] * 4)


def test_name_is_pinned_to_matching_arguments(specs):
    # calculator name with note_write arguments must not validate
    assert not valid(call_schema(specs), [{"name": "calculator", "arguments": {"title": "t"}}])


def _keywords_ok(node, path="$"):
    if not isinstance(node, dict):
        return
    extra = set(node) - SUPPORTED_KEYWORDS
    assert not extra, f"{path} uses keywords outside the supported subset: {extra}"
    for k, v in node.get("properties", {}).items():
        _keywords_ok(v, f"{path}.{k}")
    _keywords_ok(node.get("items"), f"{path}[]")


def test_builtin_parameter_schemas_use_supported_keywords_only(specs):
    for s in specs:
        _keywords_ok(s["parameters"], s["name"])
