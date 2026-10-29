"""Harness variants: prompt rendering, context views, budgets, stop rules and suite expansion."""
import json
from pathlib import Path

import pytest

from a2e.agent.harness import (MASK_KEEP, MASKED, TIGHT_SLACK, TRUNCATE_CHARS, Harness, StopWatch, apply_context,
                               call_signature, turn_budget)
from a2e.agent.parse import ToolCall
from a2e.agent.prompt import (HEADERS, PROMPT_FORMATS, SYSTEM, bfcl_system_message, render_compact, render_natural,
                              render_tools, render_tools_as, system_message)
from a2e.suite import ConfigError, expand, load_suite
from tests.rig import make_settings

SPECS = [
    {"name": "add", "description": "Add two numbers.",
     "parameters": {"type": "object", "properties": {"a": {"type": "integer", "description": "first"},
                                                       "b": {"type": "integer", "description": "second"}},
                    "required": ["a", "b"]}},
    {"name": "note", "description": "Save a note",
     "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "tag": {"type": "string"}},
                    "required": ["text"]}},
    {"name": "ping", "description": "Check the connection", "parameters": {"type": "object", "properties": {}}},
]


def tr(text):
    return {"role": "user", "content": f"<tool_response>\n{text}\n</tool_response>"}


def convo(n):
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "do it"}]
    for i in range(n):
        msgs += [{"role": "assistant", "content": f"call{i}"}, tr(f"result{i} " + "x" * 600)]
    return msgs


# ------------------------------------------------------------------------------------------------ the default
def test_default_harness_changes_nothing():
    h = Harness()
    assert h.is_default and h.non_default() == {}
    assert apply_context(convo(4), h) is not None and apply_context(convo(4), h) == convo(4)


def test_default_prompt_is_the_original_text():
    msg = system_message(SPECS)["content"]
    assert msg.startswith("You are an assistant that completes tasks by calling tools.\n"
                          "Available tools (JSON schemas):\n<tools>\n")
    assert render_tools(SPECS) in msg and msg == system_message(SPECS, "json_schema")["content"]
    assert bfcl_system_message(SPECS)["content"].startswith(
        "You are an assistant that fulfils requests by calling tools.\nAvailable tools (JSON schemas):\n<tools>\n")
    assert SYSTEM.format(header=HEADERS["json_schema"], tools="T").count("(JSON schemas)") == 1


def test_harness_rejects_unknown_choices():
    for field, bad in (("prompt_format", "xml"), ("context", "summary"), ("recovery", "pray"), ("stopping", "never")):
        with pytest.raises(ValueError, match=field):
            Harness(**{field: bad})


def test_non_default_lists_only_what_differs_and_round_trips():
    h = Harness(context="mask", stopping="early_stop")
    assert h.non_default() == {"context": "mask", "stopping": "early_stop"}
    assert Harness.from_job({**h.non_default(), "model": "x", "grammar": True}) == h
    assert Harness.from_job({"model": "x"}).is_default


# ------------------------------------------------------------------------------------------------ prompt formats
def test_compact_marks_optional_arguments_and_drops_argument_descriptions():
    lines = render_compact(SPECS).splitlines()
    assert lines[0] == "add(a: integer, b: integer) - Add two numbers."
    assert lines[1] == "note(text: string, tag?: string) - Save a note"
    assert lines[2] == "ping() - Check the connection"
    assert "first" not in render_compact(SPECS)


def test_natural_describes_each_argument_in_words():
    text = render_natural(SPECS)
    assert "- add: Add two numbers. Arguments: a (integer, required): first; b (integer, required): second." in text
    assert "tag (string, optional)" in text and "- ping: Check the connection. It takes no arguments." in text
    assert "{" not in text  # no JSON left


def test_every_format_names_every_tool_and_is_shorter_than_json():
    for fmt in PROMPT_FORMATS:
        text = render_tools_as(SPECS, fmt)
        assert all(s["name"] in text for s in SPECS)
    assert len(render_tools_as(SPECS, "compact")) < len(render_tools(SPECS))


def test_system_header_matches_the_format():
    assert "Available tools (JSON schemas):" in system_message(SPECS)["content"]
    assert "Available tools (name(arguments)" in system_message(SPECS, "compact")["content"]
    assert "Available tools:\n<tools>" in system_message(SPECS, "natural")["content"]
    with pytest.raises(ValueError):
        render_tools_as(SPECS, "yaml")


# ------------------------------------------------------------------------------------------------ context
def test_mask_replaces_all_but_the_latest_tool_results_and_keeps_history_intact():
    msgs = convo(5)
    before = json.dumps(msgs)
    out = apply_context(msgs, Harness(context="mask"))
    kept = [m for m in out if m["content"].startswith("<tool_response>") and m["content"] != MASKED]
    assert len(kept) == MASK_KEEP and kept[-1]["content"].startswith("<tool_response>\nresult4")
    assert sum(m["content"] == MASKED for m in out) == 3
    assert json.dumps(msgs) == before  # the stored history is untouched
    assert [m["role"] for m in out] == [m["role"] for m in msgs]
    assert out[0] == msgs[0] and out[1] == msgs[1]  # system and user messages are never masked


def test_mask_with_few_results_changes_nothing():
    assert apply_context(convo(MASK_KEEP), Harness(context="mask")) == convo(MASK_KEEP)


def test_truncate_cuts_long_results_and_keeps_the_closing_tag():
    out = apply_context(convo(2), Harness(context="truncate"))
    cut = out[3]["content"]
    assert len(cut) < len(convo(2)[3]["content"]) and cut.endswith("[truncated]\n</tool_response>")
    assert cut.startswith("<tool_response>\nresult0")
    short = [tr("ok")]
    assert apply_context(short, Harness(context="truncate")) == short
    assert len(out[3]["content"]) <= TRUNCATE_CHARS + len("\n[truncated]\n</tool_response>")


def test_mask_breaks_the_cached_prefix_at_the_message_it_masks():
    h = Harness(context="mask")
    a, b = apply_context(convo(2), h), apply_context(convo(3), h)
    assert a[:3] == b[:3] and a[3] != b[3]  # result0 is shown with 2 results and masked once a third arrives


# ------------------------------------------------------------------------------------------------ budget and stop
def test_turn_budget():
    assert turn_budget(Harness(), 8, 3) == 8
    assert turn_budget(Harness(stopping="tight"), 8, 3) == 3 + TIGHT_SLACK
    assert turn_budget(Harness(stopping="tight"), 4, 9) == 4  # never above the suite limit
    assert turn_budget(Harness(stopping="tight"), 8, None) == 8  # unknown need: no cap
    assert turn_budget(Harness(stopping="early_stop"), 8, 3) == 8


def c(name, **args):
    return [ToolCall(name, args)]


def test_stopwatch_ends_on_two_identical_calls_in_a_row():
    w = StopWatch(Harness(stopping="early_stop"))
    assert not w.update(c("ls"), False)
    assert not w.update(c("cat", f="a"), False)
    assert not w.update(c("ls"), False)
    assert w.update(c("ls"), False)


def test_stopwatch_ends_on_two_failed_turns_in_a_row_and_resets_on_success():
    w = StopWatch(Harness(stopping="early_stop"))
    assert not w.update(c("a"), True)
    assert not w.update(c("b"), False)  # a good turn resets the count
    assert not w.update(c("c"), True)
    assert w.update(c("d"), True)


def test_stopwatch_is_off_for_other_rules():
    for rule in ("fixed", "tight"):
        w = StopWatch(Harness(stopping=rule))
        assert not any(w.update(c("ls"), True) for _ in range(5))


def test_call_signature_ignores_argument_order():
    assert call_signature(c("f", a=1, b=2)) == call_signature([ToolCall("f", {"b": 2, "a": 1})])


# ------------------------------------------------------------------------------------------------ suites
BASE = """
[repeats]
warmup = 0
measured = 1
[run]
idle_seconds = 0
[matrix]
device = ["mac"]
model = ["qwen3-4b-instruct-2507"]
quant = ["q5_k_m"]
[axes]
decode_len = [16]
schema_len = [512]
[[tasks]]
kind = "plain"
n = 1
[[tasks]]
kind = "custom"
family = "calc"
depth = 2
n = 1
[[tasks]]
kind = "bfcl"
category = ["simple_python"]
n = 1
"""
HARNESS = """
[[harness]]
name = "baseline"
[[harness]]
name = "mask"
context = "mask"
[[harness]]
name = "lean"
prompt_format = "compact"
recovery = "feedback"
"""


def suite(tmp_path, harness=""):
    p = tmp_path / "s.toml"
    p.write_text('[suite]\nname = "s"\n' + BASE + harness)
    return load_suite(p)


def test_no_harness_section_leaves_the_suite_hash_and_job_keys_alone(tmp_path):
    s = suite(tmp_path)
    assert "harness" not in s.canonical()
    plain = {j["job_key"] for j in expand(s, make_settings(tmp_path))}
    with_base = suite(tmp_path, '[[harness]]\nname = "baseline"\n')
    assert {j["job_key"] for j in expand(with_base, make_settings(tmp_path))} == plain


def test_variants_multiply_only_the_jobs_that_call_tools(tmp_path):
    jobs = expand(suite(tmp_path, HARNESS), make_settings(tmp_path))
    by_kind = {k: [j for j in jobs if j["kind"] == k] for k in ("plain", "custom", "bfcl")}
    assert len(by_kind["plain"]) == 1 and len(by_kind["custom"]) == 3 and len(by_kind["bfcl"]) == 3
    assert not any("harness" in j for j in by_kind["plain"])
    keys = {j["job_key"] for j in jobs}
    assert len(keys) == len(jobs)


def test_variant_fields_are_stored_only_when_they_differ(tmp_path):
    jobs = [j for j in expand(suite(tmp_path, HARNESS), make_settings(tmp_path)) if j["kind"] == "custom"]
    by = {j.get("harness", "baseline"): j for j in jobs}
    assert set(by) == {"baseline", "mask", "lean"}
    assert not any(f in by["baseline"] for f in ("prompt_format", "context", "recovery", "stopping"))
    assert by["mask"]["context"] == "mask" and "prompt_format" not in by["mask"]
    assert (by["lean"]["prompt_format"], by["lean"]["recovery"]) == ("compact", "feedback")


def test_bad_harness_sections_are_rejected(tmp_path):
    for body in ('[[harness]]\nname = "a"\nprompt_format = "xml"\n',
                 '[[harness]]\nname = "a"\n[[harness]]\nname = "a"\ncontext = "mask"\n',
                 '[[harness]]\nname = "a"\n[[harness]]\nname = "b"\n',
                 '[[harness]]\nname = ""\n', '[[harness]]\nname = "a"\nunknown = 1\n'):
        with pytest.raises(ConfigError):
            suite(tmp_path, body)


def test_shipped_ablation_suite_covers_each_lever_once():
    s = load_suite(Path(__file__).parents[2] / "configs" / "suites" / "harness_ablation.toml")
    by = {v.name: v.harness for v in s.harness}
    assert by["baseline"].is_default
    singles = {n: h.non_default() for n, h in by.items() if len(h.non_default()) == 1}
    assert set(singles) == {"compact", "natural", "mask", "truncate", "feedback", "retry", "tight", "early_stop"}
    assert len(by["lean"].non_default()) == 4 and len(by["heavy"].non_default()) == 2
