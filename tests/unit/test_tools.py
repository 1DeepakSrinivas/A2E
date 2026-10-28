import json

import pytest

from a2e.tools.base import Registry, Tool
from a2e.tools.builtin import build_registry, make_corpus, safe_calc, slug


@pytest.fixture
def env(tmp_path):
    meta = make_corpus(tmp_path, n=12, seed=3)
    return build_registry(tmp_path), meta, tmp_path


@pytest.mark.parametrize("expr,want", [("2+3", "5"), ("2*(3+4)", "14"), ("7/2", "3.5"), ("8/2", "4"),
                                       ("-3+5", "2"), ("2**10", "1024"), ("10%3", "1"), ("7//2", "3"),
                                       (" 4 * 5 ", "20")])
def test_calculator_ok(expr, want):
    assert safe_calc(expr) == want


@pytest.mark.parametrize("expr", ["__import__('os').system('id')", "open('x')", "2**1000", "a+1", "1/0", "",
                                  "(1", "1" * 201, "[1,2]", "True+1", "'a'*3", "(1).real", "lambda: 1"])
def test_calculator_rejects_unsafe_or_invalid(expr):
    with pytest.raises(Exception):
        safe_calc(expr)


def test_execute_turns_exceptions_into_error_results(env):
    reg, _, _ = env
    r = reg.execute("calculator", {"expression": "1/0"})
    assert not r.ok and "ZeroDivisionError" in r.output and r.exec_ms >= 0


def test_note_roundtrip_and_list(env):
    reg, _, _ = env
    assert reg.execute("note_write", {"title": "Zed", "body": "b", "priority": 2}).ok
    reg.execute("note_write", {"title": "Alpha"})
    assert json.loads(reg.execute("note_list", {}).output) == ["Alpha", "Zed"]
    assert json.loads(reg.execute("note_read", {"title": "Zed"}).output)["priority"] == 2


def test_note_title_cannot_escape_notes_dir(env):
    reg, _, root = env
    reg.execute("note_write", {"title": "../../evil", "body": "x"})
    assert not list(root.parent.glob("evil*")) and not (root / "evil").exists()
    assert [p.name for p in (root / "notes").iterdir()] == ["evil.json"]
    assert slug("../../evil") == "evil"


@pytest.mark.parametrize("title", ["///", "...", "   "])
def test_degenerate_titles_error(env, title):
    reg, _, _ = env
    assert not reg.execute("note_write", {"title": title}).ok


def test_read_missing_note_is_error_result(env):
    assert not env[0].execute("note_read", {"title": "nope"}).ok


def test_search_ranks_named_doc_first_and_respects_top_k(env):
    reg, meta, _ = env
    d = meta[5]
    hits = json.loads(reg.execute("search_docs", {"query": d["name"], "top_k": 2}).output)
    assert hits[0]["doc_id"] == d["doc_id"] and len(hits) <= 2


def test_search_no_match(env):
    assert env[0].execute("search_docs", {"query": "zzzzqqq"}).output == "[]"


def test_doc_read_and_bad_ids(env):
    reg, meta, _ = env
    d = meta[0]
    out = reg.execute("doc_read", {"doc_id": d["doc_id"]}).output
    assert f"Its population is {d['population']}" in out
    for bad in ["../notes/x", "doc_1", "doc_001.txt", "DOC_001"]:
        assert not reg.execute("doc_read", {"doc_id": bad}).ok


def test_reminders_accumulate_and_validate(env):
    reg, _, root = env
    reg.execute("set_reminder", {"text": "a", "minutes": 5})
    assert "#2" in reg.execute("set_reminder", {"text": "b", "minutes": 1}).output
    assert not reg.execute("set_reminder", {"text": "c", "minutes": 0}).ok
    assert len(json.loads((root / "reminders.json").read_text())) == 2


def test_corpus_deterministic_unique_and_seeded(tmp_path):
    a = make_corpus(tmp_path / "a", 20, seed=7)
    b = make_corpus(tmp_path / "b", 20, seed=7)
    c = make_corpus(tmp_path / "c", 20, seed=8)
    assert a == b and a != c
    assert len({d["name"] for d in a}) == 20 and len(list((tmp_path / "a" / "corpus").glob("*.txt"))) == 20


@pytest.mark.parametrize("name,args,ok", [
    ("calculator", {"expression": "1+1"}, True),
    ("calculator", {}, False),
    ("calculator", {"expression": 5}, False),
    ("note_write", {"title": "t", "priority": "high"}, False),
    ("note_write", {"title": "t", "priority": 3}, True),
    ("nope", {}, False),
    ("calculator", [1], False),
    ("note_list", {}, True),
])
def test_validate(env, name, args, ok):
    err = env[0].validate(name, args)
    assert (err is None) == ok, err


def test_output_truncation():
    reg = Registry([Tool("big", "x", {"type": "object", "properties": {}}, lambda: "a" * 5000)])
    out = reg.execute("big", {}).output
    assert out.endswith("[truncated]") and len(out) < 1100


def test_specs_subset_and_order(env):
    reg = env[0]
    assert [s["name"] for s in reg.specs(["note_list", "calculator"])] == ["note_list", "calculator"]
    assert {"calculator", "final_answer", "doc_read"} <= {s["name"] for s in reg.specs()}
