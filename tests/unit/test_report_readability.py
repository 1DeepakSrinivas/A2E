"""Plain numbers in tables, the failure-reason table and the no-energy note in the report."""
import json

import pytest

from a2e import results as R
from a2e.store import Store


@pytest.mark.parametrize("value, digits, text", [
    (0.0, 3, "0"), (1.23456, 3, "1.23"), (12.3456, 3, "12.3"), (123.456, 3, "123"), (1234.56, 3, "1235"),
    (1020.0, 3, "1020"), (98765.4, 3, "98765"), (0.5, 3, "0.5"), (0.0000188, 3, "0.0000188"), (-2.5, 3, "-2.5"),
    (0.981, 3, "0.981"), (float("inf"), 3, "inf"), (float("-inf"), 3, "-inf"), (1e-12, 3, "1.00e-12"),
    (2.5e16, 3, "2.50e+16"), (77.5, 3, "77.5"), (5.0, 3, "5"), (1.5, 2, "1.5"),
])
def test_plain_number(value, digits, text):
    assert R.plain_number(value, digits) == text


def test_tables_never_use_scientific_notation_for_everyday_numbers():
    import pandas as pd
    t = R.md_table(pd.DataFrame([{"ms": 1020.0, "share": 0.0000188, "tok_s": 77.5, "big": 5500.0, "n": 3}]))
    row = t.splitlines()[2]
    assert "e+" not in row and "e-" not in row
    assert row == "| 1020 | 0.0000188 | 77.5 | 5500 | 3 |"


def fill(st, cases, model="m", grammar=False, with_energy=False):
    if not st.scalar("SELECT count(*) FROM experiments"):
        st.insert("experiments", {"name": "e", "started_at": 0.0, "status": "complete"})
    cfg = {"model": model, "quant": "q5_k_m", "kv_quant": "f16", "grammar": grammar, "decode_len": 256}
    run = st.insert("runs", {"exp_id": 1, "config_json": json.dumps(cfg), "device": "mac", "phase": "measured",
                             "repeat_idx": 0, "task_kind": "custom", "status": "complete", "started_at": 0.0,
                             "power_scope": "none" if not with_energy else "package",
                             "server_json": json.dumps({"cache_reset": "erase"})})
    for cid, category, valid, err in cases:
        st.insert("cases", {"run_id": run, "case_id": cid, "category": category, "valid": valid, "error_type": err})
        st.insert("turns", {"run_id": run, "task_id": cid, "wall_ms": 100.0, "prefill_ms": 10.0, "decode_ms": 50.0,
                            "prompt_tokens": 10, "cached_tokens": 0, "gen_tokens": 5,
                            "energy_j": 1.0 if with_energy else None})
    return run


def test_failures_count_reasons_and_tasks_run():
    st = Store(":memory:")
    fill(st, [("a", "custom:calc", 0, "parse_error"), ("b", "custom:calc", 0, "parse_error"),
              ("c", "custom:calc", 0, "wrong_step"), ("d", "custom:calc", 1, None), ("e", "simple_python", 1, None)])
    f = R.failures(st).set_index(["category", "reason"])
    assert f.loc[("custom:calc", "parse_error"), "tasks"] == 2 and f.loc[("custom:calc", "wrong_step"), "tasks"] == 1
    assert (f["tasks_run"].loc["custom:calc"] == 4).all() and "simple_python" not in f.index.get_level_values(0)
    assert f.reset_index().iloc[0]["reason"] == "parse_error"  # the most common reason first within a group


def test_failures_empty_when_everything_passed_and_ignores_unscored_cases():
    st = Store(":memory:")
    fill(st, [("a", "c", 1, None), ("b", "c", None, "x")])
    assert R.failures(st).empty


def test_failures_can_be_grouped_differently():
    st = Store(":memory:")
    fill(st, [("a", "c", 0, "boom")], model="x")
    fill(st, [("b", "c", 0, "boom")], model="y", grammar=True)
    assert set(R.failures(st, by=["model"])["model"]) == {"x", "y"}
    assert R.failures(st, by=["grammar"]).shape[0] == 2


def test_report_has_a_failure_table_and_says_when_there_is_no_energy():
    st = Store(":memory:")
    fill(st, [("a", "custom:calc", 0, "parse_error"), ("b", "custom:calc", 1, None)])
    text = R.report_markdown(st, 1)
    assert "## Why tasks failed" in text and "parse_error" in text
    assert "## Energy" in text and "No energy was recorded" in text and "powermetrics" in text and "root" in text
    assert "e+0" not in text.split("## Latency")[1].split("## Where")[0]


def test_report_has_no_energy_note_when_energy_exists_and_no_failure_table_when_nothing_failed():
    st = Store(":memory:")
    fill(st, [("a", "custom:calc", 1, None)], with_energy=True)
    text = R.report_markdown(st, 1)
    assert "No energy was recorded" not in text and "## Why tasks failed" not in text
    assert "## Energy" in text  # the real energy table
