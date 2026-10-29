"""Migration 3: harness fields in the views, and results written before it read as the baseline harness."""
import json
import sqlite3

from a2e.store import MIGRATIONS, Store


def test_results_from_before_harness_variants_read_as_baseline(tmp_path):
    p = tmp_path / "old.db"
    db = sqlite3.connect(p)
    for i, script in enumerate(MIGRATIONS[:2]):  # a database as version 2 left it
        db.executescript("BEGIN;\n" + script + f"\nPRAGMA user_version = {i + 1};\nCOMMIT;")
    cfg = {"model": "qwen3-4b", "quant": "q4_0", "kv_quant": "f16", "grammar": False}
    db.execute("INSERT INTO runs (config_json, device, phase, started_at) VALUES (?, 'mac', 'measured', 1.0)",
               [json.dumps(cfg)])
    db.execute("INSERT INTO turns (run_id, task_id, energy_j) VALUES (1, 't', 2.5)")
    db.commit()
    db.close()
    s = Store(p)
    assert s.version == len(MIGRATIONS)
    run = s.query("SELECT harness, prompt_format, context, recovery, stopping FROM v_runs")[0]
    assert run == {"harness": "baseline", "prompt_format": "json_schema", "context": "full", "recovery": "none",
                   "stopping": "fixed"}
    turn = s.query("SELECT energy_j, energy_harness_j, harness, context FROM v_turns")[0]
    assert turn == {"energy_j": 2.5, "energy_harness_j": None, "harness": "baseline", "context": "full"}


def test_variant_fields_in_the_config_show_up_in_every_view():
    s = Store(":memory:")
    cfg = {"model": "m", "harness": "lean", "prompt_format": "compact", "context": "mask", "recovery": "feedback",
           "stopping": "early_stop"}
    run = s.insert("runs", {"config_json": json.dumps(cfg), "device": "mac", "phase": "measured", "started_at": 0.0})
    s.insert("turns", {"run_id": run, "task_id": "a"})
    s.insert("cases", {"run_id": run, "case_id": "a", "category": "c", "valid": 1})
    want = ("lean", "compact", "mask", "feedback", "early_stop")
    cols = "harness, prompt_format, context, recovery, stopping"
    for view in ("v_runs", "v_turns", "v_cases"):
        assert tuple(s.query(f"SELECT {cols} FROM {view}")[0].values()) == want


def test_thinking_reads_as_default_for_older_runs_and_shows_the_stored_value():
    s = Store(":memory:")
    old = s.insert("runs", {"config_json": json.dumps({"model": "m"}), "device": "mac", "phase": "measured",
                            "started_at": 0.0})
    new = s.insert("runs", {"config_json": json.dumps({"model": "m", "thinking": "off"}), "device": "mac",
                            "phase": "measured", "started_at": 0.0})
    for rid in (old, new):
        s.insert("turns", {"run_id": rid, "task_id": "a"})
        s.insert("cases", {"run_id": rid, "case_id": "a", "category": "c", "valid": 1})
    for view in ("v_runs", "v_turns", "v_cases"):
        got = {r["thinking"] for r in s.query(f"SELECT thinking FROM {view}")}
        assert got == {"default", "off"}, view
