import json
import sqlite3
import threading

import pytest

from a2e.store import MIGRATIONS, TABLES, VIEWS, Store


def mk_run(store, **kw):
    cfg = {"model": "qwen3-4b-instruct-2507", "quant": "q5_k_m", "kv_quant": "f16", "grammar": True, "decode_len": 128,
           "family": "calc", "depth": 3, **kw.pop("cfg", {})}
    row = {"config_json": json.dumps(cfg), "device": "mac", "phase": "measured", "started_at": 1.0, **kw}
    return store.insert("runs", row)


def test_fresh_database_has_all_tables_views_and_version(tmp_path):
    s = Store(tmp_path / "a.db")
    assert s.version == len(MIGRATIONS)
    names = {r["name"] for r in s.query("SELECT name FROM sqlite_master WHERE type IN ('table','view')")}
    assert set(TABLES) | set(VIEWS) <= names


def test_wal_mode_and_foreign_keys(tmp_path):
    s = Store(tmp_path / "a.db")
    assert s.scalar("PRAGMA journal_mode") == "wal" and s.scalar("PRAGMA foreign_keys") == 1
    with pytest.raises(sqlite3.IntegrityError):
        s.insert("turns", {"run_id": 999})


def test_reopening_keeps_data_and_does_not_remigrate(tmp_path):
    p = tmp_path / "a.db"
    s = Store(p)
    rid = mk_run(s)
    s.close()
    s2 = Store(p)
    assert s2.scalar("SELECT count(*) FROM runs") == 1 and s2.scalar("SELECT run_id FROM runs") == rid


def test_newer_database_is_refused(tmp_path):
    p = tmp_path / "a.db"
    Store(p).close()
    con = sqlite3.connect(p)
    con.execute("PRAGMA user_version = 99")
    con.close()
    with pytest.raises(RuntimeError, match="newer"):
        Store(p)


def test_migration_sequence_upgrades_an_old_version(tmp_path, monkeypatch):
    import a2e.store as st
    p = tmp_path / "a.db"
    monkeypatch.setattr(st, "MIGRATIONS", st.MIGRATIONS[:1])
    Store(p).close()
    monkeypatch.setattr(st, "MIGRATIONS", st.MIGRATIONS + ["ALTER TABLE experiments ADD COLUMN extra TEXT;"])
    s = Store(p)
    assert s.version == 2 and "extra" in [c["name"] for c in s.query("PRAGMA table_info(experiments)")]


def test_views_hide_aborted_and_running_runs_and_expose_config_fields(tmp_path):
    s = Store(tmp_path / "a.db")
    ok = mk_run(s)
    bad = mk_run(s, status="aborted")
    live = mk_run(s, status="running")
    for r in (ok, bad, live):
        s.insert("turns", {"run_id": r, "prefill_ms": 1.0})
    rows = s.query("SELECT run_id, model, quant, grammar, decode_len, family, depth FROM v_turns")
    assert rows == [{"run_id": ok, "model": "qwen3-4b-instruct-2507", "quant": "q5_k_m", "grammar": 1, "decode_len": 128,
                     "family": "calc", "depth": 3}]
    assert s.scalar("SELECT count(*) FROM turns") == 3


def test_unique_job_key_per_experiment(tmp_path):
    s = Store(tmp_path / "a.db")
    e1 = s.insert("experiments", {"name": "x", "started_at": 0.0})
    e2 = s.insert("experiments", {"name": "y", "started_at": 0.0})
    job = {"kind": "plain", "config_json": "{}", "job_key": "k1"}
    s.insert("jobs", {**job, "exp_id": e1})
    s.insert("jobs", {**job, "exp_id": e2})  # same key in another experiment is fine
    with pytest.raises(sqlite3.IntegrityError):
        s.insert("jobs", {**job, "exp_id": e1})


def test_transaction_commits_once_and_rolls_back_on_error(tmp_path):
    s = Store(tmp_path / "a.db")
    rid = mk_run(s)
    with s.transaction():
        s.insert("turns", {"run_id": rid, "prefill_ms": 1.0})
        with s.transaction():  # nested joins the outer one
            s.insert("turns", {"run_id": rid, "prefill_ms": 2.0})
    assert s.scalar("SELECT count(*) FROM turns") == 2
    with pytest.raises(ValueError):
        with s.transaction():
            s.insert("turns", {"run_id": rid, "prefill_ms": 3.0})
            raise ValueError("boom")
    assert s.scalar("SELECT count(*) FROM turns") == 2


def test_insert_many_update_query_scalar_df(tmp_path):
    s = Store(tmp_path / "a.db")
    rid = mk_run(s)
    s.insert_many("samples", [{"run_id": rid, "t": float(i), "rss_mb": 100.0 + i} for i in range(5)])
    s.update("runs", "run_id", rid, {"status": "aborted", "idle_power_w": 2.5})
    assert s.query("SELECT status, idle_power_w FROM runs") == [{"status": "aborted", "idle_power_w": 2.5}]
    assert s.scalar("SELECT max(rss_mb) FROM samples") == 104.0
    assert list(s.df("SELECT t FROM samples ORDER BY t")["t"]) == [0, 1, 2, 3, 4]
    s.insert_many("samples", [])  # no-op


def test_second_connection_can_read_while_a_transaction_is_open(tmp_path):
    p = tmp_path / "a.db"
    w = Store(p)
    rid = mk_run(w)
    r = Store(p)
    with w.transaction():
        w.insert("turns", {"run_id": rid, "prefill_ms": 1.0})
        assert r.scalar("SELECT count(*) FROM turns") == 0  # WAL: readers see the last committed state
    assert r.scalar("SELECT count(*) FROM turns") == 1


def test_threads_can_use_one_store(tmp_path):
    s = Store(tmp_path / "a.db")
    out = []
    t = threading.Thread(target=lambda: out.append(s.scalar("SELECT count(*) FROM runs")))
    t.start()
    t.join()
    assert out == [0]


@pytest.mark.parametrize("fmt", ["parquet", "csv"])
def test_export_writes_tables_and_views(tmp_path, fmt):
    s = Store(tmp_path / "a.db")
    rid = mk_run(s)
    s.insert("turns", {"run_id": rid, "prefill_ms": 1.0})
    files = s.export(tmp_path / "out", fmt)
    assert {f.stem for f in files} == set(TABLES) | set(VIEWS)
    assert all(f.exists() for f in files)
    s.export_parquet(tmp_path / "legacy")
    assert (tmp_path / "legacy" / "turns.parquet").exists()
    with pytest.raises(ValueError):
        s.export(tmp_path / "x", "xlsx")


def test_in_memory_store_works():
    s = Store(":memory:")
    assert mk_run(s) == 1 and s.version == len(MIGRATIONS)
