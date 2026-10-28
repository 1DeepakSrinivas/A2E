import json
import sqlite3

import pytest
from typer.testing import CliRunner

import a2e.bfcl  # noqa: F401
from a2e.bfcl.load import load_subset
from a2e.bfcl.runner import rescore, run_case, run_category, safe_token_counter
from a2e.cli import app
from a2e.store import Store
from tests.bfcl_oracle import multi_outputs, single_output
from tests.stubs import StubClient

pytest.importorskip("mpmath", reason="multi-turn backends need `uv sync --extra bfcl`")


class OracleClient(StubClient):
    """Answers each case with its ground truth, looked up from the user text in the prompt."""

    def __init__(self, category, n, seed=0):
        cases = load_subset(category, n, seed)
        self.queue = [t for c in cases for t in (multi_outputs(c) if c.is_multi_turn else [single_output(c)])]
        super().__init__(self.queue)


def new_run(store):
    return store.insert("runs", {"config_json": "{}", "device": "mac", "phase": "bfcl", "started_at": 0.0})


@pytest.mark.parametrize("category,n", [("simple_python", 8), ("parallel", 6), ("multi_turn_base", 5),
                                        ("multi_turn_miss_func", 5)])
def test_run_category_writes_rows_and_raw_and_all_oracle_cases_pass(tmp_path, category, n):
    store, raw = Store(tmp_path / "r.db"), tmp_path / "raw.jsonl"
    outs = run_category(OracleClient(category, n), category, store, new_run(store), n=n, raw_path=raw)
    assert len(outs) == n and all(o.result.valid for o in outs)
    assert store.db.execute("SELECT count(*), sum(valid) FROM cases").fetchone() == (n, n)
    n_calls = sum(len(o.turns) for o in outs)
    assert store.db.execute("SELECT count(*) FROM turns").fetchone()[0] == n_calls
    assert len(raw.read_text().splitlines()) == n
    rec = json.loads(raw.read_text().splitlines()[0])
    assert set(rec) == {"run_id", "case_id", "category", "status", "texts"} and rec["texts"]


def test_rescore_reproduces_stored_scores_including_failures(tmp_path):
    store, raw = Store(tmp_path / "r.db"), tmp_path / "raw.jsonl"
    # half right, half wrong: the wrong ones answer with prose
    cases = load_subset("multiple", 10, seed=0)
    outs = [single_output(c) if i % 2 == 0 else "I do not know" for i, c in enumerate(cases)]
    run_category(StubClient(outs), "multiple", store, new_run(store), n=10, raw_path=raw)
    stored = [bool(v) for (v,) in store.db.execute("SELECT valid FROM cases ORDER BY case_row_id")]
    assert stored == [i % 2 == 0 for i in range(10)]
    assert [r.valid for r in rescore(raw)] == stored


def test_rescore_multi_turn_replays_backends(tmp_path):
    store, raw = Store(tmp_path / "r.db"), tmp_path / "raw.jsonl"
    run_category(OracleClient("multi_turn_base", 4), "multi_turn_base", store, new_run(store), n=4, raw_path=raw)
    assert [r.valid for r in rescore(raw)] == [True] * 4


def test_server_error_recorded_and_rescored_as_failure(tmp_path):
    store, raw = Store(tmp_path / "r.db"), tmp_path / "raw.jsonl"
    outs = run_category(StubClient(["[]"], fail_at=0), "simple_python", store, new_run(store), n=1, raw_path=raw)
    assert outs[0].status == "server_error" and not outs[0].result.valid
    assert store.db.execute("SELECT status, valid, error_type FROM cases").fetchone() == ("server_error", 0, "a2e:server_error")
    assert rescore(raw)[0].error_type == "a2e:server_error"


def test_schema_tokens_recorded_when_counter_available(tmp_path):
    store = Store(tmp_path / "r.db")
    client = OracleClient("multi_turn_base", 2)
    run_category(client, "multi_turn_base", store, new_run(store), n=2, count_tokens=client.tokenize)
    vals = [v for (v,) in store.db.execute("SELECT schema_tokens FROM cases")]
    assert all(v and v > 500 for v in vals)  # multi-turn schemas are large


def test_safe_token_counter_handles_missing_endpoint():
    class NoTok:
        def tokenize(self, t):
            raise RuntimeError("404")
    assert safe_token_counter(NoTok()) is None
    assert safe_token_counter(StubClient([])) is not None


def test_parquet_export_includes_cases(tmp_path):
    import pandas as pd
    store = Store(tmp_path / "r.db")
    run_category(OracleClient("simple_python", 3), "simple_python", store, new_run(store), n=3)
    store.export_parquet(tmp_path / "pq")
    assert len(pd.read_parquet(tmp_path / "pq" / "cases.parquet")) == 3


def test_run_case_options_pass_through():
    c = load_subset("simple_python", 1)[0]
    client = StubClient(["[]"])
    run_case(client, c, grammar=True, cache_prompt=False, n_predict=33)
    assert client.calls[0]["n_predict"] == 33 and client.calls[0]["cache_prompt"] is False
    assert client.calls[0]["json_schema"]["minItems"] == 0


# ------------------------------------------------------------------ CLI over real HTTP
def test_cli_list_and_run_and_rescore_against_fake_server(fake_server, tmp_path):
    srv = fake_server()  # the fake model answers 'xxxx': every case is a scored failure, none crash
    runner = CliRunner()
    assert "multi_turn_miss_func" in runner.invoke(app, ["bfcl", "list"]).output
    db, raw = tmp_path / "o.db", tmp_path / "raw.jsonl"
    res = runner.invoke(app, ["bfcl", "run", "--url", srv.url, "--category", "simple_python", "--category",
                              "irrelevance", "--model", "qwen3-4b", "--quant", "q4_0", "--device", "mac",
                              "--n", "5", "--db", str(db), "--raw", str(raw), "--n-predict", "16"])
    assert res.exit_code == 0, res.output
    assert "simple_python: 0/5 valid" in res.output and "irrelevance: 5/5 valid" in res.output  # prose = no call
    con = sqlite3.connect(db)
    assert con.execute("SELECT count(*) FROM cases").fetchone()[0] == 10
    assert con.execute("SELECT count(*) FROM turns WHERE prefill_ms > 0 AND decode_ms > 0").fetchone()[0] == 10
    assert con.execute("SELECT count(*) FROM cases WHERE schema_tokens > 0").fetchone()[0] == 10
    out = runner.invoke(app, ["bfcl", "rescore", "--raw", str(raw)])
    assert "simple_python: 0/5 valid" in out.output and "irrelevance: 5/5 valid" in out.output


def test_cli_rejects_unknown_category(fake_server, tmp_path):
    res = CliRunner().invoke(app, ["bfcl", "run", "--url", fake_server().url, "--category", "nope", "--model", "m",
                                   "--quant", "q4_0", "--device", "mac", "--db", str(tmp_path / "x.db")])
    assert res.exit_code != 0
