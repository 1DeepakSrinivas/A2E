import json

import pytest
from typer.testing import CliRunner

from a2e.cli import app
from tests.rig import make_settings

pytest.importorskip("mpmath")
runner = CliRunner()


@pytest.fixture
def env(tmp_path, monkeypatch):
    s = make_settings(tmp_path)
    for k, v in {"A2E_LLAMA_SERVER": s.llama_server, "A2E_MODELS_DIR": s.models_dir, "A2E_DB": s.db,
                 "A2E_DEVICE": "mac"}.items():
        monkeypatch.setenv(k, v)
    return s


def ok(args):
    r = runner.invoke(app, args)
    assert r.exit_code == 0, r.output
    return r.output


def test_run_smoke_suite_then_every_results_command(env, tmp_path):
    out = ok(["run", "smoke"])
    assert "new experiment 1" in out and "finished experiment 1" in out and "job 1" in out
    assert "resuming experiment 1" in ok(["run", "smoke"])  # nothing left to do, same experiment
    assert "complete" in ok(["results", "list"])
    assert "plain" in ok(["results", "jobs", "1"])
    summ = ok(["results", "summary", "--csv", str(tmp_path / "s.csv")])
    assert "decode_tok_s" in summ and (tmp_path / "s.csv").exists()
    assert "prefill" in ok(["results", "stages"])
    assert "knee_task" in ok(["results", "stability"])
    assert "rss_mb" in ok(["results", "timeline", "1"])
    ok(["results", "report", "1", "--out", str(tmp_path / "r.md")])
    assert "## Latency and throughput" in (tmp_path / "r.md").read_text()
    files = ok(["results", "export", str(tmp_path / "exp"), "--fmt", "csv"])
    assert "v_turns.csv" in files


def test_run_with_limit_reports_resume(env):
    out = ok(["run", "smoke", "--limit", "1"])
    assert "'pending': 3" in out


def test_run_unknown_suite_is_a_clear_error(env):
    r = runner.invoke(app, ["run", "nope"])
    assert r.exit_code == 2 and "suite file not found" in r.output


def test_plan_and_suites_and_lock_check(env):
    out = ok(["plan", "smoke", "--device", "mac", "--jobs"])
    assert '"runnable": 4' in out and "custom family=calc" in out
    assert "smoke" in ok(["suites"])
    assert "match" in ok(["suite-lock", "--check"])


def test_doctor_runs_and_reports(env):
    r = runner.invoke(app, ["doctor", "--models", "qwen3-4b"])
    assert "llama-server binary" in r.output and "model qwen3-4b q4_0" in r.output


def test_test_command_lists_groups_and_rejects_unknown():
    assert "quick" in ok(["test", "--list"])
    r = runner.invoke(app, ["test", "nope"])
    assert r.exit_code == 2 and "unknown group" in r.output


def test_test_group_commands_are_well_formed():
    from a2e.testgroups import load_groups
    g = load_groups()
    assert set(g) >= {"quick", "unit", "integration", "sweep", "bfcl", "runner", "hardware", "all"}
    assert g["quick"].command()[-2:] == ["-m", "unit or integration"]
    files = [p for p in g["runner"].command() if p.endswith(".py")]
    assert len(files) == 4 and all(p.startswith(str(__import__("a2e.settings").settings.REPO)) for p in files)
    assert g["hardware"].env == ["A2E_LLAMA_SERVER", "A2E_GGUF"]
