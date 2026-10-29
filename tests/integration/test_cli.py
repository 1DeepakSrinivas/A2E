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
    res = ok(["results", "resources", "--csv", str(tmp_path / "res.csv")])
    assert "power_p95_w" in res and "rss_max_mb" in res and (tmp_path / "res.csv").exists()
    assert "energy_j" in ok(["results", "task-resources"])
    ok(["results", "failures", "--csv", str(tmp_path / "fail.csv")])
    for view in ("summary", "frontier", "budget", "grammar", "stages"):
        ok(["results", "efficiency", view])
    assert "view must be one of" in runner.invoke(app, ["results", "efficiency", "nope"]).output
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
    r = runner.invoke(app, ["doctor", "--models", "qwen3-4b-instruct-2507"])
    assert "llama-server binary" in r.output and "model qwen3-4b-instruct-2507 q5_k_m" in r.output


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


def test_run_stops_when_power_is_required_but_missing(env, monkeypatch):
    import os

    import a2e.cli
    from a2e import hardware
    real = a2e.cli._settings

    def required(db=None, device=None):
        s = real(db, device)
        s.power = "required"
        return s

    monkeypatch.setattr(a2e.cli, "_settings", required)
    monkeypatch.setattr(os, "geteuid", lambda: 501, raising=False)
    monkeypatch.setattr(hardware.shutil, "which", lambda name: "/usr/bin/" + name)
    r = runner.invoke(app, ["run", "smoke"])
    assert r.exit_code == 2 and "required" in r.output and "root" in r.output
