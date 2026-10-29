from pathlib import Path

import pytest

from a2e import doctor, env
from a2e.settings import Settings, detect_device, load_settings, resolve_device

REPO = Path(__file__).resolve().parents[2]


def write(p: Path, text: str) -> Path:
    p.write_text(text)
    return p


def test_committed_registry_loads_and_has_study_models():
    s = load_settings(REPO / "configs", local=REPO / "configs" / "does-not-exist.toml")
    assert {"qwen3-4b", "gemma3-4b"} == set(s.models)
    assert s.model("gemma3-4b").params_b == 4.0


def test_default_file_template_and_local_overrides(tmp_path, monkeypatch):
    for v in ("A2E_LLAMA_SERVER", "A2E_MODELS_DIR", "A2E_DB", "A2E_DEVICE"):
        monkeypatch.delenv(v, raising=False)
    write(tmp_path / "models.toml", '[models.m1]\nparams_b = 3.0\n[models.m2]\nparams_b = 8.0\nfile = "{models_dir}/x/{QUANT}.gguf"\n')
    local = write(tmp_path / "local.toml", '[paths]\nmodels_dir = "/mm"\nllama_server = "/bin/ls"\n'
                  '[machine]\ndevice = "pi5"\ncooler = true\n[models.m1.files]\nq8_0 = "/exact/m1.gguf"\n')
    s = load_settings(tmp_path, local)
    assert s.model_path("m1", "q4_0") == Path("/mm/m1/m1-q4_0.gguf")
    assert s.model_path("m1", "q8_0") == Path("/exact/m1.gguf")
    assert s.model_path("m2", "q4_0") == Path("/mm/x/Q4_0.gguf")
    assert s.device == "pi5" and s.cooler is True and s.llama_server == "/bin/ls"


def test_env_vars_win(tmp_path, monkeypatch):
    write(tmp_path / "models.toml", "")
    monkeypatch.setenv("A2E_DEVICE", "mac")
    monkeypatch.setenv("A2E_DB", "/tmp/x.db")
    s = load_settings(tmp_path)
    assert s.device == "mac" and s.db == "/tmp/x.db"


def test_unknown_model_message():
    with pytest.raises(KeyError, match="not in configs/models.toml"):
        Settings().model("nope")


def test_resolve_device_requires_a_known_device(monkeypatch):
    monkeypatch.setattr("a2e.settings.detect_device", lambda: None)
    with pytest.raises(RuntimeError, match="pi5 or a mac"):
        resolve_device(Settings())
    assert resolve_device(Settings(device="mac")) == "mac"


def test_detect_device_returns_known_value_or_none():
    assert detect_device() in (None, "pi5", "mac")


def test_env_capture_has_traceability_fields(tmp_path):
    snap = env.capture(Settings(llama_server=None))
    for k in ("hostname", "os", "hardware", "ram_gb", "python", "a2e_commit", "llama_server_version", "governor"):
        assert k in snap
    assert snap["a2e_commit"] is None or len(snap["a2e_commit"]) == 40


def test_file_fingerprint(tmp_path):
    f = tmp_path / "m.gguf"
    f.write_bytes(b"x" * 3000)
    fp = env.file_fingerprint(f)
    assert fp["size"] == 3000 and len(fp["sha256_head_1mib"]) == 64
    assert "error" in env.file_fingerprint(tmp_path / "missing")


def test_doctor_reports_missing_pieces(tmp_path):
    s = Settings(llama_server=str(tmp_path / "nope"), device="mac", db=str(tmp_path / "r" / "a.db"),
                 models_dir=str(tmp_path))
    s.models = load_settings(REPO / "configs", local=tmp_path / "none.toml").models
    checks = {c.name: c for c in doctor.run_checks(s)}
    assert not checks["llama-server binary"].ok and checks["llama-server binary"].critical
    assert not checks["model qwen3-4b q4_0"].ok and not checks["model qwen3-4b q4_0"].critical
    assert checks["results database writable"].ok and checks["device known (pi5 or mac)"].ok
    assert "powermetrics available" in checks


def test_doctor_passes_with_a_complete_setup(tmp_path):
    srv = tmp_path / "llama-server"
    srv.write_text("#!/bin/sh\n")
    srv.chmod(0o755)
    (tmp_path / "m1").mkdir()
    for q in ("q4_0", "q8_0"):
        (tmp_path / "m1" / f"m1-{q}.gguf").write_bytes(b"x")
    s = Settings(llama_server=str(srv), device="pi5", db=str(tmp_path / "a.db"), models_dir=str(tmp_path), power="none")
    from a2e.settings import ModelEntry
    s.models = {"m1": ModelEntry(name="m1", params_b=3)}
    checks = doctor.run_checks(s)
    assert all(c.ok for c in checks if c.critical), [c for c in checks if c.critical and not c.ok]


def test_committed_registry_gives_one_doctor_check_per_model_and_quant(tmp_path):
    s = Settings(llama_server=str(tmp_path / "nope"), device="mac", db=str(tmp_path / "r" / "a.db"),
                 models_dir=str(tmp_path))
    s.models = load_settings(REPO / "configs", local=tmp_path / "none.toml").models
    model_checks = [c.name for c in doctor.run_checks(s) if c.name.startswith("model ")]
    assert len(model_checks) == 2 * 3 and "model gemma3-4b q3_k_m" in model_checks
