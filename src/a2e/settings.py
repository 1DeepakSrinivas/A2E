"""Machine settings and the model registry.

Three layers, later ones win:
  1. configs/models.toml     committed: the models in the study and where their GGUF files are expected
  2. configs/local.toml      not committed: paths and facts about this machine (copy local.example.toml)
  3. environment variables   A2E_LLAMA_SERVER, A2E_MODELS_DIR, A2E_DB, A2E_DEVICE
"""
from __future__ import annotations

import os
import platform
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

REPO = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(os.environ.get("A2E_CONFIG_DIR", REPO / "configs"))
QUANTS = ("q4_0", "q8_0")


class ModelEntry(BaseModel):
    name: str
    params_b: float
    file: str = "{models_dir}/{name}/{name}-{quant}.gguf"  # template; {models_dir} {name} {quant} {QUANT}
    files: dict[str, str] = Field(default_factory=dict)  # exact per-quant paths (override the template)
    source: str = ""  # where the GGUF came from (record it here when you download it)
    notes: str = ""

    def path_for(self, quant: str, models_dir: str) -> Path:
        tmpl = self.files.get(quant, self.file)
        return Path(os.path.expanduser(tmpl.format(models_dir=models_dir, name=self.name, quant=quant,
                                                   QUANT=quant.upper())))


class Settings(BaseModel):
    llama_server: str | None = None
    llama_bench: str | None = None  # default: llama-bench next to llama-server
    perf: str | None = None  # Linux perf binary for kernel profiles; default: found on PATH
    models_dir: str = str(REPO / "models")
    db: str = str(REPO / "results" / "a2e.db")
    raw_dir: str = str(REPO / "results" / "raw")
    work_dir: str = str(REPO / "results" / "work")
    device: str | None = None  # pi5 | mac; detected when unset
    cooler: bool | None = None  # Pi: is the active cooler fitted? recorded with every run
    power: Literal["auto", "none", "required"] = "auto"  # required: refuse to run without energy data
    threads: int | None = None
    models: dict[str, ModelEntry] = Field(default_factory=dict)

    def bench_binary(self) -> str | None:
        if self.llama_bench:
            return self.llama_bench
        if self.llama_server:
            cand = Path(self.llama_server).with_name("llama-bench")
            return str(cand) if cand.exists() else None
        return None

    def model(self, name: str) -> ModelEntry:
        if name not in self.models:
            raise KeyError(f"model {name!r} is not in configs/models.toml (known: {sorted(self.models)})")
        return self.models[name]

    def model_path(self, name: str, quant: str) -> Path:
        return self.model(name).path_for(quant, self.models_dir)


def _read(path: Path) -> dict:
    return tomllib.loads(path.read_text()) if path.exists() else {}


def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_settings(config_dir: str | Path | None = None, local: str | Path | None = None) -> Settings:
    cdir = Path(config_dir or CONFIG_DIR)
    data = _merge(_read(cdir / "models.toml"), _read(Path(local) if local else cdir / "local.toml"))
    paths = data.pop("paths", {})
    machine = data.pop("machine", {})
    models = {k: ModelEntry(name=k, **v) for k, v in data.pop("models", {}).items()}
    s = Settings(**paths, **machine, models=models)
    env = {"llama_server": "A2E_LLAMA_SERVER", "models_dir": "A2E_MODELS_DIR", "db": "A2E_DB", "device": "A2E_DEVICE"}
    for field, var in env.items():
        if os.environ.get(var):
            setattr(s, field, os.environ[var])
    for field in ("llama_server", "llama_bench", "perf", "models_dir", "db", "raw_dir", "work_dir"):  # ~ and repo-relative paths
        v = getattr(s, field)
        if v:
            p = Path(os.path.expanduser(v))
            setattr(s, field, str(p if p.is_absolute() else REPO / p))
    return s


def detect_device() -> str | None:
    """pi5 on a Raspberry Pi 5, mac on Apple Silicon macOS, otherwise None."""
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "mac"
    model = Path("/proc/device-tree/model")
    try:
        if "Raspberry Pi 5" in model.read_text(errors="ignore"):
            return "pi5"
    except OSError:
        pass
    return None


def resolve_device(settings: Settings) -> str:
    dev = settings.device or detect_device()
    if dev not in ("pi5", "mac"):
        raise RuntimeError("cannot tell whether this is a pi5 or a mac; set [machine] device in configs/local.toml "
                           "or A2E_DEVICE")
    return dev
