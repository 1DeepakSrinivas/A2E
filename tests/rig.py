"""A complete local rig for runner tests: fake llama-server binary, dummy model files, settings, suites."""
from __future__ import annotations

import sys
from pathlib import Path

from a2e.hardware import Instruments
from a2e.metrics.power_base import PowerSource
from a2e.metrics.thermal import Thermal
from a2e.settings import ModelEntry, Settings

FAKES = Path(__file__).parent / "fakes"
FAKE = FAKES / "fake_llama_server.py"


def _wrapper(path: Path, script: Path) -> str:
    path.write_text(f"#!/usr/bin/env bash\nexec {sys.executable} {script} \"$@\"\n")
    path.chmod(0o755)
    return str(path)


class ConstPower(PowerSource):
    name, scope = "const", "test constant"

    def __init__(self, w: float = 5.0):
        self.w = w

    def read_watts(self):
        return self.w


def make_settings(tmp: Path, models=("qwen3-4b", "qwen3-8b"), quants=("q4_0", "q8_0"), device="mac") -> Settings:
    bin_ = Path(_wrapper(tmp / "llama-server", FAKE))
    _wrapper(tmp / "llama-bench", FAKES / "fake_llama_bench.py")  # found next to llama-server
    perf = _wrapper(tmp / "perf", FAKES / "fake_perf.py")
    reg = {"qwen3-4b": 4.0, "qwen3-8b": 8.0, "xlam2-3b": 3.0, "xlam2-8b": 8.0, "phi4-mini": 3.8}
    for m in models:
        for q in quants:
            p = tmp / "models" / m / f"{m}-{q}.gguf"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"GGUF" + b"\0" * 64)
    return Settings(llama_server=str(bin_), perf=perf, models_dir=str(tmp / "models"), db=str(tmp / "a2e.db"),
                    raw_dir=str(tmp / "raw"), work_dir=str(tmp / "work"), device=device, power="auto",
                    models={k: ModelEntry(name=k, params_b=v) for k, v in reg.items()})


def instruments(w: float = 5.0, temp: float = 50.0):
    return lambda device, settings: Instruments(ConstPower(w), lambda: Thermal(temp_c=temp, freq_mhz=2400.0,
                                                                                throttled=False), "test constant")


def write_suite(tmp: Path, body: str, name: str = "t") -> Path:
    p = tmp / f"{name}.toml"
    p.write_text(f'[suite]\nname = "{name}"\n' + body)
    return p


SMALL = """
[repeats]
warmup = 1
measured = 2
[run]
idle_seconds = 0
sample_hz = 50
[matrix]
device = ["mac", "pi5"]
model = ["qwen3-4b"]
quant = ["q4_0"]
[axes]
decode_len = [16]
schema_len = [512]
grammar = [true]
[[tasks]]
kind = "plain"
n = 2
[[tasks]]
kind = "custom"
family = "calc"
depth = 2
n = 2
[[tasks]]
kind = "bfcl"
category = ["simple_python", "irrelevance"]
n = 2
"""
