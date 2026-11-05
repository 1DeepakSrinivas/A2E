"""`a2e doctor`: check this machine before a long run. Each check is (name, level, ok, detail)."""
from __future__ import annotations

import importlib.util
import os
import shutil
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

from a2e.settings import Settings, detect_device


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    critical: bool = True


def run_checks(settings: Settings, models: list[str] | None = None) -> list[Check]:
    out: list[Check] = []
    out.append(Check("python >= 3.11", sys.version_info >= (3, 11), sys.version.split()[0]))
    dev = settings.device or detect_device()
    out.append(Check("device known (pi5 or mac)", dev in ("pi5", "mac"),
                     f"{dev or 'unknown'}; set [machine] device in configs/local.toml if wrong"))
    srv = settings.llama_server
    srv_ok = bool(srv) and Path(srv).is_file() and os.access(srv, os.X_OK)
    out.append(Check("llama-server binary", srv_ok, srv or "not set: [paths] llama_server in configs/local.toml"))
    bench = settings.bench_binary()
    out.append(Check("llama-bench binary (bench jobs)", bool(bench) and Path(bench).exists(),
                     bench or "not found next to llama-server; set [paths] llama_bench", critical=False))
    for name in models or sorted(settings.models):
        for q in settings.model(name).available_quants():
            p = settings.model_path(name, q)
            out.append(Check(f"model {name} {q}", p.exists(), str(p), critical=False))
    if dev == "pi5":
        out.append(Check("vcgencmd (Pi power/thermals)", bool(shutil.which("vcgencmd")), shutil.which("vcgencmd") or "missing",
                         critical=settings.power != "none"))
        gov = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
        out.append(Check("cpu governor readable", gov.exists(), gov.read_text().strip() if gov.exists() else "missing",
                         critical=False))
        perf = settings.perf or shutil.which("perf")
        out.append(Check("perf (kernel profiles)", bool(perf), perf or "sudo apt install linux-perf", critical=False))
        out.append(Check("cooler recorded in local.toml", settings.cooler is not None, str(settings.cooler), critical=False))
    if dev == "mac":
        pm = shutil.which("powermetrics")
        root = hasattr(os, "geteuid") and os.geteuid() == 0
        out.append(Check("powermetrics available", bool(pm), pm or "missing", critical=settings.power != "none"))
        out.append(Check("running as root (powermetrics needs it)", root, "use sudo -E for energy runs",
                         critical=False))
    key = settings.web.api_key_env
    out.append(Check(f"web search key ({key}, web family only)", bool(settings.web.api_key()),
                     "set" if settings.web.api_key() else f"export {key}=<Brave Search API key>", critical=False))
    out.append(Check("bfcl extra (mpmath)", importlib.util.find_spec("mpmath") is not None,
                     "uv sync --extra bfcl", critical=False))
    out.append(Check("tui extra (textual)", importlib.util.find_spec("textual") is not None,
                     "uv sync --extra tui", critical=False))
    db = Path(settings.db)
    try:
        db.parent.mkdir(parents=True, exist_ok=True)
        sqlite3.connect(db).close()
        out.append(Check("results database writable", True, str(db)))
    except (OSError, sqlite3.Error) as e:
        out.append(Check("results database writable", False, f"{db}: {e}"))
    free_gb = shutil.disk_usage(db.parent).free / 2**30
    out.append(Check("free disk >= 2 GiB", free_gb >= 2, f"{free_gb:.1f} GiB free", critical=False))
    return out
