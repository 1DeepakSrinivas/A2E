"""Snapshot of the machine and software, stored with every experiment so results can be traced later."""
from __future__ import annotations

import hashlib
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import psutil

from a2e.settings import REPO, Settings, detect_device


def _run(args: list[str], timeout: float = 10) -> str:
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return (r.stdout or r.stderr).strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def git_state(repo: Path = REPO) -> tuple[str | None, bool | None]:
    commit = _run(["git", "-C", str(repo), "rev-parse", "HEAD"]) or None
    if commit is None:
        return None, None
    return commit, bool(_run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"]))


def file_fingerprint(path: Path, head_bytes: int = 1 << 20) -> dict:
    """Size, mtime and sha256 of the first MiB. Hashing whole GGUF files takes minutes on a Pi."""
    try:
        st = path.stat()
        with open(path, "rb") as f:
            head = hashlib.sha256(f.read(head_bytes)).hexdigest()
        return {"path": str(path), "size": st.st_size, "mtime": st.st_mtime, "sha256_head_1mib": head}
    except OSError as e:
        return {"path": str(path), "error": str(e)}


def hardware_model() -> str:
    try:
        return Path("/proc/device-tree/model").read_text(errors="ignore").strip("\x00\n ")
    except OSError:
        pass
    if platform.system() == "Darwin":
        return _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    return platform.processor() or platform.machine()


def governor() -> str | None:
    p = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    try:
        return p.read_text().strip()
    except OSError:
        return None


def llama_server_version(binary: str | None) -> str:
    return _run([binary, "--version"]) if binary and Path(binary).exists() else ""


def capture(settings: Settings) -> dict:
    commit, dirty = git_state()
    return {
        "captured_at": time.time(),
        "hostname": platform.node(),
        "os": platform.platform(),
        "machine": platform.machine(),
        "hardware": hardware_model(),
        "device_detected": detect_device(),
        "device_configured": settings.device,
        "cpu_count": os.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 2),
        "python": sys.version.split()[0],
        "a2e_commit": commit,
        "a2e_dirty": dirty,
        "llama_server": settings.llama_server,
        "llama_server_version": llama_server_version(settings.llama_server),
        "governor": governor(),
        "cooler": settings.cooler,
        "power_setting": settings.power,
        "vcgencmd": shutil.which("vcgencmd"),
        "powermetrics": shutil.which("powermetrics"),
        "is_root": hasattr(os, "geteuid") and os.geteuid() == 0,
    }
