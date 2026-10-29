from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass

# vcgencmd get_throttled bit meanings (Pi firmware).
THROTTLE_NOW = 0x1 | 0x2 | 0x4 | 0x8  # undervolt, freq cap, throttled, soft temp limit (current)
THROTTLE_EVER = 0x10000 | 0x20000 | 0x40000 | 0x80000  # same conditions, since boot


@dataclass
class Thermal:
    temp_c: float | None = None
    freq_mhz: float | None = None
    throttled: bool | None = None
    pressure: str | None = None  # Mac thermal pressure level


def parse_temp(text: str) -> float | None:
    m = re.search(r"temp=([\d.]+)", text)
    return float(m.group(1)) if m else None


def parse_clock_mhz(text: str) -> float | None:
    m = re.search(r"=(\d+)\s*$", text.strip())
    return int(m.group(1)) / 1e6 if m else None


def parse_throttled(text: str) -> int | None:
    m = re.search(r"throttled=(0x[0-9a-fA-F]+)", text)
    return int(m.group(1), 16) if m else None


def _vc(*args: str) -> str:
    try:
        return subprocess.run(["vcgencmd", *args], capture_output=True, text=True, timeout=2).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def read_pi() -> Thermal:
    flags = parse_throttled(_vc("get_throttled"))
    return Thermal(
        temp_c=parse_temp(_vc("measure_temp")),
        freq_mhz=parse_clock_mhz(_vc("measure_clock", "arm")),
        throttled=None if flags is None else bool(flags & THROTTLE_NOW),
    )
