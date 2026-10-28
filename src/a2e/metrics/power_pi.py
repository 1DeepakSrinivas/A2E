from __future__ import annotations

import re
import subprocess

from a2e.metrics.power_base import PowerSource

_ROW = re.compile(r"^\s*(\w+?)_([AV])\s+(?:current|volt)\(\d+\)=([\d.]+)[AV]\s*$")


def parse_pmic(text: str) -> float:
    """Sum V*I over Pi 5 PMIC rails from `vcgencmd pmic_read_adc`.

    EXT5V has only a voltage reading (the input supply) and is skipped. Rails are
    downstream of each other, so the sum is an approximate on-board consumption,
    not wall power: PSU losses and USB peripherals are not included.
    """
    cur: dict[str, float] = {}
    vol: dict[str, float] = {}
    for line in text.splitlines():
        m = _ROW.match(line)
        if not m:
            continue
        rail, kind, val = m.group(1), m.group(2), float(m.group(3))
        (cur if kind == "A" else vol)[rail] = val
    return sum(vol[r] * i for r, i in cur.items() if r in vol)


class PiPmicPower(PowerSource):
    name = "pi5-pmic"
    scope = "sum of Pi 5 PMIC rail V*I; excludes PSU losses and peripherals (estimate)"

    def read_watts(self) -> float | None:
        try:
            out = subprocess.run(["vcgencmd", "pmic_read_adc"], capture_output=True, text=True, timeout=2).stdout
        except (OSError, subprocess.TimeoutExpired):
            return None
        return parse_pmic(out) or None
