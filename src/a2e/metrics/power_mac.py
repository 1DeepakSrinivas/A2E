from __future__ import annotations

import re
import subprocess
import threading

from a2e.metrics.power_base import PowerSource
from a2e.metrics.thermal import Thermal

_COMBINED = re.compile(r"Combined Power \(CPU \+ GPU \+ ANE\):\s*([\d.]+)\s*mW")
_CPU = re.compile(r"^CPU Power:\s*([\d.]+)\s*mW", re.M)
_GPU = re.compile(r"^GPU Power:\s*([\d.]+)\s*mW", re.M)
_PRESSURE = re.compile(r"Current pressure level:\s*(\w+)")


def parse_powermetrics(block: str) -> tuple[float | None, str | None]:
    """One powermetrics sample -> (package watts, thermal pressure level)."""
    m = _COMBINED.search(block)
    if m:
        watts = float(m.group(1)) / 1000
    else:
        c, g = _CPU.search(block), _GPU.search(block)
        watts = (float(c.group(1)) + (float(g.group(1)) if g else 0.0)) / 1000 if c else None
    p = _PRESSURE.search(block)
    return watts, p.group(1) if p else None


class MacPowerMetrics(PowerSource):
    """Streams `sudo powermetrics` in the background; read_watts returns the latest sample.

    Needs root. Samples are CPU+GPU+ANE package power (estimate; excludes DRAM, display, PSU).
    """

    name = "mac-powermetrics"
    scope = "CPU+GPU+ANE combined power from powermetrics; excludes DRAM/display/PSU (estimate)"

    def __init__(self, interval_ms: int = 500):
        self.interval_ms = interval_ms
        self._watts: float | None = None
        self._pressure: str | None = None
        self._proc: subprocess.Popen | None = None

    def start(self) -> None:
        self._proc = subprocess.Popen(
            ["powermetrics", "--samplers", "cpu_power,gpu_power,thermal", "-i", str(self.interval_ms)],
            stdout=subprocess.PIPE, text=True)
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        buf: list[str] = []
        for line in self._proc.stdout:
            if line.startswith("*** Sampled system activity") and buf:
                self._watts, p = parse_powermetrics("".join(buf))
                self._pressure = p or self._pressure
                buf = []
            buf.append(line)

    def stop(self) -> None:
        if self._proc:
            self._proc.terminate()

    def read_watts(self) -> float | None:
        return self._watts

    def read_thermal(self) -> Thermal:
        return Thermal(pressure=self._pressure)
