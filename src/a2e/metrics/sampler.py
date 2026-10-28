from __future__ import annotations

import threading
import time
from collections.abc import Callable

import psutil

from a2e.metrics.power_base import PowerSource
from a2e.metrics.thermal import Thermal


class Sampler:
    """Background sampler of RSS (server + children), CPU, thermals and power at a fixed rate.

    Each sample is (t, rss_mb, cpu_pct, temp_c, freq_mhz, power_w, throttled, pressure) on the perf_counter clock,
    the same clock used for stage windows, so energy can be integrated per stage.
    """

    def __init__(self, pid: int, power: PowerSource | None = None,
                 thermal: Callable[[], Thermal] | None = None, hz: float = 2.0):
        self.proc = psutil.Process(pid)
        self.power, self.thermal, self.period = power, thermal, 1.0 / hz
        self.samples: list[tuple] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _rss_mb(self) -> float:
        total = self.proc.memory_info().rss  # raises NoSuchProcess if the server died
        for p in self.proc.children(recursive=True):
            try:
                total += p.memory_info().rss
            except psutil.Error:  # child exited or is not inspectable; keep sampling the rest
                pass
        return total / 2**20

    def _run(self) -> None:
        self.proc.cpu_percent(None)
        while not self._stop.is_set():
            t = time.perf_counter()
            try:
                rss, cpu = self._rss_mb(), self.proc.cpu_percent(None)
            except psutil.NoSuchProcess:
                break
            th = self.thermal() if self.thermal else Thermal()
            w = self.power.read_watts() if self.power else None
            self.samples.append((t, rss, cpu, th.temp_c, th.freq_mhz, w, th.throttled, th.pressure))
            self._stop.wait(self.period)

    def start(self) -> "Sampler":
        self._thread.start()
        return self

    def stop(self) -> list[tuple]:
        self._stop.set()
        self._thread.join(5)
        return self.samples

    def window(self, t0: float, t1: float) -> list[tuple]:
        return [s for s in self.samples if t0 <= s[0] <= t1]

    def power_trace(self) -> list[tuple[float, float]]:
        return [(s[0], s[5]) for s in self.samples if s[5] is not None]


def measure_idle_power(power: PowerSource, seconds: float = 30.0, hz: float = 2.0) -> float | None:
    """Mean idle power, subtracted from stage energy. Run with the server loaded but quiescent."""
    vals = []
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        w = power.read_watts()
        if w is not None:
            vals.append(w)
        time.sleep(1 / hz)
    return sum(vals) / len(vals) if vals else None
