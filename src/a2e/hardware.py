"""Per-device power/thermal sources, the Pi CPU governor, and waiting for the SoC to cool down."""
from __future__ import annotations

import glob
import os
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from a2e.metrics.power_base import PowerSource
from a2e.metrics.thermal import Thermal, read_pi
from a2e.settings import Settings

GOV_GLOB = "/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"


class PowerUnavailable(RuntimeError):
    """`power = "required"` is set but this machine cannot record energy."""


@dataclass
class Instruments:
    power: PowerSource | None = None
    thermal: Callable[[], Thermal] | None = None
    scope: str = "none"  # stored with every run so energy numbers carry their meaning
    notes: list[str] = field(default_factory=list)
    _stop: Callable[[], None] | None = None

    def close(self) -> None:
        if self._stop:
            self._stop()


def make_instruments(device: str, settings: Settings) -> Instruments:
    """The best power/thermal sources available on this device. Missing pieces are noted, not fatal,
    unless `power = "required"`, which raises PowerUnavailable instead of running without energy data."""
    ins = _make_instruments(device, settings)
    if settings.power == "required" and ins.power is None:
        raise PowerUnavailable("power = \"required\" but no energy data is available: "
                               + "; ".join(ins.notes or ["no power source"]))
    return ins


def _make_instruments(device: str, settings: Settings) -> Instruments:
    if device == "pi5":
        if not shutil.which("vcgencmd"):
            return Instruments(notes=["vcgencmd not found: no power or thermal data"])
        if settings.power == "none":
            return Instruments(thermal=read_pi, scope="none", notes=["power disabled in settings"])
        from a2e.metrics.power_pi import PiPmicPower
        p = PiPmicPower()
        return Instruments(p, read_pi, p.scope)
    if device == "mac":
        if settings.power == "none":
            return Instruments(notes=["power disabled in settings"])
        root = hasattr(os, "geteuid") and os.geteuid() == 0
        if not shutil.which("powermetrics") or not root:
            return Instruments(notes=["powermetrics needs root (sudo -E): no power or thermal data"])
        from a2e.metrics.power_mac import MacPowerMetrics
        pm = MacPowerMetrics()
        pm.start()
        return Instruments(pm, pm.read_thermal, pm.scope, _stop=pm.stop)
    return Instruments(notes=[f"unknown device {device}"])


def read_governor() -> str | None:
    paths = sorted(glob.glob(GOV_GLOB))
    if not paths:
        return None
    try:
        return open(paths[0]).read().strip()
    except OSError:
        return None


def set_governor(name: str) -> None:
    """Set every core's cpufreq governor. Needs root; raises RuntimeError with the reason otherwise."""
    paths = sorted(glob.glob(GOV_GLOB))
    if not paths:
        raise RuntimeError("no cpufreq governor files on this machine")
    try:
        for p in paths:
            with open(p, "w") as f:
                f.write(name)
    except OSError as e:
        raise RuntimeError(f"cannot set governor {name!r}: {e} (run as root)") from None
    if read_governor() != name:
        raise RuntimeError(f"governor did not change to {name!r}")


def wait_for_cooldown(thermal: Callable[[], Thermal] | None, target_c: float, timeout_s: float,
                      poll_s: float = 5.0, sleep=time.sleep) -> tuple[float, float | None]:
    """Block until temperature <= target_c (or timeout). Returns (seconds waited, last temperature)."""
    if not thermal or target_c <= 0:
        return 0.0, None
    waited, temp = 0.0, thermal().temp_c
    while temp is not None and temp > target_c and waited < timeout_s:
        sleep(poll_s)
        waited += poll_s
        temp = thermal().temp_c
    return waited, temp
