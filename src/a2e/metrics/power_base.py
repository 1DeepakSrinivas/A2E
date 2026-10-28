from __future__ import annotations

from abc import ABC, abstractmethod


class PowerSource(ABC):
    """Pluggable power reading so a USB meter can replace software estimates later."""

    name: str
    # Human-readable scope of the measurement, reported alongside results.
    scope: str

    @abstractmethod
    def read_watts(self) -> float | None: ...


def integrate_energy(samples: list[tuple[float, float]], t0: float, t1: float, idle_w: float = 0.0) -> float:
    """Trapezoidal energy (J) over [t0, t1] from (t, watts) samples, idle baseline subtracted."""
    pts = [(t, w) for t, w in samples if t0 <= t <= t1]
    return sum(
        ((w0 + w1) / 2 - idle_w) * (t1_ - t0_) for (t0_, w0), (t1_, w1) in zip(pts, pts[1:])
    )
