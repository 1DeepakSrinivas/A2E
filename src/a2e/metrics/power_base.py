from __future__ import annotations

import bisect
from abc import ABC, abstractmethod


class PowerSource(ABC):
    """Pluggable power reading so a USB meter can replace software estimates later."""

    name: str
    # Human-readable scope of the measurement, reported alongside results.
    scope: str

    @abstractmethod
    def read_watts(self) -> float | None: ...


def integrate_energy(samples: list[tuple[float, float]], t0: float, t1: float, idle_w: float = 0.0) -> float:
    """Energy (J) over [t0, t1] from (t, watts) samples, idle baseline subtracted.

    The trace is treated as piecewise linear and clipped to the window, so stages shorter than
    the sample period still get energy. Outside the sampled range the nearest sample is held.
    Fewer than two samples cannot define a trace and give 0.
    """
    if t1 <= t0 or len(samples) < 2:
        return 0.0
    pts = sorted(samples)
    ts = [t for t, _ in pts]

    def at(t: float) -> float:
        i = bisect.bisect_right(ts, t)
        if i == 0:
            return pts[0][1]
        if i == len(pts):
            return pts[-1][1]
        (ta, wa), (tb, wb) = pts[i - 1], pts[i]
        return wa + (wb - wa) * (t - ta) / (tb - ta) if tb > ta else wa

    knots = [(t0, at(t0)), *[(t, w) for t, w in pts if t0 < t < t1], (t1, at(t1))]
    return sum(((w0 + w1) / 2 - idle_w) * (b - a) for (a, w0), (b, w1) in zip(knots, knots[1:]))
