"""Apple Silicon: real powermetrics. Run with sudo on the Mac."""
import time

from a2e.metrics.power_mac import MacPowerMetrics

from .conftest import needs_mac, needs_root

pytestmark = [needs_mac, needs_root]


def test_powermetrics_streams_power_and_pressure():
    pm = MacPowerMetrics(interval_ms=500)
    pm.start()
    try:
        deadline = time.time() + 10
        while pm.read_watts() is None and time.time() < deadline:
            time.sleep(0.25)
        assert pm.read_watts() is not None, "no sample parsed; compare powermetrics output with the regexps"
        assert 0 < pm.read_watts() < 150
        assert pm.read_thermal().pressure in {"Nominal", "Light", "Moderate", "Heavy", "Trapping", "Sleeping"}
    finally:
        pm.stop()
