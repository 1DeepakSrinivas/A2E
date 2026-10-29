"""Raspberry Pi 5: real vcgencmd readings. Run on the Pi."""
import multiprocessing as mp
import time

import pytest

from a2e.metrics.power_pi import PiPmicPower, parse_pmic
from a2e.metrics.sampler import measure_idle_power
from a2e.metrics.thermal import read_pi

from .conftest import needs_pi

pytestmark = needs_pi


def test_temperature_and_clock_are_plausible():
    t = read_pi()
    assert t.temp_c is not None and 15 < t.temp_c < 100
    assert t.freq_mhz is not None and 500 < t.freq_mhz < 3200
    assert t.throttled is not None  # get_throttled parsed


def test_pmic_power_reads_and_is_plausible():
    w = PiPmicPower().read_watts()
    assert w is not None, "pmic_read_adc output did not parse; check rail names with `vcgencmd pmic_read_adc`"
    assert 0.5 < w < 30


def test_pmic_rail_parse_on_real_output():
    import subprocess
    out = subprocess.run(["vcgencmd", "pmic_read_adc"], capture_output=True, text=True).stdout
    assert "VDD_CORE_A" in out, "unexpected rail names; update the parser regexp"
    assert parse_pmic(out) > 0


def _burn(stop):
    while not stop.is_set():
        sum(i * i for i in range(10000))


@pytest.mark.slow
def test_load_raises_power_and_temperature():
    pw = PiPmicPower()
    idle = measure_idle_power(pw, seconds=5)
    t_idle = read_pi().temp_c
    stop = mp.Event()
    procs = [mp.Process(target=_burn, args=(stop,)) for _ in range(4)]
    [p.start() for p in procs]
    try:
        time.sleep(3)
        loaded = measure_idle_power(pw, seconds=5)
        t_load = read_pi().temp_c
    finally:
        stop.set()
        [p.join() for p in procs]
    assert loaded > idle * 1.2, f"idle {idle:.2f} W vs loaded {loaded:.2f} W: PMIC sum not tracking load"
    assert t_load >= t_idle
