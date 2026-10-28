import os
import time

from a2e.metrics.power_base import PowerSource
from a2e.metrics.power_mac import parse_powermetrics
from a2e.metrics.power_pi import parse_pmic
from a2e.metrics.sampler import Sampler
from a2e.metrics.thermal import Thermal, parse_clock_mhz, parse_temp, parse_throttled

PMIC = """\
   3V7_WL_SW_A current(0)=0.10000000A
   VDD_CORE_A current(2)=2.00000000A
   3V3_SYS_A current(4)=0.20000000A
   3V7_WL_SW_V volt(8)=3.70000000V
   VDD_CORE_V volt(9)=0.80000000V
   3V3_SYS_V volt(12)=3.30000000V
   EXT5V_V volt(24)=5.10000000V
"""

PM = """\
*** Sampled system activity (Tue Oct 28 16:00:00 2026 +0000) (500.2ms elapsed) ***
CPU Power: 1200 mW
GPU Power: 300 mW
ANE Power: 0 mW
Combined Power (CPU + GPU + ANE): 1500 mW
Current pressure level: Nominal
"""


def test_parse_pmic_skips_ext5v_and_sums_vi():
    assert abs(parse_pmic(PMIC) - (0.37 + 1.6 + 0.66)) < 1e-9


def test_parse_powermetrics():
    assert parse_powermetrics(PM) == (1.5, "Nominal")
    assert parse_powermetrics("CPU Power: 800 mW\nGPU Power: 200 mW\n")[0] == 1.0


def test_vcgencmd_parsers():
    assert parse_temp("temp=54.0'C") == 54.0
    assert parse_clock_mhz("frequency(48)=2400000000") == 2400.0
    assert parse_throttled("throttled=0x50005") == 0x50005


class Fake(PowerSource):
    name, scope = "fake", "test"

    def read_watts(self):
        return 3.0


def test_sampler_collects_rss_and_power():
    s = Sampler(os.getpid(), Fake(), lambda: Thermal(temp_c=40.0), hz=50).start()
    time.sleep(0.2)
    samples = s.stop()
    assert len(samples) >= 3
    assert samples[0][1] > 0 and samples[0][3] == 40.0 and samples[0][5] == 3.0
    assert len(s.power_trace()) == len(samples)
