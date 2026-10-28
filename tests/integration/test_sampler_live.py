import os
import subprocess
import sys
import time

import psutil
import pytest

from a2e.metrics.power_base import PowerSource, integrate_energy
from a2e.metrics.sampler import Sampler, measure_idle_power
from a2e.metrics.thermal import Thermal


class Const(PowerSource):
    name, scope = "const", "test"

    def __init__(self, w=5.0):
        self.w = w

    def read_watts(self):
        return self.w


class Flaky(PowerSource):
    name, scope = "flaky", "test"

    def __init__(self):
        self.n = 0

    def read_watts(self):
        self.n += 1
        return None if self.n % 2 else 4.0


def test_rss_tracks_server_memory(fake_server):
    srv = fake_server("--rss-mb", "150")
    s = Sampler(srv.pid, Const(), hz=50).start()
    time.sleep(0.3)
    samples = s.stop()
    assert max(x[1] for x in samples) > 150  # ballast shows up in RSS


def test_rss_includes_children(tmp_path):
    code = ("import subprocess,sys,time\n"
            "c=subprocess.Popen([sys.executable,'-c','b=bytearray(120*2**20)\\nfor i in range(0,len(b),4096): b[i]=1\\n"
            "import time; time.sleep(5)'])\n"
            "time.sleep(5)")
    p = subprocess.Popen([sys.executable, "-c", code])
    try:
        time.sleep(1.0)
        s = Sampler(p.pid, hz=50).start()
        time.sleep(0.2)
        assert max(x[1] for x in s.stop()) > 100
    finally:
        for c in psutil.Process(p.pid).children(recursive=True):
            c.kill()
        p.kill()


def test_none_power_readings_are_dropped_from_trace():
    s = Sampler(os.getpid(), Flaky(), hz=100).start()
    time.sleep(0.3)
    s.stop()
    assert 0 < len(s.power_trace()) < len(s.samples)
    assert all(w == 4.0 for _, w in s.power_trace())


def test_sampler_survives_process_death():
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.3)"])
    s = Sampler(p.pid, Const(), hz=50).start()
    p.wait()
    time.sleep(0.2)
    s.stop()  # must not hang or raise
    assert len(s.samples) >= 1


def test_no_power_source_leaves_none():
    s = Sampler(os.getpid(), None, hz=50).start()
    time.sleep(0.1)
    assert all(x[5] is None for x in s.stop())


def test_window_selects_samples():
    s = Sampler(os.getpid(), Const(), hz=100).start()
    time.sleep(0.3)
    t_mid = time.perf_counter()
    time.sleep(0.3)
    s.stop()
    w = s.window(t_mid, t_mid + 10)
    assert w and all(x[0] >= t_mid for x in w) and len(w) < len(s.samples)


def test_idle_power_mean():
    assert measure_idle_power(Const(3.5), seconds=0.2, hz=50) == pytest.approx(3.5)
    assert measure_idle_power(type("N", (Const,), {"read_watts": lambda self: None})(), 0.1) is None


def test_thermal_callable_recorded():
    s = Sampler(os.getpid(), None, lambda: Thermal(temp_c=61.5, freq_mhz=2400), hz=50).start()
    time.sleep(0.1)
    x = s.stop()[0]
    assert x[3] == 61.5 and x[4] == 2400
