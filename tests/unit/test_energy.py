"""Energy integration: stage windows are short relative to the sample period, so edges matter."""
import pytest

from a2e.metrics.power_base import integrate_energy

CONST = [(i * 0.5, 5.0) for i in range(0, 21)]  # 5 W for 10 s at 2 Hz


def test_full_window():
    assert integrate_energy(CONST, 0, 10) == pytest.approx(50.0)


def test_window_between_samples_is_not_zero():
    # a 0.2 s stage that starts and ends between two 0.5 s samples
    assert integrate_energy(CONST, 1.1, 1.3) == pytest.approx(1.0)


def test_window_edges_off_sample_grid():
    assert integrate_energy(CONST, 0.25, 9.75) == pytest.approx(5.0 * 9.5)


def test_ramp_is_interpolated():
    ramp = [(0.0, 0.0), (1.0, 10.0)]  # 0 -> 10 W
    assert integrate_energy(ramp, 0.0, 0.5) == pytest.approx(1.25)  # mean 2.5 W * 0.5 s


def test_idle_subtraction():
    assert integrate_energy(CONST, 0, 10, idle_w=2.0) == pytest.approx(30.0)


def test_degenerate_inputs():
    assert integrate_energy([], 0, 1) == 0.0
    assert integrate_energy([(0.0, 5.0)], 0, 1) == 0.0
    assert integrate_energy(CONST, 2.0, 2.0) == 0.0
    assert integrate_energy(CONST, 3.0, 2.0) == 0.0


def test_unsorted_samples():
    assert integrate_energy(list(reversed(CONST)), 0, 10) == pytest.approx(50.0)
