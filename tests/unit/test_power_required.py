import os

import pytest
from pydantic import ValidationError

from a2e import hardware
from a2e.cli import format_event
from a2e.hardware import PowerUnavailable, make_instruments
from a2e.settings import Settings


def _not_root(monkeypatch):
    monkeypatch.setattr(os, "geteuid", lambda: 501, raising=False)
    monkeypatch.setattr(hardware.shutil, "which", lambda name: "/usr/bin/" + name)


def test_auto_without_root_runs_with_a_note(monkeypatch):
    _not_root(monkeypatch)
    ins = make_instruments("mac", Settings(power="auto"))
    assert ins.power is None and ins.scope == "none" and "root" in ins.notes[0]


def test_required_without_root_raises(monkeypatch):
    _not_root(monkeypatch)
    with pytest.raises(PowerUnavailable, match="required"):
        make_instruments("mac", Settings(power="required"))


def test_required_on_pi_without_vcgencmd_raises(monkeypatch):
    monkeypatch.setattr(hardware.shutil, "which", lambda name: None)
    with pytest.raises(PowerUnavailable, match="vcgencmd"):
        make_instruments("pi5", Settings(power="required"))


def test_unknown_power_value_is_rejected():
    with pytest.raises(ValidationError):
        Settings(power="sometimes")


def test_no_energy_warning_is_loud():
    msg = format_event({"event": "instruments", "scope": "none", "notes": ["powermetrics needs root"]})
    assert msg.startswith("WARNING: no energy will be recorded")
    quiet = format_event({"event": "instruments", "scope": "CPU+GPU", "notes": ["x"]})
    assert not quiet.startswith("WARNING")
