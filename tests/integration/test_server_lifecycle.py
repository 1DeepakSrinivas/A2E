import time

import httpx
import pytest

from a2e.client import Client


def test_start_reports_load_time_and_health(fake_server):
    srv = fake_server("--health-delay", "0.6")
    assert srv.proc.poll() is None
    assert httpx.get(f"{srv.url}/health").status_code == 200


def test_load_time_includes_health_delay(fake_server):
    srv = fake_server(start=False)
    srv_delay = fake_server("--health-delay", "0.8", start=False)
    t = srv_delay.start()
    assert t >= 0.8
    assert srv.proc is None


def test_early_exit_raises_with_log_hint(fake_server):
    srv = fake_server("--exit-code", "3", start=False)
    with pytest.raises(RuntimeError, match="exited"):
        srv.start(timeout=10)


def test_health_timeout_kills_process(fake_server):
    srv = fake_server("--health-delay", "60", start=False)
    with pytest.raises(TimeoutError):
        srv.start(timeout=1.0)
    assert srv.proc.poll() is not None


def test_stop_is_idempotent_and_frees_port(fake_server):
    srv = fake_server()
    srv.stop()
    srv.stop()
    with pytest.raises(httpx.HTTPError):
        httpx.get(f"{srv.url}/health", timeout=1)


def test_context_manager_stops_on_exit(fake_server, tmp_path):
    srv = fake_server(start=False)
    with srv:
        pid = srv.pid
        assert srv.proc.poll() is None
    assert srv.proc.poll() is not None and pid


def test_restart_same_port(fake_server):
    srv = fake_server()
    srv.stop()
    time.sleep(0.2)
    srv.start(timeout=20)
    assert httpx.get(f"{srv.url}/health").status_code == 200
