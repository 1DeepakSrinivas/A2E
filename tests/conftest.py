from __future__ import annotations

import os
import socket
import sys
from pathlib import Path

import pytest

from a2e.server import LlamaServer

FAKE = Path(__file__).parent / "fakes" / "fake_llama_server.py"
SUITES = ("unit", "integration", "sweep", "hardware")


DEPRECATED = "Python and BFCL tests are deprecated; set A2E_LEGACY_TESTS=1 to run them"


def pytest_collection_modifyitems(items):
    """Tag every test with its suite (from its directory) so `-m <suite>` selects a group.

    All tests are skipped unless A2E_LEGACY_TESTS=1.
    """
    legacy = os.environ.get("A2E_LEGACY_TESTS") == "1"
    for item in items:
        if not legacy:
            item.add_marker(pytest.mark.skip(reason=DEPRECATED))
        for suite in SUITES:
            if f"/tests/{suite}/" in str(item.fspath).replace("\\", "/"):
                item.add_marker(getattr(pytest.mark, suite))


@pytest.fixture
def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def fake_server(tmp_path, free_port):
    """Factory: start the fake llama-server with extra flags; all servers are stopped at teardown."""
    started: list[LlamaServer] = []

    def make(*extra: str, start: bool = True, timeout: float = 30.0) -> LlamaServer:
        srv = LlamaServer([sys.executable, str(FAKE), "--port", str(free_port), *extra],
                          port=free_port, log_path=tmp_path / "server.log")
        started.append(srv)
        if start:
            srv.start(timeout=timeout)
        return srv

    yield make
    for s in started:
        s.stop()
