"""Gating for tests that need a real device. Set these to opt in:

  A2E_LLAMA_SERVER=/path/to/llama-server   A2E_GGUF=/path/to/model.gguf   A2E_DEVICE=pi5|mac
Optional: A2E_CTX (default 4096), A2E_THREADS, A2E_FLASH_ATTN (default on).
"""
import os
import platform
import shutil

import pytest

from a2e.config import RunConfig
from a2e.server import LlamaServer, build_args

needs_llama = pytest.mark.skipif(
    not (os.environ.get("A2E_LLAMA_SERVER") and os.environ.get("A2E_GGUF")),
    reason="set A2E_LLAMA_SERVER and A2E_GGUF to run real-server tests")
needs_pi = pytest.mark.skipif(shutil.which("vcgencmd") is None, reason="vcgencmd not found (not a Pi)")
needs_mac = pytest.mark.skipif(platform.system() != "Darwin", reason="not macOS")
needs_root = pytest.mark.skipif(not hasattr(os, "geteuid") or os.geteuid() != 0,
                                reason="powermetrics needs root: run with sudo")


@pytest.fixture
def real_server(tmp_path_factory, free_port):
    """Factory for a real llama-server for the configured device; stopped at teardown."""
    started = []

    def make(thinking: str = "default", **cfg_kw) -> LlamaServer:
        device = os.environ.get("A2E_DEVICE") or ("mac" if platform.system() == "Darwin" else "pi5")
        cfg = RunConfig(**{"model": "qwen3-4b", "quant": "q4_0", "device": device, **cfg_kw})
        args = build_args(cfg, os.environ["A2E_LLAMA_SERVER"], os.environ["A2E_GGUF"], port=free_port,
                          ctx=int(os.environ.get("A2E_CTX", 4096)),
                          threads=int(os.environ["A2E_THREADS"]) if os.environ.get("A2E_THREADS") else None,
                          flash_attn=os.environ.get("A2E_FLASH_ATTN", "on"),
                          slot_save_path=str(tmp_path_factory.mktemp("slots")), thinking=thinking)
        srv = LlamaServer(args, port=free_port, log_path=tmp_path_factory.mktemp("srv") / "llama.log")
        started.append(srv)
        srv.start(timeout=600)
        return srv

    yield make
    for s in started:
        s.stop()
