from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import httpx

from a2e.config import RunConfig

THINKING_OFF = {"enable_thinking": False}  # the Qwen3 chat template variable; other templates ignore it


def build_args(cfg: RunConfig, bin_path: str, model_path: str, port: int = 8080,
               ctx: int = 8192, threads: int | None = None, flash_attn: str = "on",
               slot_save_path: str | None = None, thinking: str = "default") -> list[str]:
    """llama-server command line for a RunConfig.

    KV-cache quantization needs flash attention. `flash_attn` is passed as the value of
    `--flash-attn` (newer builds take on/off/auto); override for older builds.
    Metal is used on Mac via full GPU offload; the Pi runs CPU only.
    `slot_save_path` is required for `POST /slots/0?action=erase`: without `--slot-save-path` the
    server answers 501 to every slot action, so the cache could not be reset between repeats.
    `thinking = "off"` passes `--chat-template-kwargs` so reasoning models (Qwen3) answer directly instead of
    spending the token budget on a thinking block. Templates that ignore the variable are not affected.
    """
    args = [bin_path, "-m", model_path, "--port", str(port), "-c", str(ctx), "--jinja",
            "--flash-attn", flash_attn]
    args += ["-ngl", "99" if cfg.device == "mac" else "0"]
    if threads:
        args += ["-t", str(threads)]
    if cfg.kv_quant != "f16":
        args += ["--cache-type-k", cfg.kv_quant, "--cache-type-v", cfg.kv_quant]
    if slot_save_path:
        args += ["--slot-save-path", slot_save_path]
    if thinking == "off":
        args += ["--chat-template-kwargs", json.dumps(THINKING_OFF)]
    return args


class LlamaServer:
    def __init__(self, args: list[str], port: int = 8080, log_path: str | Path = "llama-server.log"):
        self.args, self.port, self.log_path = args, port, Path(log_path)
        self.proc: subprocess.Popen | None = None
        self.url = f"http://127.0.0.1:{port}"

    def start(self, timeout: float = 300.0) -> float:
        """Launch and wait for /health. Returns model load time in seconds."""
        t0 = time.perf_counter()
        self._log = open(self.log_path, "w")
        self.proc = subprocess.Popen(self.args, stdout=self._log, stderr=subprocess.STDOUT)
        while time.perf_counter() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited ({self.proc.returncode}); see {self.log_path}")
            try:
                if httpx.get(f"{self.url}/health", timeout=2).status_code == 200:
                    return time.perf_counter() - t0
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        self.stop()
        raise TimeoutError("llama-server did not become healthy")

    @property
    def pid(self) -> int:
        assert self.proc is not None
        return self.proc.pid

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if getattr(self, "_log", None):
            self._log.close()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
