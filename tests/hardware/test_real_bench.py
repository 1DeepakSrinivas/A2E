"""llama-bench (and perf on the Pi) on a real machine: checks the JSON fields and perf output the parser expects."""
import os
import shutil
from pathlib import Path

import pytest

from a2e.bench import bench_args, run_bench

from .conftest import needs_llama

pytestmark = needs_llama


def bench_binary():
    b = os.environ.get("A2E_LLAMA_BENCH") or str(Path(os.environ["A2E_LLAMA_SERVER"]).with_name("llama-bench"))
    if not Path(b).exists():
        pytest.skip("llama-bench not found next to llama-server (set A2E_LLAMA_BENCH)")
    return b


def device():
    import platform
    return os.environ.get("A2E_DEVICE") or ("mac" if platform.system() == "Darwin" else "pi5")


@pytest.mark.parametrize("test,p,g", [("pp", 64, 0), ("tg", 0, 16)])
def test_llama_bench_json_has_expected_fields(test, p, g):
    args = bench_args(bench_binary(), os.environ["A2E_GGUF"], n_prompt=p, n_gen=g, reps=1, device=device(),
                      kv_quant="f16", flash_attn=os.environ.get("A2E_FLASH_ATTN", "on"), threads=None)
    results, _, _ = run_bench(args, test, p, g)
    assert results and results[0].avg_tok_s and results[0].avg_tok_s > 0, results[0].raw
    assert "avg_ts" in results[0].raw and "n_prompt" in results[0].raw, sorted(results[0].raw)


@pytest.mark.skipif(not shutil.which("perf"), reason="perf not installed")
def test_perf_report_parses_on_the_pi():
    if device() != "pi5":
        pytest.skip("kernel profiles are Pi only")
    args = bench_args(bench_binary(), os.environ["A2E_GGUF"], n_prompt=0, n_gen=16, reps=1, device="pi5",
                      kv_quant="f16", flash_attn="on", threads=None)
    _, kernels, note = run_bench(args, "tg", 0, 16, perf=shutil.which("perf"))
    assert kernels, f"no symbols parsed ({note}); check perf_event_paranoid or run as root"
    assert any("ggml" in k["symbol"] for k in kernels), kernels[:5]
