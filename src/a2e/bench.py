"""Operator-level bottleneck analysis: llama-bench throughput and, on the Pi, Linux perf top kernels.

Prefill (pp) runs the model over a batch of prompt tokens and is mostly limited by compute. Decode (tg)
produces one token at a time and is mostly limited by memory bandwidth, because every weight is read once
per token. llama-bench measures each on its own. On the Pi, `perf record` around llama-bench shows which
functions (ggml kernels) take the most CPU time in each phase.

The llama-bench JSON field names and the perf report line format are from memory of those tools and are
parsed tolerantly; tests/hardware/test_real_bench.py checks them on a real machine.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

PERF_LINE = re.compile(r"^\s*([\d.]+)%\s+(\S+)\s+\[[.kgu]\]\s+(.+?)\s*$")


@dataclass
class BenchResult:
    test: str  # pp | tg
    n_prompt: int
    n_gen: int
    avg_tok_s: float | None
    sd_tok_s: float | None
    avg_ms: float | None
    n_threads: int | None
    raw: dict


def bench_args(binary: str, model: str, *, n_prompt: int, n_gen: int, reps: int, device: str, kv_quant: str,
               flash_attn: str, threads: int | None) -> list[str]:
    fa = "1" if str(flash_attn).lower() in ("on", "1", "true") else "0"
    args = [binary, "-m", model, "-p", str(n_prompt), "-n", str(n_gen), "-r", str(reps), "-o", "json",
            "-ngl", "99" if device == "mac" else "0", "-fa", fa]
    if kv_quant != "f16":
        args += ["-ctk", kv_quant, "-ctv", kv_quant]
    if threads:
        args += ["-t", str(threads)]
    return args


def parse_bench_json(text: str, test: str, n_prompt: int, n_gen: int) -> list[BenchResult]:
    """llama-bench -o json prints a JSON array of test results (avg_ts / stddev_ts in tokens per second)."""
    start = text.find("[")
    if start < 0:
        raise ValueError("no JSON array in llama-bench output")
    rows = json.loads(text[start:text.rindex("]") + 1])
    out = []
    for r in rows:
        ns = r.get("avg_ns")
        out.append(BenchResult(test, int(r.get("n_prompt", n_prompt)), int(r.get("n_gen", n_gen)),
                               r.get("avg_ts"), r.get("stddev_ts"), ns / 1e6 if ns else None,
                               r.get("n_threads"), r))
    return out


def parse_perf_report(text: str, top: int = 15) -> list[dict]:
    """Lines like `  23.45%  libggml-cpu.so  [.] ggml_vec_dot_q4_0_q8_0` -> ranked symbols."""
    rows = []
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        m = PERF_LINE.match(line)
        if m:
            rows.append({"pct": float(m.group(1)), "dso": m.group(2), "symbol": m.group(3)})
    rows.sort(key=lambda r: -r["pct"])
    return [{"rank": i + 1, **r} for i, r in enumerate(rows[:top])]


def perf_available(perf: str | None = None) -> str | None:
    return perf or shutil.which("perf")


def run_bench(args: list[str], test: str, n_prompt: int, n_gen: int, *, perf: str | None = None,
              timeout: float = 3600, on_start=None) -> tuple[list[BenchResult], list[dict], str | None]:
    """Run one llama-bench invocation, optionally under `perf record`. Returns (results, top kernels, note).

    `on_start(pid)` is called with the process id once it starts (the runner samples it).
    """
    note = None
    with tempfile.TemporaryDirectory() as td:
        data = Path(td) / "perf.data"
        cmd = ([perf, "record", "-F", "999", "-o", str(data), "--"] + args) if perf else args
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if on_start:
            on_start(proc.pid)
        out, err = proc.communicate(timeout=timeout)
        if proc.returncode != 0:
            raise RuntimeError(f"llama-bench failed ({proc.returncode}): {err.strip()[-400:]}")
        results = parse_bench_json(out, test, n_prompt, n_gen)
        kernels: list[dict] = []
        if perf:
            rep = subprocess.run([perf, "report", "-i", str(data), "--stdio", "--no-children", "--sort", "dso,sym",
                                  "--percent-limit", "0.5"], capture_output=True, text=True, timeout=600)
            kernels = parse_perf_report(rep.stdout)
            if not kernels:
                note = "perf report gave no symbols: " + (rep.stderr.strip()[-200:] or "empty output")
    return results, kernels, note
