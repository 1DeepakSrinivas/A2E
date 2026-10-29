#!/usr/bin/env python3
"""Stand-in for Linux perf: `record -o FILE -- cmd...` runs cmd; `report` prints a canned symbol table."""
import subprocess
import sys

a = sys.argv[1:]
if a and a[0] == "record":
    out = a[a.index("-o") + 1]
    cmd = a[a.index("--") + 1:]
    code = subprocess.call(cmd)
    open(out, "w").write("fake perf data")
    sys.exit(code)
if a and a[0] == "report":
    print("# Samples: 12K of event 'cpu-clock'\n#\n# Overhead  Shared Object  Symbol\n# ........\n#")
    print("    41.20%  libggml-cpu.so     [.] ggml_vec_dot_q4_0_q8_0")
    print("    12.50%  libggml-cpu.so     [.] ggml_compute_forward_mul_mat")
    print("     8.00%  [kernel.kallsyms]  [k] clear_page_erms")
    print("     3.10%  libc.so.6          [.] __memcpy_generic")
    print("     0.90%  llama-bench        [.] main")
    print("garbage line that is not a symbol row")
    sys.exit(0)
sys.exit(2)
