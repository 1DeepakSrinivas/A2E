#!/usr/bin/env python3
"""Stand-in for llama-bench -o json: prints a log line then a JSON array, like the real tool's stdout."""
import json
import os
import sys
import time

a = sys.argv[1:]
get = lambda f, d: a[a.index(f) + 1] if f in a else d  # noqa: E731
if os.environ.get("FAKE_BENCH_FAIL"):
    print("error: failed to load model", file=sys.stderr)
    sys.exit(1)
p, n, r = int(get("-p", 512)), int(get("-n", 128)), int(get("-r", 5))
ballast = bytearray(50 * 2**20)
for i in range(0, len(ballast), 4096):
    ballast[i] = 1
time.sleep(0.3)
ts = 400.0 if p else 20.0
print("build: fake (0)", file=sys.stderr)
print(json.dumps([{"model_type": "fake", "n_prompt": p, "n_gen": n, "n_threads": int(get("-t", 4)),
                   "avg_ns": int(1e9 * (p or n) / ts), "stddev_ns": 1000, "avg_ts": ts, "stddev_ts": 1.5,
                   "samples_ts": [ts] * r}], indent=2))
