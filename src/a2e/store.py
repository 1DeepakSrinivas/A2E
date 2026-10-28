from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  run_id INTEGER PRIMARY KEY AUTOINCREMENT,
  config_json TEXT NOT NULL,
  device TEXT NOT NULL,
  phase TEXT NOT NULL,            -- warmup | measured
  started_at REAL NOT NULL,
  idle_power_w REAL
);
CREATE TABLE IF NOT EXISTS turns (
  turn_id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER NOT NULL REFERENCES runs(run_id),
  task_id TEXT, step INTEGER,
  prefill_ms REAL, decode_ms REAL, parse_ms REAL, validate_ms REAL, tool_wait_ms REAL,
  harness_ms REAL, wall_ms REAL,
  prompt_tokens INTEGER, cached_tokens INTEGER, gen_tokens INTEGER,
  energy_j REAL, peak_rss_mb REAL, max_temp_c REAL, throttled INTEGER,
  correct INTEGER
);
CREATE TABLE IF NOT EXISTS samples (
  run_id INTEGER NOT NULL REFERENCES runs(run_id),
  t REAL NOT NULL, rss_mb REAL, cpu_pct REAL, temp_c REAL, freq_mhz REAL, power_w REAL
);
"""


class Store:
    def __init__(self, path: str | Path):
        self.db = sqlite3.connect(str(path))
        self.db.executescript(SCHEMA)

    def insert(self, table: str, row: dict) -> int:
        cols = ", ".join(row)
        qs = ", ".join("?" for _ in row)
        cur = self.db.execute(f"INSERT INTO {table} ({cols}) VALUES ({qs})", list(row.values()))
        self.db.commit()
        return cur.lastrowid

    def export_parquet(self, outdir: str | Path) -> None:
        import pandas as pd

        out = Path(outdir)
        out.mkdir(parents=True, exist_ok=True)
        for t in ("runs", "turns", "samples"):
            pd.read_sql(f"SELECT * FROM {t}", self.db).to_parquet(out / f"{t}.parquet")
