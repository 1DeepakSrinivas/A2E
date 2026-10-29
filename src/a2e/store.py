"""SQLite metrics store.

One database file holds everything: experiments, jobs, runs, per-turn timings, per-case scores, and the
resource samples taken during each run. SQLite is enough here. Writes come from a single runner process and
readers (the TUI, the results commands, a notebook) can open the same file at the same time because the
database runs in WAL mode. The schema is versioned with `PRAGMA user_version` and upgraded by MIGRATIONS.
"""
from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

# Each entry upgrades the database by one version. Never edit an old entry; append a new one.
MIGRATIONS: list[str] = [
    # ---- version 1 ----------------------------------------------------------------------------------
    """
    CREATE TABLE experiments (
      exp_id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT NOT NULL,
      suite_hash TEXT, suite_json TEXT, env_json TEXT,
      git_commit TEXT, git_dirty INTEGER,
      status TEXT NOT NULL DEFAULT 'running',          -- running | complete | interrupted
      started_at REAL NOT NULL, finished_at REAL, notes TEXT
    );
    CREATE TABLE jobs (
      job_id INTEGER PRIMARY KEY AUTOINCREMENT,
      exp_id INTEGER NOT NULL REFERENCES experiments(exp_id),
      job_key TEXT NOT NULL,                           -- hash of the canonical job config
      kind TEXT NOT NULL,                              -- plain | custom | bfcl
      config_json TEXT NOT NULL,
      device TEXT, model TEXT, quant TEXT, kv_quant TEXT,
      status TEXT NOT NULL DEFAULT 'pending',          -- pending | running | done | failed | skipped
      reason TEXT, error TEXT,
      attempts INTEGER NOT NULL DEFAULT 0,
      started_at REAL, finished_at REAL,
      UNIQUE (exp_id, job_key)
    );
    CREATE TABLE runs (
      run_id INTEGER PRIMARY KEY AUTOINCREMENT,
      exp_id INTEGER REFERENCES experiments(exp_id),
      job_id INTEGER REFERENCES jobs(job_id),
      config_json TEXT NOT NULL,
      device TEXT NOT NULL,
      phase TEXT NOT NULL,                             -- warmup | measured (anything else for ad hoc runs)
      repeat_idx INTEGER,
      task_kind TEXT,
      status TEXT NOT NULL DEFAULT 'complete',         -- running | complete | aborted
      started_at REAL NOT NULL, finished_at REAL,
      idle_power_w REAL, power_scope TEXT, load_time_s REAL, server_json TEXT
    );
    CREATE TABLE turns (
      turn_id INTEGER PRIMARY KEY AUTOINCREMENT,
      run_id INTEGER NOT NULL REFERENCES runs(run_id),
      task_id TEXT, step INTEGER, user_turn INTEGER,
      prefill_ms REAL, decode_ms REAL, parse_ms REAL, validate_ms REAL, tool_wait_ms REAL,
      template_ms REAL, harness_ms REAL, wall_ms REAL,
      prompt_tokens INTEGER, cached_tokens INTEGER, gen_tokens INTEGER, n_calls INTEGER,
      energy_j REAL, energy_prefill_j REAL, energy_decode_j REAL, energy_tool_j REAL,
      avg_power_w REAL, peak_rss_mb REAL, avg_cpu_pct REAL, max_temp_c REAL, mean_freq_mhz REAL,
      throttled INTEGER, correct INTEGER, error TEXT
    );
    CREATE TABLE cases (
      case_row_id INTEGER PRIMARY KEY AUTOINCREMENT,
      run_id INTEGER NOT NULL REFERENCES runs(run_id),
      case_id TEXT NOT NULL, category TEXT NOT NULL,
      valid INTEGER, error_type TEXT, status TEXT,
      n_user_turns INTEGER, n_model_calls INTEGER, schema_tokens INTEGER, wall_ms REAL, extra_json TEXT
    );
    CREATE TABLE samples (
      run_id INTEGER NOT NULL REFERENCES runs(run_id),
      t REAL NOT NULL,                                 -- seconds since the run started
      rss_mb REAL, cpu_pct REAL, temp_c REAL, freq_mhz REAL, power_w REAL, throttled INTEGER, pressure TEXT
    );
    CREATE INDEX idx_jobs_exp ON jobs(exp_id, status);
    CREATE INDEX idx_runs_job ON runs(job_id);
    CREATE INDEX idx_turns_run ON turns(run_id);
    CREATE INDEX idx_cases_run ON cases(run_id);
    CREATE INDEX idx_samples_run ON samples(run_id, t);

    -- Analysis views. They keep only complete runs, so aborted partial runs never reach a table or plot,
    -- and they pull the common configuration fields out of the stored JSON.
    CREATE VIEW v_runs AS
      SELECT r.*,
             json_extract(r.config_json, '$.model') AS model,
             json_extract(r.config_json, '$.quant') AS quant,
             json_extract(r.config_json, '$.kv_quant') AS kv_quant,
             json_extract(r.config_json, '$.grammar') AS grammar,
             json_extract(r.config_json, '$.prefix_cache') AS prefix_cache,
             json_extract(r.config_json, '$.decode_len') AS decode_len,
             json_extract(r.config_json, '$.schema_len') AS schema_len,
             json_extract(r.config_json, '$.family') AS family,
             json_extract(r.config_json, '$.depth') AS depth,
             json_extract(r.config_json, '$.n_args') AS n_args,
             json_extract(r.config_json, '$.category') AS category,
             json_extract(r.config_json, '$.governor') AS governor
        FROM runs r WHERE r.status = 'complete';
    CREATE VIEW v_turns AS
      SELECT t.*, v.exp_id, v.job_id, v.phase, v.repeat_idx, v.task_kind, v.device, v.idle_power_w,
             v.model, v.quant, v.kv_quant, v.grammar, v.prefix_cache, v.decode_len, v.schema_len,
             v.family, v.depth, v.n_args, v.category, v.governor
        FROM turns t JOIN v_runs v ON v.run_id = t.run_id;
    CREATE VIEW v_cases AS
      SELECT c.*, v.exp_id, v.job_id, v.phase, v.repeat_idx, v.task_kind, v.device,
             v.model, v.quant, v.kv_quant, v.grammar, v.prefix_cache, v.decode_len, v.schema_len,
             v.family, v.depth, v.n_args, v.governor
        FROM cases c JOIN v_runs v ON v.run_id = c.run_id;
    """,
    # ---- version 2: operator-level bottleneck analysis ---------------------------------------------
    """
    CREATE TABLE bench (
      bench_id INTEGER PRIMARY KEY AUTOINCREMENT,
      run_id INTEGER NOT NULL REFERENCES runs(run_id),
      test TEXT NOT NULL,                              -- pp (prefill) | tg (decode)
      n_prompt INTEGER, n_gen INTEGER, n_threads INTEGER,
      avg_tok_s REAL, sd_tok_s REAL, avg_ms REAL, raw_json TEXT
    );
    CREATE TABLE kernels (
      run_id INTEGER NOT NULL REFERENCES runs(run_id),
      test TEXT NOT NULL,                              -- pp | tg
      n_tokens INTEGER, rank INTEGER, pct REAL, symbol TEXT, dso TEXT
    );
    CREATE INDEX idx_bench_run ON bench(run_id);
    CREATE INDEX idx_kernels_run ON kernels(run_id);
    """,
    # ---- version 3: harness variants ---------------------------------------------------------------
    # Jobs of the default harness store no harness fields, so the views fall back to the defaults. Older
    # results therefore read as the baseline harness.
    """
    ALTER TABLE turns ADD COLUMN energy_harness_j REAL;     -- templating + parsing + validating
    DROP VIEW v_cases;
    DROP VIEW v_turns;
    DROP VIEW v_runs;
    CREATE VIEW v_runs AS
      SELECT r.*,
             json_extract(r.config_json, '$.model') AS model,
             json_extract(r.config_json, '$.quant') AS quant,
             json_extract(r.config_json, '$.kv_quant') AS kv_quant,
             json_extract(r.config_json, '$.grammar') AS grammar,
             json_extract(r.config_json, '$.prefix_cache') AS prefix_cache,
             json_extract(r.config_json, '$.decode_len') AS decode_len,
             json_extract(r.config_json, '$.schema_len') AS schema_len,
             json_extract(r.config_json, '$.family') AS family,
             json_extract(r.config_json, '$.depth') AS depth,
             json_extract(r.config_json, '$.n_args') AS n_args,
             json_extract(r.config_json, '$.category') AS category,
             json_extract(r.config_json, '$.governor') AS governor,
             COALESCE(json_extract(r.config_json, '$.harness'), 'baseline') AS harness,
             COALESCE(json_extract(r.config_json, '$.prompt_format'), 'json_schema') AS prompt_format,
             COALESCE(json_extract(r.config_json, '$.context'), 'full') AS context,
             COALESCE(json_extract(r.config_json, '$.recovery'), 'none') AS recovery,
             COALESCE(json_extract(r.config_json, '$.stopping'), 'fixed') AS stopping
        FROM runs r WHERE r.status = 'complete';
    CREATE VIEW v_turns AS
      SELECT t.*, v.exp_id, v.job_id, v.phase, v.repeat_idx, v.task_kind, v.device, v.idle_power_w,
             v.model, v.quant, v.kv_quant, v.grammar, v.prefix_cache, v.decode_len, v.schema_len,
             v.family, v.depth, v.n_args, v.category, v.governor,
             v.harness, v.prompt_format, v.context, v.recovery, v.stopping
        FROM turns t JOIN v_runs v ON v.run_id = t.run_id;
    CREATE VIEW v_cases AS
      SELECT c.*, v.exp_id, v.job_id, v.phase, v.repeat_idx, v.task_kind, v.device,
             v.model, v.quant, v.kv_quant, v.grammar, v.prefix_cache, v.decode_len, v.schema_len,
             v.family, v.depth, v.n_args, v.governor,
             v.harness, v.prompt_format, v.context, v.recovery, v.stopping
        FROM cases c JOIN v_runs v ON v.run_id = c.run_id;
    """,
    # ---- version 4: thinking switch -----------------------------------------------------------------
    # Runs made before it read as "default", the model's own thinking behavior.
    """
    DROP VIEW v_cases;
    DROP VIEW v_turns;
    DROP VIEW v_runs;
    CREATE VIEW v_runs AS
      SELECT r.*,
             json_extract(r.config_json, '$.model') AS model,
             json_extract(r.config_json, '$.quant') AS quant,
             json_extract(r.config_json, '$.kv_quant') AS kv_quant,
             json_extract(r.config_json, '$.grammar') AS grammar,
             json_extract(r.config_json, '$.prefix_cache') AS prefix_cache,
             json_extract(r.config_json, '$.decode_len') AS decode_len,
             json_extract(r.config_json, '$.schema_len') AS schema_len,
             json_extract(r.config_json, '$.family') AS family,
             json_extract(r.config_json, '$.depth') AS depth,
             json_extract(r.config_json, '$.n_args') AS n_args,
             json_extract(r.config_json, '$.category') AS category,
             json_extract(r.config_json, '$.governor') AS governor,
             COALESCE(json_extract(r.config_json, '$.harness'), 'baseline') AS harness,
             COALESCE(json_extract(r.config_json, '$.prompt_format'), 'json_schema') AS prompt_format,
             COALESCE(json_extract(r.config_json, '$.context'), 'full') AS context,
             COALESCE(json_extract(r.config_json, '$.recovery'), 'none') AS recovery,
             COALESCE(json_extract(r.config_json, '$.stopping'), 'fixed') AS stopping,
             COALESCE(json_extract(r.config_json, '$.thinking'), 'default') AS thinking
        FROM runs r WHERE r.status = 'complete';
    CREATE VIEW v_turns AS
      SELECT t.*, v.exp_id, v.job_id, v.phase, v.repeat_idx, v.task_kind, v.device, v.idle_power_w,
             v.model, v.quant, v.kv_quant, v.grammar, v.prefix_cache, v.decode_len, v.schema_len,
             v.family, v.depth, v.n_args, v.category, v.governor,
             v.harness, v.prompt_format, v.context, v.recovery, v.stopping, v.thinking
        FROM turns t JOIN v_runs v ON v.run_id = t.run_id;
    CREATE VIEW v_cases AS
      SELECT c.*, v.exp_id, v.job_id, v.phase, v.repeat_idx, v.task_kind, v.device,
             v.model, v.quant, v.kv_quant, v.grammar, v.prefix_cache, v.decode_len, v.schema_len,
             v.family, v.depth, v.n_args, v.governor,
             v.harness, v.prompt_format, v.context, v.recovery, v.stopping, v.thinking
        FROM cases c JOIN v_runs v ON v.run_id = c.run_id;
    """,
]

TABLES = ("experiments", "jobs", "runs", "turns", "cases", "samples", "bench", "kernels")
VIEWS = ("v_runs", "v_turns", "v_cases")


class Store:
    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread is off so the TUI can read from a worker thread; writes still come from one process.
        self.db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.execute("PRAGMA busy_timeout = 10000")
        if self.path != ":memory:":
            self.db.execute("PRAGMA journal_mode = WAL")
            self.db.execute("PRAGMA synchronous = NORMAL")
        self._depth = 0
        self.migrate()

    # ---------------------------------------------------------------- schema
    @property
    def version(self) -> int:
        return self.db.execute("PRAGMA user_version").fetchone()[0]

    def migrate(self) -> None:
        v = self.version
        if v > len(MIGRATIONS):
            raise RuntimeError(f"{self.path} has schema version {v}, newer than this code ({len(MIGRATIONS)})")
        for i in range(v, len(MIGRATIONS)):
            self.db.executescript("BEGIN;\n" + MIGRATIONS[i] + f"\nPRAGMA user_version = {i + 1};\nCOMMIT;")

    # ---------------------------------------------------------------- writes
    @contextmanager
    def transaction(self):
        """Group many writes into one commit (much faster on an SD card). Nested use joins the outer one."""
        if self._depth == 0:
            self.db.execute("BEGIN")
        self._depth += 1
        try:
            yield self
        except BaseException:
            self._depth -= 1
            if self._depth == 0:
                self.db.execute("ROLLBACK")
            raise
        else:
            self._depth -= 1
            if self._depth == 0:
                self.db.execute("COMMIT")

    def insert(self, table: str, row: dict) -> int:
        cols = ", ".join(row)
        marks = ", ".join("?" for _ in row)
        cur = self.db.execute(f"INSERT INTO {table} ({cols}) VALUES ({marks})", list(row.values()))
        return cur.lastrowid

    def insert_many(self, table: str, rows: Iterable[dict]) -> None:
        rows = list(rows)
        if not rows:
            return
        cols = list(rows[0])
        with self.transaction():
            self.db.executemany(
                f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})",
                [[r.get(c) for c in cols] for r in rows])

    def update(self, table: str, key_col: str, key: Any, fields: dict) -> None:
        sets = ", ".join(f"{k} = ?" for k in fields)
        self.db.execute(f"UPDATE {table} SET {sets} WHERE {key_col} = ?", [*fields.values(), key])

    def execute(self, sql: str, params: Iterable = ()) -> None:
        self.db.execute(sql, list(params))

    # ---------------------------------------------------------------- reads
    def query(self, sql: str, params: Iterable = ()) -> list[dict]:
        cur = self.db.execute(sql, list(params))
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def scalar(self, sql: str, params: Iterable = ()) -> Any:
        row = self.db.execute(sql, list(params)).fetchone()
        return None if row is None else row[0]

    def df(self, sql: str, params: Iterable = ()):
        import pandas as pd
        return pd.read_sql_query(sql, self.db, params=list(params))

    # ---------------------------------------------------------------- export / housekeeping
    def export(self, outdir: str | Path, fmt: str = "parquet", include_views: bool = True) -> list[Path]:
        """Write every table (and the analysis views) as parquet or csv files."""
        if fmt not in ("parquet", "csv"):
            raise ValueError("fmt must be parquet or csv")
        out = Path(outdir)
        out.mkdir(parents=True, exist_ok=True)
        written = []
        for name in TABLES + (VIEWS if include_views else ()):
            df = self.df(f"SELECT * FROM {name}")
            path = out / f"{name}.{fmt}"
            df.to_parquet(path) if fmt == "parquet" else df.to_csv(path, index=False)
            written.append(path)
        return written

    def export_parquet(self, outdir: str | Path) -> None:  # kept for existing callers
        self.export(outdir, "parquet", include_views=False)

    def checkpoint(self) -> None:
        if self.path != ":memory:":
            self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def close(self) -> None:
        self.db.close()

    @staticmethod
    def now() -> float:
        return time.time()
