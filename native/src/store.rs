// store.rs
// SQLite storage for studies, attempts, and cached judgments
// Interrupted-run recovery, grouped summaries, and CSV export

use anyhow::{bail, Result};
use rusqlite::{params, Connection, OptionalExtension};
use serde_json::{json, Value};
use std::{
    io::Write,
    path::{Path, PathBuf},
};

pub struct Store {
    pub db: Connection,
}
impl Store {
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(p) = path.parent() {
            std::fs::create_dir_all(p)?;
        }
        let db = Connection::open(path)?;
        db.busy_timeout(std::time::Duration::from_secs(5))?;
        let version: i64 = db.pragma_query_value(None, "user_version", |r| r.get(0))?;
        let app: i64 = db.pragma_query_value(None, "application_id", |r| r.get(0))?;
        let tables: i64 = db.query_row(
            "SELECT count(*) FROM sqlite_master WHERE type='table'",
            [],
            |r| r.get(0),
        )?;
        if (tables > 0 && app != 0x4132454e) || version > 1 {
            bail!("not a supported A2E native database");
        }
        db.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;
            BEGIN;
            CREATE TABLE IF NOT EXISTS studies(id TEXT PRIMARY KEY, definition TEXT NOT NULL, judge_config TEXT);
            CREATE TABLE IF NOT EXISTS runs(id INTEGER PRIMARY KEY, study TEXT NOT NULL REFERENCES studies(id), job TEXT NOT NULL, attempt INTEGER NOT NULL, status TEXT NOT NULL, phase TEXT NOT NULL, metadata TEXT NOT NULL, result TEXT, trace TEXT NOT NULL, UNIQUE(study,job,attempt));
            CREATE TABLE IF NOT EXISTS judgments(cache_key TEXT PRIMARY KEY, rubric TEXT NOT NULL, config TEXT NOT NULL, packet TEXT NOT NULL, response TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS run_judgments(run_id INTEGER PRIMARY KEY REFERENCES runs(id), cache_key TEXT NOT NULL REFERENCES judgments(cache_key));
            PRAGMA application_id=1093813582; PRAGMA user_version=1; COMMIT;")?;
        Ok(Self { db })
    }
    pub fn study(&self, id: &str, definition: &Value) -> Result<()> {
        self.db.execute(
            "INSERT OR IGNORE INTO studies(id,definition) VALUES (?,?)",
            params![id, definition.to_string()],
        )?;
        Ok(())
    }
    /// Retry incomplete attempts without reusing task state.
    pub fn recover(&self) -> Result<usize> {
        Ok(self.db.execute(
            "UPDATE runs SET status='interrupted' WHERE status='running'",
            [],
        )?)
    }
    pub fn should_run(&self, study: &str, job: &str, retry_failed: bool) -> Result<bool> {
        let old:Option<(String,Option<String>)>=self.db.query_row("SELECT status,result FROM runs WHERE study=? AND job=? ORDER BY attempt DESC LIMIT 1",params![study,job],|r|Ok((r.get(0)?,r.get(1)?))).optional()?;
        Ok(match old {
            None => true,
            Some((s, result)) => {
                s != "complete"
                    || (retry_failed
                        && result.as_ref().is_some_and(|r| {
                            serde_json::from_str::<Value>(r).is_ok_and(|v| {
                                v["status"] != "success" && v["status"] != "awaiting_judge"
                            })
                        }))
            }
        })
    }
    pub fn begin(
        &self,
        study: &str,
        job: &str,
        phase: &str,
        meta: &Value,
        trace: &Path,
    ) -> Result<i64> {
        self.db.execute("INSERT INTO runs(study,job,attempt,status,phase,metadata,trace) SELECT ?,?,coalesce(max(attempt),0)+1,'running',?,?,? FROM runs WHERE study=? AND job=?",params![study,job,phase,meta.to_string(),trace.to_string_lossy(),study,job])?;
        Ok(self.db.last_insert_rowid())
    }
    pub fn finish(&self, id: i64, result: &Value) -> Result<()> {
        let state = if result["status"] == "interrupted" {
            "interrupted"
        } else {
            "complete"
        };
        self.db.execute(
            "UPDATE runs SET status=?,result=? WHERE id=?",
            params![state, result.to_string(), id],
        )?;
        Ok(())
    }
    pub fn rows(&self) -> Result<Vec<Value>> {
        let mut statement=self.db.prepare("SELECT r.id,r.study,r.status,r.phase,r.metadata,r.result,r.trace,j.response FROM runs r LEFT JOIN run_judgments rj ON rj.run_id=r.id LEFT JOIN judgments j ON j.cache_key=rj.cache_key WHERE r.attempt=(SELECT max(attempt) FROM runs n WHERE n.study=r.study AND n.job=r.job) ORDER BY r.id")?;
        let rows = statement.query_map([], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, String>(3)?,
                r.get::<_, String>(4)?,
                r.get::<_, Option<String>>(5)?,
                r.get::<_, String>(6)?,
                r.get::<_, Option<String>>(7)?,
            ))
        })?;
        let mut out = Vec::new();
        for r in rows {
            let (id, study, state, phase, meta, result, trace, judgment) = r?;
            out.push(json!({"id":id,"study":study,"state":state,"phase":phase,"metadata":serde_json::from_str::<Value>(&meta)?,"result":result.map(|s|serde_json::from_str::<Value>(&s)).transpose()?,"trace":trace,"judgment":judgment.map(|s|serde_json::from_str::<Value>(&s)).transpose()?}));
        }
        Ok(out)
    }
    pub fn freeze_judge(&self, study: &str, config: &str) -> Result<()> {
        self.db.execute(
            "UPDATE studies SET judge_config=? WHERE id=? AND judge_config IS NULL",
            params![config, study],
        )?;
        let saved: String = self.db.query_row(
            "SELECT judge_config FROM studies WHERE id=?",
            [study],
            |r| r.get(0),
        )?;
        if saved != config {
            bail!("judge configuration is frozen for this study; create a new study to change it");
        }
        Ok(())
    }
    pub fn cached(&self, key: &str) -> Result<Option<Value>> {
        let s: Option<String> = self
            .db
            .query_row(
                "SELECT response FROM judgments WHERE cache_key=?",
                [key],
                |r| r.get(0),
            )
            .optional()?;
        Ok(s.map(|s| serde_json::from_str(&s)).transpose()?)
    }
    pub fn judgment(
        &self,
        id: i64,
        key: &str,
        rubric: &str,
        config: &Value,
        packet: &Value,
        response: &Value,
    ) -> Result<()> {
        let tx = self.db.unchecked_transaction()?;
        tx.execute(
            "INSERT OR IGNORE INTO judgments VALUES(?,?,?,?,?)",
            params![
                key,
                rubric,
                config.to_string(),
                packet.to_string(),
                response.to_string()
            ],
        )?;
        tx.execute(
            "INSERT OR REPLACE INTO run_judgments VALUES(?,?)",
            params![id, key],
        )?;
        tx.commit()?;
        Ok(())
    }
}
pub struct RunLock {
    path: PathBuf,
}
impl RunLock {
    pub fn acquire(dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(dir)?;
        let path = dir.join("runtime.lock");
        for _ in 0..2 {
            match std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
            {
                Ok(mut f) => {
                    writeln!(f, "{}", std::process::id())?;
                    return Ok(Self { path });
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    let pid = std::fs::read_to_string(&path)?.trim().parse::<i32>()?;
                    if pid <= 0 {
                        bail!("invalid runtime lock");
                    }
                    // SAFETY: Signal zero and a validated positive PID
                    if unsafe { libc::kill(pid, 0) } == 0
                        || std::io::Error::last_os_error().raw_os_error() != Some(libc::ESRCH)
                    {
                        bail!("another native run holds the runtime lock");
                    }
                    std::fs::remove_file(&path)?;
                }
                Err(e) => return Err(e.into()),
            }
        }
        bail!("cannot acquire runtime lock")
    }
}
impl Drop for RunLock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}
/// Measured attempts grouped by experiment condition.
pub fn summary(rows: &[Value]) -> Value {
    let mut groups: std::collections::BTreeMap<String, Vec<&Value>> = Default::default();
    for r in rows
        .iter()
        .filter(|r| r["phase"] == "measured" && r["state"] == "complete")
    {
        let m = &r["metadata"];
        let k=json!({"study":r["study"],"model":m["model"],"variant":m["variant"],"context":m["context"],"memory_limit_mb":m["memory_limit_mb"],"budget":m["budget"],"category":m["case"]["category"],"expected_calls":m["case"]["scope"]["expected_calls"],"available_tools":m["case"]["tools"]["allowed"].as_array().map(Vec::len)}).to_string();
        groups.entry(k).or_default().push(r);
    }
    let mut out = Vec::new();
    for (key, rs) in groups {
        let mut success = 0;
        let mut unscored = 0;
        let mut failures = std::collections::BTreeMap::<String, usize>::new();
        let mut times = Vec::new();
        let mut calls = 0;
        let mut tool_calls = 0;
        let mut peak_bytes = None;
        let mut retrieved = 0;
        let mut retrieval_scored = 0;
        let mut per_task = std::collections::BTreeMap::<String, Vec<f64>>::new();
        for r in &rs {
            let v = &r["result"];
            let pending = v["status"] == "awaiting_judge"
                && (r["judgment"]["verdict"] == "unscored"
                    || r["judgment"].is_null()
                    || r["judgment"]["provisional"] == true);
            let pass = v["status"] == "success"
                || (v["status"] == "awaiting_judge" && r["judgment"]["verdict"] == "pass");
            if pending {
                unscored += 1;
            } else {
                success += usize::from(pass);
                per_task
                    .entry(r["metadata"]["case"]["id"].as_str().unwrap_or("").into())
                    .or_default()
                    .push(if pass { 1. } else { 0. });
            }
            *failures
                .entry(if r["state"] != "complete" {
                    r["state"].as_str().unwrap_or("unknown").into()
                } else {
                    v["status"].as_str().unwrap_or("unknown").into()
                })
                .or_default() += 1;
            if let Some(t) = v["wall_ms"].as_f64() {
                times.push(t);
            }
            calls += v["model_calls"].as_u64().unwrap_or(0);
            tool_calls += v["tool_invocations"].as_u64().unwrap_or(0);
            if let Some(bytes) = v["resources"]["combined_peak_bytes"]
                .as_u64()
                .filter(|n| *n > 0)
            {
                peak_bytes = Some(peak_bytes.map_or(bytes, |previous: u64| previous.max(bytes)));
            }
            if let Some(found) = v["retrieval_success"].as_bool() {
                retrieval_scored += 1;
                retrieved += usize::from(found);
            }
        }
        times.sort_by(f64::total_cmp);
        let rates = per_task
            .values()
            .map(|a| a.iter().sum::<f64>() / a.len() as f64)
            .collect::<Vec<_>>();
        let interval = cluster_interval(&rates);
        out.push(json!({
            "group": serde_json::from_str::<Value>(&key).expect("group key was serialized as JSON"),
            "attempts": rs.len(),
            "scored": rs.len() - unscored,
            "successes": success,
            "unscored": unscored,
            "success_rate": (rs.len() > unscored).then(|| success as f64 / (rs.len() - unscored) as f64),
            "task_bootstrap_95": interval,
            "unique_tasks": rates.len(),
            "retrieval_successes": retrieved,
            "retrieval_scored": retrieval_scored,
            "model_calls": calls,
            "tool_invocations": tool_calls,
            "combined_peak_bytes": peak_bytes,
            "median_wall_ms": times.get(times.len() / 2),
            "p95_wall_ms": times.get((times.len() as f64 * 0.95).ceil().max(1.) as usize - 1),
            "statuses": failures
        }));
    }
    json!({"groups":out,"interval_method":"Percentile bootstrap over task-level mean success, 2000 seeded resamples; unavailable below 2 unique tasks. Repeats are clustered by task."})
}
fn cluster_interval(rates: &[f64]) -> Option<[f64; 2]> {
    if rates.len() < 2 {
        return None;
    }
    let mut seed = 42u64;
    let mut means = Vec::with_capacity(2000);
    for _ in 0..2000 {
        let mut sum = 0.;
        for _ in rates {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            sum += rates[(seed >> 32) as usize % rates.len()];
        }
        means.push(sum / rates.len() as f64);
    }
    means.sort_by(f64::total_cmp);
    Some([means[50], means[1949]])
}
pub fn export_csv(rows: &[Value], out: &Path) -> Result<()> {
    let mut w = std::io::BufWriter::new(std::fs::File::create(out)?);
    w.write_all(b"id,study,state,phase,metadata,result,judgment,trace\n")?;
    for r in rows {
        let values = [
            "id", "study", "state", "phase", "metadata", "result", "judgment", "trace",
        ]
        .map(|k| format!("\"{}\"", r[k].to_string().replace('"', "\"\"")));
        writeln!(w, "{}", values.join(","))?;
    }
    Ok(())
}
