// store/mod.rs
// SQLite storage for studies, attempts, samples, and cached judgments
// Interrupted-run recovery, JSON row queries, and the run lock

mod record;
mod schema;
mod summary;

pub use summary::{cluster_interval, export_csv, summary};

use crate::{case::Case, config::Variant, measure::Resources};
use anyhow::{bail, Result};
use rusqlite::{params, types::ValueRef, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::{
    io::Write,
    path::{Path, PathBuf},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    Warmup,
    Measured,
}
/// Identity of one attempt.
pub struct RunKey<'a> {
    pub study: &'a str,
    pub job: &'a str,
    pub phase: Phase,
    pub model: &'a str,
    pub context: usize,
    pub variant: &'a Variant,
    pub repeat: usize,
    pub case: &'a Case,
    pub load_id: Option<i64>,
    pub trace: &'a Path,
}
/// One model load: timing, memory, and the server's template.
pub struct LoadRecord<'a> {
    pub study: &'a str,
    pub model: &'a str,
    pub context: usize,
    pub load_ms: f64,
    pub resources: Resources,
    pub template: Option<&'a str>,
    pub server_version: Option<&'a str>,
    pub props: Option<&'a Value>,
    pub error: Option<&'a str>,
}
/// Serde tag of a unit-variant enum, as stored in text columns.
pub fn tag(v: &impl Serialize) -> String {
    serde_json::to_value(v)
        .ok()
        .and_then(|v| v.as_str().map(String::from))
        .unwrap_or_default()
}
fn json_text(s: Option<String>) -> Value {
    s.and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or(Value::Null)
}

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
        if tables > 0 && app != schema::APPLICATION_ID {
            bail!("not an A2E native database");
        }
        if tables > 0 && version != schema::VERSION {
            bail!(
                "database schema v{version} is not supported (need v{}); use a new output directory",
                schema::VERSION
            );
        }
        db.execute_batch(
            "PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA foreign_keys=ON;",
        )?;
        if tables == 0 {
            db.execute_batch(&format!(
                "BEGIN; {}{} PRAGMA application_id={}; PRAGMA user_version={}; COMMIT;",
                schema::TABLES,
                schema::VIEWS,
                schema::APPLICATION_ID,
                schema::VERSION
            ))?;
        }
        Ok(Self { db })
    }
    /// Stores the study definition and a copy of every case once.
    pub fn study(&self, id: &str, definition: &Value, cases: &[Case]) -> Result<()> {
        let tx = self.db.unchecked_transaction()?;
        tx.execute(
            "INSERT OR IGNORE INTO studies(id,definition) VALUES (?,?)",
            params![id, definition.to_string()],
        )?;
        {
            let mut insert = tx.prepare("INSERT OR IGNORE INTO cases(study,id,hash,category,split,tags,definition) VALUES (?,?,?,?,?,?,?)")?;
            for c in cases {
                insert.execute(params![
                    id,
                    c.id,
                    crate::canonical_hash(c)?,
                    tag(&c.category),
                    tag(&c.split),
                    json!(c.tags).to_string(),
                    serde_json::to_string(c)?
                ])?;
            }
        }
        tx.commit()?;
        Ok(())
    }
    pub fn model_load(&self, l: &LoadRecord) -> Result<i64> {
        let tx = self.db.unchecked_transaction()?;
        tx.execute(
            "INSERT INTO model_loads(study,model,context,load_ms,peak_bytes,template_sha256,template,server_version,props,error) VALUES (?,?,?,?,?,?,?,?,?,?)",
            params![
                l.study,
                l.model,
                l.context,
                l.load_ms,
                l.resources.inference_peak_bytes,
                l.template.map(|t| crate::digest(t.as_bytes())),
                l.template,
                l.server_version,
                l.props.map(Value::to_string),
                l.error
            ],
        )?;
        let id = tx.last_insert_rowid();
        record::samples(&tx, record::SampleOwner::Load(id), &l.resources)?;
        tx.commit()?;
        Ok(id)
    }
    /// Retry incomplete attempts without reusing task state.
    pub fn recover(&self) -> Result<usize> {
        Ok(self.db.execute(
            "UPDATE runs SET state='interrupted' WHERE state='running'",
            [],
        )?)
    }
    pub fn should_run(&self, study: &str, job: &str, retry_failed: bool) -> Result<bool> {
        let old: Option<(String, Option<String>)> = self
            .db
            .query_row(
                "SELECT state,status FROM runs WHERE study=? AND job=? ORDER BY attempt DESC LIMIT 1",
                params![study, job],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()?;
        Ok(match old {
            None => true,
            Some((state, status)) => {
                state != "complete"
                    || (retry_failed
                        && !matches!(status.as_deref(), Some("success" | "awaiting_judge")))
            }
        })
    }
    pub fn begin(&self, k: &RunKey) -> Result<i64> {
        self.db.execute(
            "INSERT INTO runs(study,job,attempt,state,phase,model,context,variant,variant_config,mode,repeat_index,case_id,category,split,load_id,trace)
             SELECT ?,?,coalesce(max(attempt),0)+1,'running',?,?,?,?,?,?,?,?,?,?,?,? FROM runs WHERE study=? AND job=?",
            params![
                k.study,
                k.job,
                tag(&k.phase),
                k.model,
                k.context,
                k.variant.name,
                serde_json::to_string(k.variant)?,
                tag(&k.variant.mode),
                k.repeat,
                k.case.id,
                tag(&k.case.category),
                tag(&k.case.split),
                k.load_id,
                k.trace.to_string_lossy(),
                k.study,
                k.job
            ],
        )?;
        Ok(self.db.last_insert_rowid())
    }
    /// Latest attempts as JSON rows without the heavy text columns.
    pub fn rows(&self) -> Result<Vec<Value>> {
        self.query(
            "SELECT id,study,job,attempt,state,phase,model,context,variant,mode,repeat_index,case_id,category,split,status,reason,failure_class,retrieval_success,
                model_calls,tool_invocations,valid_tool_calls,argument_errors,repairs,prompt_tokens,generated_tokens,cached_tokens,
                prefill_ms,decode_ms,http_ms,template_ms,tokenize_ms,parse_ms,validate_ms,tool_ms,tool_setup_ms,wall_ms,harness_other_ms,
                harness_peak_bytes,inference_peak_bytes,combined_peak_bytes,available_min_bytes,cpu_time_ms,minor_faults,major_faults,
                peak_temperature_c,throttled,energy_j,power_scope,sample_count,memory_exceeded,verdict,provisional,pending,passed,trace
             FROM v_run_summary ORDER BY id",
            [],
        )
    }
    /// Any SELECT as JSON objects keyed by column name.
    pub fn query(&self, sql: &str, args: impl rusqlite::Params) -> Result<Vec<Value>> {
        let mut statement = self.db.prepare(sql)?;
        let names = statement
            .column_names()
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>();
        let rows = statement.query_map(args, |r| {
            let mut object = Map::new();
            for (i, name) in names.iter().enumerate() {
                object.insert(
                    name.clone(),
                    match r.get_ref(i)? {
                        ValueRef::Null | ValueRef::Blob(_) => Value::Null,
                        ValueRef::Integer(n) => json!(n),
                        ValueRef::Real(x) => json!(x),
                        ValueRef::Text(t) => json!(String::from_utf8_lossy(t)),
                    },
                );
            }
            Ok(Value::Object(object))
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }
    /// Completed attempts waiting for a judge, with the case and the handoff material.
    pub fn judge_inputs(&self) -> Result<Vec<Value>> {
        let mut rows = self.query(
            "SELECT r.id,r.study,r.answer,r.evidence,r.final_state,c.definition AS \"case\"
             FROM v_run_summary r JOIN cases c ON c.study=r.study AND c.id=r.case_id
             WHERE r.phase='measured' AND r.state='complete' AND r.status='awaiting_judge' ORDER BY r.id",
            [],
        )?;
        for r in &mut rows {
            for key in ["evidence", "final_state", "case"] {
                r[key] = json_text(r[key].as_str().map(String::from));
            }
            let id = r["id"].as_i64().unwrap_or_default();
            let mut calls = self.query(
                "SELECT tool,arguments,ok FROM tool_calls WHERE run_id=? ORDER BY idx",
                [id],
            )?;
            for c in &mut calls {
                c["arguments"] = json_text(c["arguments"].as_str().map(String::from));
                c["ok"] = json!(c["ok"] == 1);
            }
            r["calls"] = json!(calls);
        }
        Ok(rows)
    }
    /// Requests to the judge model; `run_id` is empty for calibration.
    pub fn judge_calls(
        &self,
        run_id: Option<i64>,
        purpose: &str,
        rubric: &str,
        records: &[crate::judge::CallRecord],
    ) -> Result<()> {
        let tx = self.db.unchecked_transaction()?;
        {
            let mut insert = tx.prepare_cached("INSERT INTO judge_calls(run_id,purpose,rubric,attempt,model,provider,response_id,prompt_tokens,completion_tokens,cost,wall_ms,ok,error) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)")?;
            for r in records {
                insert.execute(params![
                    run_id,
                    purpose,
                    rubric,
                    r.attempt,
                    r.model,
                    r.provider,
                    r.response_id,
                    r.prompt_tokens,
                    r.completion_tokens,
                    r.cost,
                    r.wall_ms,
                    r.ok,
                    r.error
                ])?;
            }
        }
        tx.commit()?;
        Ok(())
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
