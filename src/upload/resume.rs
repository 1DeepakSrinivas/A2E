//! Per-device resume state: finished attempts per job and loads per study.
//!
//! Append-only JSON lines; the Worker has no read route for the runner.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    io::Write,
    path::{Path, PathBuf},
};

#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Line {
    Job {
        study: String,
        job: String,
        attempt: u32,
        state: String,
        status: Option<String>,
    },
    Load {
        study: String,
    },
}
pub struct Resume {
    path: PathBuf,
    jobs: HashMap<(String, String), (u32, String, Option<String>)>,
    loads: HashMap<String, u32>,
}
impl Resume {
    /// Reads the file if present; an unreadable trailing line is ignored.
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(p) = path.parent() {
            std::fs::create_dir_all(p)?;
        }
        let mut s = Self {
            path: path.to_owned(),
            jobs: HashMap::new(),
            loads: HashMap::new(),
        };
        let text = std::fs::read_to_string(path).unwrap_or_default();
        for (n, line) in text.lines().enumerate() {
            match serde_json::from_str(line) {
                Ok(Line::Job {
                    study,
                    job,
                    attempt,
                    state,
                    status,
                }) => {
                    s.jobs.insert((study, job), (attempt, state, status));
                }
                Ok(Line::Load { study }) => *s.loads.entry(study).or_default() += 1,
                Err(e) => eprintln!(
                    "warning: {} line {} was skipped and its attempt may run again: {e}",
                    path.display(),
                    n + 1
                ),
            }
        }
        Ok(s)
    }
    fn append(&self, line: &Line) -> Result<()> {
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        f.write_all(format!("{}\n", serde_json::to_string(line)?).as_bytes())?;
        Ok(())
    }
    /// Incomplete jobs run again; with `retry_failed`, so do jobs that neither succeeded nor await a judge.
    pub fn should_run(&self, study: &str, job: &str, retry_failed: bool) -> bool {
        match self.jobs.get(&(study.to_owned(), job.to_owned())) {
            None => true,
            Some((_, state, status)) => {
                state != "complete"
                    || (retry_failed
                        && !matches!(status.as_deref(), Some("success" | "awaiting_judge")))
            }
        }
    }
    pub fn next_attempt(&self, study: &str, job: &str) -> u32 {
        self.jobs
            .get(&(study.to_owned(), job.to_owned()))
            .map_or(1, |j| j.0 + 1)
    }
    pub fn record_job(
        &mut self,
        study: &str,
        job: &str,
        attempt: u32,
        state: &str,
        status: Option<&str>,
    ) -> Result<()> {
        self.append(&Line::Job {
            study: study.into(),
            job: job.into(),
            attempt,
            state: state.into(),
            status: status.map(Into::into),
        })?;
        self.jobs.insert(
            (study.into(), job.into()),
            (attempt, state.into(), status.map(Into::into)),
        );
        Ok(())
    }
    /// Id for the next model load of a study, recorded so a resumed study never reuses it.
    pub fn next_load_id(&mut self, study: &str) -> Result<String> {
        self.append(&Line::Load {
            study: study.into(),
        })?;
        let n = self.loads.entry(study.into()).or_default();
        *n += 1;
        Ok(format!("{}-load-{n}", &study[..12.min(study.len())]))
    }
}
