//! Spool folder of write requests that are posted in order.
//!
//! A request leaves the folder once stored, or moves to rejected/ with its error.

use super::client::{Client, Reply, Request};
use anyhow::Result;
use serde::Serialize;
use std::{
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

static SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Default, Serialize, PartialEq, Eq)]
pub struct Flush {
    pub sent: usize,
    pub rejected: usize,
    pub pending: usize,
    pub stopped: Option<String>,
}
pub struct Outbox {
    dir: PathBuf,
}
impl Outbox {
    pub fn open(root: &Path) -> Result<Self> {
        let dir = root.join("outbox");
        std::fs::create_dir_all(dir.join("rejected"))?;
        Ok(Self { dir })
    }
    /// Writes the request durably under a name that sorts in posting order.
    pub fn push(&self, r: &Request) -> Result<()> {
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let n = SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let name = format!("{nanos:025}-{n:06}.json");
        let tmp = self.dir.join(format!("{name}.tmp"));
        std::fs::write(&tmp, serde_json::to_vec(r)?)?;
        std::fs::rename(tmp, self.dir.join(name))?;
        Ok(())
    }
    pub fn pending(&self) -> Result<Vec<PathBuf>> {
        let mut files = Self::json_files(&self.dir)?;
        files.sort();
        Ok(files)
    }
    pub fn rejected(&self) -> Result<usize> {
        Ok(Self::json_files(&self.dir.join("rejected"))?.len())
    }
    fn json_files(dir: &Path) -> Result<Vec<PathBuf>> {
        Ok(std::fs::read_dir(dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "json"))
            .collect())
    }
    /// Posts pending requests oldest first and stops at the first retryable failure.
    pub fn flush(&self, client: &Client) -> Result<Flush> {
        let mut out = Flush::default();
        let files = self.pending()?;
        for (i, path) in files.iter().enumerate() {
            let request: Request = serde_json::from_slice(&std::fs::read(path)?)?;
            match client.send(&request) {
                Reply::Stored(_) => {
                    std::fs::remove_file(path)?;
                    out.sent += 1;
                }
                Reply::Rejected(why) => {
                    let name = path.file_name().unwrap_or_default();
                    let target = self.dir.join("rejected").join(name);
                    std::fs::write(target.with_extension("err"), &why)?;
                    std::fs::rename(path, target)?;
                    eprintln!(
                        "warning: rejected {} {}: {why}",
                        request.method, request.path
                    );
                    out.rejected += 1;
                }
                Reply::Retry(why) => {
                    out.pending = files.len() - i;
                    out.stopped = Some(why);
                    return Ok(out);
                }
            }
        }
        Ok(out)
    }
}
