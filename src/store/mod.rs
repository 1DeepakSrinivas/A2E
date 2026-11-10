// store/mod.rs
// Attempt phase, serde tags for text fields, and the single-run lock
// Results go to the A2E-Results Worker; see the upload module

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
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
/// Serde tag of a unit-variant enum, as sent in text fields.
pub fn tag(v: &impl Serialize) -> String {
    serde_json::to_value(v)
        .ok()
        .and_then(|v| v.as_str().map(String::from))
        .unwrap_or_default()
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
                        bail!("another run holds the runtime lock");
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
