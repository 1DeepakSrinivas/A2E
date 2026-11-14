// upload/mod.rs
// Sends results to the A2E-Results Worker; the Worker judges, scores and reports
// Requests are spooled to an outbox first, so an offline device uploads later

mod client;
mod outbox;
pub mod payload;
mod resume;

pub use client::{classify, Client, Reply, Request};
pub use outbox::{Flush, Outbox};
pub use resume::Resume;

use anyhow::Result;
use std::path::Path;

/// Where this process sends results, for the notice printed before a run.
pub fn destination() -> String {
    if crate::dotenv::local_only() {
        return "kept on this device (local mode)".into();
    }
    match std::env::var("WORKER_URL").ok().filter(|v| !v.is_empty()) {
        Some(url) => format!("uploading to {url}"),
        None => "kept on this device (no Worker configured)".into(),
    }
}
pub struct Uploader {
    outbox: Outbox,
    client: Option<Client>,
    offline: bool,
}
impl Uploader {
    /// Uploads with configured credentials unless local mode is enabled.
    pub fn open(root: &Path) -> Result<Self> {
        Self::with_client(root, Client::from_env()?)
    }
    pub fn with_client(root: &Path, client: Option<Client>) -> Result<Self> {
        Ok(Self {
            outbox: Outbox::open(root)?,
            client,
            offline: false,
        })
    }
    /// Spools the requests, then posts what is pending unless an earlier post failed.
    pub fn submit(&mut self, requests: Vec<Request>) -> Result<()> {
        for r in &requests {
            self.outbox.push(r)?;
        }
        if !self.offline {
            let flush = self.flush()?;
            self.offline = flush.stopped.is_some();
        }
        Ok(())
    }
    /// Posts every pending request now.
    pub fn flush(&mut self) -> Result<Flush> {
        let Some(client) = &self.client else {
            return self.status();
        };
        let flush = self.outbox.flush(client)?;
        if let Some(why) = &flush.stopped {
            eprintln!(
                "warning: upload paused with {} pending: {why}",
                flush.pending
            );
        }
        self.offline = flush.stopped.is_some();
        Ok(flush)
    }
    pub fn status(&self) -> Result<Flush> {
        Ok(Flush {
            pending: self.outbox.pending()?.len(),
            rejected: self.outbox.rejected()?,
            stopped: self.client.is_none().then(|| {
                if crate::dotenv::local_only() {
                    "uploads are disabled in local mode".to_owned()
                } else {
                    "WORKER_URL and WRITE_TOKEN are not set".to_owned()
                }
            }),
            sent: 0,
        })
    }
}
