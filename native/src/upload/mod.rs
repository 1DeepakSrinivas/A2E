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

pub struct Uploader {
    outbox: Outbox,
    client: Option<Client>,
    offline: bool,
}
impl Uploader {
    /// Uploads only when `WORKER_URL` and `WRITE_TOKEN` are set; otherwise requests wait in the outbox.
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
            stopped: self
                .client
                .is_none()
                .then(|| "WORKER_URL and WRITE_TOKEN are not set".to_owned()),
            sent: 0,
        })
    }
}
