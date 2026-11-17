//! Blocking HTTP client for the A2E-Results Worker write routes.
//!
//! Replies are classified as stored, retryable, or rejected.

use crate::read_bounded;
use anyhow::{Context, Result};
use reqwest::blocking::Client as Http;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::Duration;

const REPLY_BYTES: usize = 1 << 20;

/// One write call; stored as a file in the outbox.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Request {
    pub method: String,
    pub path: String,
    pub body: Value,
}
pub enum Reply {
    /// Rows the Worker stored; zero when the request was already stored.
    Stored(u64),
    /// Network failure, 5xx, 429 or an authorization error; the request stays pending.
    Retry(String),
    /// The Worker refused the body for good (400, 404, 405, 409, 413, 422).
    Rejected(String),
}
pub struct Client {
    base: String,
    token: String,
    http: Http,
}
impl Client {
    pub fn new(base: &str, token: &str) -> Result<Self> {
        Ok(Self {
            base: base.trim_end_matches('/').to_owned(),
            token: token.to_owned(),
            http: Http::builder()
                .timeout(Duration::from_secs(
                    crate::settings::active().advanced.upload_timeout_s,
                ))
                .build()
                .context("cannot build the HTTP client")?,
        })
    }
    /// Local runs and runs without credentials have no upload client.
    pub fn from_env() -> Result<Option<Self>> {
        if crate::dotenv::local_only() {
            return Ok(None);
        }
        let get = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
        match (get("WORKER_URL"), get("WRITE_TOKEN")) {
            (None, None) => Ok(None),
            (Some(url), Some(token)) => Ok(Some(Self::new(&url, &token)?)),
            _ => anyhow::bail!("set both WORKER_URL and WRITE_TOKEN, or neither"),
        }
    }
    pub fn send(&self, r: &Request) -> Reply {
        let method = match r.method.as_str() {
            "PUT" => reqwest::Method::PUT,
            _ => reqwest::Method::POST,
        };
        let response = self
            .http
            .request(method, format!("{}{}", self.base, r.path))
            .bearer_auth(&self.token)
            .json(&r.body)
            .send();
        let response = match response {
            Ok(v) => v,
            Err(e) => return Reply::Retry(e.to_string()),
        };
        let status = response.status();
        let text = read_bounded(response, REPLY_BYTES)
            .map(|b| String::from_utf8_lossy(&b).into_owned())
            .unwrap_or_default();
        classify(status.as_u16(), &text)
    }
}
pub fn classify(status: u16, body: &str) -> Reply {
    let value: Value = serde_json::from_str(body).unwrap_or(Value::Null);
    if (200..300).contains(&status) {
        return Reply::Stored(value["inserted"].as_u64().unwrap_or(0));
    }
    let message = value["error"]["message"]
        .as_str()
        .map(|m| format!("{status} {m}"))
        .unwrap_or_else(|| format!("{status}"));
    if matches!(status, 400 | 404 | 405 | 409 | 413 | 422) {
        Reply::Rejected(message)
    } else {
        Reply::Retry(message)
    }
}
