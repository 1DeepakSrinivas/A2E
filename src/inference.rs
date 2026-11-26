//! Pinned llama.cpp server lifecycle and loopback HTTP client.
//!
//! Prompt rendering, token counts, cache resets, and completion timings.

use crate::{
    config::{Config, Model},
    read_bounded,
};
use anyhow::{bail, Context, Result};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    os::unix::process::CommandExt,
    path::Path,
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
};

#[derive(Clone)]
pub struct Inference {
    http: Client,
    url: String,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Timing {
    pub prefill_ms: Option<f64>,
    pub decode_ms: Option<f64>,
    pub prompt_tokens: Option<u64>,
    pub cached_tokens: Option<u64>,
    pub generated_tokens: Option<u64>,
    pub http_ms: f64,
}
pub struct Completion {
    pub text: String,
    pub timing: Timing,
    pub raw: Value,
    pub truncated: bool,
}
impl Inference {
    pub fn new(url: String) -> Result<Self> {
        Ok(Self {
            http: Client::builder()
                .no_proxy()
                .timeout(Duration::from_secs(
                    crate::settings::active().advanced.inference_timeout_s,
                ))
                .build()?,
            url,
        })
    }
    pub fn post(&self, path: &str, body: &Value, timeout: Duration) -> Result<Value> {
        let r = self
            .http
            .post(format!("{}{path}", self.url))
            .timeout(timeout.max(Duration::from_millis(1)))
            .json(body)
            .send()?
            .error_for_status()?;
        Ok(serde_json::from_slice(&read_bounded(r, 2 * 1024 * 1024)?)?)
    }
    pub fn get(&self, path: &str) -> Result<Value> {
        let r = self
            .http
            .get(format!("{}{path}", self.url))
            .timeout(Duration::from_secs(2))
            .send()?
            .error_for_status()?;
        Ok(serde_json::from_slice(&read_bounded(r, 2 * 1024 * 1024)?)?)
    }
    pub fn template(&self, body: &Value, remaining: Duration) -> Result<String> {
        Ok(self.post("/apply-template", body, remaining)?["prompt"]
            .as_str()
            .context("missing rendered prompt")?
            .into())
    }
    pub fn tokens(&self, text: &str, remaining: Duration) -> Result<usize> {
        Ok(self.post(
            "/tokenize",
            &json!({"content":text,"add_special":false,"parse_special":true}),
            remaining,
        )?["tokens"]
            .as_array()
            .context("missing token list")?
            .len())
    }
    pub fn reset(&self) -> Result<()> {
        self.post(
            "/slots/0?action=erase",
            &json!({}),
            Duration::from_secs(crate::settings::active().advanced.slot_erase_timeout_s),
        )
        .context("cache_reset_failed")?;
        Ok(())
    }
    /// Waits until no slot is processing. The server defers a cache erase while a slot is
    /// busy, and a call that timed out keeps running until its current batch ends.
    pub fn settle(&self, limit: Duration) -> bool {
        let deadline = Instant::now() + limit;
        loop {
            let busy = self.get("/slots").ok().map(|slots| {
                slots
                    .as_array()
                    .is_some_and(|a| a.iter().any(|s| s["is_processing"] == true))
            });
            if busy == Some(false) {
                return true;
            }
            if Instant::now() >= deadline {
                return false;
            }
            std::thread::sleep(Duration::from_millis(500));
        }
    }
    /// Raw completion and server-reported timings.
    pub fn complete(&self, body: &Value, remaining: Duration) -> Result<Completion> {
        let start = Instant::now();
        let raw = self.post("/completion", body, remaining)?;
        let t = &raw["timings"];
        let (prompt_tokens, cached_tokens) = prompt_counts(&raw);
        let timing = Timing {
            prefill_ms: t["prompt_ms"].as_f64(),
            decode_ms: t["predicted_ms"].as_f64(),
            prompt_tokens,
            cached_tokens,
            generated_tokens: t["predicted_n"].as_u64(),
            http_ms: start.elapsed().as_secs_f64() * 1000.,
        };
        let text = raw["content"]
            .as_str()
            .context("missing completion text")?
            .to_owned();
        let truncated = raw["truncated"].as_bool().unwrap_or(false)
            || raw["stop_type"] == "limit"
            || raw["stopped_limit"] == true;
        Ok(Completion {
            text,
            timing,
            raw,
            truncated,
        })
    }
}
/// Prompt length and the part of it reused from the slot cache.
/// `tokens_cached` is the slot's size after generation, not a reuse count, so it is not read.
fn prompt_counts(raw: &Value) -> (Option<u64>, Option<u64>) {
    let t = &raw["timings"];
    let fresh = t["prompt_n"].as_u64();
    let reused = t["cache_n"].as_u64();
    let total = raw["tokens_evaluated"]
        .as_u64()
        .or_else(|| fresh.zip(reused).map(|(f, r)| f + r));
    let cached = reused.or_else(|| total.zip(fresh).map(|(t, f)| t.saturating_sub(f)));
    (total, cached)
}
pub struct Server {
    child: Child,
    pub client: Inference,
    pub version: String,
    pub started: Instant,
}
impl Server {
    /// Pinned inference process on loopback.
    pub fn spawn(c: &Config, m: &Model, ctx: usize, dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(dir.join("slots"))?;
        let llama = &c.settings_for(m).llama;
        let binary = c.server_path(m);
        let version = Command::new(&binary)
            .arg("--version")
            .output()
            .context("run llama-server --version")?;
        let version = format!(
            "{}{}",
            String::from_utf8_lossy(&version.stdout),
            String::from_utf8_lossy(&version.stderr)
        );
        if !version.contains(&llama.commit[..8]) {
            bail!("llama_commit_mismatch: expected {}", llama.commit);
        }
        let client = Inference::new(format!("http://{}:{}", llama.host, llama.port))?;
        if client.get("/health").is_ok() {
            bail!("inference port is already occupied");
        }
        let started = Instant::now();
        let child = Command::new(binary)
            .args(llama.args(&c.model_path(m), ctx, &dir.join("slots")))
            .stdin(Stdio::null())
            .stdout(Stdio::from(std::fs::File::create(
                dir.join("server.stdout"),
            )?))
            .stderr(Stdio::from(std::fs::File::create(
                dir.join("server.stderr"),
            )?))
            .process_group(0)
            .spawn()?;
        Ok(Self {
            child,
            client,
            version,
            started,
        })
    }
    pub fn pid(&self) -> u32 {
        self.child.id()
    }
    pub fn ready(&mut self, seconds: u64, stop: &std::sync::atomic::AtomicBool) -> Result<Value> {
        let deadline = Instant::now() + Duration::from_secs(seconds);
        loop {
            if stop.load(std::sync::atomic::Ordering::Relaxed) {
                bail!("interrupted");
            }
            if let Some(status) = self.child.try_wait()? {
                bail!("model_load_failed: {status}");
            }
            if self.client.get("/health").is_ok() {
                let p = self.client.get("/props")?;
                if p["chat_template"].as_str().is_none_or(str::is_empty) {
                    bail!("missing model chat template");
                }
                return Ok(p);
            }
            if Instant::now() >= deadline {
                bail!("model_load_timeout");
            }
            std::thread::sleep(Duration::from_millis(100));
        }
    }
}
impl Drop for Server {
    fn drop(&mut self) {
        crate::process::kill_group(&mut self.child);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_counts_ignore_the_slot_size_field() {
        // First call after a cache erase: 314 prompt tokens, none reused, 16 generated
        let cold = json!({"tokens_evaluated":314,"tokens_cached":329,"timings":{"prompt_n":314,"cache_n":0,"predicted_n":16}});
        assert_eq!(prompt_counts(&cold), (Some(314), Some(0)));
        let warm = json!({"tokens_evaluated":376,"tokens_cached":423,"timings":{"prompt_n":67,"cache_n":309}});
        assert_eq!(prompt_counts(&warm), (Some(376), Some(309)));
        // Older servers report no reuse count
        let old = json!({"tokens_evaluated":376,"tokens_cached":423,"timings":{"prompt_n":67}});
        assert_eq!(prompt_counts(&old), (Some(376), Some(309)));
        assert_eq!(prompt_counts(&json!({"timings":{}})), (None, None));
    }
}
