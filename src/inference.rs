// inference.rs
// Pinned llama.cpp server lifecycle and loopback HTTP client
// Prompt rendering, token counts, cache resets, and completion timings

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
                .timeout(Duration::from_secs(300))
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
        self.post("/slots/0?action=erase", &json!({}), Duration::from_secs(10))
            .context("cache_reset_failed")?;
        Ok(())
    }
    /// Raw completion and server-reported timings.
    pub fn complete(&self, body: &Value, remaining: Duration) -> Result<Completion> {
        let start = Instant::now();
        let raw = self.post("/completion", body, remaining)?;
        let t = &raw["timings"];
        let fresh = t["prompt_n"].as_u64();
        let cached = raw["tokens_cached"]
            .as_u64()
            .or_else(|| t["cache_n"].as_u64());
        let timing = Timing {
            prefill_ms: t["prompt_ms"].as_f64(),
            decode_ms: t["predicted_ms"].as_f64(),
            prompt_tokens: fresh.zip(cached).map(|(f, c)| f + c),
            cached_tokens: cached,
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
        let binary = crate::config::expand_path(&c.runtime.llama_server);
        let version = Command::new(&binary)
            .arg("--version")
            .output()
            .context("run llama-server --version")?;
        let version = format!(
            "{}{}",
            String::from_utf8_lossy(&version.stdout),
            String::from_utf8_lossy(&version.stderr)
        );
        if !version.contains(&c.runtime.llama_commit[..7]) {
            bail!("llama_commit_mismatch: expected {}", c.runtime.llama_commit);
        }
        let client = Inference::new(format!("http://127.0.0.1:{}", c.runtime.port))?;
        if client.get("/health").is_ok() {
            bail!("inference port is already occupied");
        }
        let started = Instant::now();
        let child = Command::new(binary)
            .args([
                "--host",
                "127.0.0.1",
                "--port",
                &c.runtime.port.to_string(),
                "--model",
                &c.model_path(m).to_string_lossy(),
                "--ctx-size",
                &ctx.to_string(),
                "--threads",
                &c.runtime.threads.to_string(),
                "--n-gpu-layers",
                &c.runtime.gpu_layers.to_string(),
                "--parallel",
                "1",
                "--batch-size",
                "128",
                "--ubatch-size",
                "64",
                "--jinja",
                "--special",
                "--no-context-shift",
                "--slots",
                "--slot-save-path",
                &dir.join("slots").to_string_lossy(),
            ])
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
