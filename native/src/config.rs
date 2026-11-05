// config.rs
// Runtime settings, model registry, and experiment definitions
// Local overrides, pinned model downloads, and task and tool schema checks

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Model {
    pub id: String,
    pub adapter: String,
    pub quant: String,
    pub source: String,
    pub revision: String,
    pub file: String,
    pub sha256: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Runtime {
    pub llama_server: String,
    pub llama_commit: String,
    pub models_dir: String,
    pub output_dir: String,
    pub device: String,
    pub port: u16,
    pub threads: usize,
    pub gpu_layers: i32,
    pub memory_limit_mb: u64,
    pub reserve_mb: u64,
    pub sample_ms: u64,
    pub startup_seconds: u64,
    pub power: String,
}
impl Default for Runtime {
    fn default() -> Self {
        Self {
            llama_server: "deps/llama.cpp/build-a2e/bin/llama-server".into(),
            llama_commit: "6666a7330c5375dc310ddea1f9621a7770430010".into(),
            models_dir: "~/models".into(),
            output_dir: "results/native".into(),
            device: "auto".into(),
            port: 8197,
            threads: 4,
            gpu_layers: 0,
            memory_limit_mb: 768,
            reserve_mb: 128,
            sample_ms: 500,
            startup_seconds: 300,
            power: "auto".into(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct JudgeConfig {
    pub model: String,
    pub provider: String,
    pub max_tokens: u32,
    pub timeout_seconds: u64,
    pub max_requests: usize,
}
impl Default for JudgeConfig {
    fn default() -> Self {
        Self {
            model: String::new(),
            provider: String::new(),
            max_tokens: 2048,
            timeout_seconds: 120,
            max_requests: 100,
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    #[serde(default)]
    pub runtime: Runtime,
    #[serde(default)]
    pub judge: JudgeConfig,
    pub models: Vec<Model>,
}

pub fn expand_path(s: &str) -> PathBuf {
    s.strip_prefix("~/")
        .map(|p| PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(p))
        .unwrap_or_else(|| PathBuf::from(s))
}
fn merge(a: &mut toml::Value, b: toml::Value) {
    if let (Some(a), Some(b)) = (a.as_table_mut(), b.as_table()) {
        for (k, v) in b {
            if let Some(old) = a.get_mut(k) {
                merge(old, v.clone());
            } else {
                a.insert(k.clone(), v.clone());
            }
        }
    } else {
        *a = b;
    }
}
impl Config {
    /// Download pinned artifacts and verify their published hashes.
    pub fn fetch_models(&self) -> Result<Vec<Value>> {
        use std::io::{Read, Write};
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(1800))
            .build()?;
        let mut output = Vec::new();
        for model in &self.models {
            let expected = model
                .sha256
                .as_ref()
                .context("download requires a pinned SHA-256")?;
            let path = self.model_path(model);
            if path.exists() && crate::hash_file(&path)? == *expected {
                output.push(serde_json::json!({"model": model.id, "path": path, "cached": true}));
                continue;
            }
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let url = format!(
                "https://huggingface.co/{}/resolve/{}/{}",
                model.source, model.revision, model.file
            );
            let mut response = client.get(url).send()?.error_for_status()?;
            let part = path.with_extension("gguf.part");
            let mut file = std::fs::File::create(&part)?;
            let mut buffer = [0u8; 65536];
            let mut total = 0u64;
            loop {
                let n = response.read(&mut buffer)?;
                if n == 0 {
                    break;
                }
                total += n as u64;
                if total > 2 * 1024 * 1024 * 1024 {
                    bail!("artifact exceeds download limit");
                }
                file.write_all(&buffer[..n])?;
            }
            file.sync_all()?;
            if crate::hash_file(&part)? != *expected {
                bail!("download hash mismatch for {}", model.id);
            }
            std::fs::rename(&part, &path)?;
            output.push(serde_json::json!({"model": model.id, "path": path, "bytes": total, "verified": true}));
        }
        Ok(output)
    }
    /// Configuration with local overrides and validation.
    pub fn load(path: &Path) -> Result<Self> {
        let mut v: toml::Value = toml::from_str(&std::fs::read_to_string(path)?)?;
        let local = path.parent().unwrap_or(Path::new(".")).join("local.toml");
        if local != path && local.exists() {
            merge(&mut v, toml::from_str(&std::fs::read_to_string(local)?)?);
        }
        let mut c: Self = v.try_into()?;
        if let Ok(p) = std::env::var("A2E_NATIVE_MODELS_DIR") {
            c.runtime.models_dir = p;
        }
        if let Ok(p) = std::env::var("A2E_NATIVE_LLAMA_SERVER") {
            c.runtime.llama_server = p;
        }
        if let Ok(p) = std::env::var("A2E_NATIVE_OUTPUT_DIR") {
            c.runtime.output_dir = p;
        }
        if let Ok(m) = std::env::var("A2E_JUDGE_MODEL") {
            c.judge.model = m;
        }
        if let Ok(p) = std::env::var("A2E_JUDGE_PROVIDER") {
            c.judge.provider = p;
        }
        let mut names = BTreeSet::new();
        for m in &c.models {
            if !names.insert(&m.id) {
                bail!("duplicate model {}", m.id);
            }
            crate::adapter::Adapter::new(&m.adapter)?;
            if m.source.is_empty() || m.revision.len() != 40 {
                bail!("model {} requires a pinned source revision", m.id);
            }
        }
        if c.runtime.llama_commit.len() != 40
            || c.runtime.threads == 0
            || c.runtime.sample_ms < 100
            || c.runtime.memory_limit_mb == 0
        {
            bail!("invalid runtime limits or llama.cpp commit");
        }
        if !["auto", "none", "required"].contains(&c.runtime.power.as_str()) {
            bail!("power must be auto, none, or required");
        }
        Ok(c)
    }
    pub fn model(&self, id: &str) -> Result<&Model> {
        self.models
            .iter()
            .find(|m| m.id == id)
            .with_context(|| format!("unknown model {id}"))
    }
    pub fn model_path(&self, m: &Model) -> PathBuf {
        expand_path(&self.runtime.models_dir).join(&m.file)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Budget {
    pub generated_tokens: usize,
    pub model_calls: usize,
    pub call_seconds: u64,
    pub task_seconds: u64,
    pub tool_seconds: u64,
    pub tool_output_bytes: usize,
}
impl Default for Budget {
    fn default() -> Self {
        Self {
            generated_tokens: 256,
            model_calls: 8,
            call_seconds: 300,
            task_seconds: 900,
            tool_seconds: 20,
            tool_output_bytes: 8192,
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Variant {
    pub name: String,
    pub mode: String,
    pub filter_top_k: usize,
    pub context: String,
    pub constrained: bool,
    pub repairs: usize,
}
impl Default for Variant {
    fn default() -> Self {
        Self {
            name: "baseline".into(),
            mode: "autonomous".into(),
            filter_top_k: 0,
            context: "full".into(),
            constrained: false,
            repairs: 0,
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Suite {
    pub name: String,
    pub models: Vec<String>,
    pub contexts: Vec<usize>,
    pub tasks: Vec<String>,
    pub split: String,
    pub web_mode: String,
    pub warmup: usize,
    pub repeats: usize,
    #[serde(default)]
    pub seed: u64,
    #[serde(default)]
    pub budget: Budget,
    pub variants: Vec<Variant>,
}
impl Suite {
    pub fn load(path: &Path) -> Result<Self> {
        let s: Self = toml::from_str(&std::fs::read_to_string(path)?)?;
        if s.models.is_empty()
            || s.contexts.is_empty()
            || s.tasks.is_empty()
            || s.variants.is_empty()
            || s.repeats == 0
            || s.contexts.iter().any(|n| *n < 256)
        {
            bail!("suite has empty axes or invalid limits");
        }
        if !["dev", "heldout"].contains(&s.split.as_str())
            || !["fixture", "live"].contains(&s.web_mode.as_str())
        {
            bail!("invalid suite split or web_mode");
        }
        if s.budget.model_calls == 0
            || s.budget.model_calls > 128
            || s.budget.generated_tokens == 0
            || s.budget.call_seconds == 0
            || s.budget.task_seconds == 0
            || s.budget.tool_seconds == 0
            || !(256..=65536).contains(&s.budget.tool_output_bytes)
        {
            bail!("invalid task budget");
        }
        let mut names = BTreeSet::new();
        for v in &s.variants {
            if !names.insert(&v.name)
                || !["autonomous", "guided"].contains(&v.mode.as_str())
                || !["full", "recent"].contains(&v.context.as_str())
                || v.repairs > 1
            {
                bail!("invalid or duplicate variant {}", v.name);
            }
        }
        Ok(s)
    }
    pub fn load_tasks(&self, path: &Path) -> Result<Vec<Task>> {
        let base = path.parent().unwrap_or(Path::new("."));
        let mut tasks = Vec::new();
        let mut ids = BTreeSet::new();
        for file in &self.tasks {
            let text = std::fs::read_to_string(base.join(file))?;
            let batch: Vec<Task> = serde_json::from_str(&text)?;
            for t in batch.into_iter().filter(|t| t.split == self.split) {
                if !ids.insert(t.id.clone()) || t.tools.is_empty() || t.prompt.is_empty() {
                    bail!("duplicate or empty task {}", t.id);
                }
                if t.checks.is_empty() {
                    bail!("task {} has no outcome checks", t.id);
                }
                tasks.push(t);
            }
        }
        if tasks.is_empty() {
            bail!("no tasks for split {}", self.split);
        }
        Ok(tasks)
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Stage {
    pub instruction: String,
    pub tools: Vec<String>,
    #[serde(default = "one")]
    pub successes: usize,
}
fn one() -> usize {
    1
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Check {
    pub kind: String,
    pub key: String,
    pub expected: Value,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Task {
    pub id: String,
    pub split: String,
    pub family: String,
    pub prompt: String,
    pub tools: Vec<String>,
    pub required_calls: usize,
    #[serde(default)]
    pub fixtures: Value,
    #[serde(default)]
    pub stages: Vec<Stage>,
    pub checks: Vec<Check>,
    #[serde(default)]
    pub summary: bool,
    #[serde(default)]
    pub required_facts: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    #[serde(deserialize_with = "schema_value")]
    pub parameters: Value,
}
fn schema_value<'de, D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Value, D::Error> {
    let v = Value::deserialize(d)?;
    if let Value::String(s) = v {
        serde_json::from_str(&s).map_err(serde::de::Error::custom)
    } else {
        Ok(v)
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub protocol: u32,
    pub executable: String,
    #[serde(default)]
    pub args: Vec<String>,
    pub timeout_seconds: u64,
    pub output_limit_bytes: usize,
    pub tools: Vec<ToolSpec>,
}
pub fn validate_schema(schema: &Value) -> Result<()> {
    // Local schemas for offline validation
    fn inspect(v: &Value) -> Result<()> {
        match v {
            Value::Object(m) => {
                for (k, v) in m {
                    if k == "$ref" || k == "$dynamicRef" {
                        bail!("schema references are not supported");
                    }
                    inspect(v)?;
                }
            }
            Value::Array(a) => {
                for v in a {
                    inspect(v)?;
                }
            }
            _ => (),
        }
        Ok(())
    }
    inspect(schema)?;
    jsonschema::validator_for(schema).map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(())
}
impl Manifest {
    pub fn load(path: &Path) -> Result<Self> {
        let m: Self = toml::from_str(&std::fs::read_to_string(path)?)?;
        if m.protocol != 1
            || m.timeout_seconds == 0
            || !(256..=65536).contains(&m.output_limit_bytes)
        {
            bail!("invalid tool protocol or bounds");
        }
        let mut names = BTreeSet::new();
        for t in &m.tools {
            if t.name.is_empty()
                || !t
                    .name
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || c == b'_')
                || !names.insert(&t.name)
            {
                bail!("invalid or duplicate tool {}", t.name);
            }
            validate_schema(&t.parameters)?;
        }
        Ok(m)
    }
    pub fn specs(&self) -> BTreeMap<String, ToolSpec> {
        self.tools
            .iter()
            .cloned()
            .map(|t| (t.name.clone(), t))
            .collect()
    }
}
