// config.rs
// Runtime settings, model registry, and experiment definitions
// Local overrides, pinned model downloads, and task and tool schema checks

use crate::case::{self, Case, Split};
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
            llama_commit: String::new(),
            models_dir: "models".into(),
            output_dir: "results".into(),
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
#[serde(deny_unknown_fields)]
pub struct Config {
    #[serde(default)]
    pub runtime: Runtime,
    pub models: Vec<Model>,
    #[serde(skip)]
    pub host_id: String,
    #[serde(skip)]
    pub host_file: Option<PathBuf>,
}

/// Device identity as `user@hostname` without the domain suffix; `A2E_HOST_ID` overrides it.
pub fn host_id() -> String {
    if let Ok(id) = std::env::var("A2E_HOST_ID") {
        return id;
    }
    let user = std::env::var("USER").unwrap_or_default();
    let full = sysinfo::System::host_name().unwrap_or_default();
    let host = full.split('.').next().unwrap_or_default();
    format!("{user}@{host}")
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
        let dir = path.parent().unwrap_or(Path::new("."));
        let host_id = host_id();
        let host_file = dir.join("hosts").join(format!("{host_id}.toml"));
        let host_file = host_file.exists().then_some(host_file);
        if host_file.is_none() {
            eprintln!(
                "warning: no host file for {host_id}; using shared settings only. Add {}/hosts/{host_id}.toml or set A2E_HOST_ID.",
                dir.display()
            );
        }
        if let Some(f) = &host_file {
            merge(&mut v, toml::from_str(&std::fs::read_to_string(f)?)?);
        }
        let local = dir.join("local.toml");
        if local != path && local.exists() {
            merge(&mut v, toml::from_str(&std::fs::read_to_string(local)?)?);
        }
        let mut c: Self = v.try_into()?;
        c.host_id = host_id;
        c.host_file = host_file;
        if let Ok(p) = std::env::var("A2E_MODELS_DIR") {
            c.runtime.models_dir = p;
        }
        if let Ok(p) = std::env::var("A2E_LLAMA_SERVER") {
            c.runtime.llama_server = p;
        }
        if let Ok(p) = std::env::var("A2E_OUTPUT_DIR") {
            c.runtime.output_dir = p;
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
        if c.runtime.llama_commit.len() != 40 {
            bail!("llama_commit must be a full 40 character SHA in the [runtime] table");
        }
        if c.runtime.threads == 0 || c.runtime.sample_ms < 100 || c.runtime.memory_limit_mb == 0 {
            bail!("invalid runtime limits");
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
    /// Consecutive attempts with no completed model call before the rest of a
    /// model and context group is skipped; 0 never skips.
    pub stall_attempts: usize,
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
            stall_attempts: 3,
        }
    }
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    #[default]
    Autonomous,
    Guided,
    /// Guided with a prompt prefix that never changes, so the prompt cache survives every stage.
    GuidedStable,
}
impl Mode {
    /// True for the modes that follow a case's workflow stages.
    pub fn staged(self) -> bool {
        self != Self::Autonomous
    }
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContextWindow {
    #[default]
    Full,
    Recent,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WebMode {
    Fixture,
    Live,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Variant {
    pub name: String,
    pub mode: Mode,
    pub filter_top_k: usize,
    pub context: ContextWindow,
    pub constrained: bool,
    pub repairs: usize,
}
impl Default for Variant {
    fn default() -> Self {
        Self {
            name: "baseline".into(),
            mode: Mode::Autonomous,
            filter_top_k: 0,
            context: ContextWindow::Full,
            constrained: false,
            repairs: 0,
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Suite {
    pub name: String,
    /// Empty means every model in the registry; `resolve` fills it in.
    #[serde(default)]
    pub models: Vec<String>,
    pub contexts: Vec<usize>,
    pub cases: Vec<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    pub split: Split,
    pub web_mode: WebMode,
    pub warmup: usize,
    pub repeats: usize,
    #[serde(default)]
    pub seed: u64,
    #[serde(default)]
    pub budget: Budget,
    /// Names from `configs/variants.toml`, placed before the inline variants.
    #[serde(default, skip_serializing)]
    pub use_variants: Vec<String>,
    #[serde(default)]
    pub variants: Vec<Variant>,
}

/// Narrows a suite to the named models, contexts and variants; empty lists keep everything.
#[derive(Clone, Debug, Default)]
pub struct Selection {
    pub models: Vec<String>,
    pub contexts: Vec<usize>,
    pub variants: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SharedVariants {
    variants: Vec<Variant>,
}

/// Error for a name that is not in `known`, with the closest match as a hint.
pub fn unknown_name(kind: &str, name: &str, known: &[String]) -> anyhow::Error {
    let nearest = known
        .iter()
        .find(|k| k.starts_with(name) || k.contains(name))
        .or_else(|| {
            known
                .iter()
                .min_by_key(|k| edit_distance(name, k))
                .filter(|k| edit_distance(name, k) <= name.len().max(3) / 2)
        });
    let list = known.join(", ");
    match nearest {
        Some(k) => anyhow::anyhow!("unknown {kind} {name}; did you mean {k}? Known: {list}"),
        None => anyhow::anyhow!("unknown {kind} {name}. Known: {list}"),
    }
}

fn edit_distance(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.chars().enumerate() {
        let mut diagonal = row[0];
        row[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let above = row[j + 1];
            row[j + 1] = (above + 1)
                .min(row[j] + 1)
                .min(diagonal + usize::from(ca != *cb));
            diagonal = above;
        }
    }
    row[b.len()]
}

fn keep_named<T>(
    items: &mut Vec<T>,
    kind: &str,
    wanted: &[String],
    name: impl Fn(&T) -> &str,
) -> Result<()> {
    if wanted.is_empty() {
        return Ok(());
    }
    let known: Vec<String> = items.iter().map(|i| name(i).to_string()).collect();
    for w in wanted {
        if !known.contains(w) {
            return Err(unknown_name(kind, w, &known));
        }
    }
    items.retain(|i| wanted.iter().any(|w| w == name(i)));
    Ok(())
}

impl Suite {
    pub fn load(path: &Path) -> Result<Self> {
        let mut s: Self = toml::from_str(&std::fs::read_to_string(path)?)?;
        if !s.use_variants.is_empty() {
            let file = path
                .parent()
                .unwrap_or(Path::new("."))
                .join("../variants.toml");
            let shared: SharedVariants = toml::from_str(
                &std::fs::read_to_string(&file)
                    .with_context(|| format!("suite {} needs {}", s.name, file.display()))?,
            )?;
            let known: Vec<String> = shared.variants.iter().map(|v| v.name.clone()).collect();
            let mut chosen = Vec::new();
            for name in &s.use_variants {
                let found = shared
                    .variants
                    .iter()
                    .find(|v| &v.name == name)
                    .ok_or_else(|| unknown_name("shared variant", name, &known))?;
                chosen.push(found.clone());
            }
            chosen.append(&mut s.variants);
            s.variants = chosen;
        }
        if s.contexts.is_empty()
            || s.cases.is_empty()
            || s.variants.is_empty()
            || s.repeats == 0
            || s.contexts.iter().any(|n| *n < 256)
        {
            bail!("suite has empty axes or invalid limits");
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
            if !names.insert(&v.name) || v.repairs > 1 {
                bail!("invalid or duplicate variant {}", v.name);
            }
        }
        Ok(s)
    }
    /// Fills in the model list and applies `selection`; fails on names that do not exist.
    pub fn resolve(mut self, config: &Config, selection: &Selection) -> Result<Self> {
        let all: Vec<String> = config.models.iter().map(|m| m.id.clone()).collect();
        if self.models.is_empty() {
            self.models = all.clone();
        }
        for id in &self.models {
            if !all.contains(id) {
                return Err(unknown_name("model", id, &all));
            }
        }
        keep_named(&mut self.models, "model", &selection.models, |m| m.as_str())?;
        keep_named(&mut self.variants, "variant", &selection.variants, |v| {
            v.name.as_str()
        })?;
        if !selection.contexts.is_empty() {
            for c in &selection.contexts {
                if !self.contexts.contains(c) {
                    bail!(
                        "suite {} has no context {c}; it has {:?}",
                        self.name,
                        self.contexts
                    );
                }
            }
            self.contexts.retain(|c| selection.contexts.contains(c));
        }
        if self.models.is_empty() {
            bail!("suite {} has no models", self.name);
        }
        Ok(self)
    }
    /// Cases for this suite's split and tags; paths resolve against the suite file.
    pub fn load_cases(&self, path: &Path) -> Result<Vec<Case>> {
        let base = path.parent().unwrap_or(Path::new("."));
        let selectors = self.cases.iter().map(|c| base.join(c)).collect::<Vec<_>>();
        let cases = case::load_all(&selectors)?
            .into_iter()
            .filter(|c| c.split == self.split && self.tags.iter().all(|t| c.tags.contains(t)))
            .collect::<Vec<_>>();
        if cases.is_empty() {
            bail!(
                "no cases for split {:?} and tags {:?}",
                self.split,
                self.tags
            );
        }
        Ok(cases)
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
/// Model-visible tool definitions; every tool must have a built-in handler.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Catalog {
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
impl Catalog {
    pub fn load(path: &Path) -> Result<Self> {
        let c: Self = toml::from_str(&std::fs::read_to_string(path)?)?;
        let handlers = crate::tools::Builtins::names();
        let mut names = BTreeSet::new();
        for t in &c.tools {
            if t.name.is_empty()
                || !t
                    .name
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || c == b'_')
                || !names.insert(&t.name)
            {
                bail!("invalid or duplicate tool {}", t.name);
            }
            if !handlers.contains(&t.name.as_str()) {
                bail!("tool {} has no built-in handler", t.name);
            }
            validate_schema(&t.parameters)?;
        }
        Ok(c)
    }
    pub fn specs(&self) -> BTreeMap<String, ToolSpec> {
        self.tools
            .iter()
            .cloned()
            .map(|t| (t.name.clone(), t))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASE: &str = "[runtime]\nthreads = 4\nllama_commit = \"0123456789012345678901234567890123456789\"\n\n[[models]]\nid = \"m\"\nadapter = \"qwen3\"\nquant = \"q\"\nsource = \"o/r\"\nrevision = \"0123456789012345678901234567890123456789\"\nfile = \"m.gguf\"\n";

    fn setup(host: Option<&str>, local: Option<&str>) -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("models.toml");
        std::fs::write(&base, BASE).unwrap();
        if let Some(h) = host {
            std::fs::create_dir(dir.path().join("hosts")).unwrap();
            std::fs::write(dir.path().join("hosts/test@host.toml"), h).unwrap();
        }
        if let Some(l) = local {
            std::fs::write(dir.path().join("local.toml"), l).unwrap();
        }
        (dir, base)
    }

    // Tests share the process environment, so one test sets A2E_HOST_ID for all.
    #[test]
    fn host_file_and_precedence() {
        std::env::set_var("A2E_HOST_ID", "test@host");

        let (_d, base) = setup(None, None);
        assert!(Config::load(&base).unwrap().host_file.is_none());

        let host = "[runtime]\nthreads = 2\n";
        let (_d, base) = setup(Some(host), None);
        let c = Config::load(&base).unwrap();
        assert!(c.host_file.is_some());
        assert_eq!(c.runtime.threads, 2);

        let (_d, base) = setup(Some(host), Some("[runtime]\nthreads = 3\n"));
        assert_eq!(Config::load(&base).unwrap().runtime.threads, 3);

        let (_d, base) = setup(Some("[runtime]\nunknown = true\n"), None);
        assert!(Config::load(&base).is_err());
    }
}
