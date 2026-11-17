//! Model registry, settings loading, and experiment definitions.
//!
//! Settings come from layered files (see `crate::settings`); this module adds pinned model
//! downloads and task and tool schema checks.

use crate::case::{self, Case, Split};
use crate::settings::{self, Layers, Settings};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
    sync::OnceLock,
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
    /// Overrides of the `llama` and `sampling` sections for this model only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub settings: Option<toml::Value>,
}

#[derive(Clone, Debug)]
pub struct Config {
    pub settings: Settings,
    pub models: Vec<Model>,
    pub host_id: String,
    pub host_file: Option<PathBuf>,
    /// Which layer set each settings key, for `a2e config show`.
    pub layers: Layers,
    per_model: BTreeMap<String, Settings>,
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
pub(crate) fn merge(a: &mut toml::Value, b: toml::Value) {
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

static OVERRIDES: OnceLock<Vec<String>> = OnceLock::new();
/// `--set section.key=value` arguments, applied above every file and environment variable.
pub fn set_overrides(sets: Vec<String>) {
    let _ = OVERRIDES.set(sets);
}
/// The `--set` arguments of this process, so a detached run can repeat them.
pub fn overrides() -> &'static [String] {
    OVERRIDES.get().map(Vec::as_slice).unwrap_or_default()
}
fn cli_layer() -> Result<toml::Value> {
    let mut v = toml::Value::Table(Default::default());
    for item in overrides() {
        let (key, value) = item
            .split_once('=')
            .with_context(|| format!("--set needs section.key=value, got {item}"))?;
        settings::set_path(&mut v, key.trim(), settings::parse_value(value.trim()));
    }
    Ok(v)
}

/// Sections a model entry may override.
const MODEL_SECTIONS: [&str; 2] = ["llama", "sampling"];

impl Config {
    /// Download pinned artifacts and verify their published hashes.
    pub fn fetch_models(&self) -> Result<Vec<Value>> {
        let mut output = Vec::new();
        for model in &self.models {
            let expected = model
                .sha256
                .as_ref()
                .context("download requires a pinned SHA-256")?;
            output.push(self.download(model, expected)?);
        }
        Ok(output)
    }
    fn download(&self, model: &Model, expected: &str) -> Result<Value> {
        use std::io::{Read, Write};
        let adv = &self.settings.advanced;
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(adv.download_timeout_s))
            .build()?;
        let path = self.model_path(model);
        if path.exists() && crate::hash_file(&path)? == expected {
            return Ok(serde_json::json!({"model": model.id, "path": path, "cached": true}));
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
            if total > adv.download_max_bytes {
                bail!("artifact exceeds advanced.download_max_bytes");
            }
            file.write_all(&buffer[..n])?;
        }
        file.sync_all()?;
        if crate::hash_file(&part)? != expected {
            bail!("download hash mismatch for {}", model.id);
        }
        std::fs::rename(&part, &path)?;
        Ok(serde_json::json!({"model": model.id, "path": path, "bytes": total, "verified": true}))
    }
    pub fn load(path: &Path) -> Result<Self> {
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
        let mut files = vec![(path.to_path_buf(), "models.toml".to_string())];
        let shared = dir.join("settings.toml");
        if shared != path && shared.exists() {
            files.push((shared, "settings.toml".into()));
        }
        if let Some(f) = &host_file {
            files.push((f.clone(), format!("hosts/{host_id}.toml")));
        }
        let local = dir.join("local.toml");
        if local != path && local.exists() {
            files.push((local, "local.toml".into()));
        }
        let mut layers = Layers::new();
        let mut models = toml::Value::Array(Vec::new());
        for (file, name) in files {
            let mut v: toml::Value = toml::from_str(&std::fs::read_to_string(&file)?)
                .with_context(|| format!("read {}", file.display()))?;
            settings::upgrade_legacy(&mut v, &name)?;
            if let Some(t) = v.as_table_mut() {
                if let Some(m) = t.remove("models") {
                    models = m;
                }
            }
            layers.add(v, &name, false);
        }
        layers.add(settings::env_layer(), "environment", true);
        layers.add(cli_layer()?, "--set", true);
        let settings = layers.settings()?;
        let models: Vec<Model> = models.try_into()?;
        let mut names = BTreeSet::new();
        let mut per_model = BTreeMap::new();
        for m in &models {
            if !names.insert(&m.id) {
                bail!("duplicate model {}", m.id);
            }
            crate::adapter::Adapter::new(&m.adapter)?;
            if m.source.is_empty() || m.revision.len() != 40 {
                bail!("model {} requires a pinned source revision", m.id);
            }
            if let Some(over) = &m.settings {
                per_model.insert(m.id.clone(), model_settings(&layers, m, over)?);
            }
        }
        settings::activate(&settings);
        Ok(Self {
            settings,
            models,
            host_id,
            host_file,
            layers,
            per_model,
        })
    }
    pub fn model(&self, id: &str) -> Result<&Model> {
        self.models
            .iter()
            .find(|m| m.id == id)
            .with_context(|| format!("unknown model {id}"))
    }
    /// Settings for one model: the shared ones plus that model's `llama` and `sampling` overrides.
    pub fn settings_for(&self, m: &Model) -> &Settings {
        self.per_model.get(&m.id).unwrap_or(&self.settings)
    }
    pub fn model_path(&self, m: &Model) -> PathBuf {
        expand_path(&self.settings.host.models_dir).join(&m.file)
    }
}

fn model_settings(layers: &Layers, m: &Model, over: &toml::Value) -> Result<Settings> {
    let table = over
        .as_table()
        .with_context(|| format!("model {} settings must be a table", m.id))?;
    if let Some(bad) = table.keys().find(|k| !MODEL_SECTIONS.contains(&k.as_str())) {
        bail!(
            "model {} settings.{bad} is not allowed; use llama or sampling",
            m.id
        );
    }
    let mut merged = layers.merged.clone();
    let mut over = over.clone();
    // Environment and --set values still win over a model entry
    for key in &layers.forced {
        if let Some(t) = over.as_table_mut() {
            let mut parts = key.splitn(2, '.');
            if let (Some(section), Some(k)) = (parts.next(), parts.next()) {
                if let Some(s) = t.get_mut(section).and_then(|s| s.as_table_mut()) {
                    s.remove(k);
                }
            }
        }
    }
    merge(&mut merged, over);
    let mut s: Settings = merged.try_into()?;
    s.forced = layers.forced.clone();
    s.validate()
        .with_context(|| format!("model {} settings", m.id))?;
    Ok(s)
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
        let l = settings::Limits::default();
        Self {
            generated_tokens: l.output_tokens,
            model_calls: l.model_calls,
            call_seconds: l.call_seconds,
            task_seconds: l.task_seconds,
            tool_seconds: l.tool_seconds,
            tool_output_bytes: l.tool_output_bytes,
            stall_attempts: l.stall_attempts,
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
    #[serde(default = "default_contexts")]
    pub contexts: Vec<usize>,
    pub cases: Vec<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    pub split: Split,
    pub web_mode: WebMode,
    #[serde(default)]
    pub warmup: usize,
    #[serde(default = "default_repeats")]
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
    /// Keys the suite file sets itself; the rest follow the settings layers.
    #[serde(skip)]
    pub explicit: BTreeSet<String>,
}
fn default_contexts() -> Vec<usize> {
    settings::Limits::default().context
}
fn default_repeats() -> usize {
    settings::Sampling::default().repeats
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
        let text = std::fs::read_to_string(path)?;
        let raw: toml::Value = toml::from_str(&text)?;
        let mut s: Self = toml::from_str(&text)?;
        s.explicit = settings::leaves(&raw)
            .into_iter()
            .filter(|k| {
                ["contexts", "warmup", "repeats", "seed"].contains(&k.as_str())
                    || k.starts_with("budget.")
            })
            .collect();
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
        s.validate()?;
        Ok(s)
    }
    fn validate(&self) -> Result<()> {
        let s = self;
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
        Ok(())
    }
    /// Takes every value the suite file left out, and every value forced by the environment or
    /// `--set`, from the settings.
    pub fn apply_settings(&mut self, st: &Settings) -> Result<()> {
        let take = |own: &str, key: &str| -> bool {
            !self.explicit.contains(own) || st.forced.contains(key)
        };
        if take("contexts", "limits.context") {
            self.contexts = st.limits.context.clone();
        }
        if take("warmup", "sampling.warmup") {
            self.warmup = st.sampling.warmup;
        }
        if take("repeats", "sampling.repeats") {
            self.repeats = st.sampling.repeats;
        }
        if take("seed", "sampling.seed") {
            self.seed = st.sampling.seed;
        }
        let l = &st.limits;
        if take("budget.generated_tokens", "limits.output_tokens") {
            self.budget.generated_tokens = l.output_tokens;
        }
        if take("budget.model_calls", "limits.model_calls") {
            self.budget.model_calls = l.model_calls;
        }
        if take("budget.call_seconds", "limits.call_seconds") {
            self.budget.call_seconds = l.call_seconds;
        }
        if take("budget.task_seconds", "limits.task_seconds") {
            self.budget.task_seconds = l.task_seconds;
        }
        if take("budget.tool_seconds", "limits.tool_seconds") {
            self.budget.tool_seconds = l.tool_seconds;
        }
        if take("budget.tool_output_bytes", "limits.tool_output_bytes") {
            self.budget.tool_output_bytes = l.tool_output_bytes;
        }
        if take("budget.stall_attempts", "limits.stall_attempts") {
            self.budget.stall_attempts = l.stall_attempts;
        }
        self.validate()
    }
    /// Fills in the model list and applies `selection`; fails on names that do not exist.
    pub fn resolve(mut self, config: &Config, selection: &Selection) -> Result<Self> {
        self.apply_settings(&config.settings)?;
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

    const BASE: &str = "[llama]\ncommit = \"0123456789012345678901234567890123456789\"\nthreads = 4\n\n[[models]]\nid = \"m\"\nadapter = \"qwen3\"\nquant = \"q\"\nsource = \"o/r\"\nrevision = \"0123456789012345678901234567890123456789\"\nfile = \"m.gguf\"\n";

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

        let host = "[llama]\nthreads = 2\n";
        let (_d, base) = setup(Some(host), None);
        let c = Config::load(&base).unwrap();
        assert!(c.host_file.is_some());
        assert_eq!(c.settings.llama.threads, 2);
        assert_eq!(c.layers.source["llama.threads"], "hosts/test@host.toml");

        let (_d, base) = setup(Some(host), Some("[llama]\nthreads = 3\n"));
        let c = Config::load(&base).unwrap();
        assert_eq!(c.settings.llama.threads, 3);
        assert_eq!(c.layers.source["llama.threads"], "local.toml");
        assert_eq!(c.layers.source["llama.commit"], "models.toml");

        let (_d, base) = setup(Some("[llama]\nunknown = true\n"), None);
        assert!(Config::load(&base).is_err());
    }

    #[test]
    fn a_settings_file_sits_between_models_and_host_layers() {
        std::env::set_var("A2E_HOST_ID", "test@host");
        let (d, base) = setup(Some("[limits]\noutput_tokens = 99\n"), None);
        std::fs::write(
            d.path().join("settings.toml"),
            "[limits]\noutput_tokens = 50\nmodel_calls = 4\n",
        )
        .unwrap();
        let c = Config::load(&base).unwrap();
        assert_eq!(c.settings.limits.output_tokens, 99);
        assert_eq!(c.settings.limits.model_calls, 4);
        assert_eq!(c.layers.source["limits.model_calls"], "settings.toml");
    }

    #[test]
    fn a_model_entry_overrides_llama_and_sampling_only() {
        std::env::set_var("A2E_HOST_ID", "test@host");
        let (d, base) = setup(None, None);
        let with = |extra: &str| {
            BASE.replace(
                "file = \"m.gguf\"\n",
                &format!("file = \"m.gguf\"\n{extra}"),
            )
        };
        std::fs::write(
            &base,
            with("[models.settings.llama]\nthreads = 1\nflash_attn = \"on\"\n"),
        )
        .unwrap();
        let c = Config::load(&base).unwrap();
        let m = c.model("m").unwrap();
        assert_eq!(c.settings_for(m).llama.threads, 1);
        assert_eq!(c.settings.llama.threads, 4);
        std::fs::write(&base, with("[models.settings.limits]\nmemory_mb = 5\n")).unwrap();
        assert!(Config::load(&base).is_err());
        drop(d);
    }

    #[test]
    fn suites_follow_settings_unless_they_set_a_key() {
        let toml = "name = \"s\"\ncases = [\"c\"]\nsplit = \"heldout\"\nweb_mode = \"fixture\"\nrepeats = 3\n[budget]\nmodel_calls = 5\n\n[[variants]]\nname = \"a\"\n";
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.toml");
        std::fs::write(&path, toml).unwrap();
        let mut suite: Suite = Suite::load(&path).unwrap();
        let mut st = Settings::default();
        st.limits.output_tokens = 64;
        st.limits.model_calls = 7;
        st.sampling.repeats = 9;
        suite.apply_settings(&st).unwrap();
        assert_eq!(suite.budget.generated_tokens, 64);
        assert_eq!(suite.budget.model_calls, 5);
        assert_eq!(suite.repeats, 3);
        st.forced.insert("limits.model_calls".into());
        st.forced.insert("sampling.repeats".into());
        suite.apply_settings(&st).unwrap();
        assert_eq!(suite.budget.model_calls, 7);
        assert_eq!(suite.repeats, 9);
    }
}
