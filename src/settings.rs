//! Every setting a user can change, with the layers that supply them.
//!
//! Defaults equal the values that were hard-coded before this module existed, so runs stay comparable.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
    sync::OnceLock,
};

/// Budgets that bound one study; a suite may override any of them.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Limits {
    pub memory_mb: u64,
    pub reserve_mb: u64,
    pub context: Vec<usize>,
    /// Longest prompt in tokens; 0 means only the context size limits it.
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub model_calls: usize,
    pub call_seconds: u64,
    pub task_seconds: u64,
    pub tool_seconds: u64,
    pub tool_output_bytes: usize,
    pub stall_attempts: usize,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            memory_mb: 768,
            reserve_mb: 128,
            context: vec![1024],
            input_tokens: 0,
            output_tokens: 256,
            model_calls: 8,
            call_seconds: 300,
            task_seconds: 900,
            tool_seconds: 20,
            tool_output_bytes: 8192,
            stall_attempts: 3,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Sampling {
    pub temperature: f64,
    pub top_p: f64,
    /// 0 turns top-k off.
    pub top_k: usize,
    pub min_p: f64,
    pub seed: u64,
    pub repeats: usize,
    pub warmup: usize,
}
impl Default for Sampling {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            seed: 0,
            repeats: 1,
            warmup: 0,
        }
    }
}

/// One key per llama-server flag the harness controls.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Llama {
    /// Full 40 character commit the server binary must report.
    pub commit: String,
    pub server: String,
    pub host: String,
    pub port: u16,
    pub threads: usize,
    pub gpu_layers: i32,
    pub parallel: usize,
    pub batch_size: usize,
    pub ubatch_size: usize,
    /// Host-memory prompt cache in MiB. The server default of 8 GiB grows memory between tasks.
    pub cache_ram: u64,
    pub jinja: bool,
    pub special: bool,
    pub context_shift: bool,
    /// `auto` leaves the server default; `on` and `off` pass the flag.
    pub flash_attn: String,
    pub cache_type_k: String,
    pub cache_type_v: String,
    pub mlock: bool,
    pub mmap: bool,
    /// Empty uses the template inside the model file.
    pub chat_template: String,
    /// Raw flags appended last, for anything not listed above.
    pub extra_args: Vec<String>,
    pub startup_seconds: u64,
}
impl Default for Llama {
    fn default() -> Self {
        Self {
            commit: String::new(),
            server: "deps/llama.cpp/build-a2e/bin/llama-server".into(),
            host: "127.0.0.1".into(),
            port: 8197,
            threads: 4,
            gpu_layers: 0,
            parallel: 1,
            batch_size: 128,
            ubatch_size: 64,
            cache_ram: 0,
            jinja: true,
            special: true,
            context_shift: false,
            flash_attn: "auto".into(),
            cache_type_k: "f16".into(),
            cache_type_v: "f16".into(),
            mlock: false,
            mmap: true,
            chat_template: String::new(),
            extra_args: Vec::new(),
            startup_seconds: 300,
        }
    }
}
impl Llama {
    /// The server command line for one model load. Flags at their server default are not passed.
    pub fn args(&self, model: &Path, ctx: usize, slots: &Path) -> Vec<String> {
        let mut a: Vec<String> = Vec::new();
        let mut put = |flag: &str, value: String| {
            a.push(flag.into());
            a.push(value);
        };
        put("--host", self.host.clone());
        put("--port", self.port.to_string());
        put("--model", model.to_string_lossy().into_owned());
        put("--ctx-size", ctx.to_string());
        put("--threads", self.threads.to_string());
        put("--n-gpu-layers", self.gpu_layers.to_string());
        put("--parallel", self.parallel.to_string());
        put("--batch-size", self.batch_size.to_string());
        put("--ubatch-size", self.ubatch_size.to_string());
        if self.jinja {
            a.push("--jinja".into());
        }
        if self.special {
            a.push("--special".into());
        }
        if !self.context_shift {
            a.push("--no-context-shift".into());
        }
        a.push("--cache-ram".into());
        a.push(self.cache_ram.to_string());
        if self.flash_attn != "auto" {
            a.push("--flash-attn".into());
            a.push(self.flash_attn.clone());
        }
        if self.cache_type_k != "f16" {
            a.push("--cache-type-k".into());
            a.push(self.cache_type_k.clone());
        }
        if self.cache_type_v != "f16" {
            a.push("--cache-type-v".into());
            a.push(self.cache_type_v.clone());
        }
        if self.mlock {
            a.push("--mlock".into());
        }
        if !self.mmap {
            a.push("--no-mmap".into());
        }
        if !self.chat_template.is_empty() {
            a.push("--chat-template".into());
            a.push(self.chat_template.clone());
        }
        a.push("--slots".into());
        a.push("--slot-save-path".into());
        a.push(slots.to_string_lossy().into_owned());
        a.extend(self.extra_args.iter().cloned());
        a
    }
}

/// Facts about this machine and where files go.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Host {
    pub device: String,
    pub power: String,
    pub sample_ms: u64,
    pub thermal_path: String,
    pub models_dir: String,
    pub output_dir: String,
    pub sandbox_dir: String,
    /// Empty means a `workspace` folder inside each run directory.
    pub sandbox_tmp: String,
}
impl Default for Host {
    fn default() -> Self {
        Self {
            device: "auto".into(),
            power: "auto".into(),
            sample_ms: 500,
            thermal_path: "/sys/class/thermal/thermal_zone0/temp".into(),
            models_dir: "models".into(),
            output_dir: "results".into(),
            sandbox_dir: "sandbox".into(),
            sandbox_tmp: String::new(),
        }
    }
}

/// Rare limits. The defaults are the old constants.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Advanced {
    pub inference_timeout_s: u64,
    pub settle_timeout_s: u64,
    pub slot_erase_timeout_s: u64,
    pub upload_timeout_s: u64,
    pub upload_body_bytes: usize,
    pub max_samples: usize,
    pub web_timeout_s: u64,
    pub dns_timeout_s: u64,
    pub download_timeout_s: u64,
    pub download_max_bytes: u64,
}
impl Default for Advanced {
    fn default() -> Self {
        Self {
            inference_timeout_s: 300,
            settle_timeout_s: 60,
            slot_erase_timeout_s: 10,
            upload_timeout_s: 30,
            upload_body_bytes: 2_000_000,
            max_samples: 20_000,
            web_timeout_s: 12,
            dns_timeout_s: 3,
            download_timeout_s: 1800,
            download_max_bytes: 8 * 1024 * 1024 * 1024,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Settings {
    pub limits: Limits,
    pub sampling: Sampling,
    pub llama: Llama,
    pub host: Host,
    pub advanced: Advanced,
    /// Dotted keys set by an environment variable or `--set`; these beat a suite file.
    #[serde(skip)]
    pub forced: BTreeSet<String>,
}

static ACTIVE: OnceLock<Settings> = OnceLock::new();

/// Settings of the running process, for code that has no `Config` in reach. Defaults until a config loads.
pub fn active() -> &'static Settings {
    static DEFAULT: OnceLock<Settings> = OnceLock::new();
    ACTIVE
        .get()
        .unwrap_or_else(|| DEFAULT.get_or_init(Settings::default))
}
pub(crate) fn activate(s: &Settings) {
    let _ = ACTIVE.set(s.clone());
}

impl Settings {
    pub fn validate(&self) -> Result<()> {
        let (l, v, h, a) = (&self.llama, &self.limits, &self.host, &self.advanced);
        if l.commit.len() != 40 {
            bail!("llama.commit must be a full 40 character SHA");
        }
        if l.threads == 0 || l.parallel == 0 || l.batch_size == 0 || l.ubatch_size == 0 {
            bail!("llama.threads, parallel, batch_size and ubatch_size must be above 0");
        }
        if l.port == 0 || l.startup_seconds == 0 {
            bail!("llama.port and startup_seconds must be above 0");
        }
        if !["auto", "on", "off"].contains(&l.flash_attn.as_str()) {
            bail!("llama.flash_attn must be auto, on or off");
        }
        if h.sample_ms < 100 {
            bail!("host.sample_ms must be at least 100");
        }
        if !["auto", "none", "required"].contains(&h.power.as_str()) {
            bail!("host.power must be auto, none or required");
        }
        if v.memory_mb == 0 || v.context.is_empty() || v.context.iter().any(|n| *n < 256) {
            bail!(
                "limits.memory_mb must be above 0 and limits.context needs values of 256 or more"
            );
        }
        if v.model_calls == 0 || v.model_calls > 128 {
            bail!("limits.model_calls must be 1 to 128");
        }
        if v.output_tokens == 0 || v.call_seconds == 0 || v.task_seconds == 0 || v.tool_seconds == 0
        {
            bail!("limits.output_tokens and the time limits must be above 0");
        }
        if !(256..=65536).contains(&v.tool_output_bytes) {
            bail!("limits.tool_output_bytes must be 256 to 65536");
        }
        let s = &self.sampling;
        if s.temperature < 0.0 || !(0.0..=1.0).contains(&s.top_p) || !(0.0..=1.0).contains(&s.min_p)
        {
            bail!("sampling.temperature must be 0 or more; top_p and min_p must be 0 to 1");
        }
        if s.repeats == 0 {
            bail!("sampling.repeats must be above 0");
        }
        if a.inference_timeout_s == 0 || a.settle_timeout_s == 0 || a.download_max_bytes == 0 {
            bail!("advanced timeouts and download_max_bytes must be above 0");
        }
        Ok(())
    }
}

/// Old `[runtime]` keys and the settings keys that replace them.
const LEGACY: [(&str, &str); 13] = [
    ("llama_server", "llama.server"),
    ("llama_commit", "llama.commit"),
    ("port", "llama.port"),
    ("threads", "llama.threads"),
    ("gpu_layers", "llama.gpu_layers"),
    ("startup_seconds", "llama.startup_seconds"),
    ("memory_limit_mb", "limits.memory_mb"),
    ("reserve_mb", "limits.reserve_mb"),
    ("device", "host.device"),
    ("power", "host.power"),
    ("sample_ms", "host.sample_ms"),
    ("models_dir", "host.models_dir"),
    ("output_dir", "host.output_dir"),
];

/// Moves a legacy `[runtime]` table into the new sections and says which keys moved.
pub fn upgrade_legacy(v: &mut toml::Value, origin: &str) -> Result<()> {
    let Some(table) = v.as_table_mut() else {
        return Ok(());
    };
    let Some(toml::Value::Table(runtime)) = table.remove("runtime") else {
        return Ok(());
    };
    for (key, value) in runtime {
        let (_, new) = LEGACY
            .iter()
            .find(|(old, _)| *old == key)
            .with_context(|| format!("{origin}: unknown key runtime.{key}"))?;
        eprintln!("warning: {origin}: runtime.{key} is now {new}");
        set_path(v, new, value);
    }
    Ok(())
}

/// Sets a dotted key, creating tables on the way.
pub fn set_path(v: &mut toml::Value, path: &str, value: toml::Value) {
    let mut node = v;
    let parts: Vec<&str> = path.split('.').collect();
    for (i, part) in parts.iter().enumerate() {
        let Some(table) = node.as_table_mut() else {
            return;
        };
        if i + 1 == parts.len() {
            table.insert((*part).into(), value);
            return;
        }
        node = table
            .entry((*part).to_string())
            .or_insert_with(|| toml::Value::Table(Default::default()));
    }
}

/// Every leaf key of a settings table, as `section.key`.
pub fn leaves(v: &toml::Value) -> Vec<String> {
    fn walk(v: &toml::Value, prefix: &str, out: &mut Vec<String>) {
        match v.as_table() {
            Some(t) if !prefix.is_empty() || !t.is_empty() => {
                for (k, x) in t {
                    let p = if prefix.is_empty() {
                        k.clone()
                    } else {
                        format!("{prefix}.{k}")
                    };
                    if x.is_table() {
                        walk(x, &p, out);
                    } else {
                        out.push(p);
                    }
                }
            }
            _ => out.push(prefix.to_string()),
        }
    }
    let mut out = Vec::new();
    walk(v, "", &mut out);
    out
}

/// A TOML value from the text after `=` in `--set key=value`; bare words are strings.
pub fn parse_value(text: &str) -> toml::Value {
    toml::from_str::<toml::Table>(&format!("v = {text}"))
        .ok()
        .and_then(|mut t| t.remove("v"))
        .unwrap_or_else(|| toml::Value::String(text.into()))
}

/// Settings layers in order, with the source of every key.
#[derive(Clone, Debug)]
pub struct Layers {
    pub merged: toml::Value,
    pub source: BTreeMap<String, String>,
    pub forced: BTreeSet<String>,
}
impl Default for Layers {
    fn default() -> Self {
        Self::new()
    }
}
impl Layers {
    pub fn new() -> Self {
        Self {
            merged: toml::Value::Table(Default::default()),
            source: BTreeMap::new(),
            forced: BTreeSet::new(),
        }
    }
    pub fn add(&mut self, layer: toml::Value, name: &str, forced: bool) {
        for key in leaves(&layer) {
            self.source.insert(key.clone(), name.into());
            if forced {
                self.forced.insert(key);
            }
        }
        crate::config::merge(&mut self.merged, layer);
    }
    pub fn settings(&self) -> Result<Settings> {
        let mut s: Settings = self.merged.clone().try_into()?;
        s.forced = self.forced.clone();
        s.validate()?;
        Ok(s)
    }
}

/// `A2E_<SECTION>_<KEY>` variables as a settings layer, for example `A2E_LIMITS_MEMORY_MB`, plus the older names.
pub fn env_layer() -> toml::Value {
    let mut v = toml::Value::Table(Default::default());
    let sections = ["limits", "sampling", "llama", "host", "advanced"];
    let keys = |s: &str| -> Vec<String> {
        let d = toml::Value::try_from(Settings::default())
            .unwrap_or_else(|_| toml::Value::Table(Default::default()));
        d.get(s)
            .and_then(|t| t.as_table())
            .map(|t| t.keys().cloned().collect())
            .unwrap_or_default()
    };
    for section in sections {
        for key in keys(section) {
            let name = format!("A2E_{}_{}", section.to_uppercase(), key.to_uppercase());
            if let Ok(text) = std::env::var(&name) {
                set_path(&mut v, &format!("{section}.{key}"), parse_value(&text));
            }
        }
    }
    for (name, path) in [
        ("A2E_MODELS_DIR", "host.models_dir"),
        ("A2E_OUTPUT_DIR", "host.output_dir"),
        ("A2E_LLAMA_SERVER", "llama.server"),
        ("A2E_SANDBOX", "host.sandbox_dir"),
        ("A2E_SANDBOX_TMP", "host.sandbox_tmp"),
    ] {
        if let Ok(text) = std::env::var(name) {
            set_path(&mut v, path, toml::Value::String(text));
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_llama_arguments_match_the_old_hard_coded_list() {
        let args = Llama::default().args(Path::new("m.gguf"), 1024, Path::new("s/slots"));
        let expected = "--host 127.0.0.1 --port 8197 --model m.gguf --ctx-size 1024 --threads 4 --n-gpu-layers 0 --parallel 1 --batch-size 128 --ubatch-size 64 --jinja --special --no-context-shift --cache-ram 0 --slots --slot-save-path s/slots";
        assert_eq!(args.join(" "), expected);
    }

    #[test]
    fn optional_flags_appear_only_when_changed_and_extra_args_come_last() {
        let l = Llama {
            flash_attn: "on".into(),
            cache_type_k: "q8_0".into(),
            mlock: true,
            mmap: false,
            chat_template: "chatml".into(),
            extra_args: vec!["--verbose".into()],
            ..Llama::default()
        };
        let args = l.args(Path::new("m"), 512, Path::new("s")).join(" ");
        for flag in [
            "--flash-attn on",
            "--cache-type-k q8_0",
            "--mlock",
            "--no-mmap",
            "--chat-template chatml",
        ] {
            assert!(args.contains(flag), "{flag} missing from {args}");
        }
        assert!(args.ends_with("--verbose"));
        assert!(!args.contains("--cache-type-v"));
    }

    #[test]
    fn validation_rejects_bad_values() {
        let mut s = Settings::default();
        s.llama.commit = "0".repeat(40);
        assert!(s.validate().is_ok());
        let mut bad = s.clone();
        bad.llama.threads = 0;
        assert!(bad.validate().is_err());
        let mut bad = s.clone();
        bad.sampling.top_p = 1.5;
        assert!(bad.validate().is_err());
        let mut bad = s.clone();
        bad.limits.context = vec![100];
        assert!(bad.validate().is_err());
        let mut bad = s;
        bad.llama.flash_attn = "maybe".into();
        assert!(bad.validate().is_err());
    }

    #[test]
    fn legacy_runtime_keys_move_to_their_sections() {
        let mut v: toml::Value =
            toml::from_str("[runtime]\nthreads = 2\nmemory_limit_mb = 900\n").unwrap();
        upgrade_legacy(&mut v, "test").unwrap();
        assert_eq!(v["llama"]["threads"].as_integer(), Some(2));
        assert_eq!(v["limits"]["memory_mb"].as_integer(), Some(900));
        let mut v: toml::Value = toml::from_str("[runtime]\nnope = 1\n").unwrap();
        assert!(upgrade_legacy(&mut v, "test").is_err());
    }

    #[test]
    fn set_values_parse_as_toml_or_fall_back_to_text() {
        assert_eq!(parse_value("512").as_integer(), Some(512));
        assert_eq!(parse_value("[1, 2]").as_array().map(Vec::len), Some(2));
        assert_eq!(parse_value("on").as_str(), Some("on"));
    }
}
