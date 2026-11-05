// tools/mod.rs
// Built-in tool host: shared task state, handler registry, and fault injection
// Domain modules register handlers by tool name

mod db;
mod files;
mod misc;
mod store;
mod text;
mod timeunits;
mod web;

pub use db::check_seed;
pub use files::safe_path;
pub use timeunits::parse_date;
pub use web::{html_text, public_ip, validate_url};

use crate::{
    fixtures::{Fault, FaultMode, Fixtures, Note, Todo},
    read_bounded,
    sdk::ToolHost,
};
use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::PathBuf,
    time::Duration,
};

type Handler = fn(&mut World, &Value) -> Result<Value>;
type Registry = BTreeMap<&'static str, Handler>;

pub fn words(s: &str) -> BTreeSet<String> {
    s.split(|c: char| !c.is_alphanumeric())
        .filter(|s| s.len() > 1)
        .map(|s| s.to_lowercase().trim_end_matches('s').to_string())
        .collect()
}
fn string<'a>(v: &'a Value, key: &str) -> Result<&'a str> {
    v[key]
        .as_str()
        .with_context(|| format!("missing string {key}"))
}

fn registry() -> Registry {
    let mut r = Registry::new();
    files::register(&mut r);
    text::register(&mut r);
    timeunits::register(&mut r);
    store::register(&mut r);
    db::register(&mut r);
    misc::register(&mut r);
    web::register(&mut r);
    r
}
/// Mutable state of one task, shared by every handler.
struct World {
    root: PathBuf,
    files: BTreeSet<String>,
    devices: BTreeMap<String, Value>,
    kv: BTreeMap<String, String>,
    notes: Vec<Note>,
    todos: Vec<Todo>,
    clock: String,
    db: Option<db::Db>,
    web: web::Web,
}
pub struct Builtins {
    world: World,
    handlers: Registry,
    faults: Vec<(Fault, usize)>,
}
impl Builtins {
    pub fn new(root: PathBuf, live: bool) -> Result<Self> {
        std::fs::create_dir_all(&root)?;
        Ok(Self {
            world: World {
                root,
                files: BTreeSet::new(),
                devices: BTreeMap::new(),
                kv: BTreeMap::new(),
                notes: Vec::new(),
                todos: Vec::new(),
                clock: String::new(),
                db: None,
                web: web::Web::new(live)?,
            },
            handlers: registry(),
            faults: Vec::new(),
        })
    }
    /// Names of every registered tool, sorted.
    pub fn names() -> Vec<&'static str> {
        registry().into_keys().collect()
    }
    /// The next injected fault that still has calls left for this tool.
    fn fault(&mut self, name: &str) -> Option<Fault> {
        self.faults
            .iter_mut()
            .find(|(f, seen)| f.tool == name && *seen < f.n)
            .map(|(f, seen)| {
                *seen += 1;
                f.clone()
            })
    }
}
impl ToolHost for Builtins {
    fn initialize(&mut self, args: Value) -> Result<Value> {
        let fixtures: Fixtures = match args.get("fixtures") {
            Some(v) if !v.is_null() => serde_json::from_value(v.clone())?,
            _ => Fixtures::default(),
        };
        let w = &mut self.world;
        w.files.clear();
        for (name, body) in &fixtures.files {
            let p = safe_path(&w.root, name)?;
            if let Some(parent) = p.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(p, body)?;
            w.files.insert(name.clone());
        }
        w.devices = fixtures.devices;
        w.kv = fixtures.kv;
        w.notes = fixtures.notes;
        w.todos = fixtures.todos;
        w.clock = fixtures
            .clock
            .unwrap_or_else(|| "2026-01-01T00:00:00Z".into());
        parse_date(&w.clock)?;
        w.db = fixtures
            .db
            .map(|d| db::Db::open(d.sql.as_deref().context("database seed needs sql")?))
            .transpose()?;
        w.web.load(fixtures.pages);
        self.faults = fixtures.faults.into_iter().map(|f| (f, 0)).collect();
        Ok(json!({"ready":true}))
    }
    fn call(&mut self, name: &str, a: Value) -> Result<Value> {
        let handler = *self
            .handlers
            .get(name)
            .with_context(|| format!("unknown tool {name}"))?;
        let fault = self.fault(name);
        match fault.as_ref().map(|f| f.mode) {
            Some(FaultMode::FailN) => bail!("temporary fixture error; retry is permitted"),
            Some(FaultMode::RateLimit) => bail!("rate limit exceeded; retry later"),
            Some(FaultMode::SlowMs) => {
                std::thread::sleep(Duration::from_millis(fault.as_ref().map_or(0, |f| f.ms)))
            }
            _ => (),
        }
        let mut out = handler(&mut self.world, &a)?;
        match fault {
            Some(f) if f.mode == FaultMode::Malformed => out = json!("@@garbled output@@"),
            Some(f) if f.mode == FaultMode::OversizeBytes => {
                if let Some(o) = out.as_object_mut() {
                    o.insert("padding".into(), json!("x".repeat(f.bytes)));
                }
            }
            _ => (),
        }
        Ok(out)
    }
    fn snapshot(&mut self) -> Result<Value> {
        let w = &self.world;
        let mut files = BTreeMap::new();
        for name in &w.files {
            let p = safe_path(&w.root, name)?;
            files.insert(
                name.clone(),
                String::from_utf8(read_bounded(std::fs::File::open(p)?, 65536)?)?,
            );
        }
        let mut state = w.web.snapshot();
        state["files"] = json!(files);
        state["devices"] = json!(w.devices);
        state["kv"] = json!(w.kv);
        state["notes"] = json!(w.notes);
        state["todos"] = json!(w.todos);
        state["clock"] = json!(w.clock);
        Ok(state)
    }
}

#[cfg(test)]
mod test_support {
    use super::*;

    /// Initialized host in a temporary directory.
    pub fn host(fixtures: Value) -> (Builtins, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let mut h = Builtins::new(dir.path().into(), false).unwrap();
        h.initialize(json!({"fixtures":fixtures})).unwrap();
        (h, dir)
    }
}
#[cfg(test)]
mod tests {
    use super::test_support::host;
    use super::*;

    fn faulted(fault: Value) -> (Builtins, tempfile::TempDir) {
        host(json!({"files":{"a.txt":"hello"},"faults":[fault]}))
    }
    fn read(h: &mut Builtins) -> Result<Value> {
        h.call("read_file", json!({"path":"a.txt"}))
    }

    #[test]
    fn fail_n_and_rate_limit_fail_the_first_calls_then_recover() {
        for (mode, text) in [("fail_n", "temporary"), ("rate_limit", "rate limit")] {
            let (mut h, _dir) = faulted(json!({"tool":"read_file","mode":mode,"n":2}));
            for _ in 0..2 {
                assert!(read(&mut h).unwrap_err().to_string().contains(text));
            }
            assert_eq!(read(&mut h).unwrap()["content"], "hello");
        }
    }
    #[test]
    fn slow_calls_are_delayed_and_other_tools_are_untouched() {
        let (mut h, _dir) = faulted(json!({"tool":"read_file","mode":"slow_ms","ms":120}));
        let t = std::time::Instant::now();
        read(&mut h).unwrap();
        assert!(t.elapsed() >= Duration::from_millis(120));
        let t = std::time::Instant::now();
        read(&mut h).unwrap();
        assert!(t.elapsed() < Duration::from_millis(100));
        let t = std::time::Instant::now();
        h.call("list_dir", json!({})).unwrap();
        assert!(t.elapsed() < Duration::from_millis(100));
    }
    #[test]
    fn malformed_and_oversize_faults_change_the_result_shape() {
        let (mut h, _dir) = faulted(json!({"tool":"read_file","mode":"malformed"}));
        assert!(read(&mut h).unwrap().is_string());
        assert!(read(&mut h).unwrap().is_object());
        let (mut h, _dir) =
            faulted(json!({"tool":"read_file","mode":"oversize_bytes","bytes":5000}));
        assert_eq!(
            read(&mut h).unwrap()["padding"].as_str().unwrap().len(),
            5000
        );
    }
    #[test]
    fn unknown_tools_and_bad_fixtures_are_rejected() {
        let (mut h, _dir) = host(json!({}));
        assert!(h.call("nope", json!({})).is_err());
        assert!(h.initialize(json!({"fixtures":{"flies":{}}})).is_err());
        assert!(h
            .initialize(json!({"fixtures":{"clock":"tomorrow"}}))
            .is_err());
        assert!(h
            .initialize(json!({"fixtures":{"db":{"sql":"NOT SQL"}}}))
            .is_err());
    }
    #[test]
    fn snapshot_includes_every_state_domain() {
        let (mut h, _dir) = host(json!({"devices":{"fan":{"setting":1}},"clock":"2026-02-02"}));
        h.call("set_setting", json!({"name":"fan","value":7}))
            .unwrap();
        let s = h.snapshot().unwrap();
        assert_eq!(s["devices"]["fan"]["setting"], 7);
        assert_eq!(s["clock"], "2026-02-02");
        for key in [
            "files",
            "kv",
            "notes",
            "todos",
            "fetched_sources",
            "web_requests",
        ] {
            assert!(s.get(key).is_some(), "{key}");
        }
    }
}
