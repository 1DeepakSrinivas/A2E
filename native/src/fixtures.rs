// fixtures.rs
// Typed task fixtures shared by case files and the built-in tool host
// Files, devices, pages, stores, clock, database seed, and injected faults

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Page {
    pub id: String,
    pub url: String,
    pub title: String,
    pub text: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Note {
    pub text: String,
    pub tag: Option<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Todo {
    pub title: String,
    #[serde(default)]
    pub done: bool,
}
/// Database seed as SQL text; `sql_file` is resolved to `sql` when a case loads.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DbSeed {
    pub sql: Option<String>,
    pub sql_file: Option<String>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FaultMode {
    FailN,
    RateLimit,
    SlowMs,
    OversizeBytes,
    Malformed,
}
/// Failure injected into one tool: `n` affected calls, `ms` delay, or `bytes` of padding.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Fault {
    pub tool: String,
    pub mode: FaultMode,
    #[serde(default = "one")]
    pub n: usize,
    #[serde(default)]
    pub ms: u64,
    #[serde(default)]
    pub bytes: usize,
}
fn one() -> usize {
    1
}
/// Initial task state; `custom` carries data for external tools untouched.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Fixtures {
    /// Name of the generated workspace tree copied into the task directory.
    pub sandbox: Option<String>,
    pub files: BTreeMap<String, String>,
    pub devices: BTreeMap<String, Value>,
    pub pages: Vec<Page>,
    pub kv: BTreeMap<String, String>,
    pub notes: Vec<Note>,
    pub todos: Vec<Todo>,
    pub clock: Option<String>,
    pub db: Option<DbSeed>,
    pub faults: Vec<Fault>,
    pub custom: BTreeMap<String, Value>,
}
