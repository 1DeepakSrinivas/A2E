//! Per-turn and per-tool-call records kept in memory during an attempt.
//!
//! Written to SQLite after the attempt's wall time is fixed.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// How the model's response to one turn was understood.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParseStatus {
    Call,
    Final,
    Error,
    Truncated,
    #[default]
    Pending,
}
/// One model call; timings are milliseconds.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Turn {
    pub idx: usize,
    pub called: bool,
    pub visible_tools: usize,
    pub template_ms: f64,
    pub tokenize_ms: f64,
    pub http_ms: f64,
    pub prefill_ms: Option<f64>,
    pub decode_ms: Option<f64>,
    pub prompt_tokens: Option<u64>,
    pub cached_tokens: Option<u64>,
    pub generated_tokens: Option<u64>,
    pub parse_ms: f64,
    pub parse_status: ParseStatus,
    pub leading_text: bool,
    pub parse_error: Option<String>,
    pub raw_text: Option<String>,
}
impl Turn {
    pub fn new(idx: usize) -> Self {
        Self {
            idx,
            ..Default::default()
        }
    }
}
/// One tool call the model made, valid or not.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolEntry {
    pub turn: usize,
    pub tool: String,
    pub arguments: Value,
    pub valid: bool,
    pub ok: bool,
    pub validate_ms: f64,
    pub ms: f64,
    pub output_bytes: usize,
    pub output: Option<String>,
    pub output_truncated: bool,
    pub error: Option<String>,
}
