//! Attempt outcomes: status, metrics, failure classes and the rules that map reasons to them.

use crate::{
    case::{Case, FinalAnswer},
    error::HarnessError,
    eval::{CheckResult, FailureClass},
    measure::Resources,
    record::{ToolEntry, Turn},
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Success,
    AwaitingJudge,
    ModelFailure,
    ResourceLimit,
    InfrastructureFailure,
    Interrupted,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Outcome {
    pub status: Status,
    pub reason: Option<String>,
    pub failure_class: Option<FailureClass>,
    pub answer: Option<String>,
    pub checks: Vec<CheckResult>,
    pub milestone_score: Option<f64>,
    pub attempted_calls: u64,
    pub abstained: Option<bool>,
    pub retrieval_success: Option<bool>,
    pub evidence: Vec<Value>,
    pub model_calls: u64,
    pub tool_invocations: u64,
    pub valid_tool_calls: u64,
    pub argument_errors: u64,
    pub repairs: u64,
    pub prompt_tokens: Option<u64>,
    pub generated_tokens: Option<u64>,
    pub cached_tokens: Option<u64>,
    pub prefill_ms: Option<f64>,
    pub decode_ms: Option<f64>,
    pub http_ms: f64,
    pub template_ms: f64,
    pub tokenize_ms: f64,
    pub parse_ms: f64,
    pub validate_ms: f64,
    pub tool_ms: f64,
    pub tool_setup_ms: f64,
    pub wall_ms: f64,
    pub harness_other_ms: f64,
    pub resources: Resources,
    #[serde(skip)]
    pub turns: Vec<Turn>,
    #[serde(skip)]
    pub tool_log: Vec<ToolEntry>,
    #[serde(skip)]
    pub final_state: Option<Value>,
}
impl Default for Outcome {
    fn default() -> Self {
        Self {
            status: Status::ModelFailure,
            reason: None,
            failure_class: None,
            answer: None,
            checks: Vec::new(),
            milestone_score: None,
            attempted_calls: 0,
            abstained: None,
            retrieval_success: None,
            evidence: Vec::new(),
            model_calls: 0,
            tool_invocations: 0,
            valid_tool_calls: 0,
            argument_errors: 0,
            repairs: 0,
            prompt_tokens: Some(0),
            generated_tokens: Some(0),
            cached_tokens: Some(0),
            prefill_ms: Some(0.),
            decode_ms: Some(0.),
            http_ms: 0.,
            template_ms: 0.,
            tokenize_ms: 0.,
            parse_ms: 0.,
            validate_ms: 0.,
            tool_ms: 0.,
            tool_setup_ms: 0.,
            wall_ms: 0.,
            harness_other_ms: 0.,
            resources: Resources::default(),
            turns: Vec::new(),
            tool_log: Vec::new(),
            final_state: None,
        }
    }
}
impl Outcome {
    /// Timing and token totals from the recorded turns and tool calls.
    pub(super) fn tally(&mut self) {
        let called = || self.turns.iter().filter(|t| t.called);
        let sum = |f: fn(&Turn) -> f64| self.turns.iter().map(f).sum::<f64>();
        self.prefill_ms = called().map(|t| t.prefill_ms).sum();
        self.decode_ms = called().map(|t| t.decode_ms).sum();
        self.prompt_tokens = called().map(|t| t.prompt_tokens).sum();
        self.generated_tokens = called().map(|t| t.generated_tokens).sum();
        self.cached_tokens = called().map(|t| t.cached_tokens).sum();
        self.http_ms = sum(|t| t.http_ms);
        self.template_ms = sum(|t| t.template_ms);
        self.tokenize_ms = sum(|t| t.tokenize_ms);
        self.parse_ms = sum(|t| t.parse_ms);
        self.validate_ms = self.tool_log.iter().map(|c| c.validate_ms).sum();
        self.tool_ms = self.tool_log.iter().map(|c| c.ms).sum();
    }
}
pub(super) fn answered(out: &Outcome, case: &Case) -> bool {
    out.answer.is_some() || case.scope.final_answer == FinalAnswer::Optional
}
/// Primary reason an attempt did not pass, for error analysis.
pub(super) fn failure_class(out: &Outcome, case: &Case) -> Option<FailureClass> {
    let reason = out.reason.as_deref().unwrap_or("");
    if matches!(out.status, Status::Success | Status::AwaitingJudge) {
        None
    } else if reason.starts_with("parse_failure") {
        Some(FailureClass::ParseFailure)
    } else if reason.starts_with("argument_error") {
        Some(FailureClass::ArgumentError)
    } else if reason == "repeated_action" {
        Some(FailureClass::RepeatedAction)
    } else if out.status == Status::ResourceLimit {
        Some(FailureClass::ResourceLimit)
    } else if matches!(
        out.status,
        Status::InfrastructureFailure | Status::Interrupted
    ) {
        Some(FailureClass::Infrastructure)
    } else if out.tool_log.is_empty() && case.scope.expected_calls.unwrap_or(0) > 0 {
        Some(FailureClass::NoToolUse)
    } else if out.tool_log.last().is_some_and(|t| !t.valid) {
        // The model stopped after a call that did not fit the schema
        Some(FailureClass::ArgumentError)
    } else if out.tool_log.last().is_some_and(|t| t.valid && !t.ok) {
        Some(FailureClass::ToolErrorAbandoned)
    } else {
        out.checks
            .iter()
            .find_map(|c| c.class)
            .or_else(|| (!answered(out, case)).then_some(FailureClass::NoFinalAnswer))
    }
}
/// True when the attempt hit a limit before any model call returned.
pub fn no_call_returned(out: &Outcome) -> bool {
    out.status == Status::ResourceLimit
        && out.model_calls > 0
        && out.turns.iter().all(|t| t.raw_text.is_none())
}
/// Status for an error: harness errors map directly; text from other sources is matched by keyword.
pub fn classify_error(e: &anyhow::Error) -> Status {
    match e.chain().find_map(|c| c.downcast_ref::<HarnessError>()) {
        Some(h) => h.status(),
        None => classify(&format!("{e:#}")),
    }
}
pub fn classify(reason: &str) -> Status {
    if reason.contains("interrupted") {
        Status::Interrupted
    } else if [
        "budget",
        "context",
        "timeout",
        "timed out",
        "generation_limit",
        "output_limit",
    ]
    .iter()
    .any(|s| reason.contains(s))
    {
        Status::ResourceLimit
    } else {
        Status::InfrastructureFailure
    }
}
