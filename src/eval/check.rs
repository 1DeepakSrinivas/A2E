//! The `Check` enum read from case files, and the trait every check kind implements.
//!
//! To add a check kind: write a struct that implements `CheckKind` in `checks/`, then add one
//! variant here and one line in `Check::kind`. The match has no catch-all arm, so the compiler
//! points at anything missed.

use super::{
    checks::{
        AnswerCheck, CallCheck, CallOrderCheck, CallsAtMostCheck, FetchedCountCheck,
        FetchedSourceCheck, FetchedUrlCheck, FileCheck, NoCallCheck, StateCheck,
    },
    CheckResult, Observation,
};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub trait CheckKind {
    /// Rejects a malformed check when its case loads. `allowed` is the case's tool list.
    fn validate(&self, _allowed: &BTreeSet<&str>) -> Result<()> {
        Ok(())
    }
    fn evaluate(&self, obs: &Observation) -> CheckResult;
}

/// One typed outcome check against observed calls, final state, or the answer.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Check {
    Call(CallCheck),
    NoCall(NoCallCheck),
    CallOrder(CallOrderCheck),
    CallsAtMost(CallsAtMostCheck),
    File(FileCheck),
    State(StateCheck),
    Answer(AnswerCheck),
    FetchedUrl(FetchedUrlCheck),
    FetchedSource(FetchedSourceCheck),
    FetchedCount(FetchedCountCheck),
}

impl Check {
    fn kind(&self) -> &dyn CheckKind {
        match self {
            Self::Call(c) => c,
            Self::NoCall(c) => c,
            Self::CallOrder(c) => c,
            Self::CallsAtMost(c) => c,
            Self::File(c) => c,
            Self::State(c) => c,
            Self::Answer(c) => c,
            Self::FetchedUrl(c) => c,
            Self::FetchedSource(c) => c,
            Self::FetchedCount(c) => c,
        }
    }

    pub fn validate(&self, allowed: &BTreeSet<&str>) -> Result<()> {
        self.kind().validate(allowed)
    }

    pub fn evaluate(&self, obs: &Observation) -> CheckResult {
        self.kind().evaluate(obs)
    }
}
