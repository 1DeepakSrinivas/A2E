//! Checks over files, tool state and the final answer.

use super::{super::check::CheckKind, valid_pattern};
use crate::{
    case::{tolerance, MatchMode, Number},
    eval::{
        asks_for, found, numbers, result, values_match, CheckResult, FailureClass, Observation,
    },
};
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeSet;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileCheck {
    pub path: String,
    pub eq: Option<String>,
    pub contains: Option<String>,
    pub regex: Option<String>,
    pub json_eq: Option<Value>,
    #[serde(default)]
    pub absent: bool,
    #[serde(default)]
    pub present: bool,
}

impl CheckKind for FileCheck {
    fn validate(&self, _allowed: &BTreeSet<&str>) -> Result<()> {
        let set = usize::from(self.eq.is_some())
            + usize::from(self.contains.is_some())
            + usize::from(self.regex.is_some())
            + usize::from(self.json_eq.is_some())
            + usize::from(self.absent)
            + usize::from(self.present);
        if set != 1 || self.path.is_empty() {
            bail!(
                "file check {} needs a path and exactly one matcher",
                self.path
            );
        }
        valid_pattern(&self.regex)
    }

    fn evaluate(&self, obs: &Observation) -> CheckResult {
        let file = obs.state["files"].get(&self.path);
        let body = file.and_then(Value::as_str);
        let passed = if self.absent {
            file.is_none()
        } else if self.present {
            file.is_some()
        } else if let Some(want) = &self.eq {
            body == Some(want.as_str())
        } else if let Some(part) = &self.contains {
            body.is_some_and(|b| b.contains(part.as_str()))
        } else if let Some(p) = &self.regex {
            body.is_some_and(|b| found(p, b))
        } else if let Some(want) = &self.json_eq {
            body.and_then(|b| serde_json::from_str::<Value>(b).ok())
                .is_some_and(|v| values_match(want, &v, MatchMode::Exact, tolerance()))
        } else {
            false
        };
        result(
            "file",
            format!("file {}", self.path),
            passed,
            FailureClass::StateMismatch,
        )
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StateCheck {
    pub path: String,
    pub eq: Value,
}

impl CheckKind for StateCheck {
    fn validate(&self, _allowed: &BTreeSet<&str>) -> Result<()> {
        if !self.path.starts_with('/') {
            bail!("state path {} must be a JSON pointer", self.path);
        }
        Ok(())
    }

    fn evaluate(&self, obs: &Observation) -> CheckResult {
        result(
            "state",
            format!("state {}", self.path),
            obs.state
                .pointer(&self.path)
                .is_some_and(|v| values_match(&self.eq, v, MatchMode::Exact, tolerance())),
            FailureClass::StateMismatch,
        )
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnswerCheck {
    pub contains: Option<String>,
    pub not_contains: Option<String>,
    pub regex: Option<String>,
    pub exact: Option<String>,
    pub number: Option<Number>,
    /// Topics a clarifying question may name; see `eval::asks_for`.
    pub asks: Option<Vec<String>>,
}

impl CheckKind for AnswerCheck {
    fn validate(&self, _allowed: &BTreeSet<&str>) -> Result<()> {
        let set = usize::from(self.contains.is_some())
            + usize::from(self.not_contains.is_some())
            + usize::from(self.regex.is_some())
            + usize::from(self.exact.is_some())
            + usize::from(self.number.is_some())
            + usize::from(self.asks.is_some());
        if set != 1 {
            bail!("answer check needs exactly one matcher");
        }
        if self
            .asks
            .as_ref()
            .is_some_and(|a| a.is_empty() || a.iter().any(|t| t.trim().is_empty()))
        {
            bail!("asks needs at least one non-empty topic");
        }
        valid_pattern(&self.regex)
    }

    fn evaluate(&self, obs: &Observation) -> CheckResult {
        let lower = obs.answer.to_lowercase();
        let passed = if let Some(part) = &self.contains {
            lower.contains(&part.to_lowercase())
        } else if let Some(part) = &self.not_contains {
            !lower.contains(&part.to_lowercase())
        } else if let Some(p) = &self.regex {
            found(p, obs.answer)
        } else if let Some(want) = &self.exact {
            obs.answer.trim() == want.trim()
        } else if let Some(n) = &self.number {
            numbers(obs.answer)
                .iter()
                .any(|x| (x - n.value).abs() <= n.tol * x.abs().max(n.value.abs()).max(1.))
        } else if let Some(topics) = &self.asks {
            asks_for(topics, obs.answer)
        } else {
            false
        };
        result("answer", "answer".into(), passed, FailureClass::BadAnswer)
    }
}
