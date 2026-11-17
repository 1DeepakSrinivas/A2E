//! Checks over the tool calls the model made.

use super::{super::check::CheckKind, known_tool};
use crate::{
    case::{tolerance, MatchMode},
    eval::{result, values_match, CheckResult, FailureClass, Observation},
};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeSet;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CallCheck {
    pub tool: String,
    pub args: Option<Value>,
    #[serde(default, rename = "match")]
    pub mode: MatchMode,
    pub min: Option<usize>,
    pub max: Option<usize>,
    #[serde(default = "tolerance")]
    pub tol: f64,
}

impl CheckKind for CallCheck {
    fn validate(&self, allowed: &BTreeSet<&str>) -> Result<()> {
        known_tool(allowed, &self.tool)
    }

    fn evaluate(&self, obs: &Observation) -> CheckResult {
        let named = obs.attempted.iter().filter(|c| c.name == self.tool).count();
        let hits = obs
            .calls
            .iter()
            .filter(|c| c.name == self.tool)
            .filter(|c| {
                self.args
                    .as_ref()
                    .is_none_or(|a| values_match(a, &c.arguments, self.mode, self.tol))
            })
            .count();
        let passed = hits >= self.min.unwrap_or(1) && self.max.is_none_or(|m| hits <= m);
        let class = if self.max.is_some_and(|m| hits > m) {
            FailureClass::ExtraCall
        } else if named > 0 {
            FailureClass::WrongArgs
        } else if obs.attempted.is_empty() {
            FailureClass::MissingCall
        } else {
            FailureClass::WrongTool
        };
        result("call", format!("call {}", self.tool), passed, class)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NoCallCheck {
    pub tool: String,
}

impl CheckKind for NoCallCheck {
    fn validate(&self, allowed: &BTreeSet<&str>) -> Result<()> {
        known_tool(allowed, &self.tool)
    }

    fn evaluate(&self, obs: &Observation) -> CheckResult {
        result(
            "no_call",
            format!("no_call {}", self.tool),
            !obs.attempted.iter().any(|c| c.name == self.tool),
            FailureClass::ExtraCall,
        )
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CallOrderCheck {
    pub tools: Vec<String>,
}

impl CheckKind for CallOrderCheck {
    fn validate(&self, allowed: &BTreeSet<&str>) -> Result<()> {
        self.tools.iter().try_for_each(|t| known_tool(allowed, t))
    }

    fn evaluate(&self, obs: &Observation) -> CheckResult {
        let mut next = self.tools.iter().peekable();
        for call in obs.calls {
            if next.peek().is_some_and(|t| **t == call.name) {
                next.next();
            }
        }
        result(
            "call_order",
            format!("call_order {}", self.tools.join(" > ")),
            next.peek().is_none(),
            FailureClass::WrongOrder,
        )
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CallsAtMostCheck {
    pub max: usize,
}

impl CheckKind for CallsAtMostCheck {
    fn evaluate(&self, obs: &Observation) -> CheckResult {
        result(
            "calls_at_most",
            format!("calls_at_most {}", self.max),
            obs.attempted.len() <= self.max,
            FailureClass::ExtraCall,
        )
    }
}
