//! One struct per check kind, grouped by what they inspect.

mod calls;
mod fetched;
mod output;

pub use calls::{CallCheck, CallOrderCheck, CallsAtMostCheck, NoCallCheck};
pub use fetched::{FetchedCountCheck, FetchedSourceCheck, FetchedUrlCheck};
pub use output::{AnswerCheck, FileCheck, StateCheck};

use anyhow::{Context, Result};
use regex::Regex;
use std::collections::BTreeSet;

/// Fails unless `tool` is in the case's allowed list.
fn known_tool(allowed: &BTreeSet<&str>, tool: &str) -> Result<()> {
    allowed
        .contains(tool)
        .then_some(())
        .with_context(|| format!("check references tool {tool} outside tools.allowed"))
}

fn valid_pattern(pattern: &Option<String>) -> Result<()> {
    match pattern {
        Some(p) => Regex::new(p)
            .map(|_| ())
            .with_context(|| format!("invalid regex {p}")),
        None => Ok(()),
    }
}
