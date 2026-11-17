//! Checks over the web pages and sources the model fetched.

use super::super::check::CheckKind;
use crate::eval::{result, string_list, urls_match, CheckResult, FailureClass, Observation};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FetchedUrlCheck {
    pub url: String,
}

impl CheckKind for FetchedUrlCheck {
    fn evaluate(&self, obs: &Observation) -> CheckResult {
        result(
            "fetched_url",
            format!("fetched_url {}", self.url),
            string_list(&obs.state["fetched_urls"]).any(|u| urls_match(u, &self.url)),
            FailureClass::MissingCall,
        )
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FetchedSourceCheck {
    pub id: String,
}

impl CheckKind for FetchedSourceCheck {
    fn evaluate(&self, obs: &Observation) -> CheckResult {
        result(
            "fetched_source",
            format!("fetched_source {}", self.id),
            string_list(&obs.state["fetched_sources"]).any(|s| s == self.id),
            FailureClass::MissingCall,
        )
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FetchedCountCheck {
    pub min: usize,
}

impl CheckKind for FetchedCountCheck {
    fn evaluate(&self, obs: &Observation) -> CheckResult {
        result(
            "fetched_count",
            format!("fetched_count {}", self.min),
            obs.state["fetched_sources"].as_array().map_or(0, Vec::len) >= self.min,
            FailureClass::MissingCall,
        )
    }
}
