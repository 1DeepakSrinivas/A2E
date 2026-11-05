// rubric.rs
// Judge rubrics defined in TOML files
// Dimension descriptors, verdict thresholds, response schema, and score validation

use crate::canonical_hash;
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{collections::BTreeMap, path::Path};

pub const SCALE: u64 = 4;

/// Total below `fail_below` fails; `pass_total` with every dimension at `pass_each` passes.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Thresholds {
    pub fail_below: u64,
    pub pass_total: u64,
    pub pass_each: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Dimension {
    pub name: String,
    pub descriptors: Vec<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Rubric {
    pub name: String,
    pub version: u32,
    pub instructions: String,
    pub guidance: String,
    pub verdict: Thresholds,
    pub dimensions: Vec<Dimension>,
}
impl Rubric {
    pub fn load(path: &Path) -> Result<Self> {
        let r: Self = toml::from_str(&std::fs::read_to_string(path)?)
            .with_context(|| format!("rubric {}", path.display()))?;
        r.validate()?;
        if path.file_stem().and_then(|s| s.to_str()) != Some(r.name.as_str()) {
            bail!("rubric {} must be saved as {}.toml", r.name, r.name);
        }
        Ok(r)
    }
    /// Every rubric in a directory, keyed by name.
    pub fn load_dir(dir: &Path) -> Result<BTreeMap<String, Self>> {
        let mut out = BTreeMap::new();
        for entry in std::fs::read_dir(dir).with_context(|| format!("rubrics {}", dir.display()))? {
            let path = entry?.path();
            if path.extension().is_some_and(|e| e == "toml") {
                let r = Self::load(&path)?;
                out.insert(r.name.clone(), r);
            }
        }
        Ok(out)
    }
    fn validate(&self) -> Result<()> {
        let max = self.maximum();
        let t = &self.verdict;
        if self.dimensions.is_empty()
            || self.name.is_empty()
            || self
                .dimensions
                .iter()
                .any(|d| d.name.is_empty() || d.descriptors.len() != SCALE as usize + 1)
            || t.fail_below > t.pass_total
            || t.pass_total > max
            || t.pass_each > SCALE
        {
            bail!("rubric {} has invalid dimensions or thresholds", self.name);
        }
        let mut names = std::collections::BTreeSet::new();
        if !self.dimensions.iter().all(|d| names.insert(&d.name)) {
            bail!("rubric {} repeats a dimension", self.name);
        }
        Ok(())
    }
    pub fn maximum(&self) -> u64 {
        SCALE * self.dimensions.len() as u64
    }
    /// Stored with each judgment so a rubric edit is visible.
    pub fn label(&self) -> String {
        format!("{}-v{}", self.name, self.version)
    }
    pub fn hash(&self) -> Result<String> {
        canonical_hash(self)
    }
    /// System message: instructions, descriptor table, then guidance.
    pub fn prompt(&self) -> String {
        let mut table =
            String::from("| Dimension | 0 | 1 | 2 | 3 | 4 |\n|---|---|---|---|---|---|\n");
        for d in &self.dimensions {
            table.push_str(&format!("| {} | {} |\n", d.name, d.descriptors.join(" | ")));
        }
        format!(
            "{}\n{}\n{}",
            self.instructions.trim(),
            table,
            self.guidance.trim()
        )
    }
    /// JSON Schema for the judge response, generated from the dimensions.
    pub fn schema(&self) -> Value {
        let names = self.dimensions.iter().map(|d| &d.name).collect::<Vec<_>>();
        let dim = json!({"type":"object","properties":{"score":{"type":"integer","minimum":0,"maximum":SCALE},"reason":{"type":"string"},"evidence_ids":{"type":"array","items":{"type":"string"}}},"required":["score","reason","evidence_ids"],"additionalProperties":false});
        let props = names
            .iter()
            .map(|n| ((*n).clone(), dim.clone()))
            .collect::<serde_json::Map<_, _>>();
        json!({"type":"object","properties":{
            "dimensions":{"type":"object","properties":props,"required":names,"additionalProperties":false},
            "unsupported_claims":{"type":"array","items":{"type":"object","properties":{"claim":{"type":"string"},"reason":{"type":"string"},"central":{"type":"boolean"}},"required":["claim","reason","central"],"additionalProperties":false}},
            "insufficient_evidence":{"type":"boolean"},"insufficient_evidence_reason":{"type":"string"}
        },"required":["dimensions","unsupported_claims","insufficient_evidence","insufficient_evidence_reason"],"additionalProperties":false})
    }
    /// Schema validation, evidence-ID checks, and the rubric verdict.
    pub fn assess(&self, response: &Value, packet: &Value) -> Result<Value> {
        let validator =
            jsonschema::validator_for(&self.schema()).map_err(|e| anyhow::anyhow!("{e}"))?;
        validator
            .validate(response)
            .map_err(|e| anyhow::anyhow!("invalid judge response: {e}"))?;
        let ids = packet["evidence"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|v| v["source_id"].as_str())
            .collect::<std::collections::HashSet<_>>();
        let mut scores = Vec::new();
        for d in &self.dimensions {
            let item = &response["dimensions"][&d.name];
            let text = item["reason"].as_str().unwrap_or_default();
            if text.trim().is_empty() {
                bail!("judge omitted score explanation");
            }
            for id in item["evidence_ids"].as_array().into_iter().flatten() {
                if !id.as_str().is_some_and(|id| ids.contains(id)) {
                    bail!("judge fabricated an evidence identifier");
                }
            }
            scores.push(item["score"].as_u64().unwrap_or(0));
        }
        let total: u64 = scores.iter().sum();
        let central = response["unsupported_claims"]
            .as_array()
            .into_iter()
            .flatten()
            .any(|v| v["central"] == true);
        let t = &self.verdict;
        let verdict = if response["insufficient_evidence"] == true {
            "unscored"
        } else if central || total < t.fail_below {
            "fail"
        } else if total >= t.pass_total && scores.iter().all(|s| *s >= t.pass_each) {
            "pass"
        } else {
            "partial"
        };
        Ok(json!({"total":total,"maximum":self.maximum(),"verdict":verdict,"assessment":response}))
    }
}
