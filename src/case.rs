//! Benchmark cases defined in TOML files.
//!
//! Typed scope, judge handoff, validation, and directory loading. Checks live in `eval`.

use crate::{config::Stage, eval::Check, fixtures::Fixtures, tools};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Category {
    Simple,
    Multiple,
    Irrelevance,
    MissingParam,
    Chain,
    Recovery,
    Robustness,
    Web,
    Files,
}
impl Category {
    /// Categories that need no hand-written workflow stages.
    pub fn single_step(self) -> bool {
        matches!(
            self,
            Self::Simple | Self::Multiple | Self::Irrelevance | Self::MissingParam
        )
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Split {
    Dev,
    Heldout,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinalAnswer {
    #[default]
    Required,
    Optional,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MatchMode {
    #[default]
    Exact,
    Subset,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Handoff {
    Prompt,
    Answer,
    Calls,
    Evidence,
    Files,
    StateDiff,
    RequiredFacts,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Scope {
    pub prompt: String,
    pub max_model_calls: Option<usize>,
    pub expected_calls: Option<usize>,
    #[serde(default)]
    pub final_answer: FinalAnswer,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToolSet {
    pub allowed: Vec<String>,
    #[serde(default)]
    pub distractors: usize,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Output {
    #[serde(default)]
    pub files: Vec<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct JudgeSpec {
    pub enabled: bool,
    pub rubric: String,
    pub handoff: Vec<Handoff>,
    pub required_facts: Vec<String>,
}
impl Default for JudgeSpec {
    fn default() -> Self {
        Self {
            enabled: false,
            rubric: "retrieval_summary".into(),
            handoff: vec![
                Handoff::Prompt,
                Handoff::RequiredFacts,
                Handoff::Evidence,
                Handoff::Answer,
            ],
            required_facts: Vec::new(),
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Golden {
    pub tool: String,
    #[serde(default)]
    pub args: Value,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Number {
    pub value: f64,
    #[serde(default = "tolerance")]
    pub tol: f64,
}
pub fn tolerance() -> f64 {
    1e-9
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Case {
    pub id: String,
    pub category: Category,
    #[serde(default)]
    pub tags: Vec<String>,
    pub split: Split,
    #[serde(default)]
    pub description: String,
    pub scope: Scope,
    pub tools: ToolSet,
    #[serde(default, rename = "stage")]
    pub stages: Vec<Stage>,
    #[serde(default)]
    pub fixtures: Fixtures,
    #[serde(rename = "check")]
    pub checks: Vec<Check>,
    #[serde(default)]
    pub output: Output,
    #[serde(default)]
    pub judge: JudgeSpec,
    #[serde(default)]
    pub golden: Vec<Golden>,
    pub golden_answer: Option<String>,
}
impl Case {
    pub fn validate(&self, registry: &BTreeSet<&str>) -> Result<()> {
        let id = &self.id;
        if id.is_empty()
            || !id
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_')
        {
            bail!("invalid case id {id}");
        }
        let fail = |m: &str| -> Result<()> { bail!("case {id}: {m}") };
        if self.scope.prompt.trim().is_empty() {
            return fail("empty prompt");
        }
        if self.scope.max_model_calls == Some(0) {
            return fail("max_model_calls must be positive");
        }
        if self.tools.allowed.is_empty() {
            return fail("tools.allowed is empty");
        }
        if self
            .fixtures
            .sandbox
            .as_deref()
            .is_some_and(|p| p != "workspace")
        {
            return fail("unknown sandbox profile");
        }
        let mut allowed = BTreeSet::new();
        for t in &self.tools.allowed {
            if !registry.contains(t.as_str()) {
                return fail(&format!("unknown tool {t}"));
            }
            if !allowed.insert(t.as_str()) {
                return fail(&format!("duplicate tool {t}"));
            }
        }
        if self.tools.distractors > registry.len() - allowed.len() {
            return fail("more distractors than registered tools");
        }
        for s in &self.stages {
            if s.successes == 0 || s.tools.iter().any(|t| !allowed.contains(t.as_str())) {
                return fail("invalid workflow stage");
            }
        }
        if self.checks.is_empty() {
            return fail("no outcome checks");
        }
        for c in &self.checks {
            c.validate(&allowed).with_context(|| format!("case {id}"))?;
        }
        self.validate_fixtures(&allowed)?;
        if self.judge.enabled && (self.judge.rubric.is_empty() || self.judge.handoff.is_empty()) {
            return fail("judge needs a rubric and handoff fields");
        }
        for g in &self.golden {
            if !allowed.contains(g.tool.as_str()) {
                return fail(&format!("golden call uses unknown tool {}", g.tool));
            }
        }
        Ok(())
    }
    fn validate_fixtures(&self, allowed: &BTreeSet<&str>) -> Result<()> {
        let (id, f) = (&self.id, &self.fixtures);
        for name in f.files.keys() {
            tools::safe_path(Path::new("."), name)
                .with_context(|| format!("case {id}: fixture file {name}"))?;
        }
        if let Some(clock) = &f.clock {
            tools::parse_date(clock).with_context(|| format!("case {id}: fixture clock"))?;
        }
        if let Some(db) = &f.db {
            let sql = db.sql.as_deref().with_context(|| {
                format!("case {id}: database seed needs sql or a resolvable sql_file")
            })?;
            tools::check_seed(sql).with_context(|| format!("case {id}"))?;
        }
        for fault in &f.faults {
            if !allowed.contains(fault.tool.as_str()) {
                bail!("case {id}: fault on {} outside tools.allowed", fault.tool);
            }
        }
        Ok(())
    }
    /// Workflow for guided runs; None when the case cannot run guided.
    pub fn stage_plan(&self) -> Option<Vec<Stage>> {
        if !self.stages.is_empty() {
            Some(self.stages.clone())
        } else if self.category.single_step() {
            Some(vec![Stage {
                instruction: "Complete the user's task.".into(),
                tools: self.tools.allowed.clone(),
                successes: 1,
            }])
        } else {
            None
        }
    }
    pub fn from_toml(text: &str) -> Result<Self> {
        Ok(toml::from_str(text)?)
    }
}

fn collect(path: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    if path.is_dir() {
        let mut entries = std::fs::read_dir(path)?
            .map(|e| e.map(|e| e.path()))
            .collect::<std::io::Result<Vec<_>>>()?;
        entries.sort();
        for e in entries {
            collect(&e, out)?;
        }
    } else if path.extension().is_some_and(|e| e == "toml") {
        out.push(path.to_owned());
    }
    Ok(())
}
/// Every case under a file or directory, in path order, with unique IDs.
pub fn load_all(selectors: &[PathBuf]) -> Result<Vec<Case>> {
    let mut files = Vec::new();
    for s in selectors {
        if !s.exists() {
            bail!("case path {} does not exist", s.display());
        }
        collect(s, &mut files)?;
    }
    let mut ids = BTreeSet::new();
    let mut cases = Vec::with_capacity(files.len());
    for f in files {
        let mut case = Case::from_toml(&std::fs::read_to_string(&f)?)
            .with_context(|| format!("{}", f.display()))?;
        if let Some(seed) = case.fixtures.db.as_mut() {
            if let Some(file) = seed.sql_file.take() {
                let dir = f.parent().unwrap_or(Path::new("."));
                let path = dir.join(&file);
                seed.sql =
                    Some(std::fs::read_to_string(&path).with_context(|| {
                        format!("{}: sql_file {}", f.display(), path.display())
                    })?);
            }
        }
        if !ids.insert(case.id.clone()) {
            bail!("duplicate case id {}", case.id);
        }
        cases.push(case);
    }
    Ok(cases)
}
/// Counts by category and split with the tool range, for `cases list`.
pub fn summary(cases: &[Case]) -> Value {
    let mut by: BTreeMap<Category, [usize; 2]> = BTreeMap::new();
    for c in cases {
        by.entry(c.category).or_default()[usize::from(c.split == Split::Heldout)] += 1;
    }
    let rows = by
        .iter()
        .map(|(c, [dev, heldout])| json!({"category":c,"dev":dev,"heldout":heldout}))
        .collect::<Vec<_>>();
    let tags = cases
        .iter()
        .flat_map(|c| c.tags.iter())
        .collect::<BTreeSet<_>>();
    json!({"total":cases.len(),"categories":rows,"tags":tags})
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASE: &str = r#"
id = "t-1"
category = "simple"
split = "dev"
[scope]
prompt = "Add 1 and 2."
[tools]
allowed = ["calculate"]
[[check]]
kind = "call"
tool = "calculate"
args = { operation = "add", a = 1, b = 2 }
"#;
    fn registry() -> BTreeSet<&'static str> {
        ["calculate", "read_file"].into()
    }

    #[test]
    fn minimal_case_parses_and_validates() {
        let c = Case::from_toml(BASE).unwrap();
        assert!(c.validate(&registry()).is_ok());
        assert_eq!(c.scope.final_answer, FinalAnswer::Required);
        assert_eq!(c.stage_plan().unwrap().len(), 1);
    }
    #[test]
    fn unknown_fields_and_values_are_rejected() {
        assert!(Case::from_toml(&format!("{BASE}\nextra = 1")).is_err());
        assert!(Case::from_toml(&BASE.replace("simple", "easy")).is_err());
    }
    #[test]
    fn validation_catches_cross_reference_errors() {
        let bad_tool = BASE.replace("allowed = [\"calculate\"]", "allowed = [\"nope\"]");
        assert!(Case::from_toml(&bad_tool)
            .unwrap()
            .validate(&registry())
            .is_err());
        let outside = BASE.replace("tool = \"calculate\"", "tool = \"read_file\"");
        assert!(Case::from_toml(&outside)
            .unwrap()
            .validate(&registry())
            .is_err());
        let regex = format!("{BASE}\n[[check]]\nkind = \"answer\"\nregex = \"(\"");
        assert!(Case::from_toml(&regex)
            .unwrap()
            .validate(&registry())
            .is_err());
        let two = format!("{BASE}\n[[check]]\nkind = \"answer\"\ncontains = \"a\"\nexact = \"a\"");
        assert!(Case::from_toml(&two)
            .unwrap()
            .validate(&registry())
            .is_err());
    }
    #[test]
    fn multi_step_cases_need_stages_for_guided_runs() {
        let chain = BASE.replace("simple", "chain");
        assert!(Case::from_toml(&chain).unwrap().stage_plan().is_none());
        let staged = format!("{chain}\n[[stage]]\ninstruction = \"go\"\ntools = [\"calculate\"]");
        assert_eq!(
            Case::from_toml(&staged)
                .unwrap()
                .stage_plan()
                .unwrap()
                .len(),
            1
        );
    }
}
