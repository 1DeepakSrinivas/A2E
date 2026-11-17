//! Runs one case from a fresh tool state to a scored outcome.

use super::{
    attempt::{Attempt, TaskExecution, Trace},
    outcome::{classify, failure_class, Outcome, Status},
};
use crate::{case::Case, config::WebMode, measure::Monitor, record::Turn, tools::Builtins};
use anyhow::{Context, Result};
use serde_json::json;
use std::{
    sync::atomic::Ordering,
    time::{Duration, Instant},
};

/// Removes a temporary task directory when the attempt ends.
pub(super) struct RemoveOnDrop(Option<std::path::PathBuf>);
impl Drop for RemoveOnDrop {
    fn drop(&mut self) {
        if let Some(p) = &self.0 {
            let _ = std::fs::remove_dir_all(p);
        }
    }
}
/// Checks the generated tree when any case uses it; returns its manifest.
pub(super) fn verify_sandbox(cases: &[Case]) -> Result<Option<crate::sandbox::Manifest>> {
    cases
        .iter()
        .any(|c| c.fixtures.sandbox.is_some())
        .then(|| crate::sandbox::verify(&crate::sandbox::default_dir()))
        .transpose()
}
/// One bounded attempt with its own tool state and trace.
pub fn task_run(x: TaskExecution<'_>) -> Result<Outcome> {
    let start = Instant::now();
    let deadline = start + Duration::from_secs(x.suite.budget.task_seconds);
    let mut out = Outcome::default();
    let trace = Trace::create(&x.dir.join("trace.jsonl"))?;
    let t = Instant::now();
    let sandboxed = x.case.fixtures.sandbox.is_some();
    let root = match std::env::var_os("A2E_SANDBOX_TMP").filter(|_| sandboxed) {
        Some(tmp) => std::path::PathBuf::from(tmp)
            .join(&crate::canonical_hash(&x.dir.to_string_lossy())?[..16]),
        None => x.dir.join("workspace"),
    };
    let _cleanup = RemoveOnDrop(
        std::env::var_os("A2E_SANDBOX_TMP")
            .filter(|_| sandboxed)
            .map(|_| root.clone()),
    );
    let mut tools = Builtins::new(root, x.suite.web_mode == WebMode::Live)?;
    if sandboxed {
        tools.set_sandbox_base(Some(crate::sandbox::default_dir().join("base")));
    }
    out.tool_setup_ms = t.elapsed().as_secs_f64() * 1000.;
    let monitor = Monitor::start(x.server_pid, x.runtime, x.cancel.clone())?;
    let stages = if x.variant.mode.staged() {
        x.case
            .stage_plan()
            .with_context(|| format!("case {} has no guided workflow", x.case.id))?
    } else {
        Vec::new()
    };
    let mut attempt = Attempt {
        x,
        start,
        deadline,
        out,
        trace,
        tools,
        monitor,
        stages,
        history: Vec::new(),
        stage: 0,
        stage_successes: 0,
        calls: Vec::new(),
        attempted: Vec::new(),
        last_call: None,
        repeats: 0,
        turn: Turn::default(),
    };
    // Partial costs survive tool or inference errors
    let process = attempt.drive();
    let Attempt {
        x,
        mut out,
        mut trace,
        tools,
        monitor,
        ..
    } = attempt;
    out.resources = monitor.finish();
    if let Err(e) = process {
        let reason = format!("{e:#}");
        out.status = classify(&reason);
        out.reason = Some(reason);
    }
    if out.resources.memory_exceeded {
        out.status = Status::ResourceLimit;
        out.reason = Some("memory_budget".into());
    }
    if x.runtime.power == "required" && out.resources.energy_j.is_none() {
        out.status = Status::InfrastructureFailure;
        out.reason = Some("required power samples unavailable".into());
    }
    if x.cancel.load(Ordering::Relaxed) {
        out.status = Status::Interrupted;
    }
    drop(tools);
    out.wall_ms = start.elapsed().as_secs_f64() * 1000.;
    out.tally();
    out.harness_other_ms = (out.wall_ms
        - out.http_ms
        - out.template_ms
        - out.tokenize_ms
        - out.parse_ms
        - out.validate_ms
        - out.tool_ms
        - out.tool_setup_ms)
        .max(0.);
    out.failure_class = failure_class(&out, x.case);
    trace.event(&json!({"event":"outcome","result":out}))?;
    trace.finish()?;
    Ok(out)
}
