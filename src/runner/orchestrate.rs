//! Planning and the serial study run: study definition, resume, grouping and upload.

use super::{
    attempt::TaskExecution,
    load::{launch, LoadRequest, Loaded, SETTLE},
    outcome::{classify_error, no_call_returned, Outcome, Status},
    task::{task_run, verify_sandbox},
};
use crate::{
    adapter::Adapter,
    case::Case,
    config::{Catalog, Config, Model, Suite, Variant},
    error::HarnessError,
    inference::Server,
    progress::Progress,
    store::{Phase, RunLock},
    upload::{
        payload::{self, RunMeta},
        Resume, Uploader,
    },
};
use anyhow::{bail, Result};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{SystemTime, UNIX_EPOCH},
};

pub fn plan(config: &Config, suite: &Suite, cases: &[Case], catalog: &Catalog) -> Result<Value> {
    let registry = catalog
        .tools
        .iter()
        .map(|t| t.name.as_str())
        .collect::<BTreeSet<_>>();
    for case in cases {
        case.validate(&registry)?;
    }
    verify_sandbox(cases)?;
    let guided_unavailable = cases.iter().filter(|c| c.stage_plan().is_none()).count();
    let mut groups = Vec::new();
    for m in &suite.models {
        config.model(m)?;
        for context in &suite.contexts {
            for variant in &suite.variants {
                let n = if variant.mode.staged() {
                    cases.len() - guided_unavailable
                } else {
                    cases.len()
                };
                groups.push(json!({"model":m,"context":context,"variant":variant,"cases":n,"not_applicable":cases.len()-n,"warmup_attempts":n*suite.warmup,"measured_attempts":n*suite.repeats}));
            }
        }
    }
    Ok(
        json!({"suite":suite.name,"split":suite.split,"web_mode":suite.web_mode,"budget":suite.budget,"groups":groups}),
    )
}
/// One line of the run log for a finished attempt, with failure details and the trace path.
pub fn attempt_line(meta: &RunMeta, out: &Outcome) -> Value {
    json!({
        "run": payload::run_id(meta.study, meta.job, meta.attempt),
        "case": meta.case.id,
        "model": meta.model,
        "context": meta.context,
        "variant": meta.variant.name,
        "phase": crate::store::tag(&meta.phase),
        "status": out.status,
        "reason": out.reason,
        "failure_class": out.failure_class,
        "failed_checks": out.checks.iter().filter(|check| !check.passed).collect::<Vec<_>>(),
        "trace": meta.trace,
    })
}
/// Settings that change how a run proceeds but not what it measures.
#[derive(Clone, Copy, Debug, Default)]
pub struct RunOptions {
    pub retry_failed: bool,
    /// Run with a model file that is not on disk; its attempts fail and its hash is recorded as null.
    pub allow_missing_model: bool,
}

/// Commit and tree state of the checkout, so a result can be traced to its code.
pub(super) fn git_state() -> Value {
    let git = |args: &[&str]| {
        std::process::Command::new("git")
            .args(args)
            .output()
            .ok()
            .filter(|o| o.status.success())
    };
    let Some(head) = git(&["rev-parse", "HEAD"]) else {
        return json!({"sha":null,"dirty":null});
    };
    let dirty = git(&["status", "--porcelain"]).map(|o| !o.stdout.is_empty());
    json!({"sha":String::from_utf8_lossy(&head.stdout).trim(),"dirty":dirty})
}

/// Hash of the inference binary; null when it cannot be read, which `launch` reports later.
pub(super) fn llama_server_hash(config: &Config) -> Value {
    crate::hash_file(&crate::config::expand_path(&config.settings.llama.server))
        .map_or(Value::Null, |h| json!(h))
}

/// Version of the study definition layout, bumped when its fields change.
pub(super) const DEFINITION_SCHEMA: u32 = 1;

/// Everything a study run reads, plus the stores it writes to.
struct Study<'a> {
    config: &'a Config,
    suite: &'a Suite,
    cases: &'a [Case],
    catalog: &'a Catalog,
    options: RunOptions,
    cancel: Arc<AtomicBool>,
    progress: &'a Progress,
    root: PathBuf,
    id: String,
    resume: Resume,
    uploader: Uploader,
    counts: Counts,
}

#[derive(Default)]
struct Counts {
    completed: usize,
    skipped: usize,
    not_applicable: usize,
}

/// One loaded model at one context size.
struct Group<'a> {
    model: &'a Model,
    ctx: usize,
    dir: PathBuf,
    launched: Result<(Server, Loaded)>,
    load_id: String,
    /// Consecutive attempts in which no model call returned.
    stalled: usize,
    /// Set when a call may still be running on the server.
    busy: bool,
}

/// One attempt to run: a case under a variant, repeat and phase.
struct Job<'a> {
    case: &'a Case,
    variant: &'a Variant,
    repeat: usize,
    phase: Phase,
}

enum Flow {
    Continue,
    Interrupted,
}

/// Fails before any work when a model file is missing and the run did not allow that.
fn require_model_files(config: &Config, suite: &Suite, options: RunOptions) -> Result<()> {
    if options.allow_missing_model {
        return Ok(());
    }
    for id in &suite.models {
        let path = config.model_path(config.model(id)?);
        if !path.exists() {
            bail!(
                "model file {} is missing. Run: a2e models fetch (or pass --allow-missing-model)",
                path.display()
            );
        }
    }
    Ok(())
}

/// Model files with their hashes; fails when a file differs from its pinned hash.
fn artifacts(config: &Config, suite: &Suite) -> Result<BTreeMap<String, Value>> {
    let mut artifacts = BTreeMap::new();
    for id in &suite.models {
        let m = config.model(id)?;
        let path = config.model_path(m);
        let hash = if path.exists() {
            let h = crate::hash_file(&path)?;
            if m.sha256.as_ref().is_some_and(|s| s != &h) {
                bail!("artifact hash mismatch for {id}");
            }
            Some(h)
        } else {
            None
        };
        artifacts.insert(id.clone(), json!({"model":m,"sha256":hash}));
    }
    Ok(artifacts)
}

/// The study definition. Its hash is the study id, so anything that changes results belongs here.
fn definition(
    config: &Config,
    suite: &Suite,
    cases: &[Case],
    catalog: &Catalog,
    sandbox: Option<&crate::sandbox::Manifest>,
) -> Result<Value> {
    let machine = crate::measure::machine(&config.settings)?;
    let mut definition = json!({
        "suite": suite,
        "cases": cases,
        "tools": catalog,
        "settings": config.settings,
        "artifacts": artifacts(config, suite)?,
        "device": machine["device"],
        "kernel": machine["kernel"],
        "arch": machine["arch"],
        "total_memory_bytes": machine["total_memory_bytes"],
        "runtime_version": env!("CARGO_PKG_VERSION"),
        "parse_rules": crate::adapter::parse_rules(),
        "binary_sha256": crate::hash_file(&std::env::current_exe()?)?,
        "schema_version": DEFINITION_SCHEMA,
        "git": git_state(),
        "llama_server_sha256": llama_server_hash(config),
    });
    if let Some(m) = sandbox {
        definition["sandbox"] = json!({"version":m.version,"seed":m.seed,"sha256":m.sha256});
    }
    Ok(definition)
}

/// Serial study execution with one loaded model at a time.
pub fn run(
    config: &Config,
    suite: &Suite,
    cases: &[Case],
    catalog: &Catalog,
    options: RunOptions,
    cancel: Arc<AtomicBool>,
    progress: &Progress,
) -> Result<Value> {
    plan(config, suite, cases, catalog)?;
    require_model_files(config, suite, options)?;
    let sandbox = verify_sandbox(cases)?;
    let root = crate::config::expand_path(&config.settings.host.output_dir);
    let _lock = RunLock::acquire(&root)?;
    for w in crate::measure::preflight_warnings(&config.settings) {
        eprintln!("warning: {w}");
    }
    let definition = definition(config, suite, cases, catalog, sandbox.as_ref())?;
    let id = crate::canonical_hash(&definition)?;
    let mut study = Study {
        config,
        suite,
        cases,
        catalog,
        options,
        cancel,
        progress,
        resume: Resume::open(&root.join("resume.jsonl"))?,
        uploader: Uploader::open(&root)?,
        root,
        id,
        counts: Counts::default(),
    };
    study
        .uploader
        .submit(vec![payload::study(&study.id, &definition, cases)?])?;
    let mut interrupted = false;
    'groups: for model_id in &suite.models {
        for ctx in &suite.contexts {
            if let Flow::Interrupted = study.run_group(config.model(model_id)?, *ctx)? {
                interrupted = true;
                break 'groups;
            }
        }
    }
    study.finish(interrupted)
}

impl Study<'_> {
    fn finish(mut self, interrupted: bool) -> Result<Value> {
        self.progress.finish();
        let upload = self.uploader.flush()?;
        let Counts {
            completed,
            skipped,
            not_applicable,
        } = self.counts;
        let mut out = json!({"study":self.id,"completed":completed,"skipped":skipped,"not_applicable":not_applicable,"upload":upload});
        if interrupted {
            out["interrupted"] = json!(true);
        }
        Ok(out)
    }

    fn load(
        &mut self,
        model: &Model,
        ctx: usize,
        dir: &Path,
    ) -> Result<(Result<(Server, Loaded)>, String)> {
        launch(
            &LoadRequest {
                config: self.config,
                model,
                ctx,
                group: dir,
                study: &self.id,
                cancel: &self.cancel,
            },
            &mut self.resume,
            &mut self.uploader,
        )
    }

    fn run_group(&mut self, model: &Model, ctx: usize) -> Result<Flow> {
        let dir = self.root.join(&self.id[..16]).join(format!(
            "{}-{ctx}-{}",
            model.id,
            SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
        ));
        std::fs::create_dir_all(&dir)?;
        let (launched, load_id) = self.load(model, ctx, &dir)?;
        self.progress.set_stage(format!("{} @ {ctx}", model.id));
        let mut group = Group {
            model,
            ctx,
            dir,
            launched,
            load_id,
            stalled: 0,
            busy: false,
        };
        let suite = self.suite;
        for variant in &suite.variants {
            for repeat in 0..suite.warmup + suite.repeats {
                let phase = if repeat < suite.warmup {
                    Phase::Warmup
                } else {
                    Phase::Measured
                };
                for case in self.cases {
                    let job = Job {
                        case,
                        variant,
                        repeat,
                        phase,
                    };
                    if let Flow::Interrupted = self.run_attempt(&mut group, &job)? {
                        return Ok(Flow::Interrupted);
                    }
                }
            }
        }
        Ok(Flow::Continue)
    }

    fn run_attempt(&mut self, group: &mut Group, job: &Job) -> Result<Flow> {
        if self.cancel.load(Ordering::Relaxed) {
            return Ok(Flow::Interrupted);
        }
        let (case, variant) = (job.case, job.variant);
        if variant.mode.staged() && case.stage_plan().is_none() {
            self.counts.not_applicable += 1;
            self.progress.advance();
            return Ok(Flow::Continue);
        }
        let key = crate::canonical_hash(
            &json!({"model":group.model.id,"ctx":group.ctx,"variant":variant,"repeat":job.repeat,"case":case.id}),
        )?;
        if !self
            .resume
            .should_run(&self.id, &key, self.options.retry_failed)
        {
            self.counts.skipped += 1;
            self.progress.advance();
            return Ok(Flow::Continue);
        }
        let dir = group.dir.join(&key[..16]);
        std::fs::create_dir_all(&dir)?;
        let attempt = self.resume.next_attempt(&self.id, &key);
        let skip = self.suite.budget.stall_attempts > 0
            && group.stalled >= self.suite.budget.stall_attempts;
        if !skip {
            self.ensure_idle_server(group)?;
        }
        let out = self.execute(group, job, &dir, skip);
        if !skip {
            if no_call_returned(&out) {
                group.stalled += 1;
                group.busy = true;
            } else if out.model_calls > 0 {
                group.stalled = 0;
                group.busy = out.status == Status::ResourceLimit;
            }
        }
        self.record(group, job, &key, attempt, &dir, &out)?;
        Ok(Flow::Continue)
    }

    /// A busy slot defers the cache erase, so wait for it or start a clean server.
    fn ensure_idle_server(&mut self, group: &mut Group) -> Result<()> {
        let ready = group.launched.as_ref().is_ok_and(|(server, _)| {
            (!group.busy || server.client.settle(SETTLE))
                && (server.client.reset().is_ok()
                    || (server.client.settle(SETTLE) && server.client.reset().is_ok()))
        });
        if !ready && group.launched.is_ok() {
            eprintln!("warning: inference server did not become idle; restarting it");
            // The old process must release the port first
            drop(std::mem::replace(
                &mut group.launched,
                Err(anyhow::anyhow!("restarting")),
            ));
            (group.launched, group.load_id) = self.load(group.model, group.ctx, &group.dir)?;
        }
        group.busy = false;
        Ok(())
    }

    /// Runs the task, turning any error into an outcome so the attempt is still recorded.
    fn execute(&self, group: &Group, job: &Job, dir: &Path, skip: bool) -> Outcome {
        let result = match &group.launched {
            _ if skip => Err(HarnessError::ModelStalled {
                attempts: group.stalled,
            }
            .into()),
            Ok((server, _)) => server.client.reset().and_then(|_| {
                task_run(TaskExecution {
                    client: &server.client,
                    adapter: Adapter::new(&group.model.adapter)?,
                    case: job.case,
                    suite: self.suite,
                    variant: job.variant,
                    catalog: self.catalog,
                    ctx: group.ctx,
                    dir,
                    cancel: self.cancel.clone(),
                    server_pid: server.pid(),
                    runtime: self.config.settings_for(group.model),
                })
            }),
            Err(e) => Err(anyhow::anyhow!("{e}")),
        };
        result.unwrap_or_else(|e| Outcome {
            status: classify_error(&e),
            reason: Some(e.to_string()),
            ..Default::default()
        })
    }

    /// Submits the attempt for upload, marks it done for resume, and prints its line.
    fn record(
        &mut self,
        group: &Group,
        job: &Job,
        key: &str,
        attempt: u32,
        dir: &Path,
        out: &Outcome,
    ) -> Result<()> {
        let meta = RunMeta {
            study: &self.id,
            job: key,
            attempt,
            phase: job.phase,
            model: &group.model.id,
            context: group.ctx,
            variant: job.variant,
            repeat: job.repeat,
            case: job.case,
            load_id: Some(&group.load_id),
            trace: &dir.join("trace.jsonl"),
        };
        self.uploader.submit(payload::run(&meta, out)?)?;
        let state = if out.status == Status::Interrupted {
            "interrupted"
        } else {
            "complete"
        };
        self.resume.record_job(
            &self.id,
            key,
            attempt,
            state,
            Some(&crate::store::tag(&out.status)),
        )?;
        self.counts.completed += 1;
        self.progress.line(&attempt_line(&meta, out).to_string());
        self.progress.advance();
        Ok(())
    }
}
