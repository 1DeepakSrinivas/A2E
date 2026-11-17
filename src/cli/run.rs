//! `plan`, `run` and the shared suite selection.

use a2e::{
    config::{expand_path, unknown_name, Catalog, Config, Selection, Suite},
    progress::Progress,
};
use anyhow::Result;
use clap::Args;
use serde_json::{json, Value};
use std::{
    path::{Path, PathBuf},
    sync::{atomic::AtomicBool, Arc},
};

/// Which models, contexts and variants of a suite to use; empty keeps all.
#[derive(Args, Clone, Default)]
pub struct SelectArgs {
    /// Models to run, comma separated (default: every model in the registry)
    #[arg(long, value_delimiter = ',')]
    pub model: Vec<String>,
    /// Context sizes to run, comma separated
    #[arg(long, value_delimiter = ',')]
    pub context: Vec<usize>,
    /// Variants to run, comma separated
    #[arg(long, value_delimiter = ',')]
    pub variant: Vec<String>,
}

impl SelectArgs {
    pub fn selection(&self) -> Selection {
        Selection {
            models: self.model.clone(),
            contexts: self.context.clone(),
            variants: self.variant.clone(),
        }
    }
}

#[derive(Args, Clone)]
pub struct RunArgs {
    /// Suite name such as `smoke`, or a path to a suite file
    #[arg(default_value = "smoke")]
    pub suite: String,
    #[arg(long, default_value = "configs/tools/standard.toml")]
    pub tools: PathBuf,
    #[command(flatten)]
    pub select: SelectArgs,
    /// Run again the attempts that failed before
    #[arg(long)]
    pub retry_failed: bool,
    /// Print the runs and check the files they need, then stop
    #[arg(long)]
    pub dry_run: bool,
}

/// Names of the suite files next to the model registry.
pub fn suite_names(config: &Path) -> Vec<String> {
    let dir = config.parent().unwrap_or(Path::new(".")).join("suites");
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "toml"))
        .filter_map(|p| p.file_stem().map(|s| s.to_string_lossy().into_owned()))
        .collect();
    names.sort();
    names
}

/// A bare name becomes `<config dir>/suites/<name>.toml`; anything with a path or extension is used as given.
pub fn suite_path(config: &Path, suite: &str) -> Result<PathBuf> {
    if suite.contains('/') || suite.ends_with(".toml") {
        return Ok(PathBuf::from(suite));
    }
    let names = suite_names(config);
    if !names.iter().any(|n| n == suite) {
        return Err(unknown_name("suite", suite, &names));
    }
    Ok(config
        .parent()
        .unwrap_or(Path::new("."))
        .join("suites")
        .join(format!("{suite}.toml")))
}

pub struct Loaded {
    pub config: Config,
    pub suite: Suite,
    pub path: PathBuf,
}

pub fn load(config_path: &Path, name: &str, select: &SelectArgs) -> Result<Loaded> {
    let config = Config::load(config_path)?;
    let path = suite_path(config_path, name)?;
    let suite = Suite::load(&path)?.resolve(&config, &select.selection())?;
    Ok(Loaded {
        config,
        suite,
        path,
    })
}

pub fn plan(config_path: &Path, args: &RunArgs) -> Result<Value> {
    let l = load(config_path, &args.suite, &args.select)?;
    let cases = l.suite.load_cases(&l.path)?;
    a2e::runner::plan(&l.config, &l.suite, &cases, &Catalog::load(&args.tools)?)
}

/// Total attempts the plan lists, counting cases that do not apply, which the runner also steps over.
pub fn total_attempts(plan: &Value, suite: &Suite) -> u64 {
    let per_case = (suite.warmup + suite.repeats) as u64;
    plan["groups"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|g| {
            (g["cases"].as_u64().unwrap_or(0) + g["not_applicable"].as_u64().unwrap_or(0))
                * per_case
        })
        .sum()
}

/// What a run would do, plus whether the files it needs are there.
fn dry_run(l: &Loaded, plan: Value) -> Value {
    let models: Vec<Value> = l
        .suite
        .models
        .iter()
        .filter_map(|id| l.config.model(id).ok())
        .map(|m| {
            let path = l.config.model_path(m);
            json!({"id":m.id,"path":path,"exists":path.exists(),"size_mib":std::fs::metadata(&path).ok().map(|f| f.len() >> 20)})
        })
        .collect();
    let server = expand_path(&l.config.runtime.llama_server);
    let missing: Vec<String> = models
        .iter()
        .filter(|m| m["exists"] == false)
        .map(|m| format!("model file {}", m["path"].as_str().unwrap_or_default()))
        .chain((!server.exists()).then(|| format!("llama-server at {}", server.display())))
        .collect();
    json!({
        "dry_run": true,
        "ready": missing.is_empty(),
        "missing": missing,
        "total_attempts": total_attempts(&plan, &l.suite),
        "models": models,
        "llama_server": {"path": server, "exists": server.exists()},
        "plan": plan,
        "fix": if missing.is_empty() { Value::Null } else { json!("Run ./setup.sh, or a2e models fetch for model files.") },
    })
}

pub fn run(config_path: &Path, args: &RunArgs, stop: Arc<AtomicBool>) -> Result<Value> {
    let l = load(config_path, &args.suite, &args.select)?;
    let cases = l.suite.load_cases(&l.path)?;
    let catalog = Catalog::load(&args.tools)?;
    let plan = a2e::runner::plan(&l.config, &l.suite, &cases, &catalog)?;
    if args.dry_run {
        return Ok(dry_run(&l, plan));
    }
    eprintln!("results: {}", a2e::upload::destination());
    let progress = Progress::new(total_attempts(&plan, &l.suite));
    a2e::runner::run(
        &l.config,
        &l.suite,
        &cases,
        &catalog,
        args.retry_failed,
        stop,
        &progress,
    )
}
