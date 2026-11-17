//! Commands that read the system and report: doctor, cases, tools, sandbox and upload status.

use a2e::config::{expand_path, Catalog, Config};
use anyhow::Result;
use clap::Subcommand;
use serde_json::{json, Value};
use std::{
    path::{Path, PathBuf},
    sync::{atomic::AtomicBool, Arc},
};

#[derive(Subcommand)]
pub enum CasesCommand {
    /// Check every case against the tools and rubrics
    Validate {
        /// Case file or folder
        #[arg(default_value = "configs/cases")]
        path: PathBuf,
        /// Tool definitions the cases are checked against
        #[arg(long, default_value = "configs/tools/standard.toml")]
        tools: PathBuf,
        /// Folder of rubric files from the a2e-results project
        #[arg(long, default_value = "a2e-results/rubrics")]
        rubrics: PathBuf,
    },
    /// Count cases by category and split
    List {
        /// Case file or folder
        #[arg(default_value = "configs/cases")]
        path: PathBuf,
        /// Also print one entry per case
        #[arg(long)]
        ids: bool,
    },
}

#[derive(Subcommand)]
pub enum SandboxCommand {
    /// Generate the sandbox tree
    Init {
        /// Seed for the filler text; the planted facts do not depend on it
        #[arg(long, default_value_t = a2e::sandbox::DEFAULT_SEED)]
        seed: u64,
        /// Folder to write
        #[arg(long, default_value = "sandbox")]
        out: PathBuf,
        /// Replace an existing tree
        #[arg(long)]
        force: bool,
    },
    /// Check the tree against its manifest
    Verify {
        /// Folder holding the tree and its manifest
        #[arg(long, default_value = "sandbox")]
        dir: PathBuf,
    },
    /// Print the manifest
    Info {
        /// Folder holding the tree and its manifest
        #[arg(long, default_value = "sandbox")]
        dir: PathBuf,
    },
}

pub fn validate_tools(path: &Path) -> Result<Value> {
    let catalog = Catalog::load(path)?;
    Ok(json!({"valid":true,"tools":catalog.tools.iter().map(|t|&t.name).collect::<Vec<_>>()}))
}

pub fn sandbox(command: &SandboxCommand) -> Result<Value> {
    let manifest = match command {
        SandboxCommand::Init { seed, out, force } => a2e::sandbox::init(out, *seed, *force)?,
        SandboxCommand::Verify { dir } => a2e::sandbox::verify(dir)?,
        SandboxCommand::Info { dir } => a2e::sandbox::read_manifest(dir)?,
    };
    Ok(serde_json::to_value(manifest)?)
}

pub fn upload(config: &Path, status: bool) -> Result<Value> {
    let config = Config::load(config)?;
    let mut uploader = a2e::upload::Uploader::open(&expand_path(&config.runtime.output_dir))?;
    Ok(serde_json::to_value(if status {
        uploader.status()?
    } else {
        uploader.flush()?
    })?)
}

/// Rubric names from the results project's rubric files, or `None` when the folder is missing.
fn rubric_names(dir: &Path) -> Option<Vec<String>> {
    Some(
        std::fs::read_dir(dir)
            .ok()?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "toml"))
            .filter_map(|p| p.file_stem().map(|s| s.to_string_lossy().into_owned()))
            .collect(),
    )
}

/// Case file validation and listing; needs no model configuration.
pub fn cases(command: &CasesCommand) -> Result<Value> {
    match command {
        CasesCommand::Validate {
            path,
            tools,
            rubrics,
        } => {
            let catalog = Catalog::load(tools)?;
            let known = rubric_names(rubrics);
            if known.is_none() {
                eprintln!(
                    "warning: {} not found, so rubric names are not checked. Run: git submodule update --init",
                    rubrics.display()
                );
            }
            let registry = catalog.tools.iter().map(|t| t.name.as_str()).collect();
            let all = a2e::case::load_all(std::slice::from_ref(path))?;
            for case in &all {
                case.validate(&registry)?;
                if case.judge.enabled
                    && known
                        .as_ref()
                        .is_some_and(|k| !k.contains(&case.judge.rubric))
                {
                    anyhow::bail!(
                        "case {} names unknown rubric {}",
                        case.id,
                        case.judge.rubric
                    );
                }
            }
            Ok(json!({"valid":true,"summary":a2e::case::summary(&all)}))
        }
        CasesCommand::List { path, ids } => {
            let all = a2e::case::load_all(std::slice::from_ref(path))?;
            let mut out = a2e::case::summary(&all);
            if *ids {
                out["cases"] = json!(all
                    .iter()
                    .map(|c| json!({"id":c.id,"category":c.category,"split":c.split,"tags":c.tags}))
                    .collect::<Vec<_>>());
            }
            Ok(out)
        }
    }
}

pub fn doctor(config_path: &Path, probe: bool, stop: Arc<AtomicBool>) -> Result<Value> {
    let c = Config::load(config_path)?;
    let machine =
        a2e::measure::machine(&c.runtime).unwrap_or_else(|e| json!({"error":e.to_string()}));
    let models = c
        .models
        .iter()
        .map(|m| model_report(&c, m, probe, &machine, &stop))
        .collect::<Result<Vec<_>>>()?;
    Ok(json!({
        "host": c.host_id,
        "host_file": c.host_file,
        "machine": machine,
        "telemetry": a2e::measure::telemetry(),
        "warnings": a2e::measure::preflight_warnings(&c.runtime),
        "models": models,
        "llama_server": c.runtime.llama_server,
        "expected_llama_commit": c.runtime.llama_commit,
        "pi_energy": "unavailable without a supported measurement source",
    }))
}

fn model_report(
    c: &Config,
    m: &a2e::config::Model,
    probe: bool,
    machine: &Value,
    stop: &Arc<AtomicBool>,
) -> Result<Value> {
    let path = c.model_path(m);
    let mut item = json!({"id":m.id,"path":path,"source":m.source,"revision":m.revision,"quant":m.quant,"exists":path.exists()});
    if let Ok(file) = std::fs::metadata(&path) {
        // Weights are mapped from the file and must stay in the page cache while the model runs
        let (weights, available) = (file.len() >> 20, a2e::measure::available_memory() >> 20);
        let headroom = available as i64 - weights as i64 - c.runtime.reserve_mb as i64;
        item["memory"] = json!({
            "weights_mib": weights,
            "available_mib": available,
            "headroom_mib": headroom,
            "fits": headroom > 0,
            "note": "headroom is what remains for the context cache and buffers after the weights and reserve_mb",
        });
    }
    if path.exists() {
        let sha = a2e::hash_file(&path)?;
        item["sha256"] = json!(sha);
        item["hash_matches"] = json!(m.sha256.as_ref().map(|s| s == &sha));
        if probe {
            item["probe"] = match probe_model(c, m, machine, stop) {
                Ok(v) => v,
                Err(e) => json!({"loaded":false,"error":e.to_string()}),
            };
        }
    }
    Ok(item)
}

/// Loads the model once and asks for a single `final_answer` call.
fn probe_model(
    c: &Config,
    m: &a2e::config::Model,
    machine: &Value,
    stop: &Arc<AtomicBool>,
) -> Result<Value> {
    use std::time::Duration;
    let root = expand_path(&c.runtime.output_dir);
    let _lock = a2e::store::RunLock::acquire(&root)?;
    let dir = root.join("probes").join(&m.id);
    std::fs::create_dir_all(&dir)?;
    if machine.get("error").is_some() {
        anyhow::bail!("device preflight failed");
    }
    let mut server = a2e::inference::Server::spawn(c, m, 1024, &dir)?;
    let monitor = a2e::measure::Monitor::start(server.pid(), &c.runtime, stop.clone())?;
    let ready = server.ready(c.runtime.startup_seconds, stop);
    let resources = monitor.finish();
    if resources.memory_exceeded {
        return Err(a2e::error::HarnessError::ModelLoadMemoryBudget.into());
    }
    let props = ready?;
    let adapter = a2e::adapter::Adapter::new(&m.adapter)?;
    let message = json!({"role":"user","content":"Call final_answer with answer set to ready."});
    let prompt = server.client.template(
        &adapter.template_request(&[message], &[a2e::adapter::final_tool()]),
        Duration::from_secs(30),
    )?;
    let response = server.client.complete(
        &json!({"prompt":prompt,"n_predict":64,"temperature":0,"stop":adapter.generation_stops()}),
        Duration::from_secs(300),
    )?;
    let parsed = adapter.parse_generation(&response.text, response.raw["stopping_word"].as_str());
    Ok(
        json!({"loaded":true,"resources":resources,"props":props,"raw":response.raw,"format_valid":parsed.is_ok(),"parse_error":parsed.err().map(|e|e.to_string())}),
    )
}
