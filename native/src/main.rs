// main.rs
// Command-line entry point for native experiments and setup checks
// Command dispatch, interrupt handling, and JSON output

use a2e_native::{
    config::{Config, Manifest, Suite},
    store::Store,
};
use anyhow::Result;
use clap::{Parser, Subcommand};
use serde_json::json;
use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

#[derive(Parser)]
#[command(
    version,
    about = "Measure bounded tool-calling agents on edge hardware"
)]
struct Cli {
    #[arg(long, global = true, default_value = "configs/native/models.toml")]
    config: PathBuf,
    #[command(subcommand)]
    command: Commands,
}
#[derive(Subcommand)]
enum Commands {
    Models {
        #[command(subcommand)]
        command: Models,
    },
    Doctor {
        #[arg(long)]
        probe: bool,
    },
    Plan {
        #[arg(default_value = "configs/native/suites/smoke.toml")]
        suite: PathBuf,
        #[arg(long, default_value = "configs/native/tools/standard.toml")]
        manifest: PathBuf,
    },
    Run {
        #[arg(default_value = "configs/native/suites/smoke.toml")]
        suite: PathBuf,
        #[arg(long, default_value = "configs/native/tools/standard.toml")]
        manifest: PathBuf,
        #[arg(long)]
        retry_failed: bool,
    },
    Tools {
        #[command(subcommand)]
        command: Tools,
    },
    Results {
        #[arg(long)]
        db: Option<PathBuf>,
        #[arg(long)]
        json: bool,
        #[arg(long)]
        csv: Option<PathBuf>,
    },
    Judge {
        #[arg(long)]
        db: Option<PathBuf>,
        #[arg(long)]
        study: Option<String>,
        #[arg(long)]
        calibration: Option<PathBuf>,
    },
    #[command(hide = true)]
    ToolHost,
}
#[derive(Subcommand)]
enum Models {
    Fetch,
}
#[derive(Subcommand)]
enum Tools {
    Validate {
        #[arg(default_value = "configs/native/tools/standard.toml")]
        manifest: PathBuf,
    },
}
fn main() {
    if let Err(e) = entry() {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}
fn entry() -> Result<()> {
    let cli = Cli::parse();
    if matches!(cli.command, Commands::ToolHost) {
        let root = std::env::var("A2E_TOOL_ROOT")?;
        let live = std::env::var("A2E_WEB_MODE").as_deref() == Ok("live");
        return a2e_native::sdk::serve(&mut a2e_native::tools::Builtins::new(root.into(), live)?);
    }
    if let Commands::Tools {
        command: Tools::Validate { manifest },
    } = &cli.command
    {
        let m = Manifest::load(manifest)?;
        println!(
            "{}",
            json!({"valid":true,"protocol":m.protocol,"tools":m.tools.iter().map(|t|&t.name).collect::<Vec<_>>()})
        );
        return Ok(());
    }
    let c = Config::load(&cli.config)?;
    let stop = Arc::new(AtomicBool::new(false));
    let flag = stop.clone();
    ctrlc::set_handler(move || flag.store(true, Ordering::Relaxed))?;
    let db = |p: Option<PathBuf>| {
        Store::open(&p.unwrap_or_else(|| {
            a2e_native::config::expand_path(&c.runtime.output_dir).join("native.sqlite3")
        }))
    };
    let output = match cli.command {
        Commands::Models {
            command: Models::Fetch,
        } => serde_json::to_value(c.fetch_models()?)?,
        Commands::Doctor { probe } => doctor(&c, probe, stop)?,
        Commands::Plan { suite, manifest } => {
            let s = Suite::load(&suite)?;
            a2e_native::runner::plan(&c, &s, &s.load_tasks(&suite)?, &Manifest::load(&manifest)?)?
        }
        Commands::Run {
            suite,
            manifest,
            retry_failed,
        } => {
            let s = Suite::load(&suite)?;
            a2e_native::runner::run(
                &c,
                &s,
                &s.load_tasks(&suite)?,
                &Manifest::load(&manifest)?,
                retry_failed,
                stop,
            )?
        }
        Commands::Results { db: p, json, csv } => {
            let rows = db(p)?.rows()?;
            if let Some(path) = csv {
                a2e_native::store::export_csv(&rows, &path)?;
            }
            if json {
                serde_json::to_value(rows)?
            } else {
                a2e_native::store::summary(&rows)
            }
        }
        Commands::Judge {
            db: p,
            study,
            calibration,
        } => a2e_native::judge::run(&db(p)?, &c.judge, study.as_deref(), calibration.as_deref())?,
        Commands::Tools { .. } | Commands::ToolHost => unreachable!(),
    };
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}
fn doctor(c: &Config, probe: bool, stop: Arc<AtomicBool>) -> Result<serde_json::Value> {
    let machine =
        a2e_native::measure::machine(&c.runtime).unwrap_or_else(|e| json!({"error":e.to_string()}));
    let mut models = Vec::new();
    for m in &c.models {
        let path = c.model_path(m);
        let mut item = json!({"id":m.id,"path":path,"source":m.source,"revision":m.revision,"quant":m.quant,"exists":path.exists()});
        if path.exists() {
            let sha = a2e_native::hash_file(&path)?;
            item["sha256"] = json!(sha);
            item["hash_matches"] = json!(m.sha256.as_ref().map(|s| s == &sha));
        }
        if probe && path.exists() {
            let root = a2e_native::config::expand_path(&c.runtime.output_dir);
            let _lock = a2e_native::store::RunLock::acquire(&root)?;
            let dir = root.join("probes").join(&m.id);
            std::fs::create_dir_all(&dir)?;
            let result = (|| -> Result<serde_json::Value> {
                if machine.get("error").is_some() {
                    anyhow::bail!("device preflight failed");
                }
                let mut server = a2e_native::inference::Server::spawn(c, m, 1024, &dir)?;
                let monitor = a2e_native::measure::Monitor::start(
                    server.pid(),
                    None,
                    &c.runtime,
                    &dir.join("samples.jsonl"),
                    stop.clone(),
                )?;
                let ready = server.ready(c.runtime.startup_seconds, &stop);
                let resources = monitor.finish();
                if resources.memory_exceeded {
                    anyhow::bail!("model_load_memory_budget");
                }
                let props = ready?;
                let adapter = a2e_native::adapter::Adapter::new(&m.adapter)?;
                let prompt=server.client.template(&adapter.template_request(&[json!({"role":"user","content":"Call final_answer with answer set to ready."})],&[a2e_native::adapter::final_tool()]),std::time::Duration::from_secs(30))?;
                let response = server.client.complete(
                    &json!({"prompt":prompt,"n_predict":64,"temperature":0,"stop":adapter.generation_stops()}),
                    std::time::Duration::from_secs(300),
                )?;
                let parsed = adapter
                    .parse_generation(&response.text, response.raw["stopping_word"].as_str());
                Ok(
                    json!({"loaded":true,"resources":resources,"props":props,"raw":response.raw,"format_valid":parsed.is_ok(),"parse_error":parsed.err().map(|e|e.to_string())}),
                )
            })();
            item["probe"] = match result {
                Ok(v) => v,
                Err(e) => json!({"loaded":false,"error":e.to_string()}),
            };
        }
        models.push(item);
    }
    Ok(
        json!({"machine":machine,"models":models,"llama_server":c.runtime.llama_server,"expected_llama_commit":c.runtime.llama_commit,"pi_energy":"unavailable without a supported measurement source"}),
    )
}
