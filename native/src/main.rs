// main.rs
// Command-line entry point for native experiments and setup checks
// Command dispatch, interrupt handling, and JSON output

use a2e_native::config::{Catalog, Config, Suite};
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
        tools: PathBuf,
    },
    Run {
        #[arg(default_value = "configs/native/suites/smoke.toml")]
        suite: PathBuf,
        #[arg(long, default_value = "configs/native/tools/standard.toml")]
        tools: PathBuf,
        #[arg(long)]
        retry_failed: bool,
    },
    Tools {
        #[command(subcommand)]
        command: Tools,
    },
    Cases {
        #[command(subcommand)]
        command: Cases,
    },
    Upload {
        #[arg(long)]
        status: bool,
    },
    Sandbox {
        #[command(subcommand)]
        command: Sandbox,
    },
}
#[derive(Subcommand)]
enum Sandbox {
    Init {
        #[arg(long, default_value_t = a2e_native::sandbox::DEFAULT_SEED)]
        seed: u64,
        #[arg(long, default_value = "sandbox")]
        out: PathBuf,
        #[arg(long)]
        force: bool,
    },
    Verify {
        #[arg(long, default_value = "sandbox")]
        dir: PathBuf,
    },
    Info {
        #[arg(long, default_value = "sandbox")]
        dir: PathBuf,
    },
}
#[derive(Subcommand)]
enum Models {
    Fetch,
}
#[derive(Subcommand)]
enum Cases {
    Validate {
        #[arg(default_value = "configs/native/cases")]
        path: PathBuf,
        #[arg(long, default_value = "configs/native/tools/standard.toml")]
        tools: PathBuf,
        #[arg(long, default_value = "a2e-results/rubrics")]
        rubrics: PathBuf,
    },
    List {
        #[arg(default_value = "configs/native/cases")]
        path: PathBuf,
        #[arg(long)]
        ids: bool,
    },
}
#[derive(Subcommand)]
enum Tools {
    Validate {
        #[arg(default_value = "configs/native/tools/standard.toml")]
        tools: PathBuf,
    },
}
fn main() {
    if let Err(e) = entry() {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}
fn entry() -> Result<()> {
    a2e_native::dotenv::load_startup(std::path::Path::new(".env"))?;
    let cli = Cli::parse();
    if let Commands::Tools {
        command: Tools::Validate { tools },
    } = &cli.command
    {
        let c = Catalog::load(tools)?;
        println!(
            "{}",
            json!({"valid":true,"tools":c.tools.iter().map(|t|&t.name).collect::<Vec<_>>()})
        );
        return Ok(());
    }
    if let Commands::Cases { command } = &cli.command {
        return cases(command);
    }
    if let Commands::Sandbox { command } = &cli.command {
        let manifest = match command {
            Sandbox::Init { seed, out, force } => a2e_native::sandbox::init(out, *seed, *force)?,
            Sandbox::Verify { dir } => a2e_native::sandbox::verify(dir)?,
            Sandbox::Info { dir } => a2e_native::sandbox::read_manifest(dir)?,
        };
        println!("{}", serde_json::to_string_pretty(&manifest)?);
        return Ok(());
    }
    let c = Config::load(&cli.config)?;
    let stop = Arc::new(AtomicBool::new(false));
    let flag = stop.clone();
    ctrlc::set_handler(move || flag.store(true, Ordering::Relaxed))?;
    let output = match cli.command {
        Commands::Models {
            command: Models::Fetch,
        } => serde_json::to_value(c.fetch_models()?)?,
        Commands::Doctor { probe } => doctor(&c, probe, stop)?,
        Commands::Plan { suite, tools } => {
            let s = Suite::load(&suite)?;
            a2e_native::runner::plan(&c, &s, &s.load_cases(&suite)?, &Catalog::load(&tools)?)?
        }
        Commands::Run {
            suite,
            tools,
            retry_failed,
        } => {
            let s = Suite::load(&suite)?;
            a2e_native::runner::run(
                &c,
                &s,
                &s.load_cases(&suite)?,
                &Catalog::load(&tools)?,
                retry_failed,
                stop,
            )?
        }
        Commands::Upload { status } => {
            let mut uploader = a2e_native::upload::Uploader::open(
                &a2e_native::config::expand_path(&c.runtime.output_dir),
            )?;
            serde_json::to_value(if status {
                uploader.status()?
            } else {
                uploader.flush()?
            })?
        }
        Commands::Tools { .. } | Commands::Cases { .. } | Commands::Sandbox { .. } => {
            unreachable!()
        }
    };
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}
/// Rubric names from the cloud project's rubric files, or `None` when the folder is missing.
fn rubric_names(dir: &std::path::Path) -> Option<Vec<String>> {
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
fn cases(command: &Cases) -> Result<()> {
    let output = match command {
        Cases::Validate {
            path,
            tools,
            rubrics,
        } => {
            let c = Catalog::load(tools)?;
            let known = rubric_names(rubrics);
            if known.is_none() {
                eprintln!(
                    "warning: {} not found; rubric names are not checked (run git submodule update --init)",
                    rubrics.display()
                );
            }
            let registry = c.tools.iter().map(|t| t.name.as_str()).collect();
            let all = a2e_native::case::load_all(std::slice::from_ref(path))?;
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
            json!({"valid":true,"summary":a2e_native::case::summary(&all)})
        }
        Cases::List { path, ids } => {
            let all = a2e_native::case::load_all(std::slice::from_ref(path))?;
            let mut out = a2e_native::case::summary(&all);
            if *ids {
                out["cases"] = json!(all
                    .iter()
                    .map(|c| json!({"id":c.id,"category":c.category,"split":c.split,"tags":c.tags}))
                    .collect::<Vec<_>>());
            }
            out
        }
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
                let monitor =
                    a2e_native::measure::Monitor::start(server.pid(), &c.runtime, stop.clone())?;
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
        json!({"host":c.host_id,"host_file":c.host_file,"roles":c.roles,"machine":machine,"models":models,"llama_server":c.runtime.llama_server,"expected_llama_commit":c.runtime.llama_commit,"pi_energy":"unavailable without a supported measurement source"}),
    )
}
