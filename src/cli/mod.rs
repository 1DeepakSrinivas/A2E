//! Command line: argument parsing and dispatch.
//!
//! Data commands print JSON on stdout. Progress, warnings and prompts use stderr or the terminal.

mod inspect;
mod menu;
mod models;
mod run;
mod session;

use anyhow::Result;
use clap::{CommandFactory, Parser, Subcommand};
use clap_complete::Shell;
use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

const EXAMPLES: &str = "\
Start here:
  a2e                          open the menu (on a terminal)
  a2e doctor                   check this machine, models and llama.cpp
  a2e run smoke --dry-run      show what a run would do and check the files it needs

Run:
  a2e run smoke --local        run one suite here and keep results on this device
  a2e run core --model qwen3-0.6b --variant guided
  a2e start core,harness       run suites one after another in tmux
  a2e attach core | status core | logs core | stop core
  a2e monitor core             show a run on the device's own screen

Inspect and set up:
  a2e cases list --ids         list case ids
  a2e cases validate           check cases, tools and rubrics after any change
  a2e models fetch             download the pinned models
  a2e models add unsloth/Qwen3-0.6B-GGUF:Q4_K_M   download and pin any GGUF from Hugging Face
  a2e config show --source     print every setting and the layer that set it
  a2e run core --set limits.output_tokens=512   change one setting for one run
  a2e completions zsh          print shell completions";

#[derive(Parser)]
#[command(
    version,
    about = "Measure bounded tool-calling agents on edge hardware",
    after_help = EXAMPLES
)]
struct Cli {
    /// Model registry and shared settings
    #[arg(long, global = true, default_value = "configs/models.toml")]
    config: PathBuf,
    /// Keep results on this device; nothing is sent to the Worker (same as A2E_LOCAL=1)
    #[arg(long, global = true)]
    local: bool,
    /// Override one setting, for example --set limits.output_tokens=512; repeat for more
    #[arg(long = "set", global = true, value_name = "KEY=VALUE")]
    set: Vec<String>,
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Run one suite in this terminal
    Run(run::RunArgs),
    /// Show the runs a suite would make, without starting a model
    Plan(run::PlanArgs),
    /// Run one or more suites in a detached tmux session
    Start(session::StartArgs),
    /// Watch a running session; detach with Ctrl-b then d
    Attach(session::NameArg),
    /// Show a running session on the device's own screen
    Monitor(session::NameArg),
    /// Show whether a session is running and what waits to upload
    Status(session::NameArg),
    /// Follow the newest log of a session
    Logs(session::NameArg),
    /// Interrupt a session; start it again later to resume
    Stop(session::NameArg),
    /// Check this machine, the models and llama.cpp
    Doctor {
        /// Load each model once and check its output format
        #[arg(long)]
        probe: bool,
    },
    /// Download and verify the pinned models
    Models {
        #[command(subcommand)]
        command: ModelsCommand,
    },
    /// Show, check or create settings
    Config {
        #[command(subcommand)]
        command: inspect::ConfigCommand,
    },
    /// Check tool definitions
    Tools {
        #[command(subcommand)]
        command: ToolsCommand,
    },
    /// Check or list case files
    Cases {
        #[command(subcommand)]
        command: inspect::CasesCommand,
    },
    /// Send waiting results to the Worker, or show what is waiting
    Upload {
        /// Show what is waiting instead of sending it
        #[arg(long)]
        status: bool,
    },
    /// Build and check the file sandbox that file cases use
    Sandbox {
        #[command(subcommand)]
        command: inspect::SandboxCommand,
    },
    /// Print shell completions
    Completions { shell: Shell },
    /// Run suites in order and tee output to a log; used by `start` inside tmux
    #[command(name = "queue", hide = true)]
    Queue(session::QueueArgs),
}

#[derive(Subcommand)]
enum ModelsCommand {
    /// Download the pinned models and check their hashes
    Fetch,
    /// Download a GGUF from Hugging Face and add it as a pinned model
    Add(models::AddArgs),
}

#[derive(Subcommand)]
enum ToolsCommand {
    /// Check that every tool definition has a handler and a valid schema
    Validate {
        /// Tool definition file
        #[arg(default_value = "configs/tools/standard.toml")]
        tools: PathBuf,
    },
}

/// Prints the same help as `a2e --help`.
fn print_help() {
    let _ = Cli::command().print_help();
}

pub fn main() {
    if let Err(e) = entry() {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}

fn entry() -> Result<()> {
    let cli = Cli::parse();
    a2e::config::set_overrides(cli.set.clone());
    if cli.local {
        // Set before any thread starts and before the credentials file is read
        std::env::set_var("A2E_LOCAL", "1");
    }
    a2e::dotenv::load_startup(std::path::Path::new(".env"))?;
    let local = a2e::dotenv::local_only();
    let Some(command) = cli.command else {
        return menu::open(&cli.config, local);
    };
    let stop = Arc::new(AtomicBool::new(false));
    let flag = stop.clone();
    ctrlc::set_handler(move || flag.store(true, Ordering::Relaxed))?;
    let output = match command {
        Command::Run(args) => run::run(&cli.config, &args, stop)?,
        Command::Plan(args) => run::plan(&cli.config, &args)?,
        Command::Start(args) => session::start(&cli.config, &args, local)?,
        Command::Attach(a) => session::attach(&a.name)?,
        Command::Monitor(a) => session::monitor(&a.name)?,
        Command::Status(a) => session::status(&cli.config, &a.name)?,
        Command::Logs(a) => session::logs(&cli.config, &a.name)?,
        Command::Stop(a) => session::stop(&a.name)?,
        Command::Queue(args) => session::queue(&args, stop)?,
        Command::Doctor { probe } => inspect::doctor(&cli.config, probe, stop)?,
        Command::Models {
            command: ModelsCommand::Fetch,
        } => serde_json::to_value(a2e::config::Config::load(&cli.config)?.fetch_models()?)?,
        Command::Models {
            command: ModelsCommand::Add(args),
        } => models::add(&cli.config, &args)?,
        Command::Tools {
            command: ToolsCommand::Validate { tools },
        } => inspect::validate_tools(&tools)?,
        Command::Config { command } => inspect::config(&cli.config, &command)?,
        Command::Cases { command } => inspect::cases(&command)?,
        Command::Upload { status } => inspect::upload(&cli.config, status)?,
        Command::Sandbox { command } => inspect::sandbox(&command)?,
        Command::Completions { shell } => {
            clap_complete::generate(shell, &mut Cli::command(), "a2e", &mut std::io::stdout());
            return Ok(());
        }
    };
    if !output.is_null() {
        println!("{}", serde_json::to_string_pretty(&output)?);
    }
    Ok(())
}
