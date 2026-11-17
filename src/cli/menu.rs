//! Prompt menu opened by `a2e` with no command. Esc goes back one step; every action
//! here also exists as a command.

use super::{run, session};
use a2e::config::{expand_path, Config};
use anyhow::{bail, Result};
use inquire::{error::InquireError, Confirm, MultiSelect, Select};
use serde::{Deserialize, Serialize};
use std::{io::IsTerminal, path::Path};

/// Choices from the last started run, offered as defaults next time.
#[derive(Default, Serialize, Deserialize)]
#[serde(default)]
struct Last {
    suites: Vec<String>,
    models: Vec<String>,
    variants: Vec<String>,
    local: bool,
    retry_failed: bool,
}

impl Last {
    fn path(config: &Config) -> std::path::PathBuf {
        expand_path(&config.settings.host.output_dir).join(".last-run.toml")
    }
    fn load(config: &Config) -> Self {
        std::fs::read_to_string(Self::path(config))
            .ok()
            .and_then(|t| toml::from_str(&t).ok())
            .unwrap_or_default()
    }
    fn save(&self, config: &Config) {
        if let Ok(text) = toml::to_string(self) {
            let _ = std::fs::write(Self::path(config), text);
        }
    }
}

/// `None` when the user pressed Esc; exits quietly on Ctrl-C.
fn answer<T>(result: Result<T, InquireError>) -> Result<Option<T>> {
    match result {
        Ok(v) => Ok(Some(v)),
        Err(InquireError::OperationCanceled) => Ok(None),
        Err(InquireError::OperationInterrupted) => std::process::exit(130),
        Err(e) => Err(e.into()),
    }
}

/// Positions of the options chosen last time.
fn remembered(options: &[String], wanted: &[String]) -> Vec<usize> {
    options
        .iter()
        .enumerate()
        .filter(|(_, o)| wanted.contains(o))
        .map(|(i, _)| i)
        .collect()
}

/// Remembered positions, or every option when nothing was remembered.
fn defaults(options: &[String], wanted: &[String]) -> Vec<usize> {
    let picked = remembered(options, wanted);
    if picked.is_empty() {
        (0..options.len()).collect()
    } else {
        picked
    }
}

pub fn open(config_path: &Path, local: bool) -> Result<()> {
    if !std::io::stdin().is_terminal() || !std::io::stdout().is_terminal() {
        super::print_help();
        std::process::exit(2);
    }
    const ACTIONS: [&str; 9] = [
        "Start a run",
        "Check what a suite would do",
        "Status",
        "Attach to a run",
        "Show a log",
        "Stop a run",
        "Check this machine (doctor)",
        "Send waiting results",
        "Quit",
    ];
    loop {
        let Some(action) = answer(Select::new("A2E: what next?", ACTIONS.to_vec()).prompt())?
        else {
            return Ok(());
        };
        let outcome = match action {
            "Start a run" => start_flow(config_path, local),
            "Check what a suite would do" => dry_run_flow(config_path),
            "Status" => pick_session("Status for:").and_then(|n| {
                print_json(&session::status(
                    config_path,
                    n.as_deref().unwrap_or("none"),
                )?)
            }),
            "Attach to a run" => pick_session("Attach to:")
                .and_then(|n| n.map_or(Ok(()), |n| session::attach(&n).map(|_| ()))),
            "Show a log" => {
                let names = session::log_names(config_path);
                if names.is_empty() {
                    println!("No logs yet.");
                    Ok(())
                } else {
                    match answer(Select::new("Log:", names).prompt())? {
                        Some(n) => session::logs(config_path, &n).map(|_| ()),
                        None => Ok(()),
                    }
                }
            }
            "Stop a run" => pick_session("Stop:")
                .and_then(|n| n.map_or(Ok(()), |n| print_json(&session::stop(&n)?))),
            "Check this machine (doctor)" => print_json(&super::inspect::doctor(
                config_path,
                false,
                Default::default(),
            )?),
            "Send waiting results" => print_json(&super::inspect::upload(config_path, false)?),
            _ => return Ok(()),
        };
        if let Err(e) = outcome {
            eprintln!("error: {e:#}");
        }
        println!();
    }
}

fn print_json(v: &serde_json::Value) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(v)?);
    Ok(())
}

/// A running session; the only one is chosen without asking.
fn pick_session(prompt: &str) -> Result<Option<String>> {
    let mut running = session::sessions();
    match running.len() {
        0 => {
            println!("No run is active.");
            Ok(None)
        }
        1 => Ok(running.pop()),
        _ => answer(Select::new(prompt, running).prompt()),
    }
}

fn dry_run_flow(config_path: &Path) -> Result<()> {
    let names = run::suite_names(config_path);
    let Some(suite) = answer(Select::new("Suite:", names).prompt())? else {
        return Ok(());
    };
    let args = run::RunArgs {
        plan: run::PlanArgs {
            suite,
            tools: "configs/tools/standard.toml".into(),
            select: Default::default(),
        },
        retry_failed: false,
        allow_missing_model: false,
        dry_run: true,
    };
    print_json(&run::run(config_path, &args, Default::default())?)
}

/// Prints the attempt count per suite and the total, so the choice is clear before starting.
fn print_summary(config_path: &Path, args: &session::StartArgs, picked: &Last) -> Result<()> {
    let mut total = 0;
    for s in &picked.suites {
        let l = run::load(config_path, s, &args.select)?;
        let cases = l.suite.load_cases(&l.path)?;
        let catalog = a2e::config::Catalog::load(&args.tools)?;
        let plan = a2e::runner::plan(&l.config, &l.suite, &cases, &catalog)?;
        let n = run::total_attempts(&plan, &l.suite);
        println!("  {s}: {n} attempts");
        total += n;
    }
    println!(
        "  {total} attempts, {}",
        if picked.local {
            "results stay on this device"
        } else {
            "results go to the Worker"
        }
    );
    Ok(())
}

fn start_flow(config_path: &Path, local_flag: bool) -> Result<()> {
    let config = Config::load(config_path)?;
    let last = Last::load(&config);
    let suite_names = run::suite_names(config_path);
    let model_names: Vec<String> = config.models.iter().map(|m| m.id.clone()).collect();
    let mut picked = Last {
        local: local_flag || last.local,
        ..Default::default()
    };
    let mut step = 0;
    loop {
        match step {
            0 => {
                let chosen = answer(
                    MultiSelect::new(
                        "Suites (space selects, enter confirms):",
                        suite_names.clone(),
                    )
                    .with_default(&remembered(&suite_names, &last.suites))
                    .prompt(),
                )?;
                match chosen {
                    None => return Ok(()),
                    Some(s) if s.is_empty() => println!("Choose at least one suite."),
                    Some(s) => {
                        picked.suites = s;
                        step = 1;
                    }
                }
            }
            1 => match answer(
                MultiSelect::new("Models:", model_names.clone())
                    .with_default(&defaults(&model_names, &last.models))
                    .prompt(),
            )? {
                None => step = 0,
                Some(m) if m.is_empty() => println!("Choose at least one model."),
                Some(m) => {
                    picked.models = if m.len() == model_names.len() {
                        Vec::new()
                    } else {
                        m
                    };
                    step = if picked.suites.len() == 1 { 2 } else { 3 };
                }
            },
            2 => {
                let probe = run::load(config_path, &picked.suites[0], &run::SelectArgs::default())?;
                let names: Vec<String> = probe
                    .suite
                    .variants
                    .iter()
                    .map(|v| v.name.clone())
                    .collect();
                match answer(
                    MultiSelect::new("Variants:", names.clone())
                        .with_default(&defaults(&names, &last.variants))
                        .prompt(),
                )? {
                    None => step = 1,
                    Some(v) if v.is_empty() => println!("Choose at least one variant."),
                    Some(v) => {
                        picked.variants = if v.len() == names.len() {
                            Vec::new()
                        } else {
                            v
                        };
                        step = 3;
                    }
                }
            }
            3 => match answer(
                Confirm::new("Keep results on this device only? (No sends them to the Worker)")
                    .with_default(picked.local)
                    .prompt(),
            )? {
                None => step = if picked.suites.len() == 1 { 2 } else { 1 },
                Some(v) => {
                    picked.local = v;
                    step = 4;
                }
            },
            4 => match answer(
                Confirm::new("Run again the attempts that failed before?")
                    .with_default(last.retry_failed)
                    .prompt(),
            )? {
                None => step = 3,
                Some(v) => {
                    picked.retry_failed = v;
                    step = 5;
                }
            },
            _ => {
                let args = session::StartArgs {
                    suites: picked.suites.join(","),
                    tools: "configs/tools/standard.toml".into(),
                    select: run::SelectArgs {
                        model: picked.models.clone(),
                        context: Vec::new(),
                        variant: picked.variants.clone(),
                    },
                    retry_failed: picked.retry_failed,
                    allow_missing_model: false,
                };
                print_summary(config_path, &args, &picked)?;
                match answer(Confirm::new("Start in tmux?").with_default(true).prompt())? {
                    None => step = 4,
                    Some(false) => bail!("cancelled"),
                    Some(true) => {
                        let out = session::start(config_path, &args, picked.local)?;
                        picked.save(&config);
                        return print_json(&out);
                    }
                }
            }
        }
    }
}
