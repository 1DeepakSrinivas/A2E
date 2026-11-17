//! Runs in detached tmux sessions: start, attach, status, logs and stop.
//!
//! One suite runs in `a2e-<suite>`. Several run one after another in `a2e-queue`, because only one
//! run can hold the runtime lock.

use super::run::{self, SelectArgs};
use a2e::config::{expand_path, Config};
use anyhow::{bail, Context, Result};
use clap::Args;
use serde_json::{json, Value};
use std::{
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    time::{SystemTime, UNIX_EPOCH},
};

#[derive(Args)]
pub struct StartArgs {
    /// Suite names, comma separated; several run in order
    pub suites: String,
    #[arg(long, default_value = "configs/tools/standard.toml")]
    pub tools: PathBuf,
    #[command(flatten)]
    pub select: SelectArgs,
    /// Run again the attempts that failed before
    #[arg(long)]
    pub retry_failed: bool,
    /// Run even when a model file is missing; its attempts fail
    #[arg(long)]
    pub allow_missing_model: bool,
}

#[derive(Args)]
pub struct NameArg {
    /// Session name: the suite name, or `queue`
    pub name: String,
}

#[derive(Args)]
pub struct QueueArgs {
    #[arg(long)]
    log: PathBuf,
    #[arg(long, default_value = "configs/models.toml")]
    config: PathBuf,
    #[arg(long, value_delimiter = ',')]
    suites: Vec<String>,
    /// Options passed to each `a2e run`
    #[arg(last = true)]
    rest: Vec<String>,
}

fn tmux() -> Result<Command> {
    if Command::new("tmux").arg("-V").output().is_err() {
        bail!("tmux is not installed. Run ./setup.sh to install it.");
    }
    Ok(Command::new("tmux"))
}

/// Names of running a2e sessions without the `a2e-` prefix.
pub fn sessions() -> Vec<String> {
    let Ok(out) = Command::new("tmux")
        .args(["list-sessions", "-F", "#S"])
        .output()
    else {
        return Vec::new();
    };
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| l.strip_prefix("a2e-").map(String::from))
        .collect()
}

fn check_name(name: &str) -> Result<()> {
    if name.is_empty()
        || !name
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
    {
        bail!("session names use lowercase letters, digits and underscores");
    }
    Ok(())
}

fn require_running(name: &str) -> Result<()> {
    check_name(name)?;
    if !sessions().iter().any(|s| s == name) {
        let running = sessions();
        if running.is_empty() {
            bail!("no session a2e-{name}; nothing is running. Start one with: a2e start <suite>");
        }
        bail!("no session a2e-{name}. Running: {}", running.join(", "));
    }
    Ok(())
}

/// Log file stamp such as `20261117-142530` in UTC.
fn stamp() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let (days, rest) = (secs / 86400, secs % 86400);
    // Civil date from days since 1970-01-01 (Howard Hinnant's algorithm)
    let z = days as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}{month:02}{day:02}-{:02}{:02}{:02}",
        rest / 3600,
        rest % 3600 / 60,
        rest % 60
    )
}

pub fn session_name(suites: &[String]) -> String {
    if suites.len() == 1 {
        suites[0].clone()
    } else {
        "queue".into()
    }
}

/// Settings that change a run. Secrets are left out because they would show in the process list;
/// the run reads them from `.env` in the working directory.
const FORWARDED_ENV: [&str; 8] = [
    "A2E_HOST_ID",
    "A2E_MODELS_DIR",
    "A2E_LLAMA_SERVER",
    "A2E_OUTPUT_DIR",
    "A2E_RESULTS_ENV",
    "A2E_SANDBOX",
    "A2E_SANDBOX_TMP",
    "PATH",
];

/// The command tmux runs: this binary's hidden `queue` command with the run options.
fn queue_command(
    exe: &Path,
    config_path: &Path,
    suites: &[String],
    log: &Path,
    args: &StartArgs,
    local: bool,
) -> Vec<String> {
    // tmux starts the command with its server's environment, so pass the settings explicitly.
    let mut command = vec!["env".to_string()];
    command.extend(FORWARDED_ENV.iter().filter_map(|key| {
        std::env::var(key)
            .ok()
            .map(|value| format!("{key}={value}"))
    }));
    command.push(exe.to_string_lossy().into_owned());
    command.extend([
        "--config".into(),
        config_path.to_string_lossy().into_owned(),
    ]);
    if local {
        command.push("--local".into());
    }
    command.extend([
        "queue".into(),
        "--log".into(),
        log.to_string_lossy().into_owned(),
        "--config".into(),
        config_path.to_string_lossy().into_owned(),
        "--suites".into(),
        suites.join(","),
        "--".into(),
        "--tools".into(),
        args.tools.to_string_lossy().into_owned(),
    ]);
    for (flag, values) in [
        ("--model", args.select.model.clone()),
        (
            "--context",
            args.select
                .context
                .iter()
                .map(ToString::to_string)
                .collect(),
        ),
        ("--variant", args.select.variant.clone()),
    ] {
        if !values.is_empty() {
            command.extend([flag.into(), values.join(",")]);
        }
    }
    if args.retry_failed {
        command.push("--retry-failed".into());
    }
    if args.allow_missing_model {
        command.push("--allow-missing-model".into());
    }
    command
}

pub fn start(config_path: &Path, args: &StartArgs, local: bool) -> Result<Value> {
    let local = local || a2e::dotenv::local_only();
    let suites: Vec<String> = args
        .suites
        .split(',')
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect();
    if suites.is_empty() {
        bail!("name at least one suite; see: a2e cases list or ls configs/suites");
    }
    let running = sessions();
    if !running.is_empty() {
        bail!(
            "another a2e session is running ({}); only one run can hold the runtime lock. Stop it with: a2e stop <name>",
            running.join(", ")
        );
    }
    let exe = std::env::current_exe()?;
    if a2e::sandbox::verify(&a2e::sandbox::default_dir()).is_err() {
        bail!("the sandbox is missing or does not match its manifest. Run: a2e sandbox init");
    }
    for s in &suites {
        run::plan(
            config_path,
            &run::PlanArgs {
                suite: s.clone(),
                tools: args.tools.clone(),
                select: args.select.clone(),
            },
        )
        .with_context(|| format!("suite {s} does not plan"))?;
    }
    let config = Config::load(config_path)?;
    let logs = expand_path(&config.runtime.output_dir).join("logs");
    std::fs::create_dir_all(&logs)?;
    let name = session_name(&suites);
    let log = logs.join(format!("{name}-{}.log", stamp()));

    let command = queue_command(&exe, config_path, &suites, &log, args, local);
    let status = tmux()?
        .args(["new-session", "-d", "-s", &format!("a2e-{name}"), "-c"])
        .arg(std::env::current_dir()?)
        .arg("--")
        .args(&command)
        .status()?;
    if !status.success() {
        bail!("tmux could not start the session");
    }
    Ok(json!({
        "session": format!("a2e-{name}"),
        "suites": suites,
        "log": log,
        "uploads": if local { "kept on this device" } else { "sent to the Worker named by WORKER_URL or .env" },
        "next": format!("a2e attach {name}   (detach with Ctrl-b d)"),
    }))
}

/// Runs each suite as a child `a2e run`, copies its output to the terminal and the log, and
/// stops starting new suites after an interrupt.
pub fn queue(args: &QueueArgs, stop: Arc<AtomicBool>) -> Result<Value> {
    let log = Arc::new(Mutex::new(std::fs::File::create(&args.log)?));
    let exe = std::env::current_exe()?;
    let mut worst = 0;
    for suite in &args.suites {
        if stop.load(Ordering::Relaxed) {
            worst = worst.max(130);
            break;
        }
        let mut child = Command::new(&exe)
            .arg("--config")
            .arg(&args.config)
            .args(["run", suite])
            .args(&args.rest)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        let copies = [
            copy(child.stdout.take().context("stdout")?, log.clone(), false),
            copy(child.stderr.take().context("stderr")?, log.clone(), true),
        ];
        let code = child.wait()?.code().unwrap_or(130);
        for c in copies {
            let _ = c.join();
        }
        worst = worst.max(code);
    }
    if worst != 0 {
        std::process::exit(worst);
    }
    Ok(Value::Null)
}

fn copy(
    source: impl std::io::Read + Send + 'static,
    log: Arc<Mutex<std::fs::File>>,
    to_stderr: bool,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        for line in BufReader::new(source).lines().map_while(|l| l.ok()) {
            if to_stderr {
                eprintln!("{line}");
            } else {
                println!("{line}");
            }
            if let Ok(mut file) = log.lock() {
                let _ = writeln!(file, "{line}");
            }
        }
    })
}

pub fn attach(name: &str) -> Result<Value> {
    require_running(name)?;
    tmux()?
        .args(["attach", "-t", &format!("a2e-{name}")])
        .status()?;
    Ok(Value::Null)
}

pub fn status(config_path: &Path, name: &str) -> Result<Value> {
    check_name(name)?;
    let running = sessions().iter().any(|s| s == name);
    Ok(json!({
        "session": format!("a2e-{name}"),
        "running": running,
        "running_sessions": sessions(),
        "upload": super::inspect::upload(config_path, true)?,
    }))
}

/// Newest log for `name`, by modification time.
fn latest_log(config_path: &Path, name: &str) -> Result<PathBuf> {
    check_name(name)?;
    let config = Config::load(config_path)?;
    let dir = expand_path(&config.runtime.output_dir).join("logs");
    let prefix = format!("{name}-");
    std::fs::read_dir(&dir)
        .with_context(|| format!("no log folder at {}", dir.display()))?
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().starts_with(&prefix))
        .max_by_key(|e| e.metadata().and_then(|m| m.modified()).ok())
        .map(|e| e.path())
        .with_context(|| format!("no logs for {name} in {}", dir.display()))
}

pub fn logs(config_path: &Path, name: &str) -> Result<Value> {
    let log = latest_log(config_path, name)?;
    Command::new("tail")
        .args(["-n", "40", "-f"])
        .arg(log)
        .status()?;
    Ok(Value::Null)
}

pub fn stop(name: &str) -> Result<Value> {
    require_running(name)?;
    tmux()?
        .args(["send-keys", "-t", &format!("a2e-{name}"), "C-c"])
        .status()?;
    Ok(
        json!({"stopped": format!("a2e-{name}"), "note": "The session closes when the runner exits. Start it again to resume."}),
    )
}

/// Log names that exist, without the stamp, for the menu.
pub fn log_names(config_path: &Path) -> Vec<String> {
    let Ok(config) = Config::load(config_path) else {
        return Vec::new();
    };
    let dir = expand_path(&config.runtime.output_dir).join("logs");
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let file = e.file_name().to_string_lossy().into_owned();
            file.strip_suffix(".log")
                .and_then(|f| f.rsplit_once('-'))
                .and_then(|(rest, _)| rest.rsplit_once('-'))
                .map(|(name, _)| name.to_string())
        })
        .collect();
    names.sort();
    names.dedup();
    names
}
