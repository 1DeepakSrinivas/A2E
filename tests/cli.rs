//! Command line behavior: suite selection, dry run, name hints and completions.

use serde_json::Value;
use std::process::{Command, Output};

fn a2e(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_a2e"))
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .args(args)
        .env("A2E_LOCAL", "1")
        .env("A2E_HOST_ID", "test@host")
        .output()
        .unwrap()
}

fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

#[test]
fn dry_run_lists_only_the_selected_models_and_variants() {
    let out = a2e(&[
        "run",
        "smoke",
        "--dry-run",
        "--model",
        "qwen3-0.6b",
        "--variant",
        "guided",
    ]);
    assert!(out.status.success(), "{}", stderr(&out));
    let v: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["dry_run"], true);
    let groups = v["plan"]["groups"].as_array().unwrap();
    assert!(!groups.is_empty());
    assert!(groups
        .iter()
        .all(|g| g["model"] == "qwen3-0.6b" && g["variant"]["name"] == "guided"));
    assert_eq!(v["models"].as_array().unwrap().len(), 1);
}

#[test]
fn an_unknown_suite_or_model_gets_a_hint() {
    let out = a2e(&["run", "smke", "--dry-run"]);
    assert!(!out.status.success());
    assert!(
        stderr(&out).contains("did you mean smoke"),
        "{}",
        stderr(&out)
    );

    let out = a2e(&["run", "smoke", "--dry-run", "--model", "qwen3"]);
    assert!(!out.status.success());
    assert!(
        stderr(&out).contains("did you mean qwen3-0.6b"),
        "{}",
        stderr(&out)
    );
}

#[test]
fn completions_print_a_script() {
    let out = a2e(&["completions", "bash"]);
    assert!(out.status.success());
    assert!(String::from_utf8_lossy(&out.stdout).contains("_a2e"));
}

#[test]
fn no_command_without_a_terminal_prints_help() {
    let out = a2e(&[]);
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stdout).contains("Usage: a2e"));
}

#[test]
fn a_run_stops_early_when_a_model_file_is_missing() {
    let dir = tempfile::tempdir().unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_a2e"))
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .args(["run", "smoke", "--model", "qwen3-0.6b"])
        .env("A2E_LOCAL", "1")
        .env("A2E_HOST_ID", "test@host")
        .env("A2E_MODELS_DIR", dir.path())
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(stderr(&out).contains("is missing"), "{}", stderr(&out));
}
