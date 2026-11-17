//! Docs stay true to the code: every command, flag, environment variable and file path a
//! document names must exist.

use std::{collections::BTreeSet, path::PathBuf, process::Command};

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn docs() -> Vec<(String, String)> {
    let mut files = vec![repo().join("README.md"), repo().join("REFERENCES.md")];
    for entry in std::fs::read_dir(repo().join("docs")).unwrap() {
        files.push(entry.unwrap().path());
    }
    files
        .into_iter()
        .filter(|p| p.extension().is_some_and(|e| e == "md"))
        .map(|p| {
            let text = std::fs::read_to_string(&p).unwrap();
            (p.file_name().unwrap().to_string_lossy().into_owned(), text)
        })
        .collect()
}

/// Text inside backticks, one entry per span.
fn code_spans(text: &str) -> Vec<String> {
    text.split('`')
        .enumerate()
        .filter(|(i, _)| i % 2 == 1)
        .map(|(_, s)| s.to_string())
        .collect()
}

fn help(args: &[&str]) -> String {
    let out = Command::new(env!("CARGO_BIN_EXE_a2e"))
        .current_dir(repo())
        .args(args)
        .arg("--help")
        .env("A2E_HOST_ID", "docs@test")
        .output()
        .unwrap();
    String::from_utf8_lossy(&out.stdout).into_owned()
}

#[test]
fn every_command_in_the_docs_exists() {
    let top = help(&[]);
    let known: BTreeSet<String> = top
        .lines()
        .skip_while(|l| !l.starts_with("Commands:"))
        .skip(1)
        .take_while(|l| !l.is_empty())
        .filter_map(|l| l.split_whitespace().next().map(String::from))
        .collect();
    assert!(known.contains("run") && known.contains("start"));
    for (name, text) in docs() {
        for span in code_spans(&text) {
            let mut words = span.split_whitespace();
            let (Some("a2e"), Some(cmd)) = (words.next(), words.next()) else {
                continue;
            };
            if cmd.starts_with('-') || cmd.starts_with('<') || cmd.starts_with('[') {
                continue;
            }
            assert!(
                known.contains(cmd),
                "{name} names the command `a2e {cmd}`, which does not exist"
            );
        }
    }
}

#[test]
fn every_flag_in_cli_md_exists_in_some_help_text() {
    let mut all = help(&[]);
    for cmd in [
        "run", "plan", "start", "doctor", "upload", "cases", "sandbox", "models", "tools",
    ] {
        all.push_str(&help(&[cmd]));
    }
    for sub in [
        ["cases", "validate"],
        ["cases", "list"],
        ["sandbox", "init"],
        ["sandbox", "verify"],
        ["sandbox", "info"],
    ] {
        all.push_str(&help(&sub));
    }
    let text = std::fs::read_to_string(repo().join("docs/CLI.md")).unwrap();
    for span in code_spans(&text) {
        for word in span.split_whitespace().filter(|w| w.starts_with("--")) {
            let flag = word.trim_end_matches([',', '.', ']', ')']);
            assert!(
                all.contains(flag),
                "docs/CLI.md names {flag}, which no help text lists"
            );
        }
    }
}

#[test]
fn environment_variables_in_the_docs_are_read_by_the_code() {
    let mut source = String::new();
    let mut stack = vec![repo().join("src")];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "rs") {
                source.push_str(&std::fs::read_to_string(path).unwrap());
            }
        }
    }
    for (name, text) in docs() {
        for span in code_spans(&text) {
            for word in span.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_')) {
                if word.starts_with("A2E_") && word.len() > 4 {
                    assert!(
                        source.contains(word),
                        "{name} names {word}, which the code does not read"
                    );
                }
            }
        }
    }
}

#[test]
fn file_paths_in_the_docs_exist() {
    // Folders and files that the program creates, or that live in the results project
    let created = [
        "results",
        "deps",
        "sandbox",
        "a2e-results",
        "target",
        "models",
    ];
    let mut missing = Vec::new();
    for (name, text) in docs() {
        for span in code_spans(&text) {
            if span.contains(char::is_whitespace) || span.contains(['<', '*', '{']) {
                continue;
            }
            let path = span.trim_end_matches('/');
            let local = ["configs/", "src/", "docs/", "tests/"]
                .iter()
                .any(|p| path.starts_with(p));
            if !local || created.iter().any(|c| path.starts_with(c)) {
                continue;
            }
            if path == "configs/local.toml" {
                continue;
            }
            if !repo().join(path).exists() {
                missing.push(format!("{name} names the path {path}"));
            }
        }
    }
    assert!(
        missing.is_empty(),
        "paths that do not exist:\n{}",
        missing.join("\n")
    );
}

#[test]
fn every_setting_is_documented_in_config_md() {
    let text = std::fs::read_to_string(repo().join("docs/CONFIG.md")).unwrap();
    let defaults = serde_json::to_value(a2e::settings::Settings::default()).unwrap();
    for (section, keys) in defaults.as_object().unwrap() {
        assert!(
            text.contains(&format!("## {section}")),
            "docs/CONFIG.md has no section {section}"
        );
        for key in keys.as_object().unwrap().keys() {
            assert!(
                text.contains(&format!("`{key}`")),
                "docs/CONFIG.md does not document {section}.{key}"
            );
        }
    }
}
