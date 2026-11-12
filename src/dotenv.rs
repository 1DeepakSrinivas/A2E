// dotenv.rs
// Loads KEY=VALUE pairs from .env files into the process environment
// Variables already set in the environment take precedence over the file

use anyhow::{Context, Result};
use std::path::Path;

/// Parses `KEY=VALUE` lines, skipping blanks, comments and malformed lines.
pub fn parse(text: &str) -> Vec<(String, String)> {
    text.lines()
        .filter_map(|line| {
            let line = line.trim();
            let line = line.strip_prefix("export ").unwrap_or(line).trim_start();
            let (key, value) = line.split_once('=')?;
            let key = key.trim();
            if key.is_empty() || key.starts_with('#') || key.contains(char::is_whitespace) {
                return None;
            }
            Some((key.to_owned(), unquote(value.trim())))
        })
        .collect()
}

fn unquote(value: &str) -> String {
    for q in ['"', '\''] {
        if let Some(inner) = value
            .strip_prefix(q)
            .and_then(|v| v.find(q).map(|end| &v[..end]))
        {
            return inner.to_owned();
        }
    }
    match value.find(" #") {
        Some(i) => value[..i].trim_end().to_owned(),
        None => value.to_owned(),
    }
}

/// Sets every variable in the file that is not already set. A missing file is not an error.
/// Call before any thread starts, because it changes the process environment.
pub fn load(path: &Path) -> Result<()> {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
    };
    set_missing(&text, |_| true);
    Ok(())
}

/// Sets only the listed variables from the file when they are not already set. The file must exist.
pub fn load_only(path: &Path, keys: &[&str]) -> Result<()> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    set_missing(&text, |key| keys.contains(&key));
    Ok(())
}

/// Variables the runner takes from the file named by `A2E_RESULTS_ENV`.
const RESULTS_KEYS: [&str; 2] = ["WORKER_URL", "WRITE_TOKEN"];

pub(crate) fn local_only() -> bool {
    std::env::var_os("A2E_LOCAL").is_some_and(|value| value == "1")
}

/// Loads `path` and, outside local mode, the shared Worker credentials.
pub fn load_startup(path: &Path) -> Result<()> {
    load(path)?;
    if let Some(shared) = std::env::var_os("A2E_RESULTS_ENV").filter(|_| !local_only()) {
        load_only(Path::new(&shared), &RESULTS_KEYS)?;
    }
    Ok(())
}

fn set_missing(text: &str, wanted: impl Fn(&str) -> bool) {
    for (key, value) in parse(text) {
        if wanted(&key) && std::env::var_os(&key).is_none() {
            std::env::set_var(key, value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kv(k: &str, v: &str) -> (String, String) {
        (k.to_owned(), v.to_owned())
    }

    #[test]
    fn parses_plain_quoted_and_exported_values() {
        let got = parse("A=1\nexport B = two\nC=\"three words\"\nD='4'\n");
        assert_eq!(
            got,
            vec![
                kv("A", "1"),
                kv("B", "two"),
                kv("C", "three words"),
                kv("D", "4")
            ]
        );
    }

    #[test]
    fn skips_comments_blanks_and_malformed_lines() {
        let got = parse("# note\n\n=x\nno equals\nbad key=1\nE=5 # trailing\nF=\"a # b\"\n");
        assert_eq!(got, vec![kv("E", "5"), kv("F", "a # b")]);
    }

    #[test]
    fn existing_environment_wins_and_missing_file_is_ok() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join(".env");
        std::fs::write(
            &file,
            "A2E_DOTENV_TEST_SET=old\nA2E_DOTENV_TEST_NEW=fresh\n",
        )
        .unwrap();
        std::env::set_var("A2E_DOTENV_TEST_SET", "kept");
        load(&file).unwrap();
        assert_eq!(std::env::var("A2E_DOTENV_TEST_SET").unwrap(), "kept");
        assert_eq!(std::env::var("A2E_DOTENV_TEST_NEW").unwrap(), "fresh");
        load(&dir.path().join("absent.env")).unwrap();
    }

    #[test]
    fn load_only_takes_the_listed_keys_and_needs_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("shared.env");
        std::fs::write(
            &file,
            "A2E_DOTENV_ONLY_WANTED=yes\nA2E_DOTENV_ONLY_SKIPPED=no\n",
        )
        .unwrap();
        load_only(&file, &["A2E_DOTENV_ONLY_WANTED"]).unwrap();
        assert_eq!(std::env::var("A2E_DOTENV_ONLY_WANTED").unwrap(), "yes");
        assert!(std::env::var_os("A2E_DOTENV_ONLY_SKIPPED").is_none());
        assert!(load_only(&dir.path().join("absent.env"), &[]).is_err());
    }
}
